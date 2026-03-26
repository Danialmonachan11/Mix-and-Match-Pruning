"""
ResNet-18 Multi-Strategy Pruning with PTQ

This script:
1. Generates 10 different pruning strategies from architecture-aware ranges
2. Applies pruning and fine-tuning to each strategy
3. Applies Post-Training Quantization (PTQ) to all strategies
4. Compares all strategies against baselines (Random, MAG, SNIP, GraSP) with PTQ
5. Provides statistical analysis of multi-strategy performance
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.quantization
import copy
import numpy as np
import random
import sys
from typing import List, Dict, Tuple
import csv
import json
from thop import profile, clever_format

# Add path for benchmarking imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from resnet_layer_classifier import (
    classify_resnet_layer,
    extract_resnet_stage,
    compute_resnet_sparsity_ranges,
    get_resnet_sensitivity_ordering,
    print_resnet_layer_analysis
)

# Pruning baselines
from benchmarking.unstructured.classical.magnitude import MagnitudePruning
from benchmarking.unstructured.classical.random import RandomPruning
from benchmarking.unstructured.y2019.snip import SNIPPruning, GraSPPruning

# Default Configuration (can be overridden by command-line arguments)
DEFAULT_BATCH_SIZE = 32
DEFAULT_FINE_TUNE_EPOCHS = 30
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Dataset for Proper Test Set Mapping
class GTSRBTestRemap(Dataset):
    def __init__(self, csv_file, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.idx_map = {int(cls): idx for cls, idx in class_to_idx.items()}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])
        label = self.idx_map[row['ClassId']]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_transforms():
    """Get data transforms for GTSRB"""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_train, transform_test

def load_gtsrb_data(data_path, batch_size):
    """Load GTSRB dataset"""
    train_folder = os.path.join(data_path, 'Train')
    test_csv = os.path.join(data_path, 'Test.csv')

    if not os.path.exists(train_folder):
        raise FileNotFoundError(
            f"GTSRB Train folder not found at: {train_folder}\n"
            f"Please download GTSRB dataset from: https://benchmark.ini.rub.de/gtsrb_dataset.html\n"
            f"Extract to: {data_path}"
        )

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"GTSRB Test.csv not found at: {test_csv}")

    transform_train, transform_test = get_data_transforms()

    train_dataset = datasets.ImageFolder(train_folder, transform=transform_train)
    print("ImageFolder class_to_idx mapping:", train_dataset.class_to_idx)

    test_dataset = GTSRBTestRemap(test_csv, data_path, train_dataset.class_to_idx, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# Model Loading
def load_model(checkpoint_path, device):
    """Load ResNet-18 model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 43)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

# Pruning/Compression Utility Functions As-Routine
def calculate_sparsity(model):
    total_params, zero_params = 0, 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    return 100.0 * zero_params / total_params if total_params > 0 else 0.0

def calculate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """Calculate FLOPs for a model"""
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    input_tensor = torch.randn(input_size).to(device)
    
    try:
        flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        return flops, flops_str
    except Exception as e:
        print(f"  Warning: FLOPs calculation failed: {e}")
        return 0, "N/A"

def get_resnet_layer_info(model, score_dir_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Get prunable layer names and parameter counts (sensitivity-ordered)"""
    model_state_dict = model.state_dict()
    layer_params = {}

    # Get all prunable layers (weights only, exclude biases)
    available_layers = [
        name for name in model_state_dict.keys()
        if 'weight' in name and 'bias' not in name and len(model_state_dict[name].shape) >= 2
    ]

    # Filter to only layers that have score files
    layer_names_with_scores = []
    for name in available_layers:
        score_file = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        if os.path.exists(score_file):
            layer_names_with_scores.append(name)

    # Use sensitivity-based ordering
    layer_names = get_resnet_sensitivity_ordering(layer_names_with_scores)

    # Get parameter counts
    for name in layer_names:
        layer_params[name] = model_state_dict[name].numel()

    return layer_names, layer_params

def print_layer_sensitivity_ordering(layer_names, score_dir):
    """Print layers ordered by average sensitivity (least to most sensitive)"""
    avg_sensitivities = []

    for layer_name in layer_names:
        score_file = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')

        if not os.path.exists(score_file):
            continue

        # Read all scores and compute average
        scores = []
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    score = float(parts[1])
                    scores.append(score)

        if len(scores) > 0:
            avg_score = np.mean(scores)
            avg_sensitivities.append((layer_name, avg_score))

    # Sort by average sensitivity (ascending)
    avg_sensitivities.sort(key=lambda x: x[1])

    print("Layer ordering (least -> most sensitive):")
    for idx, (layer_name, avg_score) in enumerate(avg_sensitivities, 1):
        print(f"  {idx}. {layer_name}: avg_sensitivity={avg_score:.6e}")

def analyze_layer_sensitivity_for_strategic_pruning(layer_names, score_dir, debug=True):
    """
    Analyze sensitivity score distributions for each layer.
    Returns per-layer statistics for strategic pruning decisions.
    """
    if debug:
        print("Analyzing layer sensitivity for strategic pruning...")
        print(f"WARNING: Analyzing {len(layer_names)} layers from score files (may be incomplete)")

    layer_stats = {}

    for layer_name in layer_names:
        score_file = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')

        if not os.path.exists(score_file):
            if debug:
                print(f"  {layer_name}: No score file found, skipping")
            continue

        # Read all scores
        scores = []
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    score = float(parts[1])
                    scores.append(score)

        if len(scores) == 0:
            continue

        # Compute statistics
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()

        # Normalize to 0-1 range if needed
        if max_score > min_score:
            normalized_scores = (scores_array - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores_array)

        # Calculate sensitive ratio (weights with normalized score > 0.5)
        sensitive_count = np.sum(normalized_scores > 0.5)
        sensitive_ratio = sensitive_count / len(scores)

        # Count unique values
        unique_count = len(np.unique(scores_array))

        # Store statistics
        layer_stats[layer_name] = {
            'sensitive_ratio': sensitive_ratio,
            'min_score': min_score,
            'max_score': max_score,
            'unique_count': unique_count,
            'total_weights': len(scores)
        }

        if debug:
            print(f"  {layer_name}: sensitive_ratio={sensitive_ratio:.3f}, range={0.000:.3f}-{1.000:.3f}, variation={unique_count} unique values")

    return layer_stats

def apply_pruning_mask(model, strategy, layer_names, layer_params, score_dir, debug=False):
    """Apply pruning based on sensitivity scores"""
    masks = {}

    for layer_idx, layer_name in enumerate(layer_names):
        prune_pct = strategy[layer_idx]

        if prune_pct <= 0.0:
            param_tensor = dict(model.named_parameters())[layer_name]
            masks[layer_name] = torch.ones_like(param_tensor)
            continue

        # Load sensitivity scores
        score_file = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')

        if not os.path.exists(score_file):
            param_tensor = dict(model.named_parameters())[layer_name]
            masks[layer_name] = torch.ones_like(param_tensor)
            if debug:
                print(f"  WARNING: No sensitivity scores for {layer_name}, skipping pruning")
            continue

        # Read scores
        scores = []
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    score = float(parts[1])
                    scores.append((idx, score))

        # Sort by sensitivity (lower = less sensitive = prune first)
        scores.sort(key=lambda x: x[1])

        # Calculate number to prune
        total_weights = layer_params[layer_name]
        num_to_prune = int(total_weights * (prune_pct / 100.0))

        if debug:
            print(f"  DEBUG Layer {layer_name}: total={total_weights}, pruning {num_to_prune} ({prune_pct:.2f}%)")

        # Create mask
        mask = torch.ones(total_weights)
        for i in range(min(num_to_prune, len(scores))):
            idx = scores[i][0]
            if idx < total_weights:
                mask[idx] = 0

        # Reshape mask to match parameter shape
        param_tensor = dict(model.named_parameters())[layer_name]
        mask = mask.view(param_tensor.shape).to(param_tensor.device)

        # Apply mask
        with torch.no_grad():
            param_tensor.mul_(mask)

        masks[layer_name] = mask

    return masks

def fine_tune_model(model, masks, train_loader, test_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    print("  Fine-tuning...")
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.data.mul_(masks[name].to(param.device))
            optimizer.step()
        test_acc = evaluate_model(model, test_loader, device)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: {test_acc:.2f}%")
        if patience_counter >= 3:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    if best_model_state:
        model.load_state_dict(best_model_state)
    return best_accuracy

def create_strategy_variant(
    layer_names: List[str],
    layer_params: Dict[str, int],
    layer_ranges: Dict[str, Tuple[float, float]],
    strategy_type: str,
    target_sparsity: float = 80.0
) -> Tuple[List[float], str]:
    """
    Create a pruning strategy by sampling from layer ranges.
    Strategy types:
    1. "max_aggressive" - Start at MAX of each range
    2. "min_conservative" - Start at MIN of each range
    3. "balanced" - Start at MIDDLE of each range
    4. "fc_heavy" - Max FC, min conv
    5. "late_stage_aggressive" - Max layer4, balanced others
    6. "lower_30th_percentile" - 30% position from min
    7. "middle_50th_percentile" - 50% position (median)
    8. "upper_70th_percentile" - 70% position from min
    9. "upper_90th_percentile" - 90% position near max
    10. "graduated" - Proportional to layer size
    """
    total_params = sum(layer_params.values())
    strategy = []

    for layer_name in layer_names:
        if layer_name in layer_ranges:
            min_range, max_range = layer_ranges[layer_name]
        else:
            min_range, max_range = 0.0, 10.0

        is_fc = 'fc' in layer_name.lower()
        stage = extract_resnet_stage(layer_name)

        if strategy_type == "max_aggressive":
            prune_pct = max_range
        elif strategy_type == "min_conservative":
            prune_pct = min_range
        elif strategy_type == "balanced":
            prune_pct = (min_range + max_range) / 2.0
        elif strategy_type == "fc_heavy":
            if is_fc:
                prune_pct = max_range
            else:
                prune_pct = min_range
        elif strategy_type == "late_stage_aggressive":
            if stage == 4:
                prune_pct = max_range
            else:
                prune_pct = (min_range + max_range) / 2.0
        elif strategy_type == "lower_30th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.3
        elif strategy_type == "middle_50th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.5
        elif strategy_type == "upper_70th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.7
        elif strategy_type == "upper_90th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.9
        elif strategy_type == "graduated":
            layer_size_ratio = layer_params[layer_name] / total_params
            ratio = min(1.0, layer_size_ratio * 10)
            prune_pct = min_range + (max_range - min_range) * ratio
        else:
            prune_pct = max_range

        strategy.append(prune_pct)

    # Adjust to hit target sparsity
    strategy = adjust_to_target_sparsity(
        strategy, layer_names, layer_params, total_params, target_sparsity
    )

    description = f"{strategy_type} targeting {target_sparsity:.1f}%"
    return strategy, description

def adjust_to_target_sparsity(
    strategy: List[float],
    layer_names: List[str],
    layer_params: Dict[str, int],
    total_params: int,
    target_sparsity: float,
    max_iterations: int = 30
) -> List[float]:
    """Iteratively adjust strategy to hit target sparsity"""
    target_pruned = total_params * (target_sparsity / 100.0)
    adjustment_rate = 0.2

    for iteration in range(max_iterations):
        # Calculate current sparsity
        current_pruned = sum(
            (strategy[i] / 100.0) * layer_params[layer_names[i]]
            for i in range(len(layer_names))
        )

        current_sparsity = (current_pruned / total_params) * 100.0
        error = target_pruned - current_pruned
        error_pct = abs((current_sparsity - target_sparsity) / target_sparsity) * 100.0

        if error_pct < 0.5:  # Within 0.5%
            break

        # Adjust all layers proportionally
        for i in range(len(strategy)):
            if error > 0:  # Need MORE pruning
                strategy[i] = min(98.0, strategy[i] + adjustment_rate)
            else:  # Need LESS pruning
                strategy[i] = max(0.0, strategy[i] - adjustment_rate)

    return strategy

def apply_baseline_pruning(
    baseline_name: str,
    checkpoint_path: str,
    target_sparsity: float,
    train_loader,
    test_loader,
    device
):
    """
    Apply baseline pruning method (MAG, SNIP, GraSP, or Random) and fine-tune
    Returns: (pruned_model, masks, fp32_accuracy, sparsity)
    """
    print(f"\nApplying {baseline_name} pruning at {target_sparsity:.2f}% sparsity...")

    # Load fresh baseline model from checkpoint
    baseline_model = torchvision.models.resnet18(pretrained=False)
    baseline_model.fc = torch.nn.Linear(baseline_model.fc.in_features, 43)
    baseline_model = baseline_model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])

    # Apply pruning method
    if baseline_name == "MAG":
        pruner = MagnitudePruning()
        baseline_model, masks = pruner.prune_model(baseline_model, target_sparsity, global_pruning=True)
    elif baseline_name == "SNIP":
        pruner = SNIPPruning()
        baseline_model, masks = pruner.prune_model(
            baseline_model, target_sparsity, train_loader, device, global_pruning=True
        )
    elif baseline_name == "GraSP":
        pruner = GraSPPruning()
        baseline_model, masks = pruner.prune_model(
            baseline_model, target_sparsity, train_loader, device, global_pruning=True
        )
    elif baseline_name == "Random":
        pruner = RandomPruning()
        baseline_model, masks = pruner.prune_model(baseline_model, target_sparsity, global_pruning=True)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Calculate actual sparsity
    actual_sparsity = calculate_sparsity(baseline_model)
    print(f"   {baseline_name} sparsity achieved: {actual_sparsity:.2f}%")

    # Fine-tune
    print(f"  Fine-tuning {baseline_name} model...")
    fp32_accuracy = fine_tune_model(baseline_model, masks, train_loader, test_loader, device, epochs=30)
    print(f"  {baseline_name} FP32 accuracy (after fine-tuning): {fp32_accuracy:.2f}%")

    return baseline_model, masks, fp32_accuracy, actual_sparsity

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='ResNet-18 Multi-Strategy Pruning on GTSRB')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained ResNet-18 checkpoint (.pth file)')
    parser.add_argument('--score_dir', type=str, required=True,
                        help='Directory containing sensitivity scores')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to GTSRB dataset directory')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for training/evaluation (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_FINE_TUNE_EPOCHS,
                        help=f'Fine-tuning epochs (default: {DEFAULT_FINE_TUNE_EPOCHS})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda if available)')
    parser.add_argument('--output_dir', type=str, default='pruned_models/resnet',
                        help='Directory to save pruned models (default: pruned_models/resnet)')
    return parser.parse_args()

def main():
    print("=" * 80)
    print("ResNet-18 Multi-Strategy Pruning with PTQ")
    print("=" * 80)
    print("\nGenerating 10 different strategies and comparing against baselines")
    print("All methods evaluated with PTQ (Post-Training Quantization)\n")

    # Parse arguments
    args = parse_args()

    # Setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Setup output directory and savepoint
    os.makedirs(args.output_dir, exist_ok=True)
    savepoint_file = os.path.join(args.output_dir, 'savepoint.json')
    savepoint = json.load(open(savepoint_file)) if os.path.exists(savepoint_file) else {'completed': []}

    # Load data
    print("\nLoading GTSRB dataset...")
    train_loader, test_loader = load_gtsrb_data(args.data_path, args.batch_size)

    # Load model
    print(f"\nLoading pretrained ResNet-18 from: {args.checkpoint}")
    base_model = load_model(args.checkpoint, device)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Baseline accuracy
    print("\nEvaluating baseline model on GTSRB test set...")
    baseline_accuracy = evaluate_model(base_model, test_loader, device)
    print(f"  Baseline accuracy: {baseline_accuracy:.2f}%")

    # Get layer information
    print(f"\nGetting layer information from: {args.score_dir}")
    if not os.path.exists(args.score_dir):
        raise FileNotFoundError(
            f"Sensitivity score directory not found: {args.score_dir}\n"
            f"Please run the sensitivity score generation first:\n"
            f"  python resnet_sensitivity_simple.py --checkpoint {args.checkpoint} --score_dir {args.score_dir}"
        )

    layer_names, layer_params = get_resnet_layer_info(base_model, args.score_dir)
    print_layer_sensitivity_ordering(layer_names, args.score_dir)
    print(f"Model has {len(layer_names)} prunable layers")

    # Compute ResNet-specific ranges
    print(f"\nComputing ResNet-specific layer ranges...")
    layer_ranges = compute_resnet_sparsity_ranges(layer_names, layer_params)

    # ========================================================================
    # PART 1: Generate and Evaluate 10 Multi-Strategies with PTQ
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Generating and Evaluating 10 Multi-Strategies with PTQ")
    print("=" * 80)

    strategy_types = [
        ("max_aggressive", 90.0),
        ("min_conservative", 90.0),
        ("balanced", 90.0),
        ("fc_heavy", 90.0),
        ("late_stage_aggressive", 88.0),
        ("lower_30th_percentile", 90.0),
        ("middle_50th_percentile", 90.0),
        ("upper_70th_percentile", 92.0),
        ("upper_90th_percentile", 92.0),
        ("graduated", 90.0),
    ]

    multi_strategy_results = []

    for strategy_id, (strategy_type, target_sparsity) in enumerate(strategy_types, 1):
        sp_key = f"strategy_{strategy_id}_{strategy_type}"
        print(f"\n{'='*80}")
        print(f"Strategy {strategy_id}/10: {strategy_type} (target: {target_sparsity:.1f}%)")
        print(f"{'='*80}")

        # Skip if already completed (savepoint resume)
        if sp_key in savepoint['completed']:
            print(f"  [SKIP] Already completed — loading saved result from {args.output_dir}")
            fp32_path = os.path.join(args.output_dir, f'{sp_key}_fp32.pth')
            if os.path.exists(fp32_path):
                ckpt = torch.load(fp32_path, map_location='cpu')
                multi_strategy_results.append({
                    'strategy_id': strategy_id, 'type': strategy_type,
                    'target_sparsity': target_sparsity,
                    'actual_sparsity': ckpt['actual_sparsity'],
                    'flops': ckpt.get('flops', 0), 'flops_str': ckpt.get('flops_str', 'N/A'),
                    'fp32_accuracy': ckpt['fp32_accuracy'], 'int8_accuracy': ckpt['int8_accuracy'],
                    'total_drop': baseline_accuracy - ckpt['int8_accuracy'],
                    'quantization_drop': ckpt['fp32_accuracy'] - ckpt['int8_accuracy']
                })
            continue

        # Analyze sensitivity for first strategy only (to avoid repetition)
        if strategy_id == 1:
            print_layer_sensitivity_ordering(layer_names, SCORE_DIR)
            layer_stats = analyze_layer_sensitivity_for_strategic_pruning(layer_names, SCORE_DIR, debug=True)

        # Create strategy
        strategy, description = create_strategy_variant(
            layer_names, layer_params, layer_ranges, strategy_type, target_sparsity
        )

        # Apply pruning (enable debug for first strategy)
        print(f"\nApplying pruning masks...")
        pruned_model = copy.deepcopy(base_model)
        debug_mode = (strategy_id == 1)
        masks = apply_pruning_mask(pruned_model, strategy, layer_names, layer_params, SCORE_DIR)
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"  Actual sparsity: {actual_sparsity:.2f}%")

        # Calculate FLOPs
        flops, flops_str = calculate_flops(pruned_model, input_size=(1, 3, 224, 224), device='cpu')
        print(f"  FLOPs: {flops_str}")

        # Fine-tuning
        print(f"\nFine-tuning...")
        final_fp32_accuracy = fine_tune_model(pruned_model, masks, train_loader, test_loader, device, epochs=args.epochs)
        print(f"  FP32 Accuracy (after fine-tuning): {final_fp32_accuracy:.2f}%")

        # Apply PTQ
        print(f"\nApplying Post-Training Quantization...")
        pruned_model_cpu = pruned_model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            pruned_model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        # Evaluate quantized model
        int8_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
        total_drop = baseline_accuracy - int8_accuracy
        quantization_drop = final_fp32_accuracy - int8_accuracy
        print(f"  INT8 Accuracy: {int8_accuracy:.2f}%")
        print(f"  Total accuracy drop: {total_drop:.2f}%")

        # --- SAVE MODELS ---
        fp32_path = os.path.join(args.output_dir, f'{sp_key}_fp32.pth')
        int8_path = os.path.join(args.output_dir, f'{sp_key}_int8.pth')
        torch.save({
            'strategy_type': strategy_type, 'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity, 'flops': flops, 'flops_str': flops_str,
            'fp32_accuracy': final_fp32_accuracy, 'int8_accuracy': int8_accuracy,
            'state_dict': pruned_model.state_dict(), 'masks': masks
        }, fp32_path)
        torch.save(quantized_model, int8_path)
        print(f"  Saved FP32 → {fp32_path}")
        print(f"  Saved INT8 → {int8_path}")
        savepoint['completed'].append(sp_key)
        with open(savepoint_file, 'w') as f:
            json.dump(savepoint, f, indent=2)

        # Store results
        multi_strategy_results.append({
            'strategy_id': strategy_id,
            'type': strategy_type,
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'flops': flops,
            'flops_str': flops_str,
            'fp32_accuracy': final_fp32_accuracy,
            'int8_accuracy': int8_accuracy,
            'total_drop': total_drop,
            'quantization_drop': quantization_drop
        })

    # ========================================================================
    # PART 2: Evaluate Baselines with PTQ
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Evaluating Baselines with PTQ")
    print("=" * 80)

    # Use the mean sparsity from multi-strategies for fair comparison
    mean_sparsity = np.mean([r['actual_sparsity'] for r in multi_strategy_results])
    print(f"\nUsing mean sparsity from multi-strategies: {mean_sparsity:.2f}%")

    baseline_results = {}

    # Evaluate each baseline
    for baseline_name in ["MAG", "SNIP", "GraSP", "Random"]:
        baseline_model, baseline_masks, fp32_acc, actual_spar = apply_baseline_pruning(
            baseline_name, CHECKPOINT_PATH, mean_sparsity,
            train_loader, test_loader, device
        )

        # Calculate FLOPs
        flops, flops_str = calculate_flops(baseline_model, input_size=(1, 3, 224, 224), device='cpu')
        print(f"  {baseline_name} FLOPs: {flops_str}")

        # Apply PTQ to baseline
        print(f"\n  Applying PTQ to {baseline_name}...")
        baseline_model_cpu = baseline_model.cpu()
        baseline_quantized = torch.quantization.quantize_dynamic(
            baseline_model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        int8_acc = evaluate_model(baseline_quantized, test_loader, 'cpu')
        total_drop = baseline_accuracy - int8_acc
        print(f"  {baseline_name} INT8 accuracy: {int8_acc:.2f}%")
        print(f"  {baseline_name} total drop: {total_drop:.2f}%")

        baseline_results[baseline_name] = {
            'sparsity': actual_spar,
            'flops': flops,
            'flops_str': flops_str,
            'fp32_accuracy': fp32_acc,
            'int8_accuracy': int8_acc,
            'total_drop': total_drop
        }

    # ========================================================================
    # PART 3: Statistical Analysis of Multi-Strategies
    # ========================================================================
    print("\n" + "=" * 80)
    print("MULTI-STRATEGY STATISTICS (10 variants)")
    print("=" * 80)

    int8_accuracies = [r['int8_accuracy'] for r in multi_strategy_results]
    total_drops = [r['total_drop'] for r in multi_strategy_results]
    sparsities = [r['actual_sparsity'] for r in multi_strategy_results]

    mean_int8_acc = np.mean(int8_accuracies)
    std_int8_acc = np.std(int8_accuracies)
    mean_drop = np.mean(total_drops)
    std_drop = np.std(total_drops)
    mean_sparsity_final = np.mean(sparsities)

    best_strategy = max(multi_strategy_results, key=lambda x: x['int8_accuracy'])
    worst_strategy = min(multi_strategy_results, key=lambda x: x['int8_accuracy'])

    print(f"  Mean INT8 Accuracy: {mean_int8_acc:.2f}% +/- {std_int8_acc:.2f}%")
    print(f"  Mean Accuracy Drop: {mean_drop:.2f}% +/- {std_drop:.2f}%")
    print(f"  Mean Sparsity: {mean_sparsity_final:.2f}%")
    print(f"  Best INT8 Accuracy: {best_strategy['int8_accuracy']:.2f}% ({best_strategy['type']})")
    print(f"  Worst INT8 Accuracy: {worst_strategy['int8_accuracy']:.2f}% ({worst_strategy['type']})")

    # ========================================================================
    # PART 4: Final Comparison Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Multi-Strategy vs Baselines (with PTQ)")
    print("=" * 80)

    print(f"\n{'Method':<25} {'Sparsity':<12} {'FLOPs':<12} {'FP32 Acc':<12} {'INT8 Acc':<12} {'Total Drop':<12}")
    print("-" * 100)

    # Multi-Strategy (best)
    print(f"{'Multi-Strategy (Best)':<25} {best_strategy['actual_sparsity']:>10.2f}% "
          f"{best_strategy['flops_str']:>10} {best_strategy['fp32_accuracy']:>10.2f}% "
          f"{best_strategy['int8_accuracy']:>10.2f}% {best_strategy['total_drop']:>10.2f}%")

    # Baselines
    for baseline_name in ["MAG", "SNIP", "GraSP", "Random"]:
        res = baseline_results[baseline_name]
        method_name = f"{baseline_name}+PTQ"
        print(f"{method_name:<25} {res['sparsity']:>10.2f}% {res['flops_str']:>10} "
              f"{res['fp32_accuracy']:>10.2f}% {res['int8_accuracy']:>10.2f}% {res['total_drop']:>10.2f}%")

    # ========================================================================
    # PART 5: Detailed Multi-Strategy Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("DETAILED MULTI-STRATEGY RESULTS")
    print("=" * 80)

    print(f"\n{'ID':<4} {'Type':<25} {'Sparsity':<10} {'FLOPs':<12} {'FP32 Acc':<10} {'INT8 Acc':<10} {'Total Drop':<10}")
    print("-" * 100)
    for r in multi_strategy_results:
        print(f"{r['strategy_id']:<4} {r['type']:<25} {r['actual_sparsity']:>8.2f}% "
              f"{r['flops_str']:>10} {r['fp32_accuracy']:>8.2f}% {r['int8_accuracy']:>8.2f}% {r['total_drop']:>8.2f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Calculate improvements
    best_multi_int8 = best_strategy['int8_accuracy']
    mag_int8 = baseline_results['MAG']['int8_accuracy']
    snip_int8 = baseline_results['SNIP']['int8_accuracy']
    grasp_int8 = baseline_results['GraSP']['int8_accuracy']
    random_int8 = baseline_results['Random']['int8_accuracy']

    print(f"\n1. Multi-Strategy Performance:")
    print(f"   - Mean INT8 Accuracy: {mean_int8_acc:.2f}% +/- {std_int8_acc:.2f}%")
    print(f"   - Best INT8 Accuracy: {best_multi_int8:.2f}% ({best_strategy['type']})")
    print(f"   - Mean Sparsity: {mean_sparsity_final:.2f}%")

    print(f"\n2. Baseline Comparisons (at {mean_sparsity:.2f}% sparsity):")
    print(f"   - Best Multi-Strategy vs MAG: {best_multi_int8 - mag_int8:+.2f}%")
    print(f"   - Best Multi-Strategy vs SNIP: {best_multi_int8 - snip_int8:+.2f}%")
    print(f"   - Best Multi-Strategy vs GraSP: {best_multi_int8 - grasp_int8:+.2f}%")
    print(f"   - Best Multi-Strategy vs Random: {best_multi_int8 - random_int8:+.2f}%")

    print(f"\n3. Multi-Strategy Robustness:")
    print(f"   - All 10 variants tested with different sampling strategies")
    print(f"   - Standard deviation: {std_int8_acc:.2f}% (shows consistency)")
    print(f"   - Range: {worst_strategy['int8_accuracy']:.2f}% - {best_strategy['int8_accuracy']:.2f}%")

    print("\n" + "=" * 80)
    print("ResNet Multi-Strategy Pruning Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
