"""
VGG-11 Multi-Strategy Pruning with PTQ (CIFAR-10)
Standalone version - no dependencies on core/, config/, genetic_algorithm/

This script:
1. Generates 10 different pruning strategies from architecture-aware ranges
2. Applies pruning and fine-tuning to each strategy
3. Applies Post-Training Quantization (PTQ) to all strategies
4. Compares all strategies against baselines (Random, MAG, SNIP, GraSP) with PTQ
5. Provides statistical analysis of multi-strategy performance
"""

import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import numpy as np
import copy
import csv
import os
from typing import List, Dict, Tuple
import random
import sys
from thop import profile, clever_format
import argparse

# Add path for benchmarking imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pruning baselines
from benchmarking.unstructured.classical.magnitude import MagnitudePruning
from benchmarking.unstructured.classical.random import RandomPruning
from benchmarking.unstructured.y2019.snip import SNIPPruning, GraSPPruning

# Configuration
DEFAULT_BATCH_SIZE = 128
DEFAULT_FINE_TUNE_EPOCHS = 20
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description='VGG-11 Multi-Strategy Pruning on CIFAR-10')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained VGG-11 checkpoint (.pt or .pth file)')
    parser.add_argument('--score_dir', type=str, required=True,
                        help='Directory containing sensitivity scores')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to download/store CIFAR-10 dataset (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for training/evaluation (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_FINE_TUNE_EPOCHS,
                        help=f'Fine-tuning epochs (default: {DEFAULT_FINE_TUNE_EPOCHS})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda if available)')
    return parser.parse_args()

def load_data(data_path, batch_size):
    """Load CIFAR-10 data"""
    print("Loading CIFAR-10 data...")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

def calculate_sparsity(model):
    """Calculate overall model sparsity"""
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    return 100.0 * zero_params / total_params if total_params > 0 else 0.0

def get_layer_info(model, score_dir_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Get prunable layer names and parameter counts (sensitivity-ordered)"""
    model_state_dict = model.state_dict()
    layer_params = {}

    # Get all prunable layers (conv and fc layers)
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

    # Order by average sensitivity (ascending = least sensitive first = prune more)
    avg_sensitivities = []
    for name in layer_names_with_scores:
        score_file = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        try:
            scores = []
            with open(score_file, 'r') as f:
                next(f)  # Skip header
                for line_idx, line in enumerate(f):
                    if line_idx >= 1000:  # Sample first 1000
                        break
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        scores.append(float(parts[1]))

            if len(scores) > 0:
                avg_score = np.mean(scores)
                avg_sensitivities.append((name, avg_score))
            else:
                avg_sensitivities.append((name, 0.0))
        except Exception as e:
            print(f"    WARNING: Error reading {name}: {e}")
            avg_sensitivities.append((name, 0.0))

    # Sort by sensitivity (ascending)
    avg_sensitivities.sort(key=lambda x: x[1])
    layer_names = [name for name, _ in avg_sensitivities]

    # Get parameter counts
    for name in layer_names:
        layer_params[name] = model_state_dict[name].numel()

    return layer_names, layer_params

def compute_vgg_sparsity_ranges(layer_names: List[str], layer_params: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    """Compute layer pruning ranges using VGG architecture-aware rules"""
    layer_ranges = {}

    for layer_name in layer_names:
        param_count = layer_params.get(layer_name, 0)

        # Rule 1: Batch norm layers (if any) - no pruning
        is_batchnorm = (
            any(x in layer_name.lower() for x in ['bn', 'batchnorm', 'norm']) or
            (param_count <= 2048 and param_count in [64, 128, 256, 512, 1024, 2048])
        )
        if is_batchnorm:
            layer_ranges[layer_name] = (0.0, 0.0)
            continue

        # Rule 2: Small layers (< 10K parameters) - conservative
        if param_count < 10000:
            layer_ranges[layer_name] = (10.0, 40.0)
            continue

        # Rule 3: Classifier/FC layers - conservative (important for final decision)
        is_fc = any(x in layer_name.lower() for x in ['fc', 'classifier', 'linear', 'head'])
        if is_fc:
            layer_ranges[layer_name] = (30.0, 70.0)
            continue

        # Rule 4: Conv layers - aggressive
        is_conv = 'features' in layer_name.lower() or 'conv' in layer_name.lower()
        if is_conv:
            # Early layers (smaller param count) - more conservative
            if param_count < 100000:
                layer_ranges[layer_name] = (40.0, 80.0)
            # Middle layers
            elif param_count < 500000:
                layer_ranges[layer_name] = (50.0, 85.0)
            # Deep layers (larger param count) - most aggressive
            else:
                layer_ranges[layer_name] = (60.0, 90.0)
        else:
            # Default for any other layer type
            layer_ranges[layer_name] = (20.0, 60.0)

    return layer_ranges

def apply_pruning_mask(model, strategy, layer_names, layer_params, score_dir, debug=False):
    """Apply pruning based on sensitivity scores"""
    masks = {}

    for layer_idx, layer_name in enumerate(layer_names):
        prune_pct = strategy[layer_idx]

        if prune_pct <= 0.0:
            param_tensor = dict(model.named_parameters())[layer_name]
            masks[layer_name] = torch.ones_like(param_tensor)
            continue

        score_file = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')
        if not os.path.exists(score_file):
            param_tensor = dict(model.named_parameters())[layer_name]
            masks[layer_name] = torch.ones_like(param_tensor)
            continue

        # Read sensitivity scores
        total_weights = layer_params[layer_name]
        scores = []
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    score = float(parts[1])
                    if idx < total_weights:
                        scores.append((idx, score))

        # Sort by sensitivity (ascending = prune least sensitive first)
        scores.sort(key=lambda x: x[1])
        num_to_prune = int(total_weights * (prune_pct / 100.0))

        # Create mask
        mask = torch.ones(total_weights)
        for i in range(min(num_to_prune, len(scores))):
            idx = scores[i][0]
            if idx < total_weights:
                mask[idx] = 0

        # Apply mask
        param_tensor = dict(model.named_parameters())[layer_name]
        mask = mask.view(param_tensor.shape).to(param_tensor.device)

        with torch.no_grad():
            param_tensor.mul_(mask)

        masks[layer_name] = mask

    return masks

def fine_tune_model(model, masks, train_loader, val_loader, device, epochs=20):
    """Fine-tune pruned model with early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0
    patience = 3
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Enforce sparsity after gradient update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        mask = masks[name]
                        if mask.numel() == param.numel():
                            param.data.mul_(mask.view(param.shape).to(param.device))

            optimizer.step()

        # Validation
        model.eval()
        val_accuracy = evaluate_model(model, val_loader, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_accuracy

def create_strategy_variant(
    layer_names: List[str],
    layer_params: Dict[str, int],
    layer_ranges: Dict[str, Tuple[float, float]],
    strategy_type: str,
    target_sparsity: float = None
) -> Tuple[List[float], str]:
    """Create a pruning strategy by sampling from layer ranges."""
    strategy = []

    for layer_name in layer_names:
        if layer_name in layer_ranges:
            min_range, max_range = layer_ranges[layer_name]
        else:
            min_range, max_range = 0.0, 10.0

        if strategy_type == "max_aggressive":
            prune_pct = max_range
        elif strategy_type == "min_conservative":
            prune_pct = min_range
        elif strategy_type == "balanced":
            prune_pct = (min_range + max_range) / 2.0
        elif strategy_type == "random":
            prune_pct = np.random.uniform(min_range, max_range)
        elif strategy_type == "lower_30th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.3
        elif strategy_type == "middle_50th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.5
        elif strategy_type == "upper_70th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.7
        elif strategy_type == "upper_90th_percentile":
            prune_pct = min_range + (max_range - min_range) * 0.9
        else:
            prune_pct = max_range

        strategy.append(prune_pct)

    description = f"{strategy_type}"
    if target_sparsity:
        description += f" (target: {target_sparsity:.1f}%)"

    return strategy, description

def apply_ptq(model, val_loader, device):
    """Apply Post-Training Quantization (PTQ)"""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with validation data
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            if i >= 10:  # Use 10 batches for calibration
                break
            model(data.to(device))

    torch.quantization.convert(model, inplace=True)
    return model

def main():
    args = parse_args()

    print("=" * 80)
    print("VGG-11 Multi-Strategy Pruning with PTQ (CIFAR-10)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score Directory: {args.score_dir}")
    print("=" * 80)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader = load_data(args.data_path, args.batch_size)

    # Load model
    print(f"\nLoading pretrained VGG-11 from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at: {args.checkpoint}\n"
            f"Please provide a pretrained VGG-11 checkpoint.\n"
            f"You can:\n"
            f"  1. Train your own VGG-11 on CIFAR-10\n"
            f"  2. Use timm pretrained: python -c 'import timm; m=timm.create_model(\"vgg11_bn\", pretrained=True, num_classes=10); import torch; torch.save(m.state_dict(), \"{args.checkpoint}\")'"
        )

    # Create model
    base_model = timm.create_model('vgg11_bn', pretrained=False, num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    base_model.load_state_dict(state_dict, strict=False)
    base_model = base_model.to(device)

    initial_acc = evaluate_model(base_model, val_loader, device)
    print(f"Initial FP32 Accuracy: {initial_acc:.2f}%")

    # Check score directory
    if not os.path.exists(args.score_dir):
        raise FileNotFoundError(
            f"Sensitivity score directory not found: {args.score_dir}\n"
            f"Please run the sensitivity score generation first:\n"
            f"  python vgg_sensitivity_simple.py --checkpoint {args.checkpoint} --score_dir {args.score_dir}"
        )

    # Get layer information
    layer_names, layer_params = get_layer_info(base_model, args.score_dir)
    layer_ranges = compute_vgg_sparsity_ranges(layer_names, layer_params)

    print(f"\nFound {len(layer_names)} prunable layers with sensitivity scores")

    # Define strategies to test
    strategy_types = [
        "max_aggressive", "min_conservative", "balanced",
        "lower_30th_percentile", "middle_50th_percentile",
        "upper_70th_percentile", "upper_90th_percentile",
        "random", "random", "random"  # 3 random variants for diversity
    ]

    results = []

    print("\n" + "=" * 80)
    print("MULTI-STRATEGY PRUNING EVALUATION")
    print("=" * 80)

    for idx, strategy_type in enumerate(strategy_types, 1):
        print(f"\nStrategy {idx}/{len(strategy_types)}: {strategy_type}")

        # Create strategy
        strategy, desc = create_strategy_variant(layer_names, layer_params, layer_ranges, strategy_type)

        # Create a fresh model copy
        model = copy.deepcopy(base_model)

        # Apply pruning
        masks = apply_pruning_mask(model, strategy, layer_names, layer_params, args.score_dir)

        # Calculate sparsity
        sparsity = calculate_sparsity(model)
        print(f"  Sparsity: {sparsity:.2f}%")

        # Fine-tune
        ft_acc = fine_tune_model(model, masks, train_loader, val_loader, device, epochs=args.epochs)
        print(f"  Fine-tuned FP32 Accuracy: {ft_acc:.2f}%")

        # Apply PTQ
        model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
        int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
        print(f"  INT8 Accuracy: {int8_acc:.2f}%")

        results.append({
            'strategy': desc,
            'sparsity': sparsity,
            'ft_acc': ft_acc,
            'int8_acc': int8_acc,
            'drop': initial_acc - int8_acc
        })

    # Baseline comparisons
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (with PTQ)")
    print("=" * 80)

    # Target sparsity for baselines (use mean of multi-strategy)
    target_sparsity = np.mean([r['sparsity'] for r in results])
    print(f"\nTarget sparsity for baselines: {target_sparsity:.2f}%")

    baseline_results = []

    # Random Pruning
    print("\nBaseline 1: Random Pruning")
    model = copy.deepcopy(base_model)
    random_pruner = RandomPruning(model, train_loader, val_loader, device)
    model = random_pruner.prune(target_sparsity / 100.0)
    sparsity = calculate_sparsity(model)
    fp32_acc = evaluate_model(model, val_loader, device)
    model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
    int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
    print(f"  Sparsity: {sparsity:.2f}% | FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}%")
    baseline_results.append({'name': 'Random', 'sparsity': sparsity, 'fp32': fp32_acc, 'int8': int8_acc})

    # Magnitude Pruning
    print("\nBaseline 2: Magnitude Pruning")
    model = copy.deepcopy(base_model)
    mag_pruner = MagnitudePruning(model, train_loader, val_loader, device)
    model = mag_pruner.prune(target_sparsity / 100.0)
    sparsity = calculate_sparsity(model)
    fp32_acc = evaluate_model(model, val_loader, device)
    model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
    int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
    print(f"  Sparsity: {sparsity:.2f}% | FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}%")
    baseline_results.append({'name': 'MAG', 'sparsity': sparsity, 'fp32': fp32_acc, 'int8': int8_acc})

    # SNIP
    print("\nBaseline 3: SNIP")
    model = copy.deepcopy(base_model)
    snip_pruner = SNIPPruning(model, train_loader, val_loader, device)
    model = snip_pruner.prune(target_sparsity / 100.0)
    sparsity = calculate_sparsity(model)
    fp32_acc = evaluate_model(model, val_loader, device)
    model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
    int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
    print(f"  Sparsity: {sparsity:.2f}% | FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}%")
    baseline_results.append({'name': 'SNIP', 'sparsity': sparsity, 'fp32': fp32_acc, 'int8': int8_acc})

    # GraSP
    print("\nBaseline 4: GraSP")
    model = copy.deepcopy(base_model)
    grasp_pruner = GraSPPruning(model, train_loader, val_loader, device)
    model = grasp_pruner.prune(target_sparsity / 100.0)
    sparsity = calculate_sparsity(model)
    fp32_acc = evaluate_model(model, val_loader, device)
    model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
    int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
    print(f"  Sparsity: {sparsity:.2f}% | FP32: {fp32_acc:.2f}% | INT8: {int8_acc:.2f}%")
    baseline_results.append({'name': 'GraSP', 'sparsity': sparsity, 'fp32': fp32_acc, 'int8': int8_acc})

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\nMulti-Strategy Results:")
    print(f"{'Strategy':<30} | {'Sparsity':<10} | {'FT Acc':<10} | {'INT8 Acc':<10} | {'Drop':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['strategy']:<30} | {res['sparsity']:<10.2f} | {res['ft_acc']:<10.2f} | {res['int8_acc']:<10.2f} | {res['drop']:<10.2f}")

    print("\nMulti-Strategy Statistics:")
    int8_accs = [r['int8_acc'] for r in results]
    drops = [r['drop'] for r in results]
    print(f"  Mean INT8 Accuracy: {np.mean(int8_accs):.2f}% ± {np.std(int8_accs):.2f}%")
    print(f"  Mean Accuracy Drop: {np.mean(drops):.2f}% ± {np.std(drops):.2f}%")
    print(f"  Best INT8 Accuracy: {np.max(int8_accs):.2f}%")
    print(f"  Worst INT8 Accuracy: {np.min(int8_accs):.2f}%")

    print("\nBaseline Results:")
    print(f"{'Method':<15} | {'Sparsity':<10} | {'FP32 Acc':<10} | {'INT8 Acc':<10}")
    print("-" * 60)
    for res in baseline_results:
        print(f"{res['name']:<15} | {res['sparsity']:<10.2f} | {res['fp32']:<10.2f} | {res['int8']:<10.2f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
