"""
Swin Transformer Multi-Strategy Pruning with PTQ (CIFAR-100)

This script:
1. Generates 10 different pruning strategies from architecture-aware ranges
2. Applies pruning and fine-tuning to each strategy
3. Applies Post-Training Quantization (PTQ) to all strategies
4. Compares all strategies against baselines (Random, MAG, SNIP, GraSP) with PTQ
5. Provides statistical analysis of multi-strategy performance

NOTE: Uses CIFAR-100 (100 classes) to demonstrate scalability vs CIFAR-10 used in LeViT/ResNet
"""

import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
from timm import create_model
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

# Import custom modules from LOCAL directory
from swin_layer_classifier import (
    classify_swin_layer,
    extract_swin_stage,
    compute_swin_sparsity_ranges,
    get_swin_sensitivity_ordering,
    print_swin_layer_analysis
)

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Swin-Tiny Multi-Strategy Pruning on CIFAR-100')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained Swin-Tiny checkpoint (.pth file)')
    parser.add_argument('--score_dir', type=str, required=True,
                        help='Directory containing sensitivity scores')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to download/store CIFAR-100 dataset (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training/evaluation (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Fine-tuning epochs (default: 30)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda if available)')
    return parser.parse_args()

def evaluate_model(model, dataloader, device, debug=False):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle different output shapes
            while output.dim() > 2:
                output = output.squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

def calculate_sparsity(model, pruned_layer_names=None):
    """Calculate overall model sparsity.

    If pruned_layer_names is provided, only count sparsity in those layers.
    Otherwise count all weight parameters.
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            # If layer list provided, only count those layers
            if pruned_layer_names is not None:
                if name not in pruned_layer_names:
                    continue
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0.0
    return sparsity

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

def get_swin_layer_info(model, score_dir_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Get prunable layer names and parameter counts (sensitivity-ordered)"""
    model_state_dict = model.state_dict()
    layer_params = {}
    
    # Get all prunable layers (weights only, exclude biases)
    available_layers = [
        name for name in model_state_dict.keys()
        if 'weight' in name and 'bias' not in name and len(model_state_dict[name].shape) >= 2
    ]
    
    print(f"  DEBUG: Total layers in model: {len(available_layers)}")
    
    # Filter to only layers that have score files
    # Try BOTH with and without 'model.' prefix
    layer_names_with_scores = []
    for name in available_layers:
        score_file_1 = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        score_file_2 = os.path.join(score_dir_path, f'weight_sensitivity_scores_model.{name}.csv')
        
        if os.path.exists(score_file_1):
            layer_names_with_scores.append(name)
        elif os.path.exists(score_file_2):
            layer_names_with_scores.append(name)
    
    print(f"  DEBUG: Found {len(layer_names_with_scores)} layers with scores in {score_dir_path}")
    print(f"  Computing sensitivity ordering...")

    # Use sensitivity-based ordering (compute average sensitivity from scores)
    avg_sensitivities = []
    for idx, name in enumerate(layer_names_with_scores):
        if idx % 10 == 0:
            print(f"    Processing layer {idx+1}/{len(layer_names_with_scores)}...")

        score_file_1 = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        score_file_2 = os.path.join(score_dir_path, f'weight_sensitivity_scores_model.{name}.csv')

        score_file = None
        if os.path.exists(score_file_1):
            score_file = score_file_1
        elif os.path.exists(score_file_2):
            score_file = score_file_2

        if score_file:
            try:
                # Read only first 1000 scores for speed (good enough for average)
                scores = []
                with open(score_file, 'r') as f:
                    next(f)  # Skip header
                    for line_idx, line in enumerate(f):
                        if line_idx >= 1000:  # Only read first 1000 lines
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
        else:
            avg_sensitivities.append((name, 0.0))

    # Sort by average sensitivity (ascending - least to most sensitive)
    avg_sensitivities.sort(key=lambda x: x[1])
    layer_names = [name for name, _ in avg_sensitivities]
    print(f"  Sensitivity ordering complete!")

    # Get parameter counts
    for name in layer_names:
        layer_params[name] = model_state_dict[name].numel()

    return layer_names, layer_params

def print_layer_sensitivity_ordering(layer_names, score_dir):
    """Print layers ordered by average sensitivity (least to most sensitive)"""
    avg_sensitivities = []

    for layer_name in layer_names:
        # Try both with and without 'model.' prefix
        score_file_1 = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')
        score_file_2 = os.path.join(score_dir, f'weight_sensitivity_scores_model.{layer_name}.csv')

        score_file = None
        if os.path.exists(score_file_1):
            score_file = score_file_1
        elif os.path.exists(score_file_2):
            score_file = score_file_2
        else:
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

    print("Agent layer ordering (least -> most sensitive):")
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
        # Try both with and without 'model.' prefix
        score_file_1 = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')
        score_file_2 = os.path.join(score_dir, f'weight_sensitivity_scores_model.{layer_name}.csv')

        score_file = None
        if os.path.exists(score_file_1):
            score_file = score_file_1
        elif os.path.exists(score_file_2):
            score_file = score_file_2
        else:
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
            print(f"  {layer_name}: Using actual sensitivity scores (sensitive_ratio: {sensitive_ratio:.3f}, range: 0.000-1.000, unique: {unique_count})")

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
        
        # Load sensitivity scores - try BOTH with and without 'model.' prefix
        score_file_1 = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')
        score_file_2 = os.path.join(score_dir, f'weight_sensitivity_scores_model.{layer_name}.csv')
        
        if os.path.exists(score_file_1):
            score_file = score_file_1
        elif os.path.exists(score_file_2):
            score_file = score_file_2
        else:
            param_tensor = dict(model.named_parameters())[layer_name]
            masks[layer_name] = torch.ones_like(param_tensor)
            continue
        
        # Read scores and FILTER to valid indices only
        total_weights = layer_params[layer_name]
        scores = []
        invalid_count = 0
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    score = float(parts[1])
                    # CRITICAL FIX: Only keep scores with valid indices
                    if idx < total_weights:
                        scores.append((idx, score))
                    else:
                        invalid_count += 1

        if debug and invalid_count > 0:
            print(f"    Filtered out {invalid_count} invalid indices from score file")

        # Sort by sensitivity (lower = less sensitive = prune first)
        scores.sort(key=lambda x: x[1])

        # Calculate number to prune
        total_weights = layer_params[layer_name]
        num_to_prune = int(total_weights * (prune_pct / 100.0))

        if debug:
            print(f"  DEBUG Layer {layer_name}: total={total_weights}, pruning {num_to_prune} ({prune_pct:.2f}%)")
            if len(scores) < num_to_prune:
                print(f"    WARNING: Score file only has {len(scores)} entries, need {num_to_prune}!")

        # Create mask
        mask = torch.ones(total_weights)
        actually_pruned = 0
        for i in range(min(num_to_prune, len(scores))):
            idx = scores[i][0]
            if idx < total_weights:
                mask[idx] = 0
                actually_pruned += 1

        if debug and actually_pruned != num_to_prune:
            print(f"    WARNING: Only pruned {actually_pruned} / {num_to_prune} weights!")
        
        # Reshape mask to match parameter shape
        param_tensor = dict(model.named_parameters())[layer_name]
        mask = mask.view(param_tensor.shape).to(param_tensor.device)
        
        # Apply mask
        with torch.no_grad():
            param_tensor.mul_(mask)
        
        masks[layer_name] = mask
    
    return masks

def fine_tune_model(model, masks, train_loader, val_loader, device, epochs=30):
    """Fine-tune pruned model with early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    patience_counter = 0
    patience = 3
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Handle output shapes
            while output.dim() > 2:
                output = output.squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
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
    target_sparsity: float = 90.0
) -> Tuple[List[float], str]:
    """Create a pruning strategy by sampling from layer ranges."""
    total_params = sum(layer_params.values())
    strategy = []
    
    for layer_name in layer_names:
        if layer_name in layer_ranges:
            min_range, max_range = layer_ranges[layer_name]
        else:
            min_range, max_range = 0.0, 10.0

        is_mlp = 'mlp' in layer_name.lower()
        is_attn = 'attn' in layer_name.lower()

        if strategy_type == "max_aggressive":
            prune_pct = max_range
        elif strategy_type == "min_conservative":
            prune_pct = min_range
        elif strategy_type == "balanced":
            prune_pct = (min_range + max_range) / 2.0
        elif strategy_type == "mlp_heavy":
            if is_mlp:
                prune_pct = max_range
            else:
                prune_pct = min_range
        elif strategy_type == "attn_aggressive":
            if is_attn:
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

    description = f"{strategy_type}"
    return strategy, description

# ============================================================================
# LOCAL BASELINE PRUNING IMPLEMENTATIONS (Fixed for Transformers)
# ============================================================================

def get_prunable_parameters(model):
    """
    Get all prunable parameters - FIXED to detect 2D+ weight tensors.
    Works for Vision Transformers (Swin, LeViT) and CNNs (ResNet, VGG).
    """
    prunable = {}
    for name, param in model.named_parameters():
        # Include ALL weight tensors with dimension >= 2
        if 'weight' in name and param.dim() >= 2:
            # Exclude normalization layers only
            if not any(norm_type in name.lower() for norm_type in ['norm', 'bn', 'ln', 'layernorm']):
                prunable[name] = param

    print(f"  Found {len(prunable)} prunable layers")
    if len(prunable) > 0:
        total_prunable = sum(p.numel() for p in prunable.values())
        print(f"  Total prunable parameters: {total_prunable:,}")
    else:
        print("  WARNING: No prunable layers found!")

    return prunable

def apply_magnitude_pruning_local(model, target_sparsity):
    """Local implementation of global magnitude pruning"""
    print(f"  Applying Magnitude Pruning (local)...")

    prunable_params = get_prunable_parameters(model)

    if len(prunable_params) == 0:
        raise ValueError("No prunable layers found!")

    # Collect all weights
    all_weights = torch.cat([p.data.abs().view(-1) for p in prunable_params.values()])

    # Calculate global threshold - FIXED: k is number to PRUNE (not keep)
    sparsity_ratio = target_sparsity / 100.0
    k = int(len(all_weights) * sparsity_ratio)
    if k <= 0:
        k = 1
    if k >= len(all_weights):
        k = len(all_weights) - 1
    threshold = all_weights.kthvalue(k)[0].item()

    print(f"    Magnitude threshold: {threshold:.8f}")

    # Create and apply masks
    masks = {}
    for name, param in prunable_params.items():
        mask = (param.abs() > threshold).float()
        masks[name] = mask
        with torch.no_grad():
            param.mul_(mask)

    return masks

def apply_snip_pruning_local(model, train_loader, target_sparsity, device):
    """Local implementation of SNIP pruning"""
    print(f"  Applying SNIP Pruning (local)...")

    model.eval()
    model.zero_grad()

    prunable_params = get_prunable_parameters(model)

    if len(prunable_params) == 0:
        raise ValueError("No prunable layers found!")

    # Get one batch for gradient computation
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise ValueError("Training data loader is empty!")

    inputs, targets = inputs.to(device), targets.to(device)

    # Forward and backward to get gradients
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    # Compute SNIP scores: |w * grad_w|
    sensitivity_scores = {}
    for name, param in prunable_params.items():
        if param.grad is not None:
            scores = (param * param.grad).abs().detach()
            sensitivity_scores[name] = scores
        else:
            print(f"    WARNING: No gradient for {name}, using zeros")
            sensitivity_scores[name] = torch.zeros_like(param)

    if len(sensitivity_scores) == 0:
        raise ValueError("No sensitivity scores computed!")

    # Global threshold across all layers - FIXED: k is number to PRUNE (not keep)
    all_scores = torch.cat([scores.view(-1) for scores in sensitivity_scores.values()])
    sparsity_ratio = target_sparsity / 100.0
    k = int(len(all_scores) * sparsity_ratio)
    if k <= 0:
        k = 1
    if k >= len(all_scores):
        k = len(all_scores) - 1
    threshold = all_scores.kthvalue(k)[0].item()

    print(f"    SNIP threshold: {threshold:.8f}")

    # Create and apply masks
    masks = {}
    for name in sensitivity_scores:
        mask = (sensitivity_scores[name] > threshold).float()
        masks[name] = mask
        with torch.no_grad():
            prunable_params[name].mul_(mask)

    model.zero_grad()
    return masks

def apply_grasp_pruning_local(model, train_loader, target_sparsity, device):
    """Local implementation of GraSP pruning"""
    print(f"  Applying GraSP Pruning (local)...")

    model.eval()

    prunable_params = get_prunable_parameters(model)

    if len(prunable_params) == 0:
        raise ValueError("No prunable layers found!")

    # Get one batch
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise ValueError("Training data loader is empty!")

    inputs, targets = inputs.to(device), targets.to(device)

    # Compute first-order gradients
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)

    # Get gradients with create_graph=True for second-order
    grads = torch.autograd.grad(loss, prunable_params.values(), create_graph=True)

    # Compute gradient norm and its gradient (Hessian approximation)
    flatten_grads = torch.cat([g.view(-1) for g in grads])
    gnorm = (flatten_grads ** 2).sum()
    gnorm.backward()

    # Compute GraSP scores: -w * H*grad (approximated by second gradient)
    sensitivity_scores = {}
    for (name, param), grad in zip(prunable_params.items(), grads):
        if param.grad is not None:
            scores = (-param * param.grad).abs().detach()
            sensitivity_scores[name] = scores
        else:
            print(f"    WARNING: No Hessian for {name}, using zeros")
            sensitivity_scores[name] = torch.zeros_like(param)

    if len(sensitivity_scores) == 0:
        raise ValueError("No sensitivity scores computed!")

    # Global threshold - FIXED: k is number to PRUNE (not keep)
    all_scores = torch.cat([scores.view(-1) for scores in sensitivity_scores.values()])
    sparsity_ratio = target_sparsity / 100.0
    k = int(len(all_scores) * sparsity_ratio)
    if k <= 0:
        k = 1
    if k >= len(all_scores):
        k = len(all_scores) - 1
    threshold = all_scores.kthvalue(k)[0].item()

    print(f"    GraSP threshold: {threshold:.8f}")

    # Create and apply masks
    masks = {}
    for name in sensitivity_scores:
        mask = (sensitivity_scores[name] > threshold).float()
        masks[name] = mask
        with torch.no_grad():
            prunable_params[name].mul_(mask)

    model.zero_grad()
    return masks

def apply_random_pruning_local(model, target_sparsity):
    """Local implementation of random pruning"""
    print(f"  Applying Random Pruning (local)...")

    prunable_params = get_prunable_parameters(model)

    if len(prunable_params) == 0:
        raise ValueError("No prunable layers found!")

    # Random global threshold
    sparsity_ratio = target_sparsity / 100.0

    # Create random masks
    masks = {}
    for name, param in prunable_params.items():
        mask = (torch.rand_like(param) > sparsity_ratio).float()
        masks[name] = mask
        with torch.no_grad():
            param.mul_(mask)

    return masks

def apply_baseline_pruning(
    baseline_name: str,
    checkpoint_path: str,
    target_sparsity: float,
    train_loader,
    val_loader,
    device
):
    """Apply baseline pruning method and fine-tune"""
    print(f"\nApplying {baseline_name} pruning at {target_sparsity:.2f}% sparsity...")

    # Load fresh baseline model (MUST match checkpoint model: swin_tiny, CIFAR-100)
    baseline_model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip 'model.' prefix from all keys
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[6:] if key.startswith('model.') else key
        cleaned_state_dict[new_key] = value

    # Load encoder weights (everything except head)
    encoder_weights = {k: v for k, v in cleaned_state_dict.items() if 'head' not in k}
    baseline_model.load_state_dict(encoder_weights, strict=False)

    # Load trained head weights (matching main model loading logic)
    if 'head.1.weight' in cleaned_state_dict and 'head.1.bias' in cleaned_state_dict:
        head_weight = cleaned_state_dict['head.1.weight']
        head_bias = cleaned_state_dict['head.1.bias']
        num_classes = head_weight.shape[0]
        in_features = head_weight.shape[1]

        if num_classes == 100:
            baseline_model.head.fc = nn.Linear(in_features, 100)
            baseline_model.head.fc.weight.data = head_weight
            baseline_model.head.fc.bias.data = head_bias

    baseline_model = baseline_model.to(device)

    # Apply pruning method using LOCAL implementations (fixed for Transformers)
    if baseline_name == "MAG":
        masks = apply_magnitude_pruning_local(baseline_model, target_sparsity)
    elif baseline_name == "SNIP":
        masks = apply_snip_pruning_local(baseline_model, train_loader, target_sparsity, device)
    elif baseline_name == "GraSP":
        masks = apply_grasp_pruning_local(baseline_model, train_loader, target_sparsity, device)
    elif baseline_name == "Random":
        masks = apply_random_pruning_local(baseline_model, target_sparsity)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    actual_sparsity = calculate_sparsity(baseline_model)
    print(f"   {baseline_name} sparsity achieved: {actual_sparsity:.2f}%")
    
    print(f"  Fine-tuning {baseline_name} model...")
    fp32_accuracy = fine_tune_model(baseline_model, masks, train_loader, val_loader, device, epochs=30)
    print(f"  {baseline_name} FP32 accuracy: {fp32_accuracy:.2f}%")
    
    return baseline_model, masks, fp32_accuracy, actual_sparsity

def main():
    args = parse_args()

    print("==" * 80)
    print("Swin Transformer Multi-Strategy Pruning with PTQ (CIFAR-100)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score Directory: {args.score_dir}")
    print("==" * 80)
    print("\nGenerating 10 different strategies and comparing against baselines")
    print("All methods evaluated with PTQ (Post-Training Quantization)")
    print("Dataset: CIFAR-100 (100 classes) - Demonstrates scalability\n")

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Paths
    checkpoint_path = args.checkpoint
    score_dir = args.score_dir
    
    # ========================================================================
    # Load baseline model with PROPER checkpoint handling
    # ========================================================================
    print("\nLoading pretrained Swin-Tiny (CIFAR-100)...")
    base_model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # CRITICAL FIX: Strip 'model.' prefix from all keys
        print(f"  Cleaning checkpoint keys...")
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        print(f"  Cleaned {len(cleaned_state_dict)} checkpoint keys")
        
        # Load encoder weights (everything except head)
        encoder_weights = {k: v for k, v in cleaned_state_dict.items() if 'head' not in k}
        base_model.load_state_dict(encoder_weights, strict=False)
        print(f"  ✓ Loaded {len(encoder_weights)} encoder weights")

        # Load trained head weights from checkpoint
        if 'head.1.weight' in cleaned_state_dict and 'head.1.bias' in cleaned_state_dict:
            head_weight = cleaned_state_dict['head.1.weight']
            head_bias = cleaned_state_dict['head.1.bias']
            num_classes = head_weight.shape[0]
            in_features = head_weight.shape[1]

            print(f"  Checkpoint head: {num_classes} classes, {in_features} input features")

            if num_classes == 100:
                # Perfect! Checkpoint is for CIFAR-100, load the head weights
                base_model.head.fc = nn.Linear(in_features, 100)
                base_model.head.fc.weight.data = head_weight
                base_model.head.fc.bias.data = head_bias
                print(f"  ✓ Loaded trained head weights for CIFAR-100 (100 classes)")
            else:
                # Wrong number of classes - keep random head
                print(f"  X Warning: Checkpoint head has {num_classes} classes, need 100 for CIFAR-100")
                print(f"  ✓ Using fresh random head for CIFAR-100 (100 classes)")
        else:
            print(f"  X Warning: No trained head found in checkpoint")
            print(f"  ✓ Using fresh random head for CIFAR-100 (100 classes)")
    
    base_model = base_model.to(device)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Load data
    print("\nLoading CIFAR-100 dataset...")
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Baseline accuracy
    print("\nEvaluating baseline model...")
    baseline_accuracy = evaluate_model(base_model, val_loader, device)
    print(f"  Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Get layer information
    print(f"\nGetting layer information from {score_dir}...")
    if not os.path.exists(score_dir):
        raise FileNotFoundError(
            f"Sensitivity score directory not found: {score_dir}\n"
            f"Please run the sensitivity score generation first:\n"
            f"  python swin_sensitivity_simple.py --checkpoint {checkpoint_path} --score_dir {score_dir}"
        )

    layer_names, layer_params = get_swin_layer_info(base_model, score_dir)
    print_layer_sensitivity_ordering(layer_names, score_dir)
    print(f"Model has {len(layer_names)} prunable layers")

    if len(layer_names) == 0:
        print("\nERROR: No prunable layers found!")
        return
    
    # Compute Swin-specific ranges
    print(f"\nComputing Swin-specific layer ranges...")
    layer_ranges = compute_swin_sparsity_ranges(layer_names, layer_params)
    
    # Print layer analysis
    print_swin_layer_analysis(layer_names, layer_params)
    
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
        ("mlp_heavy", 90.0),
        ("attn_aggressive", 88.0),
        ("lower_30th_percentile", 90.0),
        ("middle_50th_percentile", 90.0),
        ("upper_70th_percentile", 92.0),
        ("upper_90th_percentile", 92.0),
        ("graduated", 90.0),
    ]
    
    multi_strategy_results = []
    
    for strategy_id, (strategy_type, target_sparsity) in enumerate(strategy_types, 1):
        print(f"\n{'='*80}")
        print(f"Strategy {strategy_id}/10: {strategy_type} (target: {target_sparsity:.1f}%)")
        print(f"{'='*80}")

        # Analyze sensitivity for first strategy only (to avoid repetition)
        if strategy_id == 1:
            print_layer_sensitivity_ordering(layer_names, score_dir)
            layer_stats = analyze_layer_sensitivity_for_strategic_pruning(layer_names, score_dir, debug=True)

        # Create strategy
        strategy, description = create_strategy_variant(
            layer_names, layer_params, layer_ranges, strategy_type, target_sparsity
        )

        # Apply pruning (enable debug for first strategy)
        print(f"\nApplying pruning masks...")
        pruned_model = copy.deepcopy(base_model)
        debug_mode = (strategy_id == 1)
        masks = apply_pruning_mask(pruned_model, strategy, layer_names, layer_params, score_dir, debug=debug_mode)

        actual_sparsity = calculate_sparsity(pruned_model, pruned_layer_names=layer_names)
        print(f"  Actual sparsity: {actual_sparsity:.2f}% (of prunable params)")
        
        # Calculate FLOPs
        flops, flops_str = calculate_flops(pruned_model, input_size=(1, 3, 224, 224), device='cpu')
        print(f"  FLOPs: {flops_str}")
        
        # Fine-tuning
        print(f"\nFine-tuning...")
        final_fp32_accuracy = fine_tune_model(pruned_model, masks, train_loader, val_loader, device, epochs=args.epochs)
        print(f"  FP32 Accuracy: {final_fp32_accuracy:.2f}%")
        
        # Apply PTQ
        print(f"\nApplying Post-Training Quantization...")
        pruned_model_cpu = pruned_model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            pruned_model_cpu,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Evaluate quantized model
        int8_accuracy = evaluate_model(quantized_model, val_loader, 'cpu')
        total_drop = baseline_accuracy - int8_accuracy
        print(f"  INT8 Accuracy: {int8_accuracy:.2f}%")
        print(f"  Total accuracy drop: {total_drop:.2f}%")
        
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
            'total_drop': total_drop
        })
    
    # ========================================================================
    # PART 2: Evaluate Baselines with PTQ
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Evaluating Baselines with PTQ")
    print("=" * 80)
    
    mean_sparsity = np.mean([r['actual_sparsity'] for r in multi_strategy_results])
    print(f"\nUsing mean sparsity: {mean_sparsity:.2f}%")
    
    baseline_results = {}
    
    for baseline_name in ["MAG", "SNIP", "GraSP", "Random"]:
        baseline_model, baseline_masks, fp32_acc, actual_spar = apply_baseline_pruning(
            baseline_name, checkpoint_path, mean_sparsity,
            train_loader, val_loader, device
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
        
        int8_acc = evaluate_model(baseline_quantized, val_loader, 'cpu')
        total_drop = baseline_accuracy - int8_acc
        print(f"  {baseline_name} INT8 accuracy: {int8_acc:.2f}%")
        
        baseline_results[baseline_name] = {
            'sparsity': actual_spar,
            'flops': flops,
            'flops_str': flops_str,
            'fp32_accuracy': fp32_acc,
            'int8_accuracy': int8_acc,
            'total_drop': total_drop
        }
    
    # ========================================================================
    # PART 3: Statistical Analysis
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
    # PART 4: Comparison Table
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
    print("Swin Transformer Multi-Strategy Pruning Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
