"""
LeViT-384 Multi-Strategy Pruning with PTQ
Supports configurable sensitivity score directory

This script:
1. Generates 10 different pruning strategies from architecture-aware ranges
2. Applies pruning and fine-tuning to each strategy
3. Applies Post-Training Quantization (PTQ) to all strategies
4. Compares all strategies against baselines (Random, MAG, SNIP, GraSP) with PTQ
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
from levit_layer_classifier import (
    classify_levit_layer,
    extract_levit_stage,
    compute_levit_sparsity_ranges,
    get_levit_sensitivity_ordering,
    print_levit_layer_analysis
)

# Pruning baselines
from benchmarking.unstructured.classical.magnitude import MagnitudePruning
from benchmarking.unstructured.classical.random import RandomPruning
from benchmarking.unstructured.y2019.snip import SNIPPruning, GraSPPruning

# Configuration
CHECKPOINT_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best3_levit_model_cifar10.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 30
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description='LeViT Multi-Strategy Pruning')
    parser.add_argument('--score_dir', type=str, default='levit_weight_sensitivity_score',
                        help='Directory containing sensitivity scores')
    return parser.parse_args()

def load_data():
    """Load CIFAR-10 data"""
    print("Loading CIFAR-10 data...")
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
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
            
            if isinstance(output, tuple):
                output = (output[0] + output[1]) / 2
                
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def calculate_sparsity(model):
    """Calculate overall model sparsity"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
    return 100.0 * zero_params / total_params if total_params > 0 else 0.0

def get_levit_layer_info(model, score_dir_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Get prunable layer names and parameter counts (sensitivity-ordered)"""
    model_state_dict = model.state_dict()
    layer_params = {}
    
    available_layers = [
        name for name in model_state_dict.keys()
        if 'weight' in name and 'bias' not in name and len(model_state_dict[name].shape) >= 2
    ]
    
    layer_names_with_scores = []
    for name in available_layers:
        score_file = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        if os.path.exists(score_file):
            layer_names_with_scores.append(name)
    
    avg_sensitivities = []
    for idx, name in enumerate(layer_names_with_scores):
        score_file = os.path.join(score_dir_path, f'weight_sensitivity_scores_{name}.csv')
        try:
            scores = []
            with open(score_file, 'r') as f:
                next(f)
                for line_idx, line in enumerate(f):
                    if line_idx >= 1000:
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

    avg_sensitivities.sort(key=lambda x: x[1])
    layer_names = [name for name, _ in avg_sensitivities]
    
    for name in layer_names:
        layer_params[name] = model_state_dict[name].numel()

    return layer_names, layer_params

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
        
        total_weights = layer_params[layer_name]
        scores = []
        with open(score_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idx = int(parts[0])
                    score = float(parts[1])
                    if idx < total_weights:
                        scores.append((idx, score))

        scores.sort(key=lambda x: x[1])
        num_to_prune = int(total_weights * (prune_pct / 100.0))

        mask = torch.ones(total_weights)
        for i in range(min(num_to_prune, len(scores))):
            idx = scores[i][0]
            if idx < total_weights:
                mask[idx] = 0
        
        param_tensor = dict(model.named_parameters())[layer_name]
        mask = mask.view(param_tensor.shape).to(param_tensor.device)
        
        with torch.no_grad():
            param_tensor.mul_(mask)
        
        masks[layer_name] = mask
    
    return masks

def fine_tune_model(model, masks, train_loader, val_loader, device, epochs=30):
    """Fine-tune pruned model with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
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
            
            if isinstance(output, tuple):
                output = (output[0] + output[1]) / 2
                
            loss = criterion(output, target)
            loss.backward()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        mask = masks[name]
                        if mask.numel() == param.numel():
                            param.data.mul_(mask.view(param.shape).to(param.device))
            
            optimizer.step()
        
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
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_accuracy

def create_strategy_variant(
    layer_names: List[str],
    layer_params: Dict[str, int],
    layer_ranges: Dict[str, Tuple[float, float]],
    strategy_type: str
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
    return strategy, description

def apply_ptq(model, val_loader, device):
    """Apply Post-Training Quantization (PTQ)"""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader):
            if i >= 10: break
            model(data.to(device))
            
    torch.quantization.convert(model, inplace=True)
    return model

def main():
    args = parse_args()
    
    print("==" * 80)
    print("LeViT-384 Multi-Strategy Pruning with PTQ")
    print(f"Score Directory: {args.score_dir}")
    print("==" * 80)
    
    train_loader, val_loader = load_data()
    
    print("\nLoading pretrained LeViT-384...")
    base_model = create_model('levit_384', pretrained=False)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    encoder_weights = {k: v for k, v in checkpoint.items() if 'head' not in k and 'dist' not in k}
    base_model.load_state_dict(encoder_weights, strict=False)
    
    if 'head.weight' in checkpoint:
        in_features = checkpoint['head.weight'].shape[1]
        base_model.head = nn.Linear(in_features, 10)
        base_model.head.weight.data = checkpoint['head.weight']
        base_model.head.bias.data = checkpoint['head.bias']
        
        if hasattr(base_model, 'head_dist'):
            base_model.head_dist = nn.Linear(in_features, 10)
            if 'head_dist.weight' in checkpoint:
                base_model.head_dist.weight.data = checkpoint['head_dist.weight']
                base_model.head_dist.bias.data = checkpoint['head_dist.bias']
            else:
                base_model.head_dist.weight.data = checkpoint['head.weight'].clone()
                base_model.head_dist.bias.data = checkpoint['head.bias'].clone()

    base_model = base_model.to(DEVICE)
    
    initial_acc = evaluate_model(base_model, val_loader, DEVICE)
    print(f"Initial FP32 Accuracy: {initial_acc:.2f}%")
    
    layer_names, layer_params = get_levit_layer_info(base_model, args.score_dir)
    layer_ranges = compute_levit_sparsity_ranges(layer_names, layer_params)
    print_levit_layer_analysis(layer_names, layer_params)
    
    strategy_types = [
        "max_aggressive", "min_conservative", "balanced",
        "lower_30th_percentile", "middle_50th_percentile", 
        "upper_70th_percentile", "upper_90th_percentile"
    ]
    
    results = []
    
    for strategy_type in strategy_types:
        print(f"\nEvaluating Strategy: {strategy_type}")
        
        strategy, desc = create_strategy_variant(layer_names, layer_params, layer_ranges, strategy_type)
        
        model = copy.deepcopy(base_model)
        masks = apply_pruning_mask(model, strategy, layer_names, layer_params, args.score_dir)
        
        sparsity = calculate_sparsity(model)
        print(f"  Sparsity: {sparsity:.2f}%")
        
        ft_acc = fine_tune_model(model, masks, train_loader, val_loader, DEVICE, epochs=FINE_TUNE_EPOCHS)
        print(f"  Fine-tuned Accuracy: {ft_acc:.2f}%")
        
        model_int8 = apply_ptq(copy.deepcopy(model).cpu(), val_loader, 'cpu')
        int8_acc = evaluate_model(model_int8, val_loader, 'cpu')
        print(f"  INT8 Accuracy: {int8_acc:.2f}%")
        
        results.append({
            'strategy': strategy_type,
            'sparsity': sparsity,
            'ft_acc': ft_acc,
            'int8_acc': int8_acc
        })
        
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<25} | {'Sparsity':<10} | {'FT Acc':<10} | {'INT8 Acc':<10}")
    print("-" * 65)
    for res in results:
        print(f"{res['strategy']:<25} | {res['sparsity']:<10.2f} | {res['ft_acc']:<10.2f} | {res['int8_acc']:<10.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
