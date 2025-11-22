"""
LeViT-384 Gradient-Based Sensitivity Score Generation
Supports configurable metrics: Product, Magnitude, Gradient

Simple, stable approach matching VGG's methodology:
sensitivity_score = |gradient| × |weight| (Product)
sensitivity_score = |weight| (Magnitude)
sensitivity_score = |gradient| (Gradient)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from timm import create_model
import numpy as np
import csv
import os
from tqdm import tqdm
import argparse

# Configuration
CHECKPOINT_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best3_levit_model_cifar10.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_BATCHES = 10

def parse_args():
    parser = argparse.ArgumentParser(description='Generate LeViT Sensitivity Scores')
    parser.add_argument('--metric', type=str, default='product', choices=['product', 'magnitude', 'gradient'],
                        help='Sensitivity metric to use: product (|w|*|g|), magnitude (|w|), or gradient (|g|)')
    parser.add_argument('--output_dir', type=str, default='levit_weight_sensitivity_score',
                        help='Directory to save sensitivity scores')
    return parser.parse_args()

def load_model():
    """Load trained LeViT model"""
    print("Loading LeViT-384 model...")
    model = create_model('levit_384', pretrained=False, num_classes=10)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Load encoder weights
    encoder_weights = {k: v for k, v in checkpoint.items() if 'head' not in k and 'dist' not in k}
    model.load_state_dict(encoder_weights, strict=False)
    
    # Load head weights
    if 'head.weight' in checkpoint and 'head.bias' in checkpoint:
        in_features = checkpoint['head.weight'].shape[1]
        model.head = nn.Linear(in_features, 10)
        model.head.weight.data = checkpoint['head.weight']
        model.head.bias.data = checkpoint['head.bias']

        if hasattr(model, 'head_dist'):
            if 'head_dist.weight' in checkpoint and 'head_dist.bias' in checkpoint:
                model.head_dist = nn.Linear(in_features, 10)
                model.head_dist.weight.data = checkpoint['head_dist.weight']
                model.head_dist.bias.data = checkpoint['head_dist.bias']
            else:
                model.head_dist = nn.Linear(in_features, 10)
                model.head_dist.weight.data = checkpoint['head.weight'].clone()
                model.head_dist.bias.data = checkpoint['head.bias'].clone()
    
    model = model.to(DEVICE)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def load_data():
    """Load CIFAR-10 validation dataset"""
    print("\nLoading CIFAR-10 validation data...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    return val_loader

def compute_gradients(model, data_loader, num_batches):
    """Compute gradients over multiple batches"""
    print(f"\nComputing gradients over {num_batches} batches...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    grad_accum = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, total=num_batches, desc="Computing gradients")):
        if batch_idx >= num_batches:
            break
        
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        model.zero_grad()
        output = model(data)

        if isinstance(output, tuple):
            output = (output[0] + output[1]) / 2

        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum[name] += param.grad.abs()
    
    for name in grad_accum:
        grad_accum[name] /= num_batches
    
    print("✓ Gradient computation complete")
    return grad_accum

def save_sensitivity_scores(model, gradients, output_dir, metric='product'):
    """Compute and save sensitivity scores to CSV files"""
    print(f"\nSaving sensitivity scores to {output_dir}/ (Metric: {metric})...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_layers = 0
    total_weights = 0
    
    for name, param in tqdm(model.named_parameters(), desc="Saving scores"):
        if 'weight' not in name or 'bias' in name:
            continue
        
        if name not in gradients:
            print(f"  WARNING: No gradient for {name}, skipping...")
            continue
        
        grad = gradients[name].cpu().numpy().flatten()
        weight = param.data.cpu().numpy().flatten()
        
        # Compute sensitivity scores based on selected metric
        if metric == 'product':
            sensitivity_scores = np.abs(grad) * np.abs(weight)
        elif metric == 'magnitude':
            sensitivity_scores = np.abs(weight)
        elif metric == 'gradient':
            sensitivity_scores = np.abs(grad)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        sensitivity_scores = np.nan_to_num(sensitivity_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        csv_file = os.path.join(output_dir, f'weight_sensitivity_scores_{name}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'sensitivity_score'])
            for idx, score in enumerate(sensitivity_scores):
                writer.writerow([idx, f'{score:.10e}'])
        
        total_layers += 1
        total_weights += len(sensitivity_scores)
        
        non_zero = np.count_nonzero(sensitivity_scores)
        print(f"  ✓ {name}: {len(sensitivity_scores):,} weights, "
              f"{non_zero:,} non-zero scores ({non_zero/len(sensitivity_scores)*100:.2f}%)")
    
    print(f"\n✓ Saved {total_layers} layers with {total_weights:,} total weights")
    print(f"✓ Sensitivity scores saved to {output_dir}/")

def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"LeViT-384 Sensitivity Score Generation")
    print(f"Metric: {args.metric.upper()}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    model = load_model()
    data_loader = load_data()
    gradients = compute_gradients(model, data_loader, NUM_BATCHES)
    save_sensitivity_scores(model, gradients, args.output_dir, args.metric)
    
    print("\n" + "=" * 80)
    print("✓ Sensitivity score generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Verify CSV files in: {args.output_dir}/")
    print(f"2. Run levit_multi_strategy.py with --score_dir {args.output_dir}")

if __name__ == "__main__":
    main()
