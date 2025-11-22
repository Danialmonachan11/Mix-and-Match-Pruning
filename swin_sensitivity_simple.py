"""
Swin Transformer Gradient-Based Sensitivity Score Generation

Simple, stable approach matching VGG's methodology:
sensitivity_score = |gradient| × |weight|

This script:
1. Loads trained Swin model
2. Computes gradients on CIFAR-10 validation set
3. Calculates sensitivity = |grad| × |weight| for each parameter
4. Saves to CSV files (one per layer)
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
# Update these paths based on your environment:
# - For server: use '/scratch/monacdan/MASTERS/...' paths
# - For local: use relative paths or full Windows paths
CHECKPOINT_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best_swin_model_cifar_changed.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_BATCHES = 10  # Use 10 batches for stable gradient estimation

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Swin Sensitivity Scores')
    parser.add_argument('--metric', type=str, default='product', choices=['product', 'magnitude', 'gradient'],
                        help='Sensitivity metric to use: product (|w|*|g|), magnitude (|w|), or gradient (|g|)')
    parser.add_argument('--output_dir', type=str, default='swin_weight_sensitivity_score',
                        help='Directory to save sensitivity scores')
    return parser.parse_args()

def load_model():
    """Load trained Swin Transformer model"""
    print("Loading Swin Transformer model...")
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)

    # Load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Using random weights (FOR TESTING ONLY)")
        return model.to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Handle 'model.' prefix in state_dict keys
    if isinstance(state_dict, dict):
        has_model_prefix = all(k.startswith('model.') for k in state_dict.keys())
        if has_model_prefix:
            print("  Stripping 'model.' prefix from checkpoint keys...")
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    # Load weights
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"  Warning: Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
        if result.unexpected_keys:
            print(f"  Note: Head mismatch likely - checkpoint trained on different number of classes")

    model = model.to(DEVICE)
    print(f"  Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def load_data():
    """Load CIFAR-100 validation dataset"""
    print("\nLoading CIFAR-100 validation data...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Use a local path if server path fails, or just standard ./data
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"  Loaded {len(val_dataset)} validation samples")
    return val_loader

def compute_gradients(model, data_loader, num_batches):
    """Compute gradients over multiple batches"""
    print(f"\nComputing gradients over {num_batches} batches...")
    model.train()  # Enable gradient computation
    criterion = nn.CrossEntropyLoss()

    # Initialize gradient accumulators
    grad_accum = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

    # Accumulate gradients over multiple batches
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, total=num_batches, desc="Computing gradients")):
        if batch_idx >= num_batches:
            break

        data, target = data.to(DEVICE), target.to(DEVICE)

        # Forward pass
        model.zero_grad()
        output = model(data)

        # Handle different output shapes
        while output.dim() > 2:
            output = output.squeeze()
        if output.dim() == 1:
            output = output.unsqueeze(0)

        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Accumulate gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum[name] += param.grad.abs()

    # Average gradients
    for name in grad_accum:
        grad_accum[name] /= num_batches

    print("  Gradient computation complete")
    return grad_accum

def save_sensitivity_scores(model, gradients, output_dir, metric='product'):
    """Compute and save sensitivity scores to CSV files"""
    print(f"\nSaving sensitivity scores to {output_dir}/ (Metric: {metric})...")
    os.makedirs(output_dir, exist_ok=True)

    total_layers = 0
    total_weights = 0

    for name, param in tqdm(model.named_parameters(), desc="Saving scores"):
        # Skip non-weight parameters (bias, normalization params, etc.)
        if 'weight' not in name or 'bias' in name:
            continue

        # Get gradients
        if name not in gradients:
            print(f"  WARNING: No gradient for {name}, skipping...")
            continue

        grad = gradients[name].cpu().numpy().flatten()
        weight = param.data.cpu().numpy().flatten()

        # Compute sensitivity scores based on selected metric
        if metric == 'product':
            # sensitivity = |gradient| * |weight| (Original)
            sensitivity_scores = np.abs(grad) * np.abs(weight)
        elif metric == 'magnitude':
            # sensitivity = |weight| (Magnitude Pruning)
            sensitivity_scores = np.abs(weight)
        elif metric == 'gradient':
            # sensitivity = |gradient| (Gradient-based Pruning)
            sensitivity_scores = np.abs(grad)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Replace any inf/nan with 0
        sensitivity_scores = np.nan_to_num(sensitivity_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Save to CSV
        csv_file = os.path.join(output_dir, f'weight_sensitivity_scores_{name}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'sensitivity_score'])
            for idx, score in enumerate(sensitivity_scores):
                writer.writerow([idx, f'{score:.10e}'])

        total_layers += 1
        total_weights += len(sensitivity_scores)

        # Print layer statistics
        non_zero = np.count_nonzero(sensitivity_scores)
        # Only print every 10th layer to reduce clutter
        if total_layers % 10 == 0:
            print(f"    {name}: {len(sensitivity_scores):,} weights, "
                  f"{non_zero:,} non-zero scores ({non_zero/len(sensitivity_scores)*100:.2f}%)")

    print(f"\n  Saved {total_layers} layers with {total_weights:,} total weights")
    print(f"  Sensitivity scores saved to {output_dir}/")

def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"Swin Transformer Sensitivity Score Generation (CIFAR-100)")
    print(f"Metric: {args.metric.upper()}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load model
    model = load_model()

    # Load data
    data_loader = load_data()

    # Compute gradients
    gradients = compute_gradients(model, data_loader, NUM_BATCHES)

    # Save sensitivity scores
    save_sensitivity_scores(model, gradients, args.output_dir, args.metric)

    print("\n" + "=" * 80)
    print("Sensitivity score generation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Verify CSV files in: {args.output_dir}/")
    print(f"2. Run swin_multi_strategy.py with --score_dir {args.output_dir}")

if __name__ == "__main__":
    main()
