"""
ResNet-18 Gradient-Based Sensitivity Score Generation for GTSRB
Supports configurable metrics: Product, Magnitude, Gradient

Simple, stable approach matching VGG's methodology:
sensitivity_score = |gradient| × |weight| (Product)
sensitivity_score = |weight| (Magnitude)
sensitivity_score = |gradient| (Gradient)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np
import csv
import os
from tqdm import tqdm
from PIL import Image
import argparse

# Configuration
CHECKPOINT_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/road_0.9994904891304348.pth'
GTSRB_ROOT = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/multi_architecture_pruning/data/archive'  # UPDATE THIS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_BATCHES = 50

def parse_args():
    parser = argparse.ArgumentParser(description='Generate ResNet Sensitivity Scores')
    parser.add_argument('--metric', type=str, default='product', choices=['product', 'magnitude', 'gradient'],
                        help='Sensitivity metric to use: product (|w|*|g|), magnitude (|w|), or gradient (|g|)')
    parser.add_argument('--output_dir', type=str, default='resnet_weight_sensitivity_score',
                        help='Directory to save sensitivity scores')
    return parser.parse_args()

# ============================================================================
# GTSRB Dataset Loader
# ============================================================================

class GTSRBDataset(Dataset):
    """Custom Dataset for GTSRB"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._load_from_csv(root_dir, split)
    
    def _load_from_csv(self, root_dir, split):
        import pandas as pd
        
        if split == 'train':
            csv_file = os.path.join(root_dir, 'Train.csv')
        else:
            csv_file = os.path.join(root_dir, 'Test.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        for idx, row in df.iterrows():
            if 'Path' in df.columns and 'ClassId' in df.columns:
                img_path = os.path.join(root_dir, row['Path'])
                class_id = int(row['ClassId'])
            elif len(df.columns) >= 2:
                img_path = os.path.join(root_dir, str(df.iloc[idx, 0]))
                class_id = int(df.iloc[idx, 1])
            else:
                continue
            
            if os.path.exists(img_path):
                self.samples.append((img_path, class_id))
                self.class_to_idx[str(class_id)] = class_id
        
        print(f"  ✓ Loaded {len(self.samples)} samples ({len(set(self.class_to_idx.values()))} classes)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_model():
    """Load trained ResNet-18 model for GTSRB (43 classes)"""
    print("Loading ResNet-18 model...")
    
    model = models.resnet18(pretrained=False)  # ← ResNet-18
    model.fc = nn.Linear(model.fc.in_features, 43)  # ← GTSRB: 43 classes
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load with strict=False to handle any mismatches
    result = model.load_state_dict(state_dict, strict=False)
    
    if result.missing_keys or result.unexpected_keys:
        print(f"  ⚠ Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
    
    model = model.to(DEVICE)
    print(f"  ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

def load_data():
    """Load GTSRB validation dataset"""
    print("\nLoading GTSRB validation data...")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = GTSRBDataset(root_dir=GTSRB_ROOT, split='test', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
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
        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum[name] += param.grad.abs()
    
    for name in grad_accum:
        grad_accum[name] /= num_batches
    
    print("  ✓ Gradient computation complete")
    return grad_accum

def save_sensitivity_scores(model, gradients, output_dir, metric='product'):
    """Compute and save sensitivity scores to CSV files"""
    print(f"\nSaving sensitivity scores to {output_dir}/ (Metric: {metric})...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_layers = 0
    total_weights = 0
    
    for name, param in tqdm(model.named_parameters(), desc="Saving scores"):
        if 'weight' not in name or param.dim() < 2:
            continue
        
        if name not in gradients:
            print(f"  ⚠ WARNING: No gradient for {name}, skipping...")
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
        
        # Prepare all rows at once
        rows = [[idx, f'{score:.10e}'] for idx, score in enumerate(sensitivity_scores)]
        
        # Write all rows in one go
        csv_file = os.path.join(output_dir, f'weight_sensitivity_scores_{name}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'sensitivity_score'])
            writer.writerows(rows)
        
        total_layers += 1
        total_weights += len(sensitivity_scores)
        
        non_zero = np.count_nonzero(sensitivity_scores)
        print(f"  ✓ {name}: {len(sensitivity_scores):,} weights, "
              f"{non_zero:,} non-zero scores ({non_zero/len(sensitivity_scores)*100:.2f}%)")
    
    print(f"\n  ✓ Saved {total_layers} layers with {total_weights:,} total weights")
    print(f"  ✓ Sensitivity scores saved to {output_dir}/")

def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"ResNet-18 Sensitivity Score Generation (GTSRB)")
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
    print(f"2. Run resnet_multi_strategy.py with --score_dir {args.output_dir}")

if __name__ == "__main__":
    main()
