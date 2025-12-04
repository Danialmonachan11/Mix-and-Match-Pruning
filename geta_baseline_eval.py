"""
GETA Baseline Evaluation Script (CVPR 2025)
============================================

Standalone script to evaluate GETA (Joint Pruning + Quantization) on:
- VGG-11 on CIFAR-10
- ResNet-18 on GTSRB
- Swin-Tiny on CIFAR-100
- LeViT-384 on CIFAR-10

This script is separate from multi-strategy pruning and only calculates
GETA baseline results for comparison.

Expected output:
================================================================================
GETA Baseline Results
================================================================================
VGG-11:     Sparsity: 90.XX%  |  FP32 Acc: XX.XX%  |  INT8 Acc: XX.XX%
ResNet-18:  Sparsity: 87.XX%  |  FP32 Acc: XX.XX%  |  INT8 Acc: XX.XX%
================================================================================
"""

import torch
import torch.nn as nn
import torch.quantization
import copy
import os
import sys
import numpy as np
from typing import Dict, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import vgg11_bn
from core.data import get_data_loaders
from config.model_config import ModelConfig


def calculate_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """Calculate FLOPs for a model using thop"""
    try:
        from thop import profile, clever_format
        model_copy = copy.deepcopy(model).to(device)
        model_copy.eval()
        input_tensor = torch.randn(input_size).to(device)
        flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        return flops, flops_str
    except ImportError:
        print("  Warning: thop not available, FLOPs calculation skipped")
        return 0, "N/A"
    except Exception as e:
        print(f"  Warning: FLOPs calculation failed: {e}")
        return 0, "N/A"


class GETAPruning:
    """
    GETA: Automatic Joint Structured Pruning and Quantization (CVPR 2025)
    
    Combines pruning and quantization in a unified optimization framework.
    Uses quantization-aware dependency graph for better compression.
    """
    
    def __init__(self, target_sparsity=0.9, target_bits=8, num_samples=100):
        self.target_sparsity = target_sparsity
        self.target_bits = target_bits
        self.num_samples = num_samples
    
    def compute_geta_scores(self, model, train_loader, device='cuda'):
        """
        Compute GETA importance scores using joint optimization criterion.
        
        GETA combines:
        1. Weight magnitude
        2. Gradient information
        3. Quantization sensitivity (how much accuracy drops when quantized)
        """
        model = model.to(device)
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        # Initialize scores
        scores = {}
        quant_sensitivity = {}
        
        print("  Computing quantization-aware importance scores...")
        
        # Step 1: Compute gradient scores
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        sample_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if sample_count >= self.num_samples:
                break
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            sample_count += 1
        
        # Step 2: Compute quantization sensitivity (simplified)
        # In full GETA, this uses quantization-aware dependency graph
        # Here we approximate by checking gradient magnitude after simulated quantization
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # GETA score: combines magnitude, gradient, and quantization impact
                grad = param.grad if param.grad is not None else torch.zeros_like(param)
                
                # Quantization sensitivity: how much parameter varies under quantization
                # Simulate 8-bit quantization impact
                param_range = param.data.max() - param.data.min()
                quant_step = param_range / (2 ** self.target_bits)
                quant_noise = torch.abs(param.data % quant_step)
                
                # GETA joint score: magnitude × gradient × (1 + quant_sensitivity)
                importance = param.data.abs() * grad.abs() * (1.0 + quant_noise)
                scores[name] = importance
        
        return scores
    
    def get_geta_pruning_mask(self, scores, target_sparsity, global_pruning=True):
        """Generate pruning mask from GETA scores"""
        if global_pruning:
            # Global ranking
            all_scores = torch.cat([scores[name].flatten() for name in scores.keys()])
            threshold_idx = int(len(all_scores) * target_sparsity)
            threshold = torch.sort(all_scores)[0][threshold_idx]
            
            masks = {}
            for name in scores.keys():
                masks[name] = (scores[name] > threshold).float()
        else:
            # Layer-wise ranking
            masks = {}
            for name in scores.keys():
                layer_scores = scores[name].flatten()
                threshold_idx = int(len(layer_scores) * target_sparsity)
                threshold = torch.sort(layer_scores)[0][threshold_idx]
                masks[name] = (scores[name] > threshold).float()
        
        return masks


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
    """Calculate global sparsity"""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params
    return sparsity


def fine_tune_model(model, masks, train_loader, val_loader, device, epochs=20):
    """Fine-tune pruned model with mask enforcement"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
            loss = criterion(output, target)
            loss.backward()
            
            # Enforce sparsity
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
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_accuracy


def evaluate_geta_on_vgg11():
    """Evaluate GETA on VGG-11 / CIFAR-10"""
    print("\n" + "="*80)
    print("Evaluating GETA on VGG-11 (CIFAR-10)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10...")
    model_config = ModelConfig()
    train_loader, val_loader = get_data_loaders(model_config)
    
    # Load pretrained model
    print("Loading pretrained VGG-11...")
    model_path = "/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/vgg11_bn.pt"
    
    model = vgg11_bn(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate_model(model, val_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply GETA pruning
    target_sparsity = 0.90  # 90% sparsity to match other baselines
    print(f"\nApplying GETA pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = GETAPruning(target_sparsity=target_sparsity, target_bits=8, num_samples=100)
    scores = pruner.compute_geta_scores(model, train_loader, device)
    masks = pruner.get_geta_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param_dict[name].data.mul_(mask.to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune
    print("\nFine-tuning for 20 epochs...")
    fp32_accuracy = fine_tune_model(model, masks, train_loader, val_loader, device, epochs=20)
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, val_loader, 'cpu')
    accuracy_drop = baseline_acc - int8_accuracy
    
    print(f"INT8 accuracy: {int8_accuracy:.2f}%")
    print(f"Total accuracy drop: {accuracy_drop:.2f}%")
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flops_val, flops_str = calculate_flops(model.to('cpu'), input_size=(1, 3, 32, 32), device='cpu')
    print(f"FLOPs: {flops_str}")
    
    return {
        'model': 'VGG-11',
        'dataset': 'CIFAR-10',
        'baseline_acc': baseline_acc,
        'sparsity': actual_sparsity,
        'fp32_acc': fp32_accuracy,
        'int8_acc': int8_accuracy,
        'accuracy_drop': accuracy_drop,
        'flops': flops_val,
        'flops_str': flops_str
    }


def evaluate_geta_on_resnet18():
    """Evaluate GETA on ResNet-18 / GTSRB"""
    print("\n" + "="*80)
    print("Evaluating GETA on ResNet-18 (GTSRB)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load GTSRB data  
    print("\nLoading GTSRB dataset...")
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import pandas as pd
    
    DATA_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/multi_architecture_pruning/data/archive'
    TRAIN_FOLDER = os.path.join(DATA_PATH, 'Train')
    TEST_CSV = os.path.join(DATA_PATH, 'Test.csv')
    
    # Custom Dataset for test set mapping
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
    
    train_dataset = datasets.ImageFolder(TRAIN_FOLDER, transform=transform_train)
    test_dataset = GTSRBTestRemap(TEST_CSV, DATA_PATH, train_dataset.class_to_idx, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained model
    print("Loading pretrained ResNet-18...")
    checkpoint_path = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/road_0.9994904891304348.pth'
    
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 43)  # GTSRB has 43 classes
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply GETA pruning
    target_sparsity = 0.87  # 87% sparsity
    print(f"\nApplying GETA pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = GETAPruning(target_sparsity=target_sparsity, target_bits=8, num_samples=100)
    scores = pruner.compute_geta_scores(model, train_loader, device)
    masks = pruner.get_geta_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param_dict[name].data.mul_(mask.to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune (use 5e-5 LR for ResNet)
    print("\nFine-tuning for 20 epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    patience_counter = 0
    patience = 3
    best_model_state = None
    
    for epoch in range(20):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Enforce sparsity
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        mask = masks[name]
                        if mask.numel() == param.numel():
                            param.data.mul_(mask.view(param.shape).to(param.device))
            
            optimizer.step()
        
        # Validation
        model.eval()
        val_accuracy = evaluate_model(model, test_loader, device)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    fp32_accuracy = best_accuracy
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
    accuracy_drop = baseline_acc - int8_accuracy
    
    print(f"INT8 accuracy: {int8_accuracy:.2f}%")
    print(f"Total accuracy drop: {accuracy_drop:.2f}%")
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flops_val, flops_str = calculate_flops(model.to('cpu'), input_size=(1, 3, 224, 224), device='cpu')
    print(f"FLOPs: {flops_str}")
    
    return {
        'model': 'ResNet-18',
        'dataset': 'GTSRB',
        'baseline_acc': baseline_acc,
        'sparsity': actual_sparsity,
        'fp32_acc': fp32_accuracy,
        'int8_acc': int8_accuracy,
        'accuracy_drop': accuracy_drop,
        'flops': flops_val,
        'flops_str': flops_str
    }


def evaluate_geta_on_swin_tiny():
    """Evaluate GETA on Swin-Tiny / CIFAR-100"""
    print("\n" + "="*80)
    print("Evaluating GETA on Swin-Tiny (CIFAR-100)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load CIFAR-100 data
    print("\nLoading CIFAR-100...")
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from timm import create_model
    
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
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained model (same as lpvit)
    print("Loading pretrained Swin-Tiny...")
    checkpoint_path = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best_swin_model_cifar_changed.pth'
    
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Strip 'model.' prefix
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[6:] if key.startswith('model.') else key
        cleaned_state_dict[new_key] = value
    
    # Load encoder
    encoder_weights = {k: v for k, v in cleaned_state_dict.items() if 'head' not in k}
    model.load_state_dict(encoder_weights, strict=False)
    
    # Load head
    if 'head.1.weight' in cleaned_state_dict and 'head.1.bias' in cleaned_state_dict:
        model.head.fc = nn.Linear(cleaned_state_dict['head.1.weight'].shape[1], 100)
        model.head.fc.weight.data = cleaned_state_dict['head.1.weight']
        model.head.fc.bias.data = cleaned_state_dict['head.1.bias']
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply GETA (iterative pruning with fine-tuning)
    target_sparsity = 0.44  # 44% sparsity
    print(f"\nApplying GETA pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = GETAPruning(target_sparsity=target_sparsity, target_bits=8, num_samples=100)
    scores = pruner.compute_geta_scores(model, train_loader, device)
    masks = pruner.get_geta_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param_dict[name].data.mul_(mask.to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune
    print("\nFine-tuning for 20 epochs...")
    fp32_accuracy = fine_tune_model(model, masks, train_loader, test_loader, device, epochs=20)
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
    accuracy_drop = baseline_acc - int8_accuracy
    
    print(f"INT8 accuracy: {int8_accuracy:.2f}%")
    print(f"Total accuracy drop: {accuracy_drop:.2f}%")
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flops_val, flops_str = calculate_flops(model.to('cpu'), input_size=(1, 3, 224, 224), device='cpu')
    print(f"FLOPs: {flops_str}")
    
    return {
        'model': 'Swin-Tiny',
        'dataset': 'CIFAR-100',
        'baseline_acc': baseline_acc,
        'sparsity': actual_sparsity,
        'fp32_acc': fp32_accuracy,
        'int8_acc': int8_accuracy,
        'accuracy_drop': accuracy_drop,
        'flops': flops_val,
        'flops_str': flops_str
    }


def evaluate_geta_on_levit384():
    """Evaluate GETA on LeViT-384 / CIFAR-10"""
    print("\n" + "="*80)
    print("Evaluating GETA on LeViT-384 (CIFAR-10)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load CIFAR-10 data
    print("\nLoading CIFAR-10...")
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import timm
    
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
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
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained model (same as lpvit)
    print("Loading pretrained LeViT-384...")
    checkpoint_path = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best3_levit_model_cifar10.pth'
    
    model = timm.create_model('levit_384', pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load encoder
    encoder_weights = {k: v for k, v in checkpoint.items() if 'head' not in k and 'dist' not in k}
    model.load_state_dict(encoder_weights, strict=False)
    
    # Load head
    if 'head.weight' in checkpoint:
        in_features = checkpoint['head.weight'].shape[1]
        model.head = nn.Linear(in_features, 10)
        model.head.weight.data = checkpoint['head.weight']
        model.head.bias.data = checkpoint['head.bias']
        
        if hasattr(model, 'head_dist'):
            model.head_dist = nn.Linear(in_features, 10)
            if 'head_dist.weight' in checkpoint:
                model.head_dist.weight.data = checkpoint['head_dist.weight']
                model.head_dist.bias.data = checkpoint['head_dist.bias']
            else:
                model.head_dist.weight.data = checkpoint['head.weight'].clone()
                model.head_dist.bias.data = checkpoint['head.bias'].clone()
    
    model = model.to(device)
    
    # Baseline accuracy
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply GETA
    target_sparsity = 0.54  # 54% sparsity
    print(f"\nApplying GETA pruning (target: {target_sparsity*100:.0f}% sparsity)...") 
    
    pruner = GETAPruning(target_sparsity=target_sparsity, target_bits=8, num_samples=100)
    scores = pruner.compute_geta_scores(model, train_loader, device)
    masks = pruner.get_geta_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param_dict[name].data.mul_(mask.to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune
    print("\nFine-tuning for 30 epochs...")
    fp32_accuracy = fine_tune_model(model, masks, train_loader, test_loader, device, epochs=30)
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
    accuracy_drop = baseline_acc - int8_accuracy
    
    print(f"INT8 accuracy: {int8_accuracy:.2f}%")
    print(f"Total accuracy drop: {accuracy_drop:.2f}%")
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    flops_val, flops_str = calculate_flops(model.to('cpu'), input_size=(1, 3, 224, 224), device='cpu')
    print(f"FLOPs: {flops_str}")
    
    return {
        'model': 'LeViT-384',
        'dataset': 'CIFAR-10',
        'baseline_acc': baseline_acc,
        'sparsity': actual_sparsity,
        'fp32_acc': fp32_accuracy,
        'int8_acc': int8_accuracy,
        'accuracy_drop': accuracy_drop,
        'flops': flops_val,
        'flops_str': flops_str
    }


def main():
    print("="*80)
    print("GETA Baseline Evaluation (CVPR 2025)")
    print("="*80)
    print("\nThis script evaluates GETA pruning method on:")
    print("  - VGG-11 on CIFAR-10")
    print("  - ResNet-18 on GTSRB")
    print("  - Swin-Tiny on CIFAR-100")
    print("  - LeViT-384 on CIFAR-10")
    
    results = []
    
    # Evaluate VGG-11
    try:
        vgg_result = evaluate_geta_on_vgg11()
        if vgg_result:
            results.append(vgg_result)
    except Exception as e:
        print(f"\nError evaluating VGG-11: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate ResNet-18
    try:
        resnet_result = evaluate_geta_on_resnet18()
        if resnet_result:
            results.append(resnet_result)
    except Exception as e:
        print(f"\nError evaluating ResNet-18: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate Swin-Tiny
    try:
        swin_result = evaluate_geta_on_swin_tiny()
        if swin_result:
            results.append(swin_result)
    except Exception as e:
        print(f"\nError evaluating Swin-Tiny: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate LeViT-384
    try:
        levit_result = evaluate_geta_on_levit384()
        if levit_result:
            results.append(levit_result)
    except Exception as e:
        print(f"\nError evaluating LeViT-384: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("GETA Baseline Results Summary")
    print("="*80)
    
    for result in results:
        print(f"\n{result['model']} on {result['dataset']}:")
        print(f"  Baseline accuracy: {result['baseline_acc']:.2f}%")
        print(f"  Sparsity: {result['sparsity']:.2f}%")
        print(f"  FP32 accuracy: {result['fp32_acc']:.2f}%")
        print(f"  INT8 accuracy: {result['int8_acc']:.2f}%")
        print(f"  Total drop: {result['accuracy_drop']:.2f}%")
    
    # Save results
    output_file = "geta_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
