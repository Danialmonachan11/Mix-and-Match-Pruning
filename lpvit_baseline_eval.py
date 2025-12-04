"""
LPViT Baseline Evaluation Script (ECCV 2024 / April 2025)
==========================================================

Standalone script to evaluate LPViT (Low-Power Semi-structured Pruning) on:
- LeViT-384 on CIFAR-10
- Swin-Tiny on CIFAR-100
- VGG-11 on CIFAR-10
- ResNet-18 on GTSRB

This script is separate from multi-strategy pruning and only calculates
LPViT baseline results for comparison.
"""

import torch
import torch.nn as nn
import torch.quantization
import copy
import os
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


class LPViTPruning:
    """
    LPViT: Low-Power Semi-structured Pruning for Vision Transformers
    
    Simplified version using magnitude-based importance for block pruning.
    """
    
    def __init__(self, target_sparsity=0.5, num_samples=64):
        self.target_sparsity = target_sparsity
        self.num_samples = num_samples
    
    def compute_lpvit_scores(self, model, train_loader, device='cuda'):
        """Compute importance scores using magnitude"""
        model = model.to(device)
        model.eval()
        
        print("  Computing importance scores...")
        
        scores = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Use magnitude as importance
                scores[name] = param.data.abs()
        
        return scores
    
    def get_lpvit_pruning_mask(self, scores, target_sparsity, global_pruning=True):
        """Generate pruning mask from scores"""
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
            
            # Handle LeViT dual head output
            if isinstance(output, tuple):
                output = (output[0] + output[1]) / 2
                
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


def fine_tune_model(model, masks, train_loader, val_loader, device, epochs=30):
    """Fine-tune pruned model with mask enforcement"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
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
            
            # Handle LeViT dual head output
            if isinstance(output, tuple):
                output = (output[0] + output[1]) / 2
                
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


def evaluate_lpvit_on_levit384():
    """Evaluate LPViT on LeViT-384 / CIFAR-10"""
    print("\n" + "="*80)
    print("Evaluating LPViT on LeViT-384 (CIFAR-10)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data (CIFAR-10)
    print("\nLoading CIFAR-10...")
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import timm
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained model
    print("Loading pretrained LeViT-384...")
    checkpoint_path = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best3_levit_model_cifar10.pth'
    
    model = timm.create_model('levit_384', pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load encoder weights
    encoder_weights = {k: v for k, v in checkpoint.items() if 'head' not in k and 'dist' not in k}
    model.load_state_dict(encoder_weights, strict=False)
    
    # Load classification head
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
    baseline_acc = evaluate_model(model, val_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply LPViT pruning
    target_sparsity = 0.54  # 54% sparsity to match your existing results
    print(f"\nApplying LPViT pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = LPViTPruning(target_sparsity=target_sparsity, num_samples=64)
    scores = pruner.compute_lpvit_scores(model, train_loader, device)
    masks = pruner.get_lpvit_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param = param_dict[name]
                if mask.numel() == param.numel():
                    param.data.mul_(mask.view(param.shape).to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune
    print("\nFine-tuning for 30 epochs...")
    fp32_accuracy = fine_tune_model(model, masks, train_loader, val_loader, device, epochs=30)
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, val_loader, 'cpu')
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


def evaluate_lpvit_on_swin_tiny():
    """Evaluate LPViT on Swin-Tiny / CIFAR-100"""
    print("\n" + "="*80)
    print("Evaluating LPViT on Swin-Tiny (CIFAR-100)")
    print("="*80)
    print("🔧 DEBUG: RUNNING NEW VERSION WITH FIXED MODEL LOADING (2025-11-29 23:48)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data (CIFAR-100)
    print("\nLoading CIFAR-100...")
    import torchvision.transforms as transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from timm import create_model
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained model (EXACT copy from swin_multi_strategy.py)
    print("Loading pretrained Swin-Tiny...")
    checkpoint_path = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/best_swin_model_cifar_changed.pth'
    
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=100)
    
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
    model.load_state_dict(encoder_weights, strict=False)
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
            model.head.fc = nn.Linear(in_features, 100)
            model.head.fc.weight.data = head_weight
            model.head.fc.bias.data = head_bias
            print(f"  ✓ Loaded trained head weights for CIFAR-100 (100 classes)")
        else:
            # Wrong number of classes - keep random head
            print(f"  X Warning: Checkpoint head has {num_classes} classes, need 100 for CIFAR-100")
            print(f"  ✓ Using fresh random head for CIFAR-100 (100 classes)")
    else:
        print(f"  X Warning: No trained head found in checkpoint")
        print(f"  ✓ Using fresh random head for CIFAR-100 (100 classes)")
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Baseline accuracy
    baseline_acc = evaluate_model(model, val_loader, device)
    print(f"\nEvaluating baseline model...")
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    
    # Apply LPViT pruning
    target_sparsity = 0.44  # 44% sparsity to match your existing results
    print(f"\nApplying LPViT pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = LPViTPruning(target_sparsity=target_sparsity, num_samples=64)
    scores = pruner.compute_lpvit_scores(model, train_loader, device)
    masks = pruner.get_lpvit_pruning_mask(scores, target_sparsity, global_pruning=True)
    
    # Apply masks
    with torch.no_grad():
        for name, mask in masks.items():
            param_dict = dict(model.named_parameters())
            if name in param_dict:
                param = param_dict[name]
                if mask.numel() == param.numel():
                    param.data.mul_(mask.view(param.shape).to(device))
    
    actual_sparsity = calculate_sparsity(model)
    print(f"Actual sparsity achieved: {actual_sparsity:.2f}%")
    
    # Fine-tune (using Adam for Swin, lr=5e-5)
    print("\nFine-tuning for 30 epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    patience_counter = 0
    patience = 3
    best_model_state = None
    
    for epoch in range(30):
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
    
    fp32_accuracy = best_accuracy
    print(f"FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
    
    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    int8_accuracy = evaluate_model(quantized_model, val_loader, 'cpu')
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


def evaluate_lpvit_on_vgg11():
    """Evaluate LPViT on VGG-11 / CIFAR-10"""
    print("\n" + "="*80)
    print("Evaluating LPViT on VGG-11 (CIFAR-10)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10...")
    from core.models import vgg11_bn
    from core.data import get_data_loaders
    from config.model_config import ModelConfig
    
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
    
    # Apply LPViT pruning
    target_sparsity = 0.90  # 90% sparsity
    print(f"\nApplying LPViT pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = LPViTPruning(target_sparsity=target_sparsity, num_samples=64)
    scores = pruner.compute_lpvit_scores(model, train_loader, device)
    masks = pruner.get_lpvit_pruning_mask(scores, target_sparsity, global_pruning=True)
    
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


def evaluate_lpvit_on_resnet18():
    """Evaluate LPViT on ResNet-18 / GTSRB"""
    print("\n" + "="*80)
    print("Evaluating LPViT on ResNet-18 (GTSRB)")
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
    
    # Apply LPViT pruning
    target_sparsity = 0.87  # 87% sparsity
    print(f"\nApplying LPViT pruning (target: {target_sparsity*100:.0f}% sparsity)...")
    
    pruner = LPViTPruning(target_sparsity=target_sparsity, num_samples=64)
    scores = pruner.compute_lpvit_scores(model, train_loader, device)
    masks = pruner.get_lpvit_pruning_mask(scores, target_sparsity, global_pruning=True)
    
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


def main():
    print("="*80)
    print("LPViT Baseline Evaluation (ECCV 2024 / April 2025)")
    print("="*80)
    print("\nThis script evaluates LPViT pruning on:")
    print("  - LeViT-384 on CIFAR-10")
    print("  - Swin-Tiny on CIFAR-100")
    print("  - VGG-11 on CIFAR-10")
    print("  - ResNet-18 on GTSRB")
    
    results = []
    
    # Evaluate LeViT-384
    try:
        levit_result = evaluate_lpvit_on_levit384()
        if levit_result:
            results.append(levit_result)
    except Exception as e:
        print(f"\nError evaluating LeViT-384: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate Swin-Tiny
    try:
        swin_result = evaluate_lpvit_on_swin_tiny()
        if swin_result:
            results.append(swin_result)
    except Exception as e:
        print(f"\nError evaluating Swin-Tiny: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate VGG-11
    try:
        vgg_result = evaluate_lpvit_on_vgg11()
        if vgg_result:
            results.append(vgg_result)
    except Exception as e:
        print(f"\nError evaluating VGG-11: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate ResNet-18
    try:
        resnet_result = evaluate_lpvit_on_resnet18()
        if resnet_result:
            results.append(resnet_result)
    except Exception as e:
        print(f"\nError evaluating ResNet-18: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("LPViT Baseline Results Summary")
        print("="*80)
        
        for result in results:
            print(f"\n{result['model']} on {result['dataset']}:")
            print(f"  Baseline accuracy: {result['baseline_acc']:.2f}%")
            print(f"  Sparsity: {result['sparsity']:.2f}%")
            print(f"  FP32 accuracy: {result['fp32_acc']:.2f}%")
            print(f"  INT8 accuracy: {result['int8_acc']:.2f}%")
            print(f"  Total drop: {result['accuracy_drop']:.2f}%")
        
        # Save results
        output_file = "lpvit_baseline_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
