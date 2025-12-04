# Example Scripts

This directory contains ready-to-use example scripts for running the Mix-and-Match Pruning framework on all four architectures.

## Quick Start

### 1. Make scripts executable

```bash
chmod +x run_*.sh
```

### 2. Update configuration

Edit each script to specify your checkpoint and data paths:

```bash
# In run_vgg_example.sh, update:
CHECKPOINT_PATH="checkpoints/vgg11_cifar10.pth"  # Your checkpoint path
DATA_PATH="./data"                                # Your data directory

# In run_resnet_example.sh, update:
CHECKPOINT_PATH="checkpoints/resnet18_gtsrb.pth"
DATA_PATH="/path/to/GTSRB"                       # IMPORTANT: Update this!
```

### 3. Run the script

```bash
# VGG-11 on CIFAR-10
./run_vgg_example.sh

# ResNet-18 on GTSRB
./run_resnet_example.sh

# LeViT-384 on CIFAR-10
./run_levit_example.sh

# Swin-Tiny on CIFAR-100
./run_swin_example.sh
```

## What Each Script Does

1. **Checks prerequisites**: Verifies checkpoint and dataset exist
2. **Generates sensitivity scores**: Computes layer-wise importance (if not already done)
3. **Runs multi-strategy pruning**: Evaluates 7-10 strategies + 4 baselines
4. **Applies PTQ**: Post-training quantization to INT8

## Expected Runtime

On NVIDIA A100 GPU:
- **Sensitivity score generation**: 30-60 minutes
- **Multi-strategy pruning**: 3-5 hours
- **Total end-to-end**: 4-6 hours per architecture

## Customization

You can modify these scripts to:

### Change batch size (if you have OOM errors)
```bash
BATCH_SIZE=64  # Reduce from default
```

### Change fine-tuning epochs
```bash
EPOCHS=20  # Reduce from 30 for faster testing
```

### Use CPU instead of GPU
```bash
DEVICE="cpu"  # Warning: 10-20× slower
```

### Skip sensitivity regeneration

The scripts automatically detect existing sensitivity scores and ask if you want to regenerate. To force skip:

```bash
# Comment out the regeneration prompt in the script
# or just press 'N' when prompted
```

## Troubleshooting

### "Checkpoint not found"
Make sure you have a pretrained model. See main README.md for options:
1. Train your own
2. Use timm pretrained models
3. Download our checkpoints (after publication)

### "GTSRB dataset not found" (ResNet only)
Download GTSRB manually from: https://benchmark.ini.rub.de/gtsrb_dataset.html

Then update `DATA_PATH` in `run_resnet_example.sh`

### "CUDA out of memory"
Reduce `BATCH_SIZE` in the script:
- VGG-11: Try 64 instead of 128
- Others: Try 16 instead of 32

### Scripts don't run on Windows
These are bash scripts for Linux/Mac. On Windows:
1. Use Git Bash / WSL
2. Or run the Python commands directly (see main README.md)

## Output

Each script will print:

```
========================================
Multi-Strategy MAG/SNIP vs Baselines with PTQ
========================================

Strategy 1/10: max_aggressive (target: 90.0%)
  FP32 Accuracy: 90.41%
  Sparsity: 90.40%
  INT8 Accuracy: 90.41%

Strategy 2/10: min_conservative (target: 70.0%)
  FP32 Accuracy: 92.15%
  Sparsity: 70.12%
  INT8 Accuracy: 92.15%

...

========================================
BASELINE COMPARISON (with PTQ)
========================================
Random (90% sparsity):
  FP32: 45.23% | INT8: 45.12%
MAG (90% sparsity):
  FP32: 87.34% | INT8: 87.21%
...
```

Look for strategies with:
- High sparsity (>80%)
- Low accuracy drop (<2%)
- Minimal FP32→INT8 degradation
