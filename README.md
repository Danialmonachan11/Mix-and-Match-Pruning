# Mix-and-Match Pruning: A Sensitivity-Driven Multi-Strategy Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**A novel neural network compression framework that combines sensitivity-driven layer analysis with multi-strategy pruning to achieve Pareto-optimal accuracy-sparsity trade-offs.**

## Overview

This repository contains the official implementation of our Mix-and-Match Pruning framework, which introduces:

- **Architecture-aware sensitivity analysis** for layer-specific pruning ranges
- **Multi-strategy exploration** across 10 different sparsity configurations
- **Hybrid pruning** combining magnitude, gradient, and structural techniques
- **Post-training quantization (PTQ)** for enhanced compression

### Key Results

| Architecture | Dataset | Max Sparsity | Accuracy Drop | PTQ Compatible |
|-------------|---------|--------------|---------------|----------------|
| VGG-11 | CIFAR-10 | 90.40% | <0.8% | ✓ |
| ResNet-18 | GTSRB | 88.76% | <1.2% | ✓ |
| LeViT-384 | CIFAR-10 | 89.23% | <0.5% | ✓ |
| Swin-Tiny | CIFAR-100 | 87.54% | <1.5% | ✓ |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [VGG-11 Pruning](#vgg-11-pruning-cifar-10)
  - [ResNet-18 Pruning](#resnet-18-pruning-gtsrb)
  - [LeViT-384 Pruning](#levit-384-pruning-cifar-10)
  - [Swin-Tiny Pruning](#swin-tiny-pruning-cifar-100)
  - [Additional Baselines](#additional-baselines-geta--lpvit)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (recommended for GPU acceleration)
- 16GB RAM minimum (32GB recommended for larger models)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/Mix-and-Match-Pruning.git
cd Mix-and-Match-Pruning

# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"
python -c "from core.models import vgg11_bn; print('✓ Core imports successful')"
```

---

## Quick Start

> **⚠️ IMPORTANT: Pretrained Models Required**
>
> This repository does **not include pretrained model checkpoints** (.pt/.pth files) due to file size constraints.
> You must provide your own pretrained models before running the scripts.
>
> See options below for obtaining models.

### 1. Prepare a Pretrained Model

**Option A: Train your own model**
```bash
# Train on your dataset using standard PyTorch/timm workflows
# Save checkpoint as .pth file
```

**Option B: Use timm pretrained models**
```python
import timm
import torch

# For VGG-11 on CIFAR-10
model = timm.create_model('vgg11_bn', pretrained=True, num_classes=10)
torch.save(model.state_dict(), 'checkpoints/vgg11_cifar10.pth')

# For ResNet-18
model = timm.create_model('resnet18', pretrained=True, num_classes=43)  # GTSRB has 43 classes
torch.save(model.state_dict(), 'checkpoints/resnet18_gtsrb.pth')

# For LeViT-384
model = timm.create_model('levit_384', pretrained=True, num_classes=10)
torch.save(model.state_dict(), 'checkpoints/levit384_cifar10.pth')

# For Swin-Tiny
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=100)
torch.save(model.state_dict(), 'checkpoints/swin_tiny_cifar100.pth')
```

**Option C: Download our pretrained checkpoints** (Coming after publication)
```bash
# Links will be provided after paper publication
# Models trained specifically on CIFAR-10/100 and GTSRB
```

### 2. Generate Sensitivity Scores

Before pruning, you must generate sensitivity scores for each layer:

```bash
# Example for VGG-11 (replace with your checkpoint path)
python vgg_sensitivity_simple.py \
    --checkpoint checkpoints/vgg11_cifar10.pth \
    --score_dir ./vgg_scores \
    --data_path ./data
```

### 3. Run Multi-Strategy Pruning

```bash
# Example for VGG-11
python VGG_multi_Strategy.py \
    --checkpoint checkpoints/vgg11_cifar10.pth \
    --score_dir ./vgg_scores \
    --batch_size 128
```

---

## Dataset Setup

### CIFAR-10 (VGG-11, LeViT-384)

Automatically downloaded by torchvision on first run. No manual setup required.

```python
# Default download location: ./data/cifar-10-batches-py/
```

### CIFAR-100 (Swin-Tiny)

Automatically downloaded by torchvision on first run. No manual setup required.

```python
# Default download location: ./data/cifar-100-python/
```

### GTSRB (ResNet-18)

**Manual download required:**

1. Download from [GTSRB Official Website](https://benchmark.ini.rub.de/gtsrb_dataset.html)
2. Extract to your preferred location
3. Verify structure:
   ```
   GTSRB/
   ├── Train/
   │   ├── 00000/
   │   ├── 00001/
   │   └── ...
   └── Test.csv
   ```
4. Use `--data_path` to specify location when running scripts

---

## Usage

### VGG-11 Pruning (CIFAR-10)

#### Step 1: Generate Sensitivity Scores

```bash
python vgg_sensitivity_simple.py \
    --checkpoint checkpoints/vgg11_cifar10.pth \
    --score_dir ./vgg_scores \
    --data_path ./data \
    --batch_size 128
```

**Expected runtime:** ~30-60 minutes on GPU

#### Step 2: Run Multi-Strategy Pruning

```bash
python VGG_multi_Strategy.py \
    --checkpoint checkpoints/vgg11_cifar10.pth \
    --score_dir ./vgg_scores \
    --data_path ./data \
    --batch_size 128 \
    --device cuda
```

**Arguments:**
- `--checkpoint` (required): Path to pretrained VGG-11 checkpoint
- `--score_dir` (required): Directory containing sensitivity scores
- `--data_path` (optional): Path to CIFAR-10 data (default: `./data`)
- `--batch_size` (optional): Batch size for training/evaluation (default: 128)
- `--device` (optional): Device to use - `cuda` or `cpu` (default: `cuda`)

**Expected output:**
```
================================================================================
Multi-Strategy MAG/SNIP vs Baselines with PTQ
================================================================================

Strategy 1/10: max_aggressive (target: 90.0%)
  FP32 Accuracy: 90.41%
  Sparsity: 90.40%
  INT8 Accuracy: 90.41%

Strategy 2/10: min_conservative (target: 70.0%)
  FP32 Accuracy: 92.15%
  Sparsity: 70.12%
  INT8 Accuracy: 92.15%

...

================================================================================
BASELINE COMPARISON (with PTQ)
================================================================================
Random (90% sparsity):
  FP32: 45.23% | INT8: 45.12%
MAG (90% sparsity):
  FP32: 87.34% | INT8: 87.21%
...
```

---

### ResNet-18 Pruning (GTSRB)

#### Step 1: Generate Sensitivity Scores

```bash
python resnet_sensitivity_simple.py \
    --checkpoint checkpoints/resnet18_gtsrb.pth \
    --score_dir ./resnet_scores \
    --data_path /path/to/GTSRB \
    --batch_size 64
```

#### Step 2: Run Multi-Strategy Pruning

```bash
python resnet_multi_strategy.py \
    --checkpoint checkpoints/resnet18_gtsrb.pth \
    --score_dir ./resnet_scores \
    --data_path /path/to/GTSRB \
    --batch_size 64 \
    --epochs 30 \
    --device cuda
```

**Arguments:**
- `--checkpoint` (required): Path to pretrained ResNet-18 checkpoint
- `--score_dir` (required): Directory containing sensitivity scores
- `--data_path` (required): Path to GTSRB dataset root directory
- `--batch_size` (optional): Batch size (default: 64)
- `--epochs` (optional): Fine-tuning epochs (default: 30)
- `--device` (optional): `cuda` or `cpu` (default: `cuda`)

---

### LeViT-384 Pruning (CIFAR-10)

#### Step 1: Generate Sensitivity Scores

```bash
python levit_sensitivity_simple.py \
    --checkpoint checkpoints/levit384_cifar10.pth \
    --score_dir ./levit_scores \
    --data_path ./data
```

#### Step 2: Run Multi-Strategy Pruning

```bash
python levit_multi_strategy.py \
    --checkpoint checkpoints/levit384_cifar10.pth \
    --score_dir ./levit_scores \
    --data_path ./data \
    --batch_size 32 \
    --epochs 30 \
    --device cuda
```

**Arguments:**
- `--checkpoint` (required): Path to pretrained LeViT-384 checkpoint (.pth file)
- `--score_dir` (required): Directory containing sensitivity scores
- `--data_path` (optional): Path to CIFAR-10 data (default: `./data`)
- `--batch_size` (optional): Batch size (default: 32)
- `--epochs` (optional): Fine-tuning epochs (default: 30)
- `--device` (optional): `cuda` or `cpu` (default: `cuda`)

---

### Swin-Tiny Pruning (CIFAR-100)

#### Step 1: Generate Sensitivity Scores

```bash
python swin_sensitivity_simple.py \
    --checkpoint checkpoints/swin_tiny_cifar100.pth \
    --score_dir ./swin_scores \
    --data_path ./data
```

#### Step 2: Run Multi-Strategy Pruning

```bash
python swin_multi_strategy.py \
    --checkpoint checkpoints/swin_tiny_cifar100.pth \
    --score_dir ./swin_scores \
    --data_path ./data \
    --batch_size 32 \
    --epochs 30 \
    --device cuda
```

**Arguments:**
- `--checkpoint` (required): Path to pretrained Swin-Tiny checkpoint
- `--score_dir` (required): Directory containing sensitivity scores
- `--data_path` (optional): Path to CIFAR-100 data (default: `./data`)
- `--batch_size` (optional): Batch size (default: 32)
- `--epochs` (optional): Fine-tuning epochs (default: 30)
- `--device` (optional): `cuda` or `cpu` (default: `cuda`)

---

### Additional Baselines (GETA & LPViT)

In addition to the standard baselines (Random, MAG, SNIP, GraSP), we also provide comparisons with recent pruning methods:

#### GETA (Gradient-Enhanced Taylor Approximation)
```bash
python geta_baseline_eval.py \
    --checkpoint checkpoints/your_model.pth \
    --data_path ./data \
    --target_sparsity 90.0
```

#### LPViT (Learned Pruning for Vision Transformers)
```bash
python lpvit_baseline_eval.py \
    --checkpoint checkpoints/your_transformer.pth \
    --data_path ./data \
    --target_sparsity 90.0
```

**Purpose**: These scripts provide additional baseline comparisons for the paper, demonstrating that the Mix-and-Match multi-strategy framework outperforms both classical (MAG, SNIP) and recent (GETA, LPViT) pruning methods.

---

## Methodology

### Three-Phase Framework

#### Phase 1: Sensitivity-Driven Range Establishment
- Compute layer-wise sensitivity scores using gradient-based analysis
- Classify layers by architecture type (convolution, attention, MLP, etc.)
- Establish sparsity ranges based on layer criticality

#### Phase 2: Multi-Strategy Exploration
Generate 10 diverse pruning strategies:
1. **Max-Aggressive**: Maximum allowable sparsity per layer
2. **Min-Conservative**: Minimum sparsity (safest)
3. **Balanced**: 50th percentile between min/max
4. **Lower 30th Percentile**: Conservative bias
5. **Middle 50th Percentile**: Balanced
6. **Upper 70th Percentile**: Aggressive bias
7. **Upper 90th Percentile**: Near-maximum sparsity

Each strategy is independently pruned and fine-tuned.

#### Phase 3: Baseline Comparison
Compare against classical methods:
- **Random Pruning**: Random weight removal
- **Magnitude Pruning**: Remove smallest weights
- **SNIP**: Gradient-based single-shot pruning
- **GraSP**: Gradient signal preservation

### Post-Training Quantization (PTQ)

All pruned models undergo INT8 quantization using PyTorch's default quantization:
- **Calibration**: 10 batches from validation set
- **Backend**: `fbgemm` (optimized for x86 CPUs)
- **Quantization-aware**: No retraining required

---

## Repository Structure

```
Mix-and-Match-Pruning/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── CONTRIBUTING.md                     # Contribution guidelines
├── CITATION.bib                        # Citation information
├── reference.bib                       # Paper references
├── .gitignore                          # Git ignore rules
│
├── paper_new.tex                       # Research paper (LaTeX)
├── Paper.tex                           # Original paper reference
├── Figure_1.png                        # Methodology diagram
│
├── VGG_multi_Strategy.py               # VGG-11 multi-strategy pruning
├── resnet_multi_strategy.py            # ResNet-18 multi-strategy pruning
├── levit_multi_strategy.py             # LeViT-384 multi-strategy pruning
├── swin_multi_strategy.py              # Swin-Tiny multi-strategy pruning
│
├── resnet_layer_classifier.py          # ResNet layer type classifier
├── levit_layer_classifier.py           # LeViT layer type classifier
├── swin_layer_classifier.py            # Swin layer type classifier
│
├── resnet_sensitivity_simple.py        # ResNet sensitivity score generator
├── levit_sensitivity_simple.py         # LeViT sensitivity score generator
├── swin_sensitivity_simple.py          # Swin sensitivity score generator
│
├── geta_baseline_eval.py               # GETA baseline evaluation
├── lpvit_baseline_eval.py              # LPViT baseline evaluation
│
├── core/                               # Core utilities
│   ├── __init__.py
│   └── utils.py                        # Helper functions
│
└── benchmarking/                       # Baseline pruning methods
    ├── unstructured/
    │   ├── classical/
    │   │   ├── magnitude.py            # Magnitude pruning
    │   │   └── random.py               # Random pruning
    │   └── y2019/
    │       └── snip.py                 # SNIP & GraSP
    └── ...
```

---

## FAQ

### Q: Do I need pretrained models to run this code?

**A:** Yes, pretrained models are **required**. **This repository does not include .pt/.pth checkpoint files** due to size constraints.

You must obtain models by:
1. **Training your own**: Use standard PyTorch/timm workflows on CIFAR-10/100 or GTSRB
2. **Using timm pretrained**: Download and adapt timm models (see Quick Start section for code)
3. **Waiting for our checkpoints**: Will be released after paper publication

Without pretrained models, the scripts will show an error with instructions on how to obtain them.

### Q: Why do I need to generate sensitivity scores separately?

**A:** Sensitivity score generation is computationally expensive (requires gradient calculations over the entire dataset). Separating this step allows you to:
- Generate scores once, reuse for multiple pruning experiments
- Parallelize score generation across multiple GPUs
- Debug pruning strategies without re-running sensitivity analysis

### Q: Can I use custom datasets?

**A:** Yes, but you'll need to modify the data loading functions in each script. Key changes:
1. Update `load_data()` function to load your dataset
2. Adjust image transformations (resize, normalization values)
3. Update number of output classes in model configuration

### Q: How long does pruning take?

**A:** Approximate runtimes on NVIDIA A100:
- **Sensitivity score generation**: 30-60 minutes per model
- **Single strategy pruning + fine-tuning**: 15-30 minutes
- **Full 10-strategy exploration**: 3-5 hours per architecture

### Q: Why are there no VGG layer classifier scripts?

**A:** VGG-11 uses architecture-aware rules directly in the script (see `compute_vgg_sparsity_ranges()` in `VGG_multi_Strategy.py`). VGG's simple sequential structure doesn't require a separate classifier module.

### Q: Can I run this on CPU?

**A:** Yes, use `--device cpu`. However:
- Expect 10-20× slower execution
- Fine-tuning may take several hours per strategy
- Recommended only for testing/debugging

### Q: What GPU memory is required?

**A:** Minimum GPU memory requirements:
- **VGG-11**: 4GB
- **ResNet-18**: 6GB
- **LeViT-384**: 8GB
- **Swin-Tiny**: 10GB

Reduce `--batch_size` if you encounter OOM errors.

### Q: How do I interpret the results?

**A:** Each strategy outputs:
- **FP32 Accuracy**: Accuracy after pruning and fine-tuning (full precision)
- **Sparsity**: Percentage of weights set to zero
- **INT8 Accuracy**: Accuracy after post-training quantization

Look for strategies with:
- High sparsity (>80%)
- Low accuracy drop (<2%)
- Minimal FP32→INT8 degradation

### Q: Can I combine this with structured pruning?

**A:** Currently, this framework focuses on unstructured (weight-level) pruning. Structured pruning (channel/filter removal) would require:
1. Modifying the pruning mask application
2. Physically removing pruned structures
3. Adjusting layer dimensions

This is planned for future releases.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{mixandmatch2024,
  title={Mix-and-Match Pruning: A Sensitivity-Driven Multi-Strategy Framework for Neural Network Compression},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

See `CITATION.bib` for complete citation information.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **timm library** for pretrained model architectures
- **PyTorch** for deep learning framework
- **SNIP/GraSP authors** for baseline implementations
- **GTSRB dataset** providers for traffic sign recognition benchmark

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For bugs, feature requests, or questions, please open an issue on GitHub.

---

## Contact

For questions or collaboration inquiries, please contact:
- **Email**: [your.email@domain.com]
- **GitHub Issues**: [Repository Issues Page]

---

**Last Updated**: December 2024
