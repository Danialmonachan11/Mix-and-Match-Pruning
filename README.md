# Mix-and-Match Pruning: Globally Guided Layer-Wise Sparsification of DNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **Mix-and-Match Pruning**, a globally guided, layer-wise sparsification framework for deep neural networks that systematically coordinates multiple pruning criteria to produce architecture-adaptive sparsity profiles.

## Overview

Deploying deep neural networks (DNNs) on edge platforms requires aggressive compression with minimal accuracy loss. This framework addresses a fundamental challenge: different layers and architectures respond differently to each pruning criterion, and no single method is universally optimal.

### Key Features

- **Architecture-Aware**: Automatically establishes layer-wise sparsity ranges based on architecture characteristics
- **Multi-Strategy Generation**: Produces 10 diverse pruning strategies spanning conservative to aggressive configurations
- **Flexible Sensitivity Metrics**: Supports magnitude-based, gradient-based, and combined pruning criteria
- **Post-Training Quantization**: Includes INT8 quantization for multiplicative compression gains
- **Proven Results**: Demonstrates up to 40% reduction in accuracy degradation compared to single-criterion pruning on Vision Transformers

### Supported Architectures

- **CNNs**: VGG-11, ResNet-18
- **Hybrid Transformers**: LeViT-384
- **Vision Transformers**: Swin-Tiny

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mix-and-match-pruning.git
cd mix-and-match-pruning

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Each architecture has dedicated scripts for multi-strategy pruning:

#### VGG-11 on CIFAR-10
```bash
python VGG_multi_Strategy.py
```

#### ResNet-18 on GTSRB
```bash
python resnet_multi_strategy.py
```

#### LeViT-384 on CIFAR-10
```bash
python levit_multi_strategy.py
```

#### Swin-Tiny on CIFAR-100
```bash
python swin_multi_strategy.py
```

### Layer Sensitivity Analysis

Before running multi-strategy pruning, you can analyze layer sensitivity:

```bash
# For ResNet-18
python resnet_sensitivity_simple.py

# For LeViT-384
python levit_sensitivity_simple.py

# For Swin-Tiny
python swin_sensitivity_simple.py
```

### Layer Classification

Classify layers by type and parameter count:

```bash
# For ResNet-18
python resnet_layer_classifier.py

# For LeViT-384
python levit_layer_classifier.py

# For Swin-Tiny
python swin_layer_classifier.py
```

## Methodology

### Three-Phase Framework

#### Phase 1: Architecture-Aware Range Establishment
- Execute baseline pruning methods (MAG, SNIP, GraSP) at multiple sparsity targets
- Record per-layer sparsity observations
- Compute minimum and maximum sparsity bounds for each layer
- Apply structural constraints (normalization, small layers, patch embeddings)

#### Phase 2: Multi-Strategy Generation
Generate 10 pruning strategies through systematic sampling:
- **Conservative**: Minimum observed sparsity per layer
- **Aggressive**: Maximum observed sparsity per layer
- **Balanced**: Midpoint of sparsity range
- **Percentile-based**: 30th, 50th, 70th, 90th percentile interpolations
- **Parameter-proportional**: Scale sparsity by layer size
- **Structure-aware**: Classifier-heavy and feature-heavy variants

#### Phase 3: Pruning Execution
- Compute sensitivity scores (magnitude, gradient, or combined)
- Apply binary masks to prune lowest-importance weights
- Fine-tune with mask enforcement
- Apply post-training INT8 quantization

## Results

### Performance Summary

| Model | Dataset | Sparsity | INT8 Accuracy | Accuracy Drop | Best Baseline | Improvement |
|-------|---------|----------|---------------|---------------|---------------|-------------|
| VGG-11 | CIFAR-10 | 90% | 90.42% | 1.97% | MAG: 2.86% | **31% better** |
| ResNet-18 | GTSRB | 87% | 98.12% | 1.20% | GraSP: 1.62% | **26% better** |
| LeViT-384 | CIFAR-10 | 54% | 97.31% | 0.44% | GraSP: 0.59% | **25% better** |
| Swin-Tiny | CIFAR-100 | 42% | 87.46% | 1.68% | MAG: 2.79% | **40% better** |

### Key Findings

1. **Architecture-Dependent Sensitivity**: Different architectures require different sensitivity metrics
   - VGG-11: Magnitude × Gradient product
   - ResNet-18, Swin-Tiny: Magnitude only
   - LeViT-384: Gradient only

2. **Low Variance**: Mean accuracy drop across strategies shows minimal variance (±0.14% to ±0.44%)

3. **Structural Constraints**: Critical for maintaining performance
   - Normalization layers: [0%, 0%] sparsity
   - Small layers (<10K params): [0%, 10%] sparsity
   - Transformer patch embeddings: [15%, 30%] sparsity

## Project Structure

```
mix-and-match-pruning/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── CITATION.bib                       # Citation information
├── Paper.tex                          # LaTeX source for research paper
│
├── VGG_multi_Strategy.py              # VGG-11 multi-strategy pruning
├── resnet_multi_strategy.py           # ResNet-18 multi-strategy pruning
├── levit_multi_strategy.py            # LeViT-384 multi-strategy pruning
├── swin_multi_strategy.py             # Swin-Tiny multi-strategy pruning
│
├── resnet_sensitivity_simple.py       # ResNet-18 sensitivity analysis
├── levit_sensitivity_simple.py        # LeViT-384 sensitivity analysis
├── swin_sensitivity_simple.py         # Swin-Tiny sensitivity analysis
│
├── resnet_layer_classifier.py         # ResNet-18 layer classification
├── levit_layer_classifier.py          # LeViT-384 layer classification
└── swin_layer_classifier.py           # Swin-Tiny layer classification
```

## Configuration

### Data Paths

Update the `DATA_PATH` variable in each script to point to your dataset location:

```python
# Example from resnet_multi_strategy.py
DATA_PATH = '/path/to/your/data/GTSRB'
TRAIN_FOLDER = os.path.join(DATA_PATH, 'Train')
```

### Hyperparameters

Key hyperparameters can be adjusted in each script:

- `BATCH_SIZE`: Training batch size (default: 128)
- `FINE_TUNE_EPOCHS`: Number of fine-tuning epochs (20-30 depending on architecture)
- `SEED`: Random seed for reproducibility (default: 42)
- Target sparsity levels for baseline methods

## Sensitivity Metrics

The framework supports three sensitivity formulations:

### 1. Magnitude-Only
```python
S_i = |w_i|
```
Best for: ResNet-18, Swin-Tiny

### 2. Gradient-Only
```python
S_i = |∂L/∂w_i|
```
Best for: LeViT-384

### 3. Magnitude-Gradient Product
```python
S_i = |w_i| × |∂L/∂w_i|
```
Best for: VGG-11

## Baseline Comparison

The framework compares against four established pruning methods:

- **MAG**: Magnitude-based pruning (smallest weights removed)
- **SNIP**: Single-shot Network Pruning using gradient-weight products
- **GraSP**: Gradient Signal Preservation with Hessian information
- **Random**: Random pruning (sanity check baseline)

All baselines use identical fine-tuning and quantization settings for fair comparison.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mix-and-match-pruning2025,
  title={Mix-and-Match Pruning: Globally Guided Layer-Wise Sparsification of DNNs},
  author={[Author Names]},
  journal={[Conference/Journal]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research implements and extends pruning methods from:
- **MAG**: Han et al., "Learning both Weights and Connections for Efficient Neural Networks" (NIPS 2015)
- **SNIP**: Lee et al., "SNIP: Single-shot Network Pruning based on Connection Sensitivity" (ICLR 2019)
- **GraSP**: Wang et al., "Picking Winning Tickets Before Training by Preserving Gradient Flow" (ICLR 2020)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [maintainer email].

## Future Work

- [ ] Structured pruning support
- [ ] Additional architecture support (EfficientNet, ConvNext, etc.)
- [ ] Automated sensitivity metric selection
- [ ] Hardware-aware optimization
- [ ] Integration with popular model compression frameworks

---

**Note**: This is research code. For production deployment, additional optimization and testing may be required.
