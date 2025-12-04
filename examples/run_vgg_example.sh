#!/bin/bash
# Example script for running VGG-11 multi-strategy pruning on CIFAR-10
# This is a complete end-to-end example

set -e  # Exit on error

echo "========================================"
echo "VGG-11 Multi-Strategy Pruning Example"
echo "========================================"

# Configuration
CHECKPOINT_PATH="checkpoints/vgg11_cifar10.pth"
SCORE_DIR="./vgg_scores"
DATA_PATH="./data"
BATCH_SIZE=128
DEVICE="cuda"

# Step 1: Check if checkpoint exists
echo ""
echo "Step 1: Checking for pretrained checkpoint..."
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please provide a pretrained VGG-11 checkpoint."
    echo "You can:"
    echo "  1. Train your own model"
    echo "  2. Download from timm: python -c 'import timm; m=timm.create_model(\"vgg11_bn\", pretrained=True, num_classes=10); import torch; torch.save(m.state_dict(), \"$CHECKPOINT_PATH\")'"
    echo "  3. Use our pretrained models (link in README)"
    exit 1
fi
echo "✓ Checkpoint found: $CHECKPOINT_PATH"

# Step 2: Generate sensitivity scores (if not already generated)
echo ""
echo "Step 2: Generating sensitivity scores..."
if [ -d "$SCORE_DIR" ] && [ "$(ls -A $SCORE_DIR)" ]; then
    echo "⚠ Sensitivity scores already exist at: $SCORE_DIR"
    read -p "Regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping sensitivity score generation..."
    else
        echo "Regenerating sensitivity scores..."
        python vgg_sensitivity_simple.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --score_dir "$SCORE_DIR" \
            --data_path "$DATA_PATH" \
            --batch_size "$BATCH_SIZE"
    fi
else
    echo "Generating sensitivity scores (this may take 30-60 minutes)..."
    python vgg_sensitivity_simple.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --score_dir "$SCORE_DIR" \
        --data_path "$DATA_PATH" \
        --batch_size "$BATCH_SIZE"
fi
echo "✓ Sensitivity scores ready"

# Step 3: Run multi-strategy pruning
echo ""
echo "Step 3: Running multi-strategy pruning..."
echo "This will evaluate 10 strategies + 4 baselines (may take 3-5 hours)"
python VGG_multi_Strategy.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --score_dir "$SCORE_DIR" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "✓ VGG-11 pruning complete!"
echo "Check the output for accuracy vs sparsity results"
echo "========================================"
