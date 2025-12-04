#!/bin/bash
# Example script for running ResNet-18 multi-strategy pruning on GTSRB
# This is a complete end-to-end example

set -e  # Exit on error

echo "========================================"
echo "ResNet-18 Multi-Strategy Pruning Example"
echo "========================================"

# Configuration
CHECKPOINT_PATH="checkpoints/resnet18_gtsrb.pth"
SCORE_DIR="./resnet_scores"
DATA_PATH="/path/to/GTSRB"  # UPDATE THIS PATH!
BATCH_SIZE=64
EPOCHS=30
DEVICE="cuda"

# Step 1: Check if GTSRB dataset exists
echo ""
echo "Step 1: Checking for GTSRB dataset..."
if [ ! -d "$DATA_PATH/Train" ]; then
    echo "ERROR: GTSRB dataset not found at: $DATA_PATH"
    echo ""
    echo "Please download GTSRB dataset from:"
    echo "  https://benchmark.ini.rub.de/gtsrb_dataset.html"
    echo ""
    echo "Then update DATA_PATH in this script to point to the extracted folder."
    exit 1
fi
echo "✓ GTSRB dataset found: $DATA_PATH"

# Step 2: Check if checkpoint exists
echo ""
echo "Step 2: Checking for pretrained checkpoint..."
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please provide a pretrained ResNet-18 checkpoint trained on GTSRB."
    exit 1
fi
echo "✓ Checkpoint found: $CHECKPOINT_PATH"

# Step 3: Generate sensitivity scores (if not already generated)
echo ""
echo "Step 3: Generating sensitivity scores..."
if [ -d "$SCORE_DIR" ] && [ "$(ls -A $SCORE_DIR)" ]; then
    echo "⚠ Sensitivity scores already exist at: $SCORE_DIR"
    read -p "Regenerate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping sensitivity score generation..."
    else
        echo "Regenerating sensitivity scores..."
        python resnet_sensitivity_simple.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --score_dir "$SCORE_DIR" \
            --data_path "$DATA_PATH" \
            --batch_size "$BATCH_SIZE"
    fi
else
    echo "Generating sensitivity scores (this may take 30-60 minutes)..."
    python resnet_sensitivity_simple.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --score_dir "$SCORE_DIR" \
        --data_path "$DATA_PATH" \
        --batch_size "$BATCH_SIZE"
fi
echo "✓ Sensitivity scores ready"

# Step 4: Run multi-strategy pruning
echo ""
echo "Step 4: Running multi-strategy pruning..."
echo "This will evaluate 10 strategies + 4 baselines (may take 3-5 hours)"
python resnet_multi_strategy.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --score_dir "$SCORE_DIR" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "✓ ResNet-18 pruning complete!"
echo "Check the output for accuracy vs sparsity results"
echo "========================================"
