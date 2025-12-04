#!/bin/bash
# Example script for running LeViT-384 multi-strategy pruning on CIFAR-10
# This is a complete end-to-end example

set -e  # Exit on error

echo "========================================"
echo "LeViT-384 Multi-Strategy Pruning Example"
echo "========================================"

# Configuration
CHECKPOINT_PATH="checkpoints/levit384_cifar10.pth"
SCORE_DIR="./levit_scores"
DATA_PATH="./data"
BATCH_SIZE=32
EPOCHS=30
DEVICE="cuda"

# Step 1: Check if checkpoint exists
echo ""
echo "Step 1: Checking for pretrained checkpoint..."
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please provide a pretrained LeViT-384 checkpoint."
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
        python levit_sensitivity_simple.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --score_dir "$SCORE_DIR" \
            --data_path "$DATA_PATH"
    fi
else
    echo "Generating sensitivity scores (this may take 30-60 minutes)..."
    python levit_sensitivity_simple.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --score_dir "$SCORE_DIR" \
        --data_path "$DATA_PATH"
fi
echo "✓ Sensitivity scores ready"

# Step 3: Run multi-strategy pruning
echo ""
echo "Step 3: Running multi-strategy pruning..."
echo "This will evaluate 7 strategies + 4 baselines (may take 3-5 hours)"
python levit_multi_strategy.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --score_dir "$SCORE_DIR" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "✓ LeViT-384 pruning complete!"
echo "Check the output for accuracy vs sparsity results"
echo "========================================"
