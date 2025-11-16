#!/bin/bash
# Batch Hyperparameter Optimization for Marvis TTS
# Runs all trial configs for 2000 steps each and logs results

set -e

echo "========================================================================"
echo "  Marvis TTS Hyperparameter Optimization"
echo "  Total trials: 18
echo "  Steps per trial: 2000"
echo "========================================================================"
echo ""

# Activate environment
source venv/bin/activate

# Create results directory
mkdir -p optimization_results

# Run each trial

echo "Starting Trial 1/18: trial_01_lr5e-05_df0.0625_dw0.5.json"
echo "  lr=5e-05, decoder_frac=0.0625, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_01_lr5e-05_df0.0625_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_01.log 2>&1 || true

echo "  ✓ Trial 1 complete"
echo ""

echo "Starting Trial 2/18: trial_02_lr5e-05_df0.0625_dw1.0.json"
echo "  lr=5e-05, decoder_frac=0.0625, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_02_lr5e-05_df0.0625_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_02.log 2>&1 || true

echo "  ✓ Trial 2 complete"
echo ""

echo "Starting Trial 3/18: trial_03_lr5e-05_df0.0625_dw1.5.json"
echo "  lr=5e-05, decoder_frac=0.0625, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_03_lr5e-05_df0.0625_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_03.log 2>&1 || true

echo "  ✓ Trial 3 complete"
echo ""

echo "Starting Trial 4/18: trial_04_lr5e-05_df0.1250_dw0.5.json"
echo "  lr=5e-05, decoder_frac=0.125, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_04_lr5e-05_df0.1250_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_04.log 2>&1 || true

echo "  ✓ Trial 4 complete"
echo ""

echo "Starting Trial 5/18: trial_05_lr5e-05_df0.1250_dw1.0.json"
echo "  lr=5e-05, decoder_frac=0.125, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_05_lr5e-05_df0.1250_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_05.log 2>&1 || true

echo "  ✓ Trial 5 complete"
echo ""

echo "Starting Trial 6/18: trial_06_lr5e-05_df0.1250_dw1.5.json"
echo "  lr=5e-05, decoder_frac=0.125, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_06_lr5e-05_df0.1250_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_06.log 2>&1 || true

echo "  ✓ Trial 6 complete"
echo ""

echo "Starting Trial 7/18: trial_07_lr1e-04_df0.0625_dw0.5.json"
echo "  lr=0.0001, decoder_frac=0.0625, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_07_lr1e-04_df0.0625_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_07.log 2>&1 || true

echo "  ✓ Trial 7 complete"
echo ""

echo "Starting Trial 8/18: trial_08_lr1e-04_df0.0625_dw1.0.json"
echo "  lr=0.0001, decoder_frac=0.0625, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_08_lr1e-04_df0.0625_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_08.log 2>&1 || true

echo "  ✓ Trial 8 complete"
echo ""

echo "Starting Trial 9/18: trial_09_lr1e-04_df0.0625_dw1.5.json"
echo "  lr=0.0001, decoder_frac=0.0625, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_09_lr1e-04_df0.0625_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_09.log 2>&1 || true

echo "  ✓ Trial 9 complete"
echo ""

echo "Starting Trial 10/18: trial_10_lr1e-04_df0.1250_dw0.5.json"
echo "  lr=0.0001, decoder_frac=0.125, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_10_lr1e-04_df0.1250_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_10.log 2>&1 || true

echo "  ✓ Trial 10 complete"
echo ""

echo "Starting Trial 11/18: trial_11_lr1e-04_df0.1250_dw1.0.json"
echo "  lr=0.0001, decoder_frac=0.125, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_11_lr1e-04_df0.1250_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_11.log 2>&1 || true

echo "  ✓ Trial 11 complete"
echo ""

echo "Starting Trial 12/18: trial_12_lr1e-04_df0.1250_dw1.5.json"
echo "  lr=0.0001, decoder_frac=0.125, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_12_lr1e-04_df0.1250_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_12.log 2>&1 || true

echo "  ✓ Trial 12 complete"
echo ""

echo "Starting Trial 13/18: trial_13_lr2e-04_df0.0625_dw0.5.json"
echo "  lr=0.0002, decoder_frac=0.0625, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_13_lr2e-04_df0.0625_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_13.log 2>&1 || true

echo "  ✓ Trial 13 complete"
echo ""

echo "Starting Trial 14/18: trial_14_lr2e-04_df0.0625_dw1.0.json"
echo "  lr=0.0002, decoder_frac=0.0625, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_14_lr2e-04_df0.0625_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_14.log 2>&1 || true

echo "  ✓ Trial 14 complete"
echo ""

echo "Starting Trial 15/18: trial_15_lr2e-04_df0.0625_dw1.5.json"
echo "  lr=0.0002, decoder_frac=0.0625, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_15_lr2e-04_df0.0625_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_15.log 2>&1 || true

echo "  ✓ Trial 15 complete"
echo ""

echo "Starting Trial 16/18: trial_16_lr2e-04_df0.1250_dw0.5.json"
echo "  lr=0.0002, decoder_frac=0.125, decoder_weight=0.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_16_lr2e-04_df0.1250_dw0.5.json \
    --max-steps 2000 \
    > optimization_results/trial_16.log 2>&1 || true

echo "  ✓ Trial 16 complete"
echo ""

echo "Starting Trial 17/18: trial_17_lr2e-04_df0.1250_dw1.0.json"
echo "  lr=0.0002, decoder_frac=0.125, decoder_weight=1.0"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_17_lr2e-04_df0.1250_dw1.0.json \
    --max-steps 2000 \
    > optimization_results/trial_17.log 2>&1 || true

echo "  ✓ Trial 17 complete"
echo ""

echo "Starting Trial 18/18: trial_18_lr2e-04_df0.1250_dw1.5.json"
echo "  lr=0.0002, decoder_frac=0.125, decoder_weight=1.5"

# Run training for 2000 steps
timeout 15m accelerate launch train.py configs/optimization/trial_18_lr2e-04_df0.1250_dw1.5.json \
    --max-steps 2000 \
    > optimization_results/trial_18.log 2>&1 || true

echo "  ✓ Trial 18 complete"
echo ""

echo "========================================================================"
echo "  Optimization Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Check wandb to compare all trials"
echo "  2. Identify trial with lowest final loss"
echo "  3. Use that config for full training"
echo ""
