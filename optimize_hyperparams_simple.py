#!/usr/bin/env python3
"""
Simple Hyperparameter Optimization for Marvis TTS

Creates multiple config variations and you run them manually to find the best one.
Much simpler than full Optuna integration.

Usage:
    python optimize_hyperparams_simple.py

This creates 10-15 config files in configs/optimization/
Then on RunPod, run short training runs with each and compare losses.
"""

import json
import logging
from pathlib import Path
from itertools import product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_hyperparameter_configs():
    """Generate a grid of hyperparameter configurations to test."""

    # Load base config
    base_config_path = Path("configs/elise_finetune_gpu.json")
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Hyperparameter search space (based on Speechmatics recommendations)
    learning_rates = [5e-5, 1e-4, 2e-4]  # Original was 1e-4
    decoder_fractions = [0.0625, 0.125]  # 1/16 and 1/8
    decoder_loss_weights = [0.5, 1.0, 1.5]  # Test different weightings

    # Optional: batch sizes (only if GPU memory allows)
    # max_batch_sizes = [48, 64, 80]

    # Create output directory
    output_dir = Path("configs/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating hyperparameter configuration grid...")
    logger.info(f"  Learning rates: {learning_rates}")
    logger.info(f"  Decoder fractions: {decoder_fractions}")
    logger.info(f"  Decoder loss weights: {decoder_loss_weights}")

    configs = []
    config_id = 0

    # Generate all combinations
    for lr, dec_frac, dec_weight in product(learning_rates, decoder_fractions, decoder_loss_weights):
        config_id += 1

        # Create config
        trial_config = base_config.copy()
        trial_config.update({
            "learning_rate": lr,
            "decoder_fraction": dec_frac,
            "decoder_loss_weight": dec_weight,
        })

        # Save config
        config_name = f"trial_{config_id:02d}_lr{lr:.0e}_df{dec_frac:.4f}_dw{dec_weight:.1f}.json"
        config_path = output_dir / config_name

        with open(config_path, 'w') as f:
            json.dump(trial_config, f, indent=2)

        configs.append({
            'id': config_id,
            'path': str(config_path),
            'lr': lr,
            'dec_frac': dec_frac,
            'dec_weight': dec_weight
        })

        logger.info(f"  ✓ Created: {config_name}")

    # Create summary
    summary = {
        'total_configs': len(configs),
        'configs': configs,
        'instructions': {
            'step1': 'Upload marvis-tts to RunPod',
            'step2': 'For each config, run training for 2000 steps:',
            'command': 'accelerate launch train.py configs/optimization/trial_XX_*.json',
            'step3': 'Monitor loss in wandb',
            'step4': 'Choose config with lowest loss after 2000 steps',
            'estimated_time': f'{len(configs)} configs × 10 min = {len(configs) * 10} minutes',
            'estimated_cost': f'{len(configs)} configs × $0.073 = ${len(configs) * 0.073:.2f} (RTX 4090)'
        }
    }

    summary_path = output_dir / "optimization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Generated {len(configs)} configuration files")
    logger.info(f"✓ Summary saved to: {summary_path}")
    logger.info(f"{'='*70}\n")

    # Create batch training script for RunPod
    batch_script = create_batch_training_script(configs)
    batch_script_path = output_dir / "run_all_trials.sh"
    with open(batch_script_path, 'w') as f:
        f.write(batch_script)
    batch_script_path.chmod(0o755)

    logger.info("📝 RunPod Instructions:")
    logger.info("  1. Upload marvis-tts to RunPod")
    logger.info("  2. Run batch optimization:")
    logger.info(f"     bash {batch_script_path}")
    logger.info(f"  3. Monitor wandb to see which config performs best")
    logger.info(f"  4. Use best config for full 50K step training")
    logger.info(f"\n  Estimated time: {len(configs) * 10} minutes")
    logger.info(f"  Estimated cost: ${len(configs) * 0.073:.2f}")
    logger.info("")

    return configs


def create_batch_training_script(configs):
    """Create a bash script to run all trials sequentially."""

    script = """#!/bin/bash
# Batch Hyperparameter Optimization for Marvis TTS
# Runs all trial configs for 2000 steps each and logs results

set -e

echo "========================================================================"
echo "  Marvis TTS Hyperparameter Optimization"
echo "  Total trials: """ + str(len(configs)) + """
echo "  Steps per trial: 2000"
echo "========================================================================"
echo ""

# Activate environment
source venv/bin/activate

# Create results directory
mkdir -p optimization_results

# Run each trial
"""

    for config in configs:
        config_name = Path(config['path']).name
        trial_id = config['id']

        script += f"""
echo "Starting Trial {trial_id}/{len(configs)}: {config_name}"
echo "  lr={config['lr']}, decoder_frac={config['dec_frac']}, decoder_weight={config['dec_weight']}"

# Run training for 2000 steps
timeout 15m accelerate launch train.py {config['path']} \\
    --max-steps 2000 \\
    > optimization_results/trial_{trial_id:02d}.log 2>&1 || true

echo "  ✓ Trial {trial_id} complete"
echo ""
"""

    script += """
echo "========================================================================"
echo "  Optimization Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Check wandb to compare all trials"
echo "  2. Identify trial with lowest final loss"
echo "  3. Use that config for full training"
echo ""
"""

    return script


def analyze_results_script():
    """Create a Python script to analyze optimization results from wandb."""

    script = """#!/usr/bin/env python3
'''
Analyze Hyperparameter Optimization Results

Fetches results from wandb and identifies best configuration.

Usage:
    python analyze_optimization_results.py
'''

import wandb
import pandas as pd

# Initialize wandb
api = wandb.Api()

# Fetch all runs from optimization
runs = api.runs("sethdford/marvis-tts")

results = []
for run in runs:
    if 'trial_' in run.name:
        config = run.config
        summary = run.summary

        results.append({
            'name': run.name,
            'learning_rate': config.get('learning_rate'),
            'decoder_fraction': config.get('decoder_fraction'),
            'decoder_loss_weight': config.get('decoder_loss_weight'),
            'final_loss': summary.get('loss'),
            'final_c0_accuracy': summary.get('c0_acc'),
            'url': run.url
        })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values('final_loss')

print("\\nHyperparameter Optimization Results:")
print("="*70)
print(df.to_string(index=False))
print("="*70)
print(f"\\nBest configuration:")
best = df.iloc[0]
print(f"  Run: {best['name']}")
print(f"  Learning rate: {best['learning_rate']}")
print(f"  Decoder fraction: {best['decoder_fraction']}")
print(f"  Decoder loss weight: {best['decoder_loss_weight']}")
print(f"  Final loss: {best['final_loss']:.4f}")
print(f"  URL: {best['url']}")
print("")
"""

    analysis_path = Path("configs/optimization/analyze_results.py")
    with open(analysis_path, 'w') as f:
        f.write(script)
    analysis_path.chmod(0o755)

    logger.info(f"✓ Created analysis script: {analysis_path}")


def main():
    logger.info("="*70)
    logger.info("  Marvis TTS Hyperparameter Optimization Generator")
    logger.info("="*70)
    logger.info("")

    configs = generate_hyperparameter_configs()
    analyze_results_script()

    logger.info("="*70)
    logger.info("  Setup Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("Quick Start:")
    logger.info("  1. Push to GitHub (includes optimization configs)")
    logger.info("  2. On RunPod:")
    logger.info("     git clone YOUR_REPO")
    logger.info("     cd marvis-tts-main")
    logger.info("     bash runpod_setup.sh")
    logger.info("     bash configs/optimization/run_all_trials.sh")
    logger.info("  3. Check wandb to see best config")
    logger.info("")


if __name__ == "__main__":
    main()
