#!/usr/bin/env python3
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

print("\nHyperparameter Optimization Results:")
print("="*70)
print(df.to_string(index=False))
print("="*70)
print(f"\nBest configuration:")
best = df.iloc[0]
print(f"  Run: {best['name']}")
print(f"  Learning rate: {best['learning_rate']}")
print(f"  Decoder fraction: {best['decoder_fraction']}")
print(f"  Decoder loss weight: {best['decoder_loss_weight']}")
print(f"  Final loss: {best['final_loss']:.4f}")
print(f"  URL: {best['url']}")
print("")
