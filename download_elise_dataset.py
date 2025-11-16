#!/usr/bin/env python3
"""
Download and explore the Elise dataset from Hugging Face
"""

import os
from datasets import load_dataset
import pandas as pd

print("Downloading Elise dataset from Hugging Face...")
print("=" * 60)

# Download the dataset
dataset = load_dataset("Jinsaryko/Elise", split="train")

print(f"✓ Dataset loaded successfully!")
print(f"  Total samples: {len(dataset)}")
print()

# Show dataset structure
print("Dataset columns:")
print("-" * 60)
print(dataset.column_names)
print()

# Show first few examples (without decoding audio yet)
print("First 3 samples (text only):")
print("-" * 60)
df_preview = dataset.to_pandas()
for i in range(min(3, len(df_preview))):
    print(f"\nSample {i+1}:")
    print(f"  text: {df_preview.iloc[i]['text']}")
    print(f"  speaker: {df_preview.iloc[i]['speaker_name']}")
    print(f"  utterance_pitch_mean: {df_preview.iloc[i]['utterance_pitch_mean']}")
    print(f"  speaking_rate: {df_preview.iloc[i]['speaking_rate']}")
    print(f"  SNR: {df_preview.iloc[i]['snr']}")

# Save to local directory
print()
print("=" * 60)
print("Saving dataset locally...")

# Create data directory
os.makedirs("data/elise", exist_ok=True)

# Convert to pandas and save as parquet
df = dataset.to_pandas()
output_path = "data/elise/dataset.parquet"
df.to_parquet(output_path)
print(f"✓ Dataset saved to: {output_path}")

# Save dataset info
info_path = "data/elise/dataset_info.txt"
with open(info_path, 'w') as f:
    f.write(f"Elise Dataset Information\n")
    f.write(f"=" * 60 + "\n")
    f.write(f"Total samples: {len(dataset)}\n")
    f.write(f"Columns: {', '.join(dataset.column_names)}\n")
    f.write(f"\nDataFrame Info:\n")
    f.write(df.info().__str__())

print(f"✓ Dataset info saved to: {info_path}")
print()
print("Dataset download complete!")
