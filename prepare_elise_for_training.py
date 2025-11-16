#!/usr/bin/env python3
"""
Prepare the Elise dataset for Marvis TTS training.
This script:
1. Loads the Elise dataset
2. Tokenizes audio using mimi codec
3. Converts to WebDataset format (.tar shards)
"""

import os
import json
import io
import tarfile
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from moshi.models import loaders
from marvis_tts.utils import load_smollm2_tokenizer

print("=" * 70)
print("Preparing Elise Dataset for Marvis TTS Training")
print("=" * 70)

# Configuration
OUTPUT_DIR = Path("data/elise_webdataset")
SAMPLES_PER_SHARD = 100  # Number of samples per .tar file
DEVICE = "cpu"  # Use CPU for preprocessing

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load the mimi audio codec
print("\n[1/5] Loading mimi audio codec...")
try:
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=DEVICE)
    mimi.eval()
    mimi.set_num_codebooks(32)  # Marvis uses 32 codebooks
    print(f"✓ Mimi codec loaded (sample rate: {mimi.sample_rate} Hz)")
except Exception as e:
    print(f"Warning: Could not load mimi with moshi loaders: {e}")
    print("Using alternative loading method...")
    # Fall back to direct model creation
    from moshi.models.compression import MimiModel
    mimi = MimiModel.get_model()
    mimi = mimi.to(device=DEVICE)
    mimi.eval()
    mimi.set_num_codebooks(32)
    print(f"✓ Mimi codec loaded (sample rate: {mimi.sample_rate} Hz)")

# Step 2: Load text tokenizer
print("\n[2/5] Loading text tokenizer...")
text_tokenizer = load_smollm2_tokenizer()
print(f"✓ Text tokenizer loaded (vocab size: {text_tokenizer.vocab_size})")

# Step 3: Load Elise dataset
print("\n[3/5] Loading Elise dataset...")
dataset = load_dataset("Jinsaryko/Elise", split="train")
print(f"✓ Dataset loaded with {len(dataset)} samples")

# Step 4: Process and tokenize audio
print("\n[4/5] Processing audio and creating WebDataset shards...")
print(f"Creating shards with {SAMPLES_PER_SHARD} samples each")

current_shard = []
shard_index = 0
total_processed = 0
skipped_count = 0

def process_sample(idx, sample):
    """Process a single sample: tokenize audio and prepare data."""
    try:
        # Get audio data
        audio_data = sample['audio']
        audio_array = np.array(audio_data['array'], dtype=np.float32)
        sample_rate = audio_data['sampling_rate']

        # Convert to mono if stereo
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # Resample if needed (mimi expects 24kHz)
        if sample_rate != mimi.sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=mimi.sample_rate
            )

        # Convert to tensor: [batch=1, channels=1, time]
        audio_tensor = torch.tensor(audio_array[None, None, :], dtype=torch.float32, device=DEVICE)

        # Tokenize audio using mimi
        with torch.no_grad():
            audio_tokens = mimi.encode(audio_tensor)  # [batch, codebooks, frames]

        # Remove batch dimension and convert to list
        audio_tokens = audio_tokens[0].cpu().numpy().tolist()  # [32, T]

        # Get text
        text = sample['text']

        # Create WebDataset-compatible sample
        sample_key = f"elise_{idx:06d}"

        # Prepare JSON metadata
        sample_json = {
            "__key__": sample_key,
            "audio_tokens": audio_tokens,
            "text": text,
            "speaker": 0,  # Single speaker dataset
            "text_tokens_length": len(text_tokenizer.encode(text)),
            "speaker_name": sample.get('speaker_name', 'Ceylia'),
        }

        return sample_key, sample_json

    except Exception as e:
        print(f"\n  ✗ Error processing sample {idx}: {e}")
        return None, None

def write_shard(shard_data, shard_idx):
    """Write a shard to a .tar file."""
    shard_path = OUTPUT_DIR / f"elise_shard_{shard_idx:04d}.tar"

    with tarfile.open(shard_path, 'w') as tar:
        for sample_key, sample_json in shard_data:
            # Create JSON file content
            json_bytes = json.dumps(sample_json).encode('utf-8')

            # Create tarinfo for the JSON file
            json_info = tarfile.TarInfo(name=f"{sample_key}.json")
            json_info.size = len(json_bytes)

            # Add to tar
            tar.addfile(json_info, io.BytesIO(json_bytes))

    print(f"  ✓ Wrote shard {shard_idx}: {shard_path} ({len(shard_data)} samples)")

# Process all samples
for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    sample = dataset[idx]

    sample_key, sample_json = process_sample(idx, sample)

    if sample_key is None:
        skipped_count += 1
        continue

    current_shard.append((sample_key, sample_json))
    total_processed += 1

    # Write shard when it reaches the target size
    if len(current_shard) >= SAMPLES_PER_SHARD:
        write_shard(current_shard, shard_index)
        shard_index += 1
        current_shard = []

# Write remaining samples
if current_shard:
    write_shard(current_shard, shard_index)
    shard_index += 1

print(f"\n✓ Processing complete!")
print(f"  Total samples processed: {total_processed}")
print(f"  Samples skipped: {skipped_count}")
print(f"  Total shards created: {shard_index}")

# Step 5: Create dataset info file
print("\n[5/5] Creating dataset info...")
info = {
    "dataset_name": "Elise (Ceylia voice)",
    "total_samples": total_processed,
    "num_shards": shard_index,
    "samples_per_shard": SAMPLES_PER_SHARD,
    "speaker": "Ceylia",
    "language": "English",
    "audio_codec": "mimi",
    "num_codebooks": 32,
    "sample_rate": mimi.sample_rate,
}

info_path = OUTPUT_DIR / "dataset_info.json"
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"✓ Dataset info saved to: {info_path}")
print("\n" + "=" * 70)
print("Dataset Preparation Complete!")
print("=" * 70)
print(f"\nDataset location: {OUTPUT_DIR}")
print(f"Number of shards: {shard_index}")
print(f"\nNext steps:")
print(f"1. Create a training config file pointing to this dataset")
print(f"2. Run training with: accelerate launch train.py config.json")
