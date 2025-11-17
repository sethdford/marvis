#!/usr/bin/env python3
"""
Augment Elise Dataset with Prosody Markers

This script adds prosody and emotion markers to the Elise dataset,
enabling Marvis to learn expressive, natural speech synthesis.

Based on FERN 2025 prosody control research.

Usage:
    python scripts/augment_elise_prosody.py

Output:
    data/elise_prosody_webdataset/ - Augmented WebDataset shards
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import io
import tarfile
from tqdm import tqdm

import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import MimiModel, AutoFeatureExtractor

# Import prosody controller from FERN
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "voice"))
from fern.tts.prosody_control import create_prosody_controller

from marvis_tts.utils import load_smollm2_tokenizer

print("=" * 70)
print("Augmenting Elise Dataset with Prosody Markers")
print("=" * 70)
print()
print("This will add emotion codes, emphasis, and pause markers to enable")
print("Marvis to generate expressive, natural-sounding speech!")
print()

# Configuration
OUTPUT_DIR = Path("data/elise_prosody_webdataset")
SAMPLES_PER_SHARD = 100
DEVICE = "cpu"  # Use CPU for preprocessing

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Load prosody controller
print("[1/6] Loading prosody controller...")
prosody = create_prosody_controller(
    use_sentiment_model=False,  # Use rule-based (faster)
    enable_all=True
)
print("✓ Prosody controller loaded")

# Step 2: Load mimi audio codec
print("\n[2/6] Loading mimi audio codec...")
try:
    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Mimi codec loaded")
except Exception as e:
    print(f"✗ Error loading mimi codec: {e}")
    exit(1)

# Step 3: Load text tokenizer
print("\n[3/6] Loading text tokenizer...")
text_tokenizer = load_smollm2_tokenizer()
print(f"✓ Text tokenizer loaded (vocab size: {text_tokenizer.vocab_size})")

# Step 4: Load Elise dataset
print("\n[4/6] Loading Elise dataset from HuggingFace...")
dataset = load_dataset("Jinsaryko/Elise", split="train")
print(f"✓ Dataset loaded with {len(dataset)} samples")

# Step 5: Process with prosody augmentation
print("\n[5/6] Processing audio and adding prosody markers...")
print(f"Creating shards with {SAMPLES_PER_SHARD} samples each")

current_shard = []
shard_index = 0
total_processed = 0
skipped_count = 0

# Statistics
prosody_stats = {
    "happy": 0,
    "sad": 0,
    "excited": 0,
    "angry": 0,
    "neutral": 0,
    "calm": 0,
    "surprised": 0,
    "with_emphasis": 0,
    "with_pauses": 0,
}

def process_sample(idx, sample):
    """Process a single sample: add prosody and tokenize audio."""
    try:
        # Get audio data
        audio_data = sample['audio']

        # Handle AudioDecoder type
        if hasattr(audio_data, '__class__') and 'AudioDecoder' in audio_data.__class__.__name__:
            if hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
                audio_array = np.array(audio_data.array, dtype=np.float32)
                sample_rate = audio_data.sampling_rate
            else:
                try:
                    audio_array = np.array(audio_data['array'], dtype=np.float32)
                    sample_rate = audio_data['sampling_rate']
                except:
                    print(f"\n  ✗ Cannot extract audio from AudioDecoder for sample {idx}")
                    return None, None
        elif isinstance(audio_data, dict):
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                audio_bytes = audio_data['bytes']
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            elif 'array' in audio_data:
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                sample_rate = audio_data.get('sampling_rate', 24000)
            else:
                print(f"\n  ✗ Unknown audio format for sample {idx}")
                return None, None
        else:
            print(f"\n  ✗ Unexpected audio data type for sample {idx}: {type(audio_data)}")
            return None, None

        # Convert to mono if stereo
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # Resample if needed
        if sample_rate != feature_extractor.sampling_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=feature_extractor.sampling_rate
            )

        # Preprocess with feature extractor
        inputs = feature_extractor(
            raw_audio=audio_array,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt"
        )

        # Move to device and encode
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])
            audio_codes = encoder_outputs.audio_codes

        # Remove batch dimension
        audio_tokens = audio_codes[0].cpu().numpy().tolist()

        # Get original text
        original_text = sample['text']

        # ADD PROSODY MARKERS! (This is the magic!)
        prosody_text = prosody.add_prosody(original_text)

        # Update statistics
        if "[HAPPY]" in prosody_text:
            prosody_stats["happy"] += 1
        elif "[SAD]" in prosody_text:
            prosody_stats["sad"] += 1
        elif "[EXCITED]" in prosody_text:
            prosody_stats["excited"] += 1
        elif "[ANGRY]" in prosody_text:
            prosody_stats["angry"] += 1
        elif "[SURPRISED]" in prosody_text:
            prosody_stats["surprised"] += 1
        elif "[CALM]" in prosody_text:
            prosody_stats["calm"] += 1
        else:
            prosody_stats["neutral"] += 1

        if "[EMPHASIS]" in prosody_text:
            prosody_stats["with_emphasis"] += 1

        if "[PAUSE:" in prosody_text:
            prosody_stats["with_pauses"] += 1

        # Create WebDataset-compatible sample
        sample_key = f"elise_prosody_{idx:06d}"

        sample_json = {
            "__key__": sample_key,
            "audio_tokens": audio_tokens,
            "text": prosody_text,  # Use prosody-augmented text!
            "original_text": original_text,  # Keep original for reference
            "speaker": 0,
            "text_tokens_length": len(text_tokenizer.encode(prosody_text)),
            "speaker_name": sample.get('speaker_name', 'Ceylia'),
        }

        return sample_key, sample_json

    except Exception as e:
        print(f"\n  ✗ Error processing sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def write_shard(shard_data, shard_idx):
    """Write a shard to a .tar file."""
    shard_path = OUTPUT_DIR / f"elise_prosody_shard_{shard_idx:04d}.tar"

    with tarfile.open(shard_path, 'w') as tar:
        for sample_key, sample_json in shard_data:
            # Create JSON file content
            json_bytes = json.dumps(sample_json).encode('utf-8')

            # Create tarinfo
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

    # Write shard when it reaches target size
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

# Step 6: Create dataset info
print("\n[6/6] Creating dataset info...")

# Print prosody statistics
print("\n📊 Prosody Statistics:")
print(f"  Emotion Distribution:")
for emotion, count in prosody_stats.items():
    if emotion not in ["with_emphasis", "with_pauses"]:
        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"    {emotion.upper()}: {count} ({percentage:.1f}%)")

print(f"\n  Prosody Features:")
print(f"    Samples with EMPHASIS: {prosody_stats['with_emphasis']} ({(prosody_stats['with_emphasis']/total_processed)*100:.1f}%)")
print(f"    Samples with PAUSES: {prosody_stats['with_pauses']} ({(prosody_stats['with_pauses']/total_processed)*100:.1f}%)")

info = {
    "dataset_name": "Elise with Prosody Markers (Ceylia voice)",
    "total_samples": total_processed,
    "num_shards": shard_index,
    "samples_per_shard": SAMPLES_PER_SHARD,
    "speaker": "Ceylia",
    "language": "English",
    "audio_codec": "mimi (via transformers)",
    "sample_rate": feature_extractor.sampling_rate,
    "prosody_augmentation": True,
    "prosody_stats": prosody_stats,
}

info_path = OUTPUT_DIR / "dataset_info.json"
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print(f"✓ Dataset info saved to: {info_path}")

# Example outputs
print("\n" + "=" * 70)
print("Example Prosody Augmentations:")
print("=" * 70)
print()

examples = [
    "I'm so excited to share this!",
    "Unfortunately, the project failed.",
    "Hello, how are you?",
    "STOP! That's dangerous!",
]

for example in examples:
    prosody_version = prosody.add_prosody(example)
    print(f"Original:  {example}")
    print(f"Prosody:   {prosody_version}")
    print()

print("=" * 70)
print("Dataset Preparation Complete!")
print("=" * 70)
print(f"\nDataset location: {OUTPUT_DIR}")
print(f"Number of shards: {shard_index}")
print()
print("Next steps:")
print("1. Update configs/elise_finetune.json:")
print(f"   Set 'dataset_path' to: '{OUTPUT_DIR.absolute()}/elise_prosody_shard_{{0000..{shard_index-1:04d}}}.tar'")
print("2. Run training:")
print("   accelerate launch train.py configs/elise_finetune.json")
print()
print("3. After training, generate expressive speech:")
print("   marvis.generate('[EXCITED] Hello world!')")
print("   marvis.generate('[SAD] I have bad news.')")
print("   marvis.generate('This is [EMPHASIS]really[/EMPHASIS] important!')")
print()
print("🎉 Marvis will learn to speak with emotion, emphasis, and natural pauses!")
print()
