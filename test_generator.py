#!/usr/bin/env python3
"""
Test script to isolate audio generation issues.
"""
import mlx.core as mx
import numpy as np
import soundfile as sf
from marvis_mlx.generator import Generator
from marvis_tts.utils import load_smollm2_tokenizer

def test_generation():
    checkpoint_path = "checkpoints/marvis_mlx.safetensors"
    text = "Hello. This is a test of the Marvis text to speech system."
    output_file = "test_output.wav"
    
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = load_smollm2_tokenizer()
    generator = Generator(checkpoint_path, tokenizer)
    
    print(f"Generating audio for: '{text}'")
    
    audio_parts = []
    
    try:
        stream = generator.generate_stream(
            text=text,
            speaker=0,
            max_audio_length_ms=5000,
            chunk_size=4
        )
        
        for i, chunk in enumerate(stream):
            print(f"Received chunk {i}, shape: {chunk.shape}, range: [{chunk.min():.4f}, {chunk.max():.4f}]")
            audio_parts.append(chunk)
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()
    
    if audio_parts:
        full_audio = np.concatenate(audio_parts)
        print(f"Saving {len(full_audio)} samples to {output_file}...")
        sf.write(output_file, full_audio, 24000)
        print("Done.")
    else:
        print("No audio generated.")

if __name__ == "__main__":
    test_generation()
