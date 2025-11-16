#!/usr/bin/env python3
"""
Test script to verify Marvis TTS setup is working correctly.
"""

print("Testing Marvis TTS setup...")
print("-" * 50)

# Test 1: Import core modules
try:
    from marvis_tts.generator import Generator
    from marvis_tts.models import Model, ModelArgs
    from marvis_tts.utils import Segment, load_smollm2_tokenizer
    print("✓ Core modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import core modules: {e}")
    exit(1)

# Test 2: Import torch and check version
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded successfully")
    print(f"  Device available: {'CUDA' if torch.cuda.is_available() else 'CPU (MPS on Apple Silicon)'}")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    exit(1)

# Test 3: Import dependencies
try:
    import moshi
    import einops
    import torchao
    print("✓ All additional dependencies loaded (moshi, einops, torchao)")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    exit(1)

# Test 4: Check if scripts can be imported
try:
    import inference
    import train
    print("✓ inference.py and train.py have no syntax errors")
except ImportError as e:
    print(f"✗ Failed to import scripts: {e}")
    exit(1)

print("-" * 50)
print("✓ All tests passed! Marvis TTS is ready to use.")
print()
print("Next steps:")
print("  1. Download or train a model checkpoint")
print("  2. Prepare reference audio (10 seconds) and text")
print("  3. Run inference:")
print("     python inference.py --model_path MODEL.pt \\")
print("       --ref_audio REF.wav --ref_text REF.txt \\")
print("       --text \"Your text here\" --output output.wav")
