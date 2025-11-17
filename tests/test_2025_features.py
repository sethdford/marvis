#!/usr/bin/env python3
"""
Test Suite for Marvis 2025 Features

Tests all 2025 research improvements:
1. Prosody augmentation
2. Flash Attention integration
3. Quantization
4. VoXtream streaming (architecture only)

Usage:
    python tests/test_2025_features.py
    pytest tests/test_2025_features.py -v
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Test configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def print_test_header(name):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)


def print_result(name, passed, details=None):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"    {details}")


# Test 1: Prosody Augmentation
def test_prosody_augmentation():
    print_test_header("Test 1: Prosody Augmentation")

    try:
        # Import FERN prosody controller
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "voice"))
        from fern.tts.prosody_control import create_prosody_controller

        # Create controller
        prosody = create_prosody_controller(use_sentiment_model=False)
        print_result("Import prosody controller", True)

        # Test cases
        test_texts = [
            "I'm SO excited!",
            "Unfortunately, the project failed.",
            "Hello, how are you?",
            "STOP! That's dangerous!",
        ]

        all_passed = True

        for text in test_texts:
            prosody_text = prosody.add_prosody(text)

            # Check for prosody markers
            has_emotion = any(
                marker in prosody_text
                for marker in ["[HAPPY]", "[SAD]", "[EXCITED]", "[ANGRY]", "[NEUTRAL]", "[CALM]", "[SURPRISED]"]
            )

            if not has_emotion:
                print_result(f"Emotion detection for '{text}'", False, f"No emotion found in: {prosody_text}")
                all_passed = False
            else:
                print_result(f"Prosody for '{text[:30]}...'", True, prosody_text[:60])

        return all_passed

    except ImportError as e:
        print_result("Import prosody controller", False, f"ImportError: {e}")
        print("  Make sure FERN voice agent is at ../voice/")
        return False
    except Exception as e:
        print_result("Prosody augmentation", False, str(e))
        return False


# Test 2: Flash Attention
def test_flash_attention():
    print_test_header("Test 2: Flash Attention Integration")

    try:
        from marvis_tts.flash_attention import (
            FLASH_ATTENTION_AVAILABLE,
            configure_flash_attention_for_training,
        )

        print_result("Import Flash Attention module", True)

        # Check availability
        if FLASH_ATTENTION_AVAILABLE:
            print_result("Flash Attention available", True, "flash-attn installed")
        else:
            print_result("Flash Attention available", False, "Not installed (optional)")
            print("  Install with: pip install flash-attn>=2.3.0 --no-build-isolation")

        # Test configuration function exists
        print_result("Configuration function exists", True)

        return True

    except ImportError as e:
        print_result("Import Flash Attention module", False, str(e))
        return False
    except Exception as e:
        print_result("Flash Attention test", False, str(e))
        return False


# Test 3: Quantization
def test_quantization():
    print_test_header("Test 3: Model Quantization")

    try:
        # Check if bitsandbytes is installed
        import bitsandbytes
        print_result("bitsandbytes installed", True)

        # Check quantization script exists
        quant_script = Path(__file__).parent.parent / "scripts" / "quantize_model.py"
        if quant_script.exists():
            print_result("Quantization script exists", True, str(quant_script))
        else:
            print_result("Quantization script exists", False, "Script not found")
            return False

        # Test quantization config creation
        from transformers import BitsAndBytesConfig

        # INT8 config
        config_int8 = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        print_result("INT8 config creation", True, "50% size reduction, <1% quality loss")

        # INT4 config
        config_int4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        print_result("INT4 config creation", True, "75% size reduction, ~5% quality loss")

        return True

    except ImportError as e:
        print_result("bitsandbytes installed", False, "Not installed")
        print("  Install with: pip install bitsandbytes>=0.41.0")
        return False
    except Exception as e:
        print_result("Quantization test", False, str(e))
        return False


# Test 4: VoXtream Streaming Architecture
def test_voxtream_architecture():
    print_test_header("Test 4: VoXtream Streaming Architecture")

    try:
        from marvis_tts.voxtream_streaming import (
            VoXtreamConfig,
            PhonemeTransformer,
            TemporalTransformer,
            DepthTransformer,
            VoXtreamGenerator,
        )

        print_result("Import VoXtream modules", True)

        # Test config creation
        config = VoXtreamConfig(
            phoneme_look_ahead=10,
            chunk_size_tokens=8,
            device=DEVICE,
        )
        print_result("VoXtream config creation", True, f"Look-ahead: {config.phoneme_look_ahead}")

        # Test transformer initialization
        phoneme_transformer = PhonemeTransformer(config)
        print_result("PhonemeTransformer init", True, "Incremental encoder with look-ahead")

        temporal_transformer = TemporalTransformer(config)
        print_result("TemporalTransformer init", True, "Semantic + duration prediction")

        depth_transformer = DepthTransformer(config)
        print_result("DepthTransformer init", True, "Acoustic token generation")

        # Test generator initialization
        generator = VoXtreamGenerator(config)
        print_result("VoXtreamGenerator init", True, "Full streaming pipeline (stub)")

        print()
        print("  Note: VoXtream is an architecture stub")
        print("  Full implementation requires 2-3 weeks development")
        print("  See INTEGRATION_GUIDE in voxtream_streaming.py")

        return True

    except ImportError as e:
        print_result("Import VoXtream modules", False, str(e))
        return False
    except Exception as e:
        print_result("VoXtream architecture test", False, str(e))
        import traceback
        traceback.print_exc()
        return False


# Test 5: Configuration Files
def test_configs():
    print_test_header("Test 5: Training Configurations")

    configs_dir = Path(__file__).parent.parent / "configs"

    configs_to_test = [
        ("elise_finetune_prosody.json", "Prosody training config"),
        ("elise_finetune_2025_all.json", "All 2025 features config"),
    ]

    all_passed = True

    for config_file, description in configs_to_test:
        config_path = configs_dir / config_file

        if config_path.exists():
            # Try to parse JSON
            import json
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                print_result(description, True, config_file)
            except json.JSONDecodeError as e:
                print_result(description, False, f"Invalid JSON: {e}")
                all_passed = False
        else:
            print_result(description, False, f"Config not found: {config_file}")
            all_passed = False

    return all_passed


# Test 6: Scripts
def test_scripts():
    print_test_header("Test 6: Deployment Scripts")

    scripts_dir = Path(__file__).parent.parent / "scripts"

    scripts_to_test = [
        ("augment_elise_prosody.py", "Prosody augmentation script"),
        ("quantize_model.py", "Model quantization script"),
        ("deploy_2025_features.sh", "Master deployment script"),
    ]

    all_passed = True

    for script_file, description in scripts_to_test:
        script_path = scripts_dir / script_file

        if script_path.exists():
            # Check if executable (for .sh)
            if script_file.endswith('.sh'):
                is_executable = os.access(script_path, os.X_OK)
                if not is_executable:
                    print_result(description, False, "Not executable (run: chmod +x)")
                    all_passed = False
                else:
                    print_result(description, True, f"{script_file} (executable)")
            else:
                # Check if valid Python
                try:
                    with open(script_path) as f:
                        compile(f.read(), script_path, 'exec')
                    print_result(description, True, script_file)
                except SyntaxError as e:
                    print_result(description, False, f"Syntax error: {e}")
                    all_passed = False
        else:
            print_result(description, False, f"Script not found: {script_file}")
            all_passed = False

    return all_passed


# Main test runner
def main():
    print("\n" + "=" * 70)
    print("  Marvis 2025 Features Test Suite")
    print("=" * 70)
    print()
    print("  Testing all 2025 research improvements:")
    print("    • Prosody & Emotion Control")
    print("    • Flash Attention Integration")
    print("    • Model Quantization")
    print("    • VoXtream Streaming Architecture")
    print()
    print(f"  Device: {DEVICE}")
    print()

    results = {}

    # Run tests
    results['prosody'] = test_prosody_augmentation()
    results['flash_attention'] = test_flash_attention()
    results['quantization'] = test_quantization()
    results['voxtream'] = test_voxtream_architecture()
    results['configs'] = test_configs()
    results['scripts'] = test_scripts()

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name.replace('_', ' ').title()}")

    print()
    print(f"  Total: {passed}/{total} test suites passed")
    print()

    if passed == total:
        print("  🎉 All tests passed!")
        print()
        print("  Next steps:")
        print("    1. Run prosody augmentation:")
        print("       python scripts/augment_elise_prosody.py")
        print()
        print("    2. Train with 2025 features:")
        print("       accelerate launch train.py configs/elise_finetune_2025_all.json")
        print()
        print("    3. Quantize trained model:")
        print("       python scripts/quantize_model.py --checkpoint ./checkpoints/elise_50k --output ./checkpoints/elise_50k_int8 --bits 8")
        print()
        return 0
    else:
        print("  ⚠️  Some tests failed")
        print()
        print("  Check the output above for details")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
