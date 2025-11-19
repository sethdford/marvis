import torch
from pathlib import Path
from marvis_tts.generator import Generator
from marvis_tts.models import Model, ModelArgs
from marvis_tts.utils import load_smollm2_tokenizer
import soundfile as sf
import argparse


def create_model(tokenizer, device="cuda"):
    model_args = ModelArgs(
        backbone_flavor="llama-250M",
        decoder_flavor="llama-60M",
        text_vocab_size=tokenizer.vocab_size,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    return Model(model_args).to(device=device, dtype=torch.bfloat16)


def load_checkpoint(model, ckpt_path):
    obj = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    state_dict = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Simple TTS generation without reference audio")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_100000.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--text", type=str,
                        default="Hello, this is a test of the Elise voice model.",
                        help="Text to generate")
    parser.add_argument("--output", type=str, default="test_output.wav",
                        help="Output audio file path")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_ms", type=float, default=10000,
                        help="Maximum audio length in milliseconds")
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = load_smollm2_tokenizer()
    print("Loading model...")
    model = create_model(tokenizer, device=device)
    model = load_checkpoint(model, Path(args.checkpoint))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Generate
    print(f"Generating: '{args.text}'")
    generator = Generator(model, text_tokenizer=tokenizer, device=device)
    with torch.inference_mode():
        audio = generator.generate(
            text=args.text,
            speaker=0,
            max_audio_length_ms=args.max_ms,
            temperature=args.temperature,
            topk=args.topk,
        )

    # Save
    sf.write(args.output, audio.cpu().numpy(), samplerate=24000, subtype="PCM_16")
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
