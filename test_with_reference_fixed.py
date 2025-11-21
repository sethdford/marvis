import torch
import json
from pathlib import Path
from marvis_tts.generator import Generator
from marvis_tts.models import Model, ModelArgs
from marvis_tts.utils import Segment, load_smollm2_tokenizer
import soundfile as sf
import argparse
import webdataset as wds


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


def get_reference_from_dataset(generator, device):
    """Get first sample from Elise dataset as reference"""
    dataset_path = Path("data/elise_webdataset")
    shards = sorted(list(dataset_path.glob("*.tar")))

    dataset = wds.WebDataset(str(shards[0]))
    sample = next(iter(dataset))

    # Parse JSON to get audio tokens and text
    sample_data = json.loads(sample['json'])
    audio_tokens = sample_data['audio_tokens']  # Already encoded by Mimi
    text = sample_data['text']
    speaker = sample_data['speaker']

    # Convert audio tokens to tensor (already encoded, no need to re-encode)
    audio_tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)

    # Tokenize text
    text_tokens = generator._text_tokenizer.encode(f"[{speaker}]{text}")

    return Segment(
        text=torch.tensor(text_tokens, dtype=torch.long, device=device),
        audio=audio_tokens_tensor[:32, :],  # Use first 32 codebooks
        speaker=speaker,
    ), text


def main():
    parser = argparse.ArgumentParser(description="Test TTS with reference from Elise dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_final.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--text", type=str,
                        default="Hello, this is a test of the trained voice model.",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = load_smollm2_tokenizer()
    print("Loading model...")
    model = create_model(tokenizer, device=device)
    model = load_checkpoint(model, Path(args.checkpoint))
    print(f"Loaded checkpoint: {args.checkpoint}")

    print("Creating generator...")
    generator = Generator(model, text_tokenizer=tokenizer, device=device)

    print("Getting reference from dataset...")
    ref_segment, ref_text = get_reference_from_dataset(generator, device)
    print(f"Reference text: {ref_text}")

    print(f"Generating: '{args.text}'")
    with torch.inference_mode():
        audio = generator.generate(
            text=args.text,
            speaker=0,
            context=[ref_segment],
            max_audio_length_ms=args.max_ms,
            temperature=args.temperature,
            topk=args.topk,
            voice_match=False,
        )

    sf.write(args.output, audio.cpu().numpy(), samplerate=24000, subtype="PCM_16")
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
