from typing import List

from tqdm import tqdm, trange

import torch

from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.modules import SEANetEncoder, SEANetDecoder, transformer
from moshi.quantization import SplitResidualVectorQuantizer
from moshi.models.compression import MimiModel
from safetensors.torch import load_file
from pathlib import Path

from .models import Model
from .utils import Segment, _tokenize_segment, _tokenize_text_segment

# Copied from moshi/models/loaders.py to allow patching
_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

def get_mimi_patched(filename, device="cpu"):
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
    decoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
    quantizer = SplitResidualVectorQuantizer(**_quantizer_kwargs)
    
    model = MimiModel(
        encoder, decoder, quantizer,
        channels=1, sample_rate=24000, frame_rate=12.5,
        encoder_frame_rate=24000 / encoder.hop_length,
        causal=True, resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()

    if filename:
        if str(filename).endswith(".safetensors"):
            state_dict = load_file(filename, device=str(device))
        else:
            state_dict = torch.load(filename, map_location="cpu")["model"]
        
        # Let moshi's _load_hook handle the weight transformations
        # The _load_hook will automatically split fused weights (in_proj_weight, out_proj.weight)
        # into the expected format (in_projs.{i}.weight, out_projs.{i}.weight)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"Warning during mimi load: {e}")
            
    return model

class Generator:
    def __init__(
        self,
        model: Model,
        text_tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = text_tokenizer

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        try:
            mimi = loaders.get_mimi(mimi_weight, device=device)
        except RuntimeError:
            print("Standard Mimi load failed, trying patched loader...")
            mimi = get_mimi_patched(mimi_weight, device=device)
            
        mimi.eval()

        mimi.set_num_codebooks(self._model.args.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self.sample_rate = mimi.sample_rate
        self.device = device

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        voice_match: bool = False,
        chunk_size: int = 8,  # Decode every N frames
    ):
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = _tokenize_segment(
                self._text_tokenizer,
                self._audio_tokenizer,
                segment,
                self._model.args.audio_num_codebooks,
                add_eos=not voice_match,
                device=self.device,
            )
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        if not voice_match:
            gen_segment_tokens, gen_segment_tokens_mask = _tokenize_text_segment(
                self._text_tokenizer, text, speaker, self._model.args.audio_num_codebooks, self.device
            )
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        pending_samples = []
        
        for _ in trange(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            pending_samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

            if len(pending_samples) >= chunk_size:
                # Decode chunk
                chunk_tensor = torch.stack(pending_samples).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(chunk_tensor)[-1, -1, :]
                yield audio_chunk.cpu().numpy()
                pending_samples = []

        # Decode remaining
        if pending_samples:
            chunk_tensor = torch.stack(pending_samples).permute(1, 2, 0)
            audio_chunk = self._audio_tokenizer.decode(chunk_tensor)[-1, -1, :]
            yield audio_chunk.cpu().numpy()

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        voice_match: bool = False,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = _tokenize_segment(
                self._text_tokenizer,
                self._audio_tokenizer,
                segment,
                self._model.args.audio_num_codebooks,
                add_eos=not voice_match,
                device=self.device,
            )
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        if not voice_match:
            gen_segment_tokens, gen_segment_tokens_mask = _tokenize_text_segment(
                self._text_tokenizer, text, speaker, self._model.args.audio_num_codebooks, self.device
            )
            tokens.append(gen_segment_tokens)
            tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in trange(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        print(f"semantic codebook 0: {torch.stack(samples)[:, 0, 0].tolist()}")

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))[-1, -1, :]
        return audio
