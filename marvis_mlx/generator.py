import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from .models import MarvisModel, create_model
from huggingface_hub import hf_hub_download

# HYBRID APPROACH: Use PyTorch Mimi for reliable audio decoding
try:
    from moshi.models import loaders
    from marvis_tts.generator import get_mimi_patched
except ImportError:
    print("Error: moshi not installed. Run 'pip install git+https://github.com/kyutai-labs/moshi'")
    raise

# Hardcoded from moshi source
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"

def load_mimi_pytorch(device="cpu"):
    print(f"Loading Mimi (PyTorch) on {device}...")
    try:
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        try:
            # Try standard loader first
            mimi = loaders.get_mimi(mimi_weight, device=device)
        except RuntimeError:
            # Fallback to patched loader if version mismatch
            print("Standard Mimi load failed, using patched loader...")
            mimi = get_mimi_patched(mimi_weight, device=device)
    except Exception as e:
        print(f"Failed to load Mimi weights: {e}")
        return None
        
    mimi.eval()
    return mimi

def sample_topk(logits: mx.array, topk: int = 50, temperature: float = 0.9):
    """Sample from logits with top-k filtering (matching original Marvis)."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering
    if topk > 0:
        # Get the top-k values
        top_values = mx.topk(logits, k=topk)
        min_top_value = top_values[..., -1:]
        
        # Mask out values below the topk threshold
        logits = mx.where(logits >= min_top_value, logits, float('-inf'))
    
    # Sample from filtered distribution using categorical on log-probs
    # Subtract max for numerical stability before softmax
    logits_max = mx.max(logits, axis=-1, keepdims=True)
    logits = logits - logits_max
    
    return mx.random.categorical(logits)

class Generator:
    def __init__(self, model_path: str, tokenizer):
        print("Loading Marvis MLX model...")
        self.model = create_model()
        self.model.load_weights(model_path)
        self.model.eval()
        
        # Use PyTorch Mimi
        self.mimi_device = "cpu" # CPU is fast enough for decoding
        self.mimi = load_mimi_pytorch(self.mimi_device)
        
        # CRITICAL: Set number of codebooks (matching original implementation)
        if hasattr(self.mimi, 'set_num_codebooks'):
            self.mimi.set_num_codebooks(32)
        
        self.audio_num_codebooks = 32
        self.tokenizer = tokenizer

    def generate_stream(self, text: str, speaker: int, context=None, max_audio_length_ms=10000, 
                        chunk_size=8, temperature=0.9, topk=50):
        # 1. Prepare Text
        prompt_text = f"[{speaker}]{text}"
        text_ids = self.tokenizer.encode(prompt_text)
        L_text = len(text_ids)
        
        # Create Text Frames: (1, L_text, 33). Audio=0. Text=text_ids.
        text_frames = mx.zeros((1, L_text, self.audio_num_codebooks + 1), dtype=mx.int32)
        text_frames[:, :, -1] = mx.array(text_ids)
        
        # Create mask: True where we have actual tokens (only text in this case)
        text_mask = mx.zeros((1, L_text, self.audio_num_codebooks + 1), dtype=mx.bool_)
        text_mask[:, :, -1] = True  # Only text column is active
        
        # 2. Prefill Backbone
        backbone_cache = [None] * len(self.model.backbone.layers)
        
        embeds = self.model.embed_tokens(text_frames, text_mask)
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(L_text)
        h, backbone_cache = self.model.backbone(embeds, mask=causal_mask, cache=backbone_cache)
        
        # Current backbone output (from last text token)
        curr_h = h[:, -1:, :] # (1, 1, D)
        
        # 3. Generate Audio Codes (all frames first)
        all_codes = []
        
        # Max frames based on duration (80ms per frame at 12.5Hz)
        max_frames = int(max_audio_length_ms / 80) 
        
        for i in range(max_frames):
            # Predict C0 (semantic codebook) with top-k sampling
            c0_logits = self.model.codebook0_head(curr_h[:, -1, :])
            c0_token = sample_topk(c0_logits, topk=topk, temperature=temperature)
            
            # Decoder - Reset cache for each frame (critical for audio quality)
            c0_embed = self.model.embed_audio(0, c0_token[:, None])
            decoder_h = mx.concatenate([curr_h, c0_embed], axis=1) # (1, 2, D)
            
            # Fresh decoder cache for each frame (matching original implementation)
            decoder_cache = [None] * len(self.model.decoder.layers)
            decoder_tokens = [c0_token]
            
            for cb in range(1, self.audio_num_codebooks):
                decoder_input = self.model.projection(decoder_h)
                
                if cb == 1:
                    inp = decoder_input # (1, 2, D)
                else:
                    inp = decoder_input[:, -1:, :] # (1, 1, D)
                
                dec_out, decoder_cache = self.model.decoder(inp, mask=None, cache=decoder_cache)
                
                head_weight = self.model.audio_head[cb-1]
                logits = dec_out[:, -1, :] @ head_weight
                token = sample_topk(logits, topk=topk, temperature=temperature)
                decoder_tokens.append(token)
                
                next_embed = self.model.embed_audio(cb, token[:, None])
                decoder_h = next_embed
            
            # Collect codes and validate range (Mimi codebooks only support 0-2047)
            frame_codes_list = []
            for t in decoder_tokens:
                code = t.item()
                # Clip to valid range for Mimi
                if code < 0 or code >= 2048:
                    print(f"Warning: Generated code {code} out of range [0, 2047], clipping")
                    code = max(0, min(2047, code))
                frame_codes_list.append(code)
            
            # Check for EOS: all codebooks == 0 (matching original implementation)
            if all(c == 0 for c in frame_codes_list):
                break
            
            # Store frame codes: (32,)
            all_codes.append(frame_codes_list)
            
            # Prepare next backbone input (audio frame)
            frame_codes_mx = mx.array(frame_codes_list).reshape(1, 32, 1)
            next_frame = mx.zeros((1, 1, 33), dtype=mx.int32)
            next_frame[:, :, :-1] = frame_codes_mx.transpose(0, 2, 1)
            next_frame[:, :, -1] = 0  # No text token
            
            # Mask for audio frame: True for audio codebooks, False for text
            audio_mask = mx.ones((1, 1, 33), dtype=mx.bool_)
            audio_mask[:, :, -1] = False  # Text column is not active
            
            embeds = self.model.embed_tokens(next_frame, audio_mask)
            curr_h, backbone_cache = self.model.backbone(embeds, mask=None, cache=backbone_cache)

        # 4. Decode all codes at once (no streaming state issues)
        if all_codes:
            # Shape: (1, 32, T) where T is number of frames
            all_codes_np = np.array(all_codes).T  # (32, T)
            all_codes_np = all_codes_np[np.newaxis, :, :]  # (1, 32, T)
            all_codes_pt = torch.from_numpy(all_codes_np).long().to(self.mimi_device)
            
            with torch.no_grad():
                # Decode everything at once - no streaming state corruption
                decoded = self.mimi.decode(all_codes_pt)
            
            # decoded is (1, 1, total_samples)
            audio_np = decoded.cpu().numpy().squeeze()
            
            # Yield in chunks for streaming playback
            samples_per_chunk = chunk_size * 1920  # 1920 samples per frame at 24kHz
            for start_idx in range(0, len(audio_np), samples_per_chunk):
                end_idx = min(start_idx + samples_per_chunk, len(audio_np))
                yield audio_np[start_idx:end_idx]