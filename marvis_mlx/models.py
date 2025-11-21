import mlx.core as mx
import mlx.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 500000
    rope_traditional: bool = True  # Changed to True for Marvis/Torchtune parity

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        
        # Use traditional=True to match Torchtune
        self.rope = nn.RoPE(args.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, D = x.shape
        
        queries = self.wq(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.wk(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.wv(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # Transpose to (B, H, L, D) for RoPE and Cache
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            # key_cache shape: (B, H, L_prev, D)
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Save unexpanded keys/values for cache (B, H_kv, L_tot, D)
        keys_cache = keys
        values_cache = values

        # GQA Support: Repeat KV to match heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            # (B, H_kv, L, D) -> (B, H_heads, L, D)
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        # Attention (B, H, L_new, L_tot)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1)
        output = scores @ values
        
        # Transpose back to (B, L, H, D) -> (B, L, D_total)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys_cache, values_cache)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Tuple[mx.array, mx.array]] = None) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        h, cache = self.attention(self.attention_norm(x), mask, cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, cache

class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[list] = None) -> Tuple[mx.array, list]:
        new_cache = []
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache else None
            x, c = layer(x, mask, c)
            new_cache.append(c)
        return self.norm(x), new_cache

class MarvisModel(nn.Module):
    def __init__(self, backbone_args: ModelArgs, decoder_args: ModelArgs, 
                 text_vocab_size: int, audio_vocab_size: int, audio_num_codebooks: int):
        super().__init__()
        self.backbone = Llama(backbone_args)
        self.decoder = Llama(decoder_args)
        
        self.text_embeddings = nn.Embedding(text_vocab_size, backbone_args.dim)
        self.audio_embeddings = nn.Embedding(audio_vocab_size * audio_num_codebooks, backbone_args.dim)
        
        self.projection = nn.Linear(backbone_args.dim, decoder_args.dim, bias=False)
        
        # Heads
        self.codebook0_head = nn.Linear(backbone_args.dim, audio_vocab_size, bias=False)
        
        # audio_head is a parameter tensor (num_codebooks-1, dim, vocab)
        # In MLX we can treat it as a parameter
        self.audio_head = mx.zeros((audio_num_codebooks - 1, decoder_args.dim, audio_vocab_size))
        
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks

    def embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        # tokens: (B, L)
        offset = codebook * self.audio_vocab_size
        return self.audio_embeddings(tokens + offset)

    def embed_tokens(self, tokens: mx.array, mask: mx.array = None) -> mx.array:
        # tokens: (B, L, C+1) where last is text, first C are audio
        # mask: (B, L, C+1) boolean mask indicating which tokens are active
        text_tokens = tokens[..., -1]
        audio_tokens = tokens[..., :-1]
        
        text_embeds = self.text_embeddings(text_tokens) # (B, L, D)
        
        # Let's reimplement Marvis embedding logic precisely
        offsets = mx.arange(self.audio_num_codebooks) * self.audio_vocab_size
        flat_audio = (audio_tokens + offsets).reshape(-1) # Flatten to look up
        # (B*L*C, D) -> (B, L, C, D)
        audio_embeds = self.audio_embeddings(flat_audio).reshape(
            audio_tokens.shape[0], audio_tokens.shape[1], self.audio_num_codebooks, -1
        )
        
        # Concatenate audio and text embeddings: (B, L, C+1, D)
        combined = mx.concatenate([audio_embeds, text_embeds[:, :, None, :]], axis=2)
        
        # CRITICAL: Apply mask before summing (prevents ghost/mixing artifacts)
        if mask is not None:
            # mask shape: (B, L, C+1) -> (B, L, C+1, 1) for broadcasting
            combined = combined * mask[..., None]
        
        # Sum over codebook dimension (axis 2) to get (B, L, D) compatible with Transformer
        return combined.sum(axis=2)

    def __call__(self, x):
        # This is a complex model, usually we call generate_step
        pass

# Configuration
def create_model():
    # Llama-250M config
    backbone_args = ModelArgs(
        dim=1536, n_layers=6, head_dim=1536//12, hidden_dim=8192, 
        n_heads=12, n_kv_heads=3, norm_eps=1e-5, vocab_size=49152
    )
    
    # Llama-60M config
    decoder_args = ModelArgs(
        dim=1024, n_layers=4, head_dim=1024//8, hidden_dim=4096, 
        n_heads=8, n_kv_heads=2, norm_eps=1e-5, vocab_size=49152
    )
    
    model = MarvisModel(
        backbone_args=backbone_args,
        decoder_args=decoder_args,
        text_vocab_size=49152, # SmolLM2 vocab
        audio_vocab_size=2051,
        audio_num_codebooks=32
    )
    return model
