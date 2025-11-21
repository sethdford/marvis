# THE ROOT CAUSE: Missing Token Mask

## 🎭 The "Ghostly" Sound Problem

The **horrifying ghostly/evil spirits sound** was caused by **mixing embeddings without masking**.

## What Was Wrong

### Original PyTorch Implementation (CORRECT):
```python
# marvis_tts/models.py line 247-249
embeds = self._embed_tokens(tokens)
masked_embeds = embeds * tokens_mask.unsqueeze(-1)  # 🔥 CRITICAL!
h = masked_embeds.sum(dim=2)
```

The mask zeros out:
- **Audio embeddings** when only text is present
- **Text embeddings** when only audio is present

### Our MLX Implementation (WRONG):
```python
# marvis_mlx/models.py (before fix)
def embed_tokens(self, tokens):
    combined = mx.concatenate([audio_embeds, text_embeds[:, :, None, :]], axis=2)
    return combined.sum(axis=2)  # ❌ NO MASK! Summing everything!
```

## Why This Caused "Ghostly" Sound

### Text Frames (no audio):
- **Should be**: `text_embedding * 1 + (32 × audio_embedding * 0) = text_embedding`
- **We had**: `text_embedding + (32 × audio_embedding_of_zero_tokens) = GARBAGE`
  
The embedding of token 0 is NOT a zero vector! It's a learned embedding that adds noise.

### Audio Frames (no text):
- **Should be**: `(32 × audio_embedding) * 1 + text_embedding * 0 = 32 audio embeddings`
- **We had**: `(32 × audio_embedding) + text_embedding_of_zero_token = GARBAGE`

Again, token 0's embedding pollutes the sum.

### Result:
Every frame had **contaminated embeddings** → Model generated **corrupted audio codes** → Mimi decoded them as **ghostly/demonic sounds**.

## The Fix

### 1. Added mask parameter to `embed_tokens`:
```python
def embed_tokens(self, tokens: mx.array, mask: mx.array = None) -> mx.array:
    combined = mx.concatenate([audio_embeds, text_embeds[:, :, None, :]], axis=2)
    
    # CRITICAL: Apply mask before summing
    if mask is not None:
        combined = combined * mask[..., None]
    
    return combined.sum(axis=2)
```

### 2. Create masks in generator for text frames:
```python
# Text frame mask: only text column is active
text_mask = mx.zeros((1, L_text, 33), dtype=mx.bool_)
text_mask[:, :, -1] = True  # Only text
embeds = self.model.embed_tokens(text_frames, text_mask)
```

### 3. Create masks in generator for audio frames:
```python
# Audio frame mask: only audio columns are active
audio_mask = mx.ones((1, 1, 33), dtype=mx.bool_)
audio_mask[:, :, -1] = False  # No text
embeds = self.model.embed_tokens(next_frame, audio_mask)
```

## Complete List of What We Missed/Got Wrong

### 🔥 CRITICAL (Caused Ghostly Sound):
1. **Missing tokens_mask** - Never masking embeddings before summing
2. **No mask parameter** in `embed_tokens`
3. **No mask creation** in generator

### ⚠️ IMPORTANT (Lower Audio Quality):
4. **No top-k sampling** - Using simple categorical instead of top-50 filtering
5. **Wrong temperature** - 0.7 instead of 0.9
6. **Missing `set_num_codebooks(32)`** - Never configured Mimi
7. **Wrong EOS detection** - Only checking c0 instead of all codebooks
8. **Small chunk size** - 4 instead of 8 frames

### 📋 NICE-TO-HAVE (Not Blocking):
9. **No context support** - Can't maintain voice across turns
10. **No KV-cache system** - Missing speed optimization
11. **No position tracking** - Not using positional embeddings properly
12. **Missing `reset_caches()`** - No cache management

## Expected Result

With the mask fix applied, audio should now be:
- ✅ **Clear** (no ghost/demon sounds)
- ✅ **Natural** (correct temperature + top-k)
- ✅ **Stable** (proper Mimi config)
- ✅ **Consistent** (proper termination)

The embedding contamination was **THE** root cause of the horrible quality.

## Technical Deep Dive

### Why Embedding of Token 0 ≠ Zero Vector

In neural networks, even "padding" or "zero" tokens have learned embeddings:

```python
# Token 0 is learned, not zeros:
self.text_embeddings = nn.Embedding(vocab_size, dim)
# embedding(0) → [-0.05, 0.12, -0.08, ...] (random learned values)
```

When we summed without masking:
```python
# Text frame (should be pure text):
sum = text_emb + audio_emb[0] + audio_emb[0] + ... (32 times)
    = text_emb + 32 × random_vector  # CONTAMINATED!
```

This random contamination confused the model completely, causing it to generate nonsensical audio codes that Mimi decoded as horrible ghostly sounds.

## Lesson Learned

**ALWAYS match the original implementation exactly**, especially for:
- Masking/attention patterns
- Embedding computations
- Token handling

A "small" difference like missing a mask can completely destroy output quality.

