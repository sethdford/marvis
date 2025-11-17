# 🚀 Marvis TTS - 2025 Research Integration Roadmap

**Applying cutting-edge 2025 research to make Marvis the BEST open-source TTS!**

Based on the breakthroughs we implemented in FERN, here's what we can apply to Marvis to create a state-of-the-art, ultra-low-latency TTS system.

---

## 🎯 What We Can Apply from FERN 2025 Research

### 1. **VoXtream Streaming Architecture** ⚡ (HIGHEST IMPACT)

**What it is**: 3-transformer architecture for 102ms initial delay

**Current Marvis**: Generates full utterance before outputting audio
- ❌ ~400-600ms before first audio plays
- ❌ User waits for complete synthesis
- ❌ Feels "slow" even though it's fast

**With VoXtream Integration**:
- ✅ **102ms initial delay** on GPU
- ✅ Starts speaking after FIRST word
- ✅ Word-level streaming synthesis
- ✅ Perceived as "instant"

**Implementation Plan**:

```
Current Marvis Pipeline:
Text → Llama Backbone → Decoder → Full Audio Codes → Mimi Decode → Audio
        (wait for all)                                            ↓
                                                              400ms delay

VoXtream Marvis Pipeline:
Text → Phoneme Transformer → Temporal Transformer → Depth Transformer → Stream!
       (incremental)         (semantic+duration)     (acoustic codes)    ↓
       10-phoneme look-ahead                         8 tokens/chunk  102ms delay!
```

**Architecture Changes**:
1. **Phoneme Transformer** - Replace text embedding with incremental phoneme encoder
2. **Temporal Transformer** - Add stay/go prediction flags for streaming
3. **Depth Transformer** - Generate acoustic tokens in chunks (8 tokens = ~50ms)
4. **Streaming Mimi Decode** - Decode chunks as they arrive

**Files to Modify**:
- `marvis_tts/models.py` - Add streaming transformers
- `marvis_tts/generator.py` - Implement incremental generation
- `marvis_tts/streaming.py` - NEW: Streaming inference wrapper

**Research**: [VoXtream (arXiv:2509.15969)](https://arxiv.org/abs/2509.15969)

---

### 2. **Prosody & Emotion Control** 🎭 (HIGH IMPACT)

**What it is**: Train Marvis to understand emotion codes and prosody markers

**Current Marvis**: Plain text input
- ❌ No control over emotion
- ❌ No emphasis on important words
- ❌ Monotone delivery

**With Prosody Training**:
- ✅ Emotion codes: `[HAPPY] Hello!` `[SAD] I'm sorry.`
- ✅ Emphasis: `I'm [EMPHASIS]SO[/EMPHASIS] excited!`
- ✅ Pauses: `Wait...[PAUSE:500ms] okay!`
- ✅ Natural, expressive speech

**Implementation Plan**:

**Phase 1: Data Preparation**
```python
# Augment Elise dataset with prosody markers
from fern.tts.prosody_control import create_prosody_controller

prosody = create_prosody_controller()

# For each sample in dataset:
text = "I'm so excited to share this!"
prosody_text = prosody.add_prosody(text)
# → "[EXCITED] I'm [EMPHASIS]so[/EMPHASIS] excited to share this[PAUSE:200ms]!"

# Save augmented dataset
augmented_dataset.append({
    'text': prosody_text,
    'audio_tokens': audio_tokens,
    'speaker': 0
})
```

**Phase 2: Model Training**
- Train Marvis on prosody-augmented dataset
- Model learns to map `[EXCITED]` → excited voice quality
- Model learns to map `[PAUSE:200ms]` → silence tokens

**Phase 3: Inference**
```python
# Users can control prosody!
marvis.generate("[HAPPY] Hello, how are you today?")
marvis.generate("[SAD] Unfortunately, I have bad news.")
marvis.generate("This is [EMPHASIS]really[/EMPHASIS] important!")
```

**Files to Create**:
- `scripts/augment_elise_prosody.py` - Add prosody to dataset
- `marvis_tts/prosody.py` - Prosody marker tokenizer
- Update `configs/elise_finetune.json` - Use augmented dataset

**Inspired by**: [Chatterbox (resemble-ai/chatterbox)](https://github.com/resemble-ai/chatterbox)

---

### 3. **Emotion Embeddings** 🎨 (MEDIUM IMPACT)

**What it is**: Add emotion as a conditioning signal (like speaker embeddings)

**Architecture**:
```python
class MarvisWithEmotion(nn.Module):
    def __init__(self):
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        self.emotion_embeddings = nn.Embedding(num_emotions, embedding_dim)  # NEW!

    def forward(self, text_tokens, speaker_id, emotion_id):
        speaker_emb = self.speaker_embeddings(speaker_id)
        emotion_emb = self.emotion_embeddings(emotion_id)  # NEW!

        # Combine embeddings
        conditioning = speaker_emb + emotion_emb  # or concatenate

        # Generate with emotion control
        audio_codes = self.generate(text_tokens, conditioning)
```

**Emotion Types**:
```python
EMOTIONS = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "excited",
    4: "angry",
    5: "surprised",
    6: "calm"
}
```

**Inference**:
```python
marvis.generate(
    text="Hello world!",
    speaker_id=0,       # Elise voice
    emotion_id=1        # Happy emotion
)
```

**Files to Modify**:
- `marvis_tts/models.py` - Add emotion embeddings
- `marvis_tts/generator.py` - Support emotion_id parameter
- `configs/elise_finetune.json` - Add emotion vocab size

---

### 4. **SyncSpeech Dual-Stream Architecture** 🔄 (ADVANCED)

**What it is**: Parallel text + speech streams for 6.4-8.5x faster generation

**Current Marvis**: Sequential generation
```
Text tokens → [wait] → Semantic tokens → [wait] → Acoustic tokens → [wait] → Audio
     T1          T2          T3              T4          T5             T6      T7
```

**SyncSpeech Approach**: Dual-stream (generates speech tokens while reading text!)
```
Text stream:     T1 → T2 → T3 → T4 → T5 → T6 → T7
                  ↓    ↓    ↓    ↓    ↓    ↓    ↓
Speech stream:   S1   S2   S3   S4   S5   S6   S7
                 (start at T2! look-ahead q=1)
```

**Performance**:
- Generates speech from **2nd text token** (vs waiting for all text)
- 6.4-8.5x faster than sequential
- Look-ahead mechanism (q=1) for quality

**Implementation**: More complex, requires architectural changes to Llama backbone

**Research**: [SyncSpeech (arXiv:2502.11094)](https://arxiv.org/abs/2502.11094)

---

### 5. **Flash Attention for Training** ⚡ (PERFORMANCE)

**What it is**: Memory-efficient attention for 2-4x training speedup

**Current**: Standard PyTorch attention
- ❌ O(n²) memory usage
- ❌ Slower on long sequences
- ❌ Limited by VRAM

**With Flash Attention**:
- ✅ 2-4x faster training
- ✅ 10-20x less memory
- ✅ Train with longer sequences
- ✅ Larger batch sizes

**Installation**:
```bash
pip install flash-attn --no-build-isolation
```

**Integration**:
```python
# In marvis_tts/models.py
from flash_attn import flash_attn_qkvpacked_func

class FlashAttentionLlama(LlamaForCausalLM):
    def forward(self, ...):
        # Replace standard attention
        attn_output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            causal=True
        )
```

**Files to Modify**:
- `marvis_tts/models.py` - Add Flash Attention
- `requirements.txt` - Add flash-attn
- `train.py` - Enable Flash Attention flag

---

### 6. **Model Quantization** 📦 (DEPLOYMENT)

**What it is**: Reduce model size for faster inference

**Quantization Levels**:
- **FP16** (current): 500MB model, baseline speed
- **INT8**: 250MB model, 1.5-2x faster
- **INT4**: 125MB model, 2-3x faster (slight quality loss)

**Implementation**:
```python
from transformers import BitsAndBytesConfig

# INT8 quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = MarvisTTS.from_pretrained(
    "your-checkpoint",
    quantization_config=quantization_config,
    device_map="auto"
)

# INT4 quantization (more aggressive)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

**Trade-offs**:
- INT8: Negligible quality loss, 2x faster
- INT4: 5-10% quality loss, 3x faster

**Files to Create**:
- `scripts/quantize_model.py` - Quantization script
- `marvis_tts/quantization.py` - Quantized inference

---

### 7. **Voice Cloning Improvements** 🎤 (FUTURE)

**What it is**: Better speaker embeddings for few-shot voice cloning

**Current Elise Training**: Single speaker (Ceylia)
- ✅ Great for Elise voice
- ❌ Can't clone other voices

**Multi-Speaker Training**:
1. **Train on LibriTTS** (2,456 speakers, 585 hours)
2. **Add speaker encoder** (like ResembleAI)
3. **Few-shot cloning** (3-10 seconds of audio)

**Architecture**:
```python
# Speaker encoder (extracts voice characteristics)
speaker_encoder = SpeakerEncoder()

# Extract embedding from reference audio
reference_audio = load_audio("target_voice.wav")
speaker_embedding = speaker_encoder(reference_audio)

# Generate in target voice
marvis.generate(
    text="Hello world!",
    speaker_embedding=speaker_embedding  # Clone this voice!
)
```

**Datasets for Multi-Speaker**:
- LibriTTS (2,456 speakers, free)
- VCTK (110 speakers, free)
- LibriLight (60,000 hours, free)

---

## 🗓️ Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ **Flash Attention** - 2-4x training speedup
2. ✅ **Prosody Data Augmentation** - Add markers to Elise dataset
3. ✅ **Quantization** - INT8 for faster inference

### Phase 2: Streaming (2-3 weeks)
1. ✅ **VoXtream Architecture** - Incremental generation
2. ✅ **Streaming Inference** - 102ms initial delay
3. ✅ **Chunked Mimi Decode** - Stream audio output

### Phase 3: Advanced Features (3-4 weeks)
1. ✅ **Emotion Embeddings** - Emotion conditioning
2. ✅ **SyncSpeech Dual-Stream** - Parallel generation
3. ✅ **Voice Cloning** - Multi-speaker training

---

## 📊 Expected Performance Gains

| Improvement | Metric | Before | After | Gain |
|-------------|--------|--------|-------|------|
| **VoXtream Streaming** | Initial delay | 400ms | 102ms | **-75%** |
| **Flash Attention** | Training speed | Baseline | 2-4x faster | **2-4x** |
| **Quantization (INT8)** | Inference speed | Baseline | 1.5-2x faster | **1.5-2x** |
| **Quantization (INT8)** | Model size | 500MB | 250MB | **-50%** |
| **Prosody Control** | Speech naturalness | 7/10 | 9/10 | **+29%** |
| **SyncSpeech** | Generation speed | Baseline | 6.4-8.5x faster | **6-8x** |

---

## 🚀 Getting Started

### Option A: Quick Wins First (Recommended)

```bash
# 1. Add Flash Attention to training
pip install flash-attn --no-build-isolation

# 2. Augment Elise dataset with prosody
python scripts/augment_elise_prosody.py

# 3. Train with prosody
accelerate launch train.py configs/elise_finetune_prosody.json

# 4. Quantize for deployment
python scripts/quantize_model.py --checkpoint ./checkpoints/elise_50k
```

### Option B: Full VoXtream Integration (Advanced)

```bash
# 1. Implement streaming architecture
# (Requires modifying marvis_tts/models.py)

# 2. Train streaming model
accelerate launch train.py configs/elise_finetune_streaming.json

# 3. Test streaming inference
python test_streaming.py
```

---

## 📁 New Files to Create

```
marvis-tts/
├── scripts/
│   ├── augment_elise_prosody.py      # Add prosody to dataset
│   ├── quantize_model.py             # INT8/INT4 quantization
│   └── test_streaming.py             # Test streaming inference
├── marvis_tts/
│   ├── streaming.py                  # VoXtream streaming (NEW)
│   ├── prosody.py                    # Prosody marker handling (NEW)
│   ├── quantization.py               # Quantized inference (NEW)
│   └── flash_attention.py            # Flash Attention integration (NEW)
├── configs/
│   ├── elise_finetune_prosody.json   # With prosody markers
│   ├── elise_finetune_streaming.json # Streaming architecture
│   └── elise_finetune_emotion.json   # With emotion embeddings
└── MARVIS_2025_ROADMAP.md            # This file!
```

---

## 🎓 Research Papers to Reference

1. **VoXtream** (Sept 2025) - [arXiv:2509.15969](https://arxiv.org/abs/2509.15969)
   - 102ms streaming TTS architecture
2. **SyncSpeech** (Feb 2025) - [arXiv:2502.11094](https://arxiv.org/abs/2502.11094)
   - Dual-stream parallel generation
3. **Chatterbox** - [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)
   - Emotion exaggeration control
4. **Flash Attention** - [dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
   - Memory-efficient attention

---

## 💡 Which Should We Implement First?

**My Recommendation**: Start with **Prosody + Flash Attention**

**Why?**
1. **Prosody** is easy to implement (just data augmentation)
2. **Flash Attention** gives immediate training speedup
3. Both work with existing Marvis architecture (no major changes)
4. You can train a prosody-aware Elise voice in 50K steps

**Then**: Move to **VoXtream Streaming** for ultra-low latency

**Finally**: Add **Emotion Embeddings** and **Voice Cloning**

---

## 🤔 Questions?

- **Q**: Will prosody markers work with current Marvis?
  - **A**: Yes! Just need to augment the dataset and retrain.

- **Q**: Is VoXtream compatible with Llama backbone?
  - **A**: Yes, but requires architectural modifications.

- **Q**: Will quantization hurt quality?
  - **A**: INT8 has negligible loss (<1%), INT4 has ~5-10% loss.

- **Q**: Can we do streaming without VoXtream?
  - **A**: Yes! We can implement chunked generation with current architecture.

---

## 🎉 Let's Build the Best Open-Source TTS!

**What do you want to implement first?**

1. 🎭 **Prosody Control** (easiest, high impact)
2. ⚡ **VoXtream Streaming** (complex, highest impact)
3. 🔥 **Flash Attention** (easy, immediate training speedup)
4. 📦 **Quantization** (easy, faster inference)
5. 🎨 **Emotion Embeddings** (medium, expressive control)

Let me know and we'll start implementing! 🚀

---

**Created**: November 16, 2025
**Status**: 📝 Roadmap - Ready to implement!
