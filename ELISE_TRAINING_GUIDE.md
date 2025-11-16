# Fine-Tuning Marvis TTS with Elise Dataset

## Current Status

✅ **Completed:**
- Downloaded Elise dataset (1,195 samples, speaker: Ceylia)
- Set up Marvis TTS environment with Python 3.13
- Fixed bugs in `inference.py`
- All dependencies installed

⚠️ **Issue Found:**
- Moshi library version incompatibility with mimi codec loading
- The installed moshi v0.2.11 has breaking changes in the model architecture

## Dataset Information

**Elise Dataset** (`Jinsaryko/Elise`)
- **Samples:** 1,195 audio clips
- **Speaker:** Ceylia (single female speaker)
- **Language:** English
- **Format:** Audio + text transcriptions
- **Quality:** High SNR (signal-to-noise ratio)
- **Location:** `data/elise/dataset.parquet`

## Solutions to Proceed

### Option 1: Use Pre-tokenized Dataset (Recommended)

Instead of tokenizing the Elise dataset yourself, look for or create a dataset that's already in WebDataset format with mimi-tokenized audio. Check the Marvis AI organization on Hugging Face for compatible datasets.

### Option 2: Fix Moshi Version Incompatibility

Try downgrading moshi to a compatible version:

```bash
source venv/bin/activate
pip uninstall moshi sphn
pip install 'moshi<0.2.0'  # Try older version
```

Then run:
```bash
python prepare_elise_for_training.py
```

### Option 3: Use Alternative Dataset

The Marvis TTS README mentions training on "Emilia-YODAS" dataset. You can:

1. Find a compatible pre-processed dataset
2. Or use the exact dataset the authors used for training

## Training Configuration

I've created a template configuration file for fine-tuning with the Elise voice:

**File:** `configs/elise_finetune.json`

```json
{
  "backbone_flavor": "llama-250M",
  "decoder_flavor": "llama-60M",
  "tokenizer": "smollm2",
  "dataset_repo_id": "YOUR_DATASET_HERE",
  "audio_num_codebooks": 32,
  "learning_rate": 1e-4,
  "max_tokens": 10000,
  "max_batch_size": 64,
  "device": "cpu",
  "precision": "bf16",
  "pad_multiple": 64,
  "decoder_fraction": 0.0625,
  "freeze_backbone": false,
  "resume_from_checkpoint": null,
  "finetune": true
}
```

## When Dataset is Ready

Once you have a compatible WebDataset format dataset:

1. **Update the configuration:**
   ```bash
   # Edit configs/elise_finetune.json
   # Set "dataset_repo_id" to your dataset path or HF repo
   ```

2. **Start training:**
   ```bash
   source venv/bin/activate
   accelerate launch train.py configs/elise_finetune.json
   ```

3. **Monitor training:**
   - Training will save checkpoints periodically
   - If wandb is configured, you'll see training metrics

## Expected WebDataset Format

Each sample in the .tar shards should contain a `.json` file with:

```json
{
  "__key__": "sample_000001",
  "audio_tokens": [[...], [...], ...],  // 32 x T array of mimi tokens
  "text": "The text transcription",
  "speaker": 0,
  "text_tokens_length": 42
}
```

## Next Steps

1. **Resolve moshi incompatibility** (try Option 2 above)
2. **Or find a pre-tokenized dataset** (Option 1)
3. **Update training config** with dataset path
4. **Run training!**

## Hardware Recommendations

- **Minimum:** 16GB RAM, GPU recommended
- **Optimal:** NVIDIA GPU with 24GB+ VRAM
- **For CPU training:** Reduce batch size in config

## Troubleshooting

### If training runs out of memory:
```json
{
  "max_tokens": 5000,      // Reduce from 10000
  "max_batch_size": 32,    // Reduce from 64
}
```

### If using CPU (Apple Silicon):
```json
{
  "device": "cpu",
  "precision": "fp32",     // Change from bf16
}
```

## Useful Commands

```bash
# Activate environment
source venv/bin/activate

# Test imports
python test_setup.py

# Check dataset
python download_elise_dataset.py

# Prepare dataset (when moshi fixed)
python prepare_elise_for_training.py

# Train
accelerate launch train.py configs/elise_finetune.json

# Resume from checkpoint
# Edit config: "resume_from_checkpoint": "path/to/checkpoint.pt"
```

## References

- Marvis TTS: https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.1
- Elise Dataset: https://huggingface.co/datasets/Jinsaryko/Elise
- Mimi Codec: https://huggingface.co/kyutai/mimi
