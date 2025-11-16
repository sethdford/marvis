# RunPod Deployment Guide for Marvis TTS Training

## Quick Summary

**Local MPS (Apple Silicon)**: ~4-8 seconds/step
**RunPod GPU (RTX 4090)**: Expected ~0.5-1 second/step (5-10x faster!)

---

## Step 1: Prepare Your Files for Upload

### Create a deployment package:

```bash
cd /Users/sethford/Downloads/marvis-tts-main

# Create a tarball with everything needed
tar -czf marvis-tts-elise.tar.gz \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='wandb' \
  --exclude='checkpoints' \
  .
```

This creates a compressed archive (~20MB) containing:
- Source code
- Configuration files
- **Prepared Elise dataset** (12 WebDataset shards)
- Training scripts

---

## Step 2: Launch RunPod Instance

### Recommended Pod Configuration:

**GPU**: RTX 4090 (24GB VRAM) or RTX A6000 (48GB VRAM)
**Template**: PyTorch 2.0+
**Storage**: 50GB minimum
**Estimated Cost**: $0.44-0.79/hour

### Launch Steps:

1. Go to https://runpod.io/console/pods
2. Click **Deploy**
3. Select **GPU**: RTX 4090 or A6000
4. Choose **PyTorch** template (or Ubuntu with CUDA)
5. Set **Container Disk**: 50GB
6. Click **Deploy**

---

## Step 3: Upload and Setup on RunPod

### Once your pod is running:

```bash
# SSH into your RunPod instance (get SSH command from RunPod console)
ssh root@<your-pod-ip> -p <port> -i ~/.ssh/id_ed25519

# Update system
apt-get update && apt-get install -y git wget

# Upload your tarball (from your local machine)
# Use the RunPod web interface "Upload Files" or scp:
# scp -P <port> marvis-tts-elise.tar.gz root@<your-pod-ip>:/workspace/

# Extract on RunPod
cd /workspace
tar -xzf marvis-tts-elise.tar.gz
cd marvis-tts-main

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Step 4: Configure for GPU Training

### Update the training config for CUDA:

```bash
cat > configs/elise_finetune_gpu.json << 'EOF'
{
  "backbone_flavor": "llama-250M",
  "decoder_flavor": "llama-60M",
  "tokenizer": "smollm2",
  "dataset_repo_id": "/workspace/marvis-tts-main/data/elise_webdataset",
  "audio_num_codebooks": 32,
  "learning_rate": 1e-4,
  "max_tokens": 10000,
  "max_batch_size": 64,
  "device": "cuda",
  "precision": "bf16",
  "pad_multiple": 64,
  "decoder_fraction": 0.0625,
  "freeze_backbone": false,
  "resume_from_checkpoint": null,
  "finetune": true
}
EOF
```

---

## Step 5: Start Training

```bash
source venv/bin/activate

# Optional: Set wandb API key for experiment tracking
export WANDB_API_KEY=your_wandb_api_key

# Start training (use nohup to keep running after disconnect)
nohup accelerate launch train.py configs/elise_finetune_gpu.json > training.log 2>&1 &

# Monitor training
tail -f training.log

# Or use tmux/screen for persistent sessions
tmux new -s training
accelerate launch train.py configs/elise_finetune_gpu.json
# Detach: Ctrl+b, then d
# Reattach: tmux attach -t training
```

---

## Step 6: Monitor Training Progress

### Check wandb dashboard:
- Visit: https://wandb.ai/sethdford/marvis-tts
- View real-time loss curves, accuracy metrics

### Check local logs:
```bash
tail -f training.log
```

### Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

---

## Step 7: Download Checkpoints

### Checkpoints are saved every 10,000 steps in `checkpoints/`:

```bash
# From your local machine, download checkpoints
scp -P <port> -r root@<your-pod-ip>:/workspace/marvis-tts-main/checkpoints ./

# Or compress and download
# On RunPod:
tar -czf checkpoints.tar.gz checkpoints/
# On local machine:
scp -P <port> root@<your-pod-ip>:/workspace/marvis-tts-main/checkpoints.tar.gz ./
```

---

## Expected Training Performance

### RTX 4090 (24GB VRAM):
- **Speed**: ~0.5-1 second/step
- **Batch size**: 64 samples
- **10,000 steps**: ~3-4 hours
- **50,000 steps**: ~14-20 hours (recommended for good quality)

### Cost Estimation:
- **10K steps** @ $0.44/hr: ~$1.50-2.00
- **50K steps** @ $0.44/hr: ~$7.00-10.00

---

## Training Recommendations

### For Voice Cloning Quality:

1. **Minimum viable**: 10,000 steps (~3-4 hours, ~$2)
2. **Good quality**: 50,000 steps (~14-20 hours, ~$8-10)
3. **Production quality**: 100,000+ steps (~30-40 hours, ~$15-20)

### Checkpointing Strategy:

Checkpoints are automatically saved every 10,000 steps. Download intermediate checkpoints to test voice quality before continuing.

---

## Testing a Checkpoint

### On RunPod or locally:

```bash
source venv/bin/activate

python inference.py \
  --checkpoint checkpoints/model_10000.pt \
  --prompt "Hello, this is a test of the fine-tuned Elise voice." \
  --output test_output.wav
```

---

## Troubleshooting

### Out of Memory Error:
```json
// Reduce in configs/elise_finetune_gpu.json:
{
  "max_tokens": 8000,
  "max_batch_size": 48
}
```

### Slow data loading:
- Ensure dataset is on fast storage (NVMe SSD)
- Increase num_workers in train.py if needed

### Connection timeout:
- Use `tmux` or `screen` for persistent sessions
- Run with `nohup` to continue after disconnect

---

## Alternative: Jupyter Notebook on RunPod

If you prefer a web interface:

1. Deploy RunPod with Jupyter template
2. Upload files via web interface
3. Open terminal in Jupyter
4. Run commands from Step 3 onwards

---

## Comparison Summary

| Platform | Speed/Step | 10K Steps | 50K Steps | Cost (50K) |
|----------|-----------|-----------|-----------|------------|
| CPU (local) | 72s | 8 days | 42 days | N/A |
| MPS (M1/M2/M3) | 4-8s | 11-22 hrs | 56-111 hrs | N/A |
| RTX 4090 (RunPod) | 0.5-1s | 3-4 hrs | 14-20 hrs | ~$7-10 |
| RTX A6000 (RunPod) | 0.4-0.8s | 2-3 hrs | 11-16 hrs | ~$12-15 |

---

## Next Steps After Training

1. Download best checkpoint
2. Test voice quality with various prompts
3. If quality is good, use for inference
4. If more training needed, resume from checkpoint

---

## Contact & Support

- **RunPod Docs**: https://docs.runpod.io/
- **Marvis TTS Issues**: https://github.com/Marvis-AI/marvis-tts/issues
- **Wandb Dashboard**: https://wandb.ai/sethdford/marvis-tts
