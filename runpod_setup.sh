#!/bin/bash
# RunPod Setup Script for Marvis TTS Training
# Auto-configures environment and downloads dataset

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║         🚀 Marvis TTS RunPod Setup - Elise Voice Training 🚀       ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: System dependencies
echo -e "${BLUE}[1/6]${NC} Installing system dependencies..."
apt-get update -qq
apt-get install -y git wget curl tmux -qq
echo -e "${GREEN}✓${NC} System dependencies installed"
echo ""

# Step 2: Python environment
echo -e "${BLUE}[2/6]${NC} Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Step 3: Install dependencies
echo -e "${BLUE}[3/6]${NC} Installing Python dependencies..."
echo "  This may take a few minutes..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Step 4: Verify CUDA
echo -e "${BLUE}[4/6]${NC} Verifying CUDA setup..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
print(f'✓ CUDA version: {torch.version.cuda}')
print(f'✓ PyTorch version: {torch.__version__}')
"
echo -e "${GREEN}✓${NC} CUDA verified"
echo ""

# Step 5: Download and prepare dataset
echo -e "${BLUE}[5/6]${NC} Downloading Elise dataset..."
echo "  Dataset: Jinsaryko/Elise (1,195 samples)"
echo "  This will take 5-10 minutes..."

python3 << 'EOF'
import sys
from pathlib import Path

print("  → Downloading dataset from HuggingFace...")
exec(open('prepare_elise_for_training_v2.py').read())
print("\n  ✓ Dataset prepared successfully!")
EOF

echo -e "${GREEN}✓${NC} Dataset ready"
echo ""

# Step 6: Verify setup
echo -e "${BLUE}[6/6]${NC} Verifying setup..."

# Check dataset
DATASET_DIR="data/elise_webdataset"
if [ -d "$DATASET_DIR" ]; then
    SHARD_COUNT=$(find "$DATASET_DIR" -name "*.tar" | wc -l)
    echo -e "  ✓ Dataset: $SHARD_COUNT shards found"
else
    echo -e "  ${YELLOW}⚠ Warning: Dataset directory not found${NC}"
fi

# Check config
if [ -f "configs/elise_finetune_gpu.json" ]; then
    echo "  ✓ Training config: GPU config ready"
else
    echo -e "  ${YELLOW}⚠ Warning: GPU config not found${NC}"
fi

echo -e "${GREEN}✓${NC} Verification complete"
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              ✅  Setup Complete!  ✅                                ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Ready to Train!"
echo ""
echo "📊 Dataset Information:"
echo "  • Name: Elise (Ceylia voice)"
echo "  • Samples: 1,195 audio clips"
echo "  • Format: WebDataset (12 shards)"
echo "  • Codec: Mimi 24kHz, 32 codebooks"
echo ""
echo "🚀 Start Training:"
echo ""
echo "  Option 1: Quick start (in tmux):"
echo "    tmux new -s training"
echo "    source venv/bin/activate"
echo "    accelerate launch train.py configs/elise_finetune_gpu.json"
echo "    # Detach: Ctrl+B, then D"
echo ""
echo "  Option 2: Direct (foreground):"
echo "    source venv/bin/activate"
echo "    accelerate launch train.py configs/elise_finetune_gpu.json"
echo ""
echo "📈 Monitor Training:"
echo "  • Wandb: https://wandb.ai/your-username/marvis-tts"
echo "  • GPU usage: watch -n 1 nvidia-smi"
echo "  • Logs: tail -f wandb/latest-run/logs/debug.log"
echo ""
echo "⏱️  Estimated Time:"
echo "  • 50,000 steps: 14-20 hours on RTX 4090"
echo ""
echo "💰 Estimated Cost:"
echo "  • RTX 4090 @ \$0.44/hr: ~\$7-10 total"
echo ""
echo "🎉 Ready to create Elise's voice!"
echo ""
