#!/bin/bash
# RunPod Setup Script for Concordia + TransformerLens Deception Experiment
# Optimized for H100 PCIe (80GB VRAM)
#
# BEFORE RUNNING:
#   1. Create pod with: H100 PCIe, 30GB container, 100GB volume
#   2. Open web terminal
#   3. Run these commands:
#
#   export HF_TOKEN="your_huggingface_token"
#   export HF_HOME=/workspace/.cache/huggingface
#   cd /workspace
#   git clone https://github.com/tesims/concordia.git
#   cd concordia
#   git checkout emergent-deception-v2
#   chmod +x scripts/runpod_setup.sh
#   ./scripts/runpod_setup.sh

set -e

echo "=============================================="
echo "CONCORDIA + TRANSFORMERLENS SETUP (H100)"
echo "=============================================="

# Set persistent cache location
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
mkdir -p $HF_HOME
echo "HuggingFace cache: $HF_HOME"

# Check GPU
echo ""
echo "[1/7] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {gpu_name}')
    print(f'  VRAM: {vram:.1f} GB')
    if 'H100' in gpu_name:
        print('  H100 detected - optimal settings will be used')
    elif vram < 40:
        print('  WARNING: Less than 40GB VRAM - may have issues with 9B model')
else:
    print('  ERROR: No GPU detected!')
    exit(1)
"

# Setup repository
echo ""
echo "[2/7] Setting up repository..."
cd /workspace
if [ -d "concordia" ]; then
    echo "  Repository exists, pulling latest..."
    cd concordia
    git fetch origin
    git checkout emergent-deception-v2
    git pull origin emergent-deception-v2
else
    echo "  Cloning repository..."
    git clone https://github.com/tesims/concordia.git
    cd concordia
    git checkout emergent-deception-v2
fi

# Install Concordia
echo ""
echo "[3/7] Installing Concordia..."
pip install -e . --quiet

# Install ML dependencies
echo ""
echo "[4/7] Installing ML dependencies..."
pip install --quiet \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    transformer-lens \
    scikit-learn \
    matplotlib \
    pandas \
    huggingface_hub \
    tqdm

# Install tmux for long-running jobs
echo ""
echo "[5/7] Installing tmux (for persistent sessions)..."
apt-get update -qq && apt-get install -y -qq tmux

# HuggingFace login
echo ""
echo "[6/7] HuggingFace login..."
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set!"
    echo "  Run: export HF_TOKEN='your_token' && huggingface-cli login --token \$HF_TOKEN"
else
    huggingface-cli login --token $HF_TOKEN
    echo "  Logged in to HuggingFace"
fi

# Pre-download model to persistent cache
echo ""
echo "[7/7] Pre-downloading Gemma 9B to persistent cache..."
echo "  (This takes 5-10 minutes on first run, instant on restart)"
python3 -c "
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'google/gemma-2-9b-it'
print(f'  Downloading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
print('  Tokenizer ready')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
print('  Model loaded successfully')
print(f'  Memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB')

del model
torch.cuda.empty_cache()
print('  Cache cleared, ready to run')
"

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "IMPORTANT: Add this to your shell (or ~/.bashrc):"
echo ""
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo ""
echo "=============================================="
echo "TO RUN THE EXPERIMENT"
echo "=============================================="
echo ""
echo "Option 1: Quick test (10-15 min on H100)"
echo ""
echo "  cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation"
echo "  python run_deception_experiment.py --mode emergent --scenarios 2 --trials 10"
echo ""
echo "Option 2: Full experiment in tmux (4-6 hours on H100)"
echo ""
echo "  tmux new -s experiment"
echo "  cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation"
echo "  python run_deception_experiment.py --mode emergent --scenarios 6 --trials 100"
echo ""
echo "  # Detach with: Ctrl+B, then D"
echo "  # Reattach with: tmux attach -t experiment"
echo ""
echo "=============================================="
echo "STORAGE INFO"
echo "=============================================="
echo ""
echo "Persistent (survives restart):"
echo "  /workspace/.cache/huggingface  - Model weights (~18GB)"
echo "  /workspace/concordia           - Code repository"
echo "  /workspace/concordia/.../evaluation/experiment_output - Results"
echo ""
echo "If pod restarts, just run:"
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo "  cd /workspace/concordia"
echo "  # Continue experiment or check results"
echo ""
