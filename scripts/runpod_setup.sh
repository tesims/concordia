#!/bin/bash
# RunPod Setup Script for Concordia + TransformerLens Deception Experiment
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   chmod +x runpod_setup.sh
#   ./runpod_setup.sh
#
# Then run:
#   python run_experiment.py --trials 50

set -e

echo "=============================================="
echo "CONCORDIA + TRANSFORMERLENS SETUP"
echo "=============================================="

# Check GPU
echo ""
echo "[1/6] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected!')
"

# Clone repo
echo ""
echo "[2/6] Setting up repository..."
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
echo "[3/6] Installing Concordia..."
pip install -e . --quiet

# Install ML dependencies
echo ""
echo "[4/6] Installing ML dependencies..."
pip install --quiet \
    torch \
    transformers \
    accelerate \
    bitsandbytes \
    transformer-lens \
    scikit-learn \
    matplotlib \
    pandas \
    huggingface_hub

# HuggingFace login
echo ""
echo "[5/6] HuggingFace login..."
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN not set. Run: huggingface-cli login"
else
    huggingface-cli login --token $HF_TOKEN
    echo "  Logged in to HuggingFace"
fi

# Pre-download model
echo ""
echo "[6/6] Pre-downloading Gemma 9B (this takes a few minutes)..."
python3 -c "
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
print('  Model ready')
del model
torch.cuda.empty_cache()
print('  Cache cleared')
"

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "To run the experiment:"
echo ""
echo "  cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation"
echo ""
echo "  # Quick test (5 min)"
echo "  python run_deception_experiment.py --mode emergent --scenarios 2 --trials 10"
echo ""
echo "  # Full experiment (2-3 hours)"
echo "  python run_deception_experiment.py --mode emergent --scenarios 6 --trials 50"
echo ""
