# RunPod Experiment Guide - Deception Detection Research

Complete guide to running the emergent deception detection experiment on RunPod.

## Pod Configuration

| Setting | Value |
|---------|-------|
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| **GPU** | **A100 80GB PCIe** (recommended) |
| **Container Disk** | 20GB |
| **Volume Disk** | 100GB+ |
| **Volume Path** | `/workspace` |

### GPU Options

| GPU | VRAM | Price/hr | Works with Gemma 9B Hybrid |
|-----|------|----------|---------------------------|
| **A100 80GB PCIe** | 80GB | $1.19/hr | ✅ Recommended |
| A100 80GB SXM | 80GB | $1.39/hr | ✅ Slightly faster |
| A40 | 48GB | $0.35/hr | ❌ OOM with hybrid mode |
| RTX 6000 Ada | 48GB | $0.74/hr | ❌ OOM with hybrid mode |

**Estimated Cost:** ~$26 for full experiment (~22 hours) on A100 80GB

---

## Step 1: Set Up Storage (IMPORTANT)

**Run this first** to avoid disk space errors. ALL model caches must go to persistent storage instead of the small container disk (20GB):

```bash
# Create persistent cache directories
mkdir -p /workspace/persistent/huggingface_cache
mkdir -p /workspace/persistent/torch_cache
mkdir -p /workspace/persistent/sae_cache
mkdir -p /workspace/persistent/pip_cache

# Set ALL cache environment variables
export HF_HOME=/workspace/persistent/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/persistent/huggingface_cache
export HF_DATASETS_CACHE=/workspace/persistent/huggingface_cache/datasets
export TORCH_HOME=/workspace/persistent/torch_cache
export SAE_LENS_CACHE=/workspace/persistent/sae_cache
export PIP_CACHE_DIR=/workspace/persistent/pip_cache

# Make it permanent (add to bashrc)
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/workspace/persistent/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/persistent/huggingface_cache
export HF_DATASETS_CACHE=/workspace/persistent/huggingface_cache/datasets
export TORCH_HOME=/workspace/persistent/torch_cache
export SAE_LENS_CACHE=/workspace/persistent/sae_cache
export PIP_CACHE_DIR=/workspace/persistent/pip_cache
EOF
source ~/.bashrc

# Symlink default cache locations to persistent storage (backup)
rm -rf ~/.cache/huggingface 2>/dev/null
ln -sf /workspace/persistent/huggingface_cache ~/.cache/huggingface

# Verify setup
echo "=== Cache Locations ==="
echo "HF_HOME: $HF_HOME"
echo "TORCH_HOME: $TORCH_HOME"
echo "SAE_LENS_CACHE: $SAE_LENS_CACHE"
df -h /workspace
```

---

## Step 2: Install Dependencies

```bash
cd /workspace && \
git clone https://github.com/tesims/concordia.git && \
cd concordia && \
git checkout hybrid-sae-experiment && \
pip install -e . && \
pip install -r concordia/prefabs/entity/negotiation/evaluation/requirements.in && \
pip install transformers==4.44.0 accelerate==0.33.0 && \
pip install huggingface_hub
```

---

## Step 3: HuggingFace Login

```bash
huggingface-cli login
```

**When prompted:** Paste your token from https://huggingface.co/settings/tokens

> **Note:** You must also accept the Gemma license at https://huggingface.co/google/gemma-2-9b-it

---

## Step 4: Validate Setup

```bash
python << 'EOF'
import torch
import os
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"HF Cache: {os.environ.get('HF_HOME', 'NOT SET')}")
print(f"Torch Cache: {os.environ.get('TORCH_HOME', 'NOT SET')}")
print(f"SAE Cache: {os.environ.get('SAE_LENS_CACHE', 'NOT SET')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    if vram < 70:
        print("⚠️  WARNING: VRAM < 70GB - may OOM with Gemma 9B hybrid mode!")
        print("   Recommended: A100 80GB ($1.19/hr)")

from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f"HuggingFace: {user['name']}")

from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner, EMERGENT_SCENARIOS
print(f"Scenarios: {list(EMERGENT_SCENARIOS.keys())}")
print("✅ All imports OK - Ready to run!")
EOF
```

**Expected output (A100 80GB):**
```
PyTorch: 2.4.0
CUDA: True
HF Cache: /workspace/persistent/huggingface_cache
Torch Cache: /workspace/persistent/torch_cache
SAE Cache: /workspace/persistent/sae_cache
GPU: NVIDIA A100 80GB PCIe
VRAM: 80.0 GB
HuggingFace: <your-username>
Scenarios: ['ultimatum_bluff', 'capability_bluff', 'hidden_value', 'info_withholding', 'promise_break', 'alliance_betrayal']
✅ All imports OK - Ready to run!
```

---

## Step 5: Quick Test (1 trial)

Verify everything works before the full run:

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/test_output && \
python -u run_deception_experiment.py \
    --mode emergent \
    --trials 1 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/test_output
```

**Expected:** Completes in ~2-5 minutes with no errors.

---

## Step 6: Full Experiment (Conference Quality)

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/full_experiment && \
mkdir -p /workspace/persistent/checkpoints && \
python -u run_deception_experiment.py \
    --mode emergent \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log
```

### What This Runs:
- **6 scenarios** × 2 conditions × 50 trials = **600 trials**
- **Theory of Mind** enabled (rich agent labels)
- **SAE feature extraction** (Gemma Scope)
- **Causal validation** (activation patching, ablation, steering)
- **Checkpointing** for crash recovery
- All output saved + logged

**Estimated time:** ~20-22 hours

---

## Step 7: Monitor Progress

In a separate terminal:

```bash
# Watch the log
tail -f /workspace/persistent/full_experiment/experiment.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## Step 8: Download Results

```bash
# Check output files
ls -la /workspace/persistent/full_experiment/

# Zip for download
cd /workspace/persistent && \
zip -r full_experiment_results.zip full_experiment/

# Results location
echo "Download from: /workspace/persistent/full_experiment_results.zip"
```

---

## Output Files

```
/workspace/persistent/full_experiment/
├── activations_emergent_YYYYMMDD_HHMMSS.pt   # Raw activations + labels
├── probe_results_v2.json                      # Probe training results
├── causal_validation_results.json             # Causal test results
├── experiment.log                             # Full log
└── probe_results_v2.png                       # Visualization
```

---

## Alternative: Run Scenarios Separately

Useful for parallel execution or crash recovery:

### Scenario 1: ultimatum_bluff
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name ultimatum_bluff \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/ultimatum_bluff
```

### Scenario 2: capability_bluff
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name capability_bluff \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/capability_bluff
```

### Scenario 3: hidden_value
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name hidden_value \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/hidden_value
```

### Scenario 4: info_withholding
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name info_withholding \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/info_withholding
```

### Scenario 5: promise_break
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name promise_break \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/promise_break
```

### Scenario 6: alliance_betrayal
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name alliance_betrayal \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/outputs/alliance_betrayal
```

### Merge Results After All Complete
```bash
python << 'EOF'
import torch
from pathlib import Path
import glob

output_base = Path("/workspace/persistent/outputs")
all_activations = {}
all_labels = {"gm_labels": [], "agent_labels": [], "scenario": []}

for pt_file in glob.glob(str(output_base / "*/activations_*.pt")):
    print(f"Loading: {pt_file}")
    data = torch.load(pt_file, weights_only=False)

    for layer, acts in data.get("activations", {}).items():
        if layer not in all_activations:
            all_activations[layer] = []
        all_activations[layer].append(acts)

    labels = data.get("labels", {})
    all_labels["gm_labels"].extend(labels.get("gm_labels", []))
    all_labels["agent_labels"].extend(labels.get("agent_labels", []))
    all_labels["scenario"].extend(labels.get("scenario", []))

for layer in all_activations:
    all_activations[layer] = torch.cat(all_activations[layer], dim=0)

merged = {"activations": all_activations, "labels": all_labels}
torch.save(merged, output_base / "merged_activations.pt")
print(f"✅ Merged {len(all_labels['gm_labels'])} samples")
EOF
```

---

## Run All Scenarios Sequentially (Loop)

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/outputs && \
for scenario in ultimatum_bluff capability_bluff hidden_value info_withholding promise_break alliance_betrayal; do
    echo "========================================"
    echo "Starting: $scenario at $(date)"
    echo "========================================"
    python -u run_deception_experiment.py \
        --mode emergent \
        --scenario-name $scenario \
        --trials 50 \
        --max-rounds 3 \
        --hybrid \
        --sae \
        --device cuda \
        --dtype bfloat16 \
        --output /workspace/persistent/outputs/$scenario
done && \
echo "✅ All scenarios complete at $(date)!"
```

---

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `emergent` | `emergent` (incentive-based) or `instructed` (explicit) |
| `--model` | `google/gemma-2-9b-it` | HuggingFace model name |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `bfloat16` | `float32`, `float16`, or `bfloat16` |
| `--scenario-name` | None | Single scenario (for parallel runs) |
| `--scenarios` | `3` | Number of scenarios (1-6) |
| `--trials` | `40` | Trials per scenario per condition |
| `--max-rounds` | `3` | Max negotiation rounds per trial |
| `--max-tokens` | `128` | Max tokens per LLM response |
| `--hybrid` | False | HuggingFace + TransformerLens (20x faster) |
| `--sae` | False | Enable SAE feature extraction |
| `--sae-layer` | `21` | Layer for SAE (middle layer for 9B) |
| `--fast` | False | Disable ToM module (~3x speedup) |
| `--ultrafast` | False | Minimal agents (~5x additional speedup) |
| `--causal` | False | Run causal validation tests |
| `--causal-samples` | `20` | Samples for causal validation |
| `--output` | `./experiment_output` | Output directory |
| `--checkpoint-dir` | None | Enable crash recovery |

---

## Experiment Configurations

### Minimum Viable (Workshop Paper)
```bash
--trials 30 --max-rounds 3 --hybrid --sae
# ~360 trials, ~3 hours
```

### Recommended (Conference Submission)
```bash
--trials 50 --max-rounds 3 --hybrid --sae --causal --causal-samples 30
# ~600 trials, ~5 hours
```

### Strong (Top Venue)
```bash
--trials 100 --max-rounds 3 --hybrid --sae --causal --causal-samples 50
# ~1200 trials, ~10 hours
```

---

## Troubleshooting

### CUDA out of memory
```bash
# Add --fast flag to disable ToM (reduces memory)
--fast

# Or reduce max tokens
--max-tokens 64
```

### TransformerLens errors
```bash
pip install transformers==4.44.0 accelerate==0.33.0 --force-reinstall
```

### HuggingFace access denied
```bash
# Re-login
huggingface-cli login

# Then visit and accept license:
# https://huggingface.co/google/gemma-2-9b-it
```

### Pod disconnects mid-run
```bash
# Add checkpoint recovery to your command:
--checkpoint-dir /workspace/persistent/checkpoints
```

### SAE loading hangs
```bash
# First-time SAE download can take 5-10 minutes
# Just wait, or pre-download with:
python -c "from sae_lens import SAE; SAE.from_pretrained('gemma-scope-9b-pt-res-canonical', 'layer_21/width_16k/canonical')"
```

### Check what's using GPU memory
```bash
nvidia-smi
# or
fuser -v /dev/nvidia*
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Setup | Step 1 |
| Validate | Step 2 |
| Quick test | Step 3 |
| Full experiment | Step 4 |
| Monitor log | `tail -f /workspace/persistent/full_experiment/experiment.log` |
| GPU usage | `nvidia-smi` |
| Zip results | `zip -r results.zip full_experiment/` |

---

## What You Get

### Probe Results (`probe_results_v2.json`)
```json
{
  "best_layer": 21,
  "gm_labels": {
    "r2": 0.25,
    "auc": 0.72
  },
  "agent_labels": {
    "r2": 0.14,
    "auc": 0.64
  },
  "cross_scenario_auc": 0.68,
  "interpretation": "GM R² > Agent R² suggests model encodes deception it doesn't self-report"
}
```

### Causal Validation (`causal_validation_results.json`)
```json
{
  "activation_patching": {"passed": true, "effect_ratio": 2.3},
  "ablation": {"passed": true, "kl_divergence": 0.8},
  "steering": {"passed": true, "dose_response": true},
  "evidence_strength": "STRONG",
  "tests_passed": "5/5"
}
```

---

## Support

- **Issues:** https://github.com/tesims/concordia/issues
- **Documentation:** See `SYSTEM_DOCUMENTATION.md` in repo

---

## Checklist

- [ ] Pod created with correct template
- [ ] Step 1: Setup completed
- [ ] Step 2: Validation passed
- [ ] Step 3: Quick test successful
- [ ] Step 4: Full experiment started
- [ ] Results downloaded


cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/full_experiment && \
mkdir -p /workspace/persistent/checkpoints && \
python -u run_deception_experiment.py \
    --mode emergent \
    --trials 50 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log

Step 1: Set Up Storage

export HF_HOME=/workspace/persistent/huggingface_cache
mkdir -p $HF_HOME
echo 'export HF_HOME=/workspace/persistent/huggingface_cache' >> ~/.bashrc
source ~/.bashrc

Step 2: Install (with latest code)

cd /workspace && \
git clone https://github.com/tesims/concordia.git && \
cd concordia && \
git checkout hybrid-sae-experiment && \
pip install -e . && \
pip install -r concordia/prefabs/entity/negotiation/evaluation/requirements.in && \
pip install transformers==4.44.0 accelerate==0.33.0 && \
pip install huggingface_hub

Step 3: HuggingFace Login

huggingface-cli login

Step 4: Quick Test

cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/test_output && \
python -u run_deception_experiment.py \
    --mode emergent \
    --trials 1 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --output /workspace/persistent/test_output

Step 5: Full Run (Conference Quality)

cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/full_experiment && \
mkdir -p /workspace/persistent/checkpoints && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenarios 6 \
    --trials 1 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log


cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && \
mkdir -p /workspace/persistent/full_experiment && \
mkdir -p /workspace/persistent/checkpoints && \
python -u run_deception_experiment.py \
    --mode emergent \
    --scenario-name alliance_betrayal \
    --scenarios 6 \
    --trials 1 \
    --max-rounds 3 \
    --hybrid \
    --sae \
    --causal \
    --causal-samples 30 \
    --device cuda \
    --dtype bfloat16 \
    --checkpoint-dir /workspace/persistent/checkpoints \
    --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log

  Available scenario names:
  - ultimatum_bluff
  - capability_bluff
  - hidden_value
  - info_withholding
  - promise_break
  - alliance_betrayal

---

## Quick Copy-Paste Commands (A100 80GB)

### Step 1: Storage Setup
```bash
mkdir -p /workspace/persistent/huggingface_cache /workspace/persistent/torch_cache /workspace/persistent/sae_cache /workspace/persistent/pip_cache
```

```bash
echo 'export HF_HOME=/workspace/persistent/huggingface_cache' >> ~/.bashrc && echo 'export TRANSFORMERS_CACHE=/workspace/persistent/huggingface_cache' >> ~/.bashrc && echo 'export TORCH_HOME=/workspace/persistent/torch_cache' >> ~/.bashrc && echo 'export SAE_LENS_CACHE=/workspace/persistent/sae_cache' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=/workspace/persistent/pip_cache' >> ~/.bashrc && source ~/.bashrc
```

```bash
ln -sf /workspace/persistent/huggingface_cache ~/.cache/huggingface && echo "HF_HOME=$HF_HOME"
```

### Step 2: Install
```bash
cd /workspace && git clone https://github.com/tesims/concordia.git && cd concordia && git checkout hybrid-sae-experiment
```

```bash
pip install -e . && pip install -r concordia/prefabs/entity/negotiation/evaluation/requirements.in && pip install transformers==4.44.0 accelerate==0.33.0 huggingface_hub
```

### Step 3: HuggingFace Login
```bash
huggingface-cli login
```
Paste your token from https://huggingface.co/settings/tokens

### Step 4: Quick Test (1 scenario, 1 trial)
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && mkdir -p /workspace/persistent/full_experiment /workspace/persistent/checkpoints && python -u run_deception_experiment.py --mode emergent --scenario-name alliance_betrayal --trials 1 --max-rounds 3 --hybrid --sae --causal --causal-samples 30 --device cuda --dtype bfloat16 --checkpoint-dir /workspace/persistent/checkpoints --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log
```

### Step 5: Full Experiment (6 scenarios, 50 trials each)
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && python -u run_deception_experiment.py --mode emergent --scenarios 6 --trials 50 --max-rounds 3 --hybrid --sae --causal --causal-samples 30 --device cuda --dtype bfloat16 --checkpoint-dir /workspace/persistent/checkpoints --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log
```

### Update Existing Pod (already has 9B model)
```bash
cd /workspace/concordia && git pull
```

```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && python -u run_deception_experiment.py --mode emergent --scenarios 6 --trials 1 --max-rounds 3 --hybrid --sae --causal --causal-samples 30 --device cuda --dtype bfloat16 --checkpoint-dir /workspace/persistent/checkpoints --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log
```

### Gemma 2B (Faster, ~4GB VRAM)
```bash
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation && python -u run_deception_experiment.py --mode emergent --scenarios 6 --trials 50 --max-rounds 3 --model google/gemma-2-2b-it --hybrid --sae --sae-layer 13 --causal --causal-samples 30 --device cuda --dtype bfloat16 --checkpoint-dir /workspace/persistent/checkpoints --output /workspace/persistent/full_experiment 2>&1 | tee /workspace/persistent/full_experiment/experiment.log
```
