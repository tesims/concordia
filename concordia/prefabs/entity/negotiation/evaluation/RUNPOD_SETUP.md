# RunPod Setup Guide: Emergent Deception Experiment

Complete setup steps for running the deception detection experiment on RunPod with H100 GPU.

## Prerequisites

- RunPod account with H100 GPU pod
- HuggingFace account with access to Gemma 2 9B

### HuggingFace Setup (do this first)

1. Go to [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
2. Click "Accept License" to get access
3. Go to [HuggingFace Settings > Tokens](https://huggingface.co/settings/tokens)
4. Create a new token (read access is sufficient)

---

## Quick Setup (Copy-Paste Commands)

```bash
# 1. Clone and checkout
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2

# 2. Install Concordia base
pip install -e .

# 3. Install interpretability dependencies
pip install transformer-lens huggingface_hub scikit-learn

# 4. Fix transformer-lens compatibility
pip install transformers==4.44.0 accelerate==0.33.0

# 5. Login to HuggingFace
huggingface-cli login
# (paste your token when prompted)

# 6. Run the experiment
cd concordia/prefabs/entity/negotiation/evaluation
nohup python run_deception_experiment.py --mode emergent --scenarios 6 --trials 100 --device cuda --dtype bfloat16 > experiment.log 2>&1 &

# 7. Monitor progress
tail -f experiment.log
```

---

## Detailed Steps

### Step 1: Clone Repository

```bash
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2
```

### Step 2: Install Concordia

```bash
pip install -e .
```

This installs:
- torch, numpy, pandas
- transformers
- matplotlib
- All other Concordia dependencies

### Step 3: Install Interpretability Dependencies

```bash
pip install transformer-lens huggingface_hub scikit-learn
```

These are needed for:
- `transformer-lens`: Activation capture via HookedTransformer
- `huggingface_hub`: Downloading Gemma model
- `scikit-learn`: Training linear probes

### Step 4: Fix Version Compatibility

```bash
pip install transformers==4.44.0 accelerate==0.33.0
```

TransformerLens requires specific transformers version.

### Step 5: HuggingFace Login

```bash
huggingface-cli login
```

Paste your HuggingFace token when prompted. This is required because Gemma is a gated model.

### Step 6: Run Experiment

```bash
cd concordia/prefabs/entity/negotiation/evaluation

# Quick test (2 trials)
nohup python run_deception_experiment.py --mode emergent --scenarios 6 --trials 2 --device cuda --dtype bfloat16 > experiment.log 2>&1 &

# Full experiment (100 trials per scenario)
nohup python run_deception_experiment.py --mode emergent --scenarios 6 --trials 100 --device cuda --dtype bfloat16 > experiment.log 2>&1 &
```

### Step 7: Monitor Progress

```bash
# Live output
tail -f experiment.log

# Check GPU usage
nvidia-smi

# Check if process is running
ps aux | grep python
```

---

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | emergent | `emergent` (incentive-based) or `instructed` (explicit) |
| `--scenarios` | 6 | Number of scenarios (max 6) |
| `--trials` | 50 | Trials per scenario per condition |
| `--model` | google/gemma-2-9b-it | HuggingFace model name |
| `--device` | cuda | `cuda` or `cpu` |
| `--dtype` | bfloat16 | `bfloat16`, `float16`, or `float32` |
| `--output` | ./experiment_output | Output directory |

---

## Expected Output

The experiment runs:
- 6 scenarios
- 2 conditions each (HIGH_INCENTIVE, LOW_INCENTIVE)
- N trials per condition
- **Total trials: 6 × 2 × N = 12N**

For 100 trials: 1,200 total negotiations.

### Runtime Estimates (H100)

| Trials | Total Negotiations | Estimated Time |
|--------|-------------------|----------------|
| 2 | 24 | ~10-15 min |
| 10 | 120 | ~1-2 hours |
| 50 | 600 | ~5-8 hours |
| 100 | 1,200 | ~10-16 hours |

---

## Output Files

After completion, find results in `experiment_output/`:

```
experiment_output/
├── activations_emergent_YYYYMMDD_HHMMSS.pt  # Activation dataset
└── probe_results.json                        # Probe training results
```

### Loading Results

```python
import torch

# Load activations
data = torch.load('experiment_output/activations_emergent_*.pt')
print(f"Samples: {data['config']['n_samples']}")
print(f"Layers: {data['config']['layers']}")

# Check deception rates
for scenario, rate in data['deception_rates'].items():
    print(f"{scenario}: {rate:.1%}")
```

---

## Troubleshooting

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "ModuleNotFoundError: Could not import module 'BertForPreTraining'"
```bash
pip install transformers==4.44.0 accelerate==0.33.0
```

### "Access denied" for Gemma model
1. Accept license at https://huggingface.co/google/gemma-2-9b-it
2. Run `huggingface-cli login` with valid token

### Process killed / OOM
- Gemma 9B needs ~20GB VRAM
- Use H100 (80GB) or A100 (40GB/80GB)
- Or try smaller model: `--model google/gemma-2-2b-it`

### Check if experiment is running
```bash
ps aux | grep python
nvidia-smi  # Should show ~20-40GB VRAM used
```

---

## Quick Reference

```bash
# Start experiment
cd /workspace/concordia/concordia/prefabs/entity/negotiation/evaluation
nohup python run_deception_experiment.py --mode emergent --scenarios 6 --trials 100 --device cuda --dtype bfloat16 > experiment.log 2>&1 &

# Monitor
tail -f experiment.log

# GPU status
nvidia-smi -l 1

# Kill experiment
pkill -f run_deception_experiment
```
