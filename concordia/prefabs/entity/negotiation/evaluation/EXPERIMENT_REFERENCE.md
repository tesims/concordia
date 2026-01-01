# Deception Detection Experiment - Complete Reference

## Quick Start (RunPod RTX 4090)

```bash
# Setup
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia
git checkout emergent-deception-v2
pip install -e .
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in
pip install transformers==4.44.0 accelerate==0.33.0
huggingface-cli login

# Quick test (2-3 min)
python run_deception_experiment.py --scenario-name ultimatum_bluff --trials 1 --fast --device cuda --dtype bfloat16

# Full experiment (single pod, ~2 hrs with --fast)
python run_deception_experiment.py --fast --device cuda --dtype bfloat16
```

---

## Configuration

### Optimized Settings (Current)

| Setting | Value | Reason |
|---------|-------|--------|
| Model | `google/gemma-2-2b-it` | Fast inference, fits in 24GB |
| GPU | RTX 4090 (24GB) | Best cost/performance for 2B |
| Scenarios | 3 | ultimatum_bluff, hidden_value, alliance_betrayal |
| Trials | 40 per condition | Statistical significance |
| Rounds | 3 per trial | Captures deception decision |
| Max tokens | 128 | Sufficient for negotiation responses |
| Fast mode | `--fast` | Disables ToM, ~3x speedup |

### Command Line Arguments

```
--mode          emergent|instructed|both (default: emergent)
--model         HuggingFace model name (default: google/gemma-2-2b-it)
--device        cuda|cpu (default: cuda)
--dtype         bfloat16|float16|float32 (default: bfloat16)
--scenarios     Number of scenarios 1-6 (default: 3)
--scenario-name Run specific scenario (for parallel pods)
--trials        Trials per condition (default: 40)
--max-rounds    Rounds per trial (default: 3)
--max-tokens    Max tokens per response (default: 128)
--fast          Disable ToM module for ~3x speedup
--output        Output directory (default: ./experiment_output)
--train-only    Only train probes on existing data
--data          Path to activations file for --train-only
```

---

## Parallel Execution (3 Pods)

### Setup (same for all pods)
```bash
cd /workspace
git clone https://github.com/tesims/concordia.git concordia
cd concordia && git checkout emergent-deception-v2
pip install -e .
cd concordia/prefabs/entity/negotiation/evaluation
pip install -r requirements.in
pip install transformers==4.44.0 accelerate==0.33.0
huggingface-cli login
```

### Run Commands
- **Pod 1:** `python run_deception_experiment.py --scenario-name ultimatum_bluff --fast --device cuda --dtype bfloat16 --output ./outputs/ultimatum_bluff`
- **Pod 2:** `python run_deception_experiment.py --scenario-name hidden_value --fast --device cuda --dtype bfloat16 --output ./outputs/hidden_value`
- **Pod 3:** `python run_deception_experiment.py --scenario-name alliance_betrayal --fast --device cuda --dtype bfloat16 --output ./outputs/alliance_betrayal`

### Merge Results
```bash
python merge_results.py outputs/ --train-probes
```

---

## Time & Cost Estimates

| Configuration | Trials | Time | Cost (RunPod) |
|---------------|--------|------|---------------|
| 1 trial test `--fast` | 2 | ~2-3 min | ~$0.02 |
| Single pod `--fast` | 240 | ~2-3 hrs | ~$1-2 |
| 3 parallel pods `--fast` | 240 | ~1-1.5 hrs | ~$1.50 |
| Single pod with ToM | 240 | ~8-12 hrs | ~$4-5 |

---

## Known Issues & Fixes

### 1. `ModuleNotFoundError: No module named 'absl'`
```bash
pip install absl-py
```

### 2. `AttributeError: module 'enum' has no attribute 'StrEnum'`
Python version too old. Need Python 3.11+.
Use RunPod template with Python 3.11+ (e.g., "RunPod Pytorch 2.4")

### 3. `ModuleNotFoundError: Could not import module 'BertForPreTraining'`
```bash
pip install transformers==4.44.0 accelerate==0.33.0
```

### 4. `Access denied` for Gemma model
1. Accept license at https://huggingface.co/google/gemma-2-2b-it
2. Run `huggingface-cli login` with valid token

### 5. Theory of Mind (ToM) too slow
ToM makes multiple LLM calls per round (emotion detection, mental modeling).
Use `--fast` flag to disable ToM for ~3x speedup.

### 6. Stop running experiment
```bash
# In same terminal
Ctrl+C

# Or from another terminal
pkill -f run_deception_experiment
```

---

## File Structure

```
concordia/prefabs/entity/negotiation/evaluation/
├── run_deception_experiment.py    # Main entry point
├── interpretability_evaluation.py # InterpretabilityRunner, TransformerLensWrapper
├── emergent_prompts.py            # 6 emergent deception scenarios
├── deception_scenarios.py         # Instructed deception scenarios
├── train_probes.py                # Ridge probe training
├── sanity_checks.py               # Probe validation
├── mech_interp_tools.py           # TransformerLens utilities
├── merge_results.py               # Merge parallel pod outputs
├── run_parallel.sh                # Parallel execution script
├── requirements.in                # Dependencies
├── RUNPOD_SETUP.md                # Setup guide
└── EXPERIMENT_REFERENCE.md        # This file
```

---

## Architecture Overview

### Data Flow
```
run_deception_experiment.py
    │
    ├── InterpretabilityRunner (interpretability_evaluation.py)
    │   └── TransformerLensWrapper (captures activations)
    │
    ├── Scenarios (emergent_prompts.py)
    │   ├── ultimatum_bluff (false final offer claims)
    │   ├── hidden_value (inflated asking price)
    │   └── alliance_betrayal (assure ally, consider betrayal)
    │
    ├── Agents (advanced_negotiator.py)
    │   ├── Base negotiator components
    │   └── Optional: theory_of_mind module (disabled with --fast)
    │
    └── Analysis
        ├── train_probes.py (ridge regression on activations)
        └── sanity_checks.py (validation)
```

### Key Classes
- `InterpretabilityRunner`: Main orchestrator, runs trials, collects activations
- `TransformerLensWrapper`: Wraps HookedTransformer, captures layer activations
- `ActivationSample`: Dataclass storing activations + labels per sample

### Labels Captured
- **Agent labels** (from ToM module, empty if --fast):
  - perceived_deception, emotion_intensity, trust_level, cooperation_intent
- **GM labels** (ground truth from scenario rules):
  - actual_deception, commitment_violation, manipulation_score, consistency_score

---

## ToM Performance Issue

The Theory of Mind module in `components/theory_of_mind.py` makes extra LLM calls:
1. `_detect_emotions()` - analyzes counterpart's emotional state
2. `_build_mental_model()` - constructs opponent model
3. `_assess_deception_indicators()` - checks for deception signals

Each call adds ~30-60 seconds per round. With 3 rounds × 2 agents × multiple calls = very slow.

**Workaround:** Use `--fast` flag to disable ToM entirely.

**Potential fix:** Batch these calls or cache results within a round.

---

## Output Files

After experiment completes:
```
experiment_output/
├── activations_emergent_YYYYMMDD_HHMMSS.pt  # Activation dataset
└── probe_results.json                        # Probe training results
```

### Loading Results
```python
import torch

data = torch.load('experiment_output/activations_emergent_*.pt')
print(f"Samples: {len(data['samples'])}")

# Check deception rates
for sample in data['samples']:
    print(f"{sample.emergent_scenario}: deception={sample.actual_deception}")
```

---

## Research Quality Notes

### What --fast preserves:
- Activation capture from main agent responses
- GM ground truth labels (from scenario rules)
- Cross-scenario generalization testing
- Probe training on layer activations

### What --fast loses:
- Rich agent labels (perceived_deception, trust_level, etc.)
- Agent perspective on counterpart's mental state
- Recursive belief modeling

For initial research validation, --fast is acceptable. GM labels are the primary target for deception probes.

---

## Version Info

- Branch: `emergent-deception-v2`
- Model: Gemma 2 2B (`google/gemma-2-2b-it`)
- TransformerLens: requires `transformers==4.44.0`
- Python: 3.11+ required
