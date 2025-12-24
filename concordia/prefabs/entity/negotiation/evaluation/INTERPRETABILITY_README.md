# Interpretability + Evaluation Module

This module enables **mechanistic interpretability research** on Concordia negotiation agents. It captures model activations during negotiations and pairs them with behavioral labels from cognitive modules like Theory of Mind.

## What This Does

When an agent negotiates, it makes decisions by calling a language model. Normally you only see the output text. This module lets you see **inside** the model:

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT HAPPENS ON EACH LLM CALL                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "What should I do?"  ──►  [Language Model]  ──►  "I cooperate" │
│                                   │                             │
│                                   ▼                             │
│                        ┌─────────────────┐                      │
│                        │  ACTIVATIONS    │                      │
│                        │  Layer 0: [...]  │  ◄── Captured!      │
│                        │  Layer 6: [...]  │                      │
│                        │  Layer 12: [...] │                      │
│                        └─────────────────┘                      │
│                                   │                             │
│                                   ▼                             │
│                        ┌─────────────────┐                      │
│                        │  ToM MODULE     │                      │
│                        │  deception: 0.3 │  ◄── Labels!         │
│                        │  emotion: 0.7   │                      │
│                        │  trust: 0.5     │                      │
│                        └─────────────────┘                      │
│                                                                 │
│         OUTPUT: (activations, labels) paired together           │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Matters

This enables research questions like:

- **Can we predict deception from activations?** Train a probe on `deception_risk` labels
- **Where does trust get computed?** Compare activation patterns across trust levels
- **Do emotions have neural correlates?** Map `emotion_intensity` to activation space

## Quick Start

### Basic Usage

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    InterpretabilityRunner,
    run_quick_study,
)

# Option 1: One-liner
runner, results = run_quick_study(
    model_name="google/gemma-2-2b-it",  # Or any HuggingFace model
    device="cuda",
    scenario="fishery",
    num_trials=50,
    use_gm=True,  # Enable GM for ground truth labels
    output_file="negotiation_data.pt"
)

# Option 2: Full control
runner = InterpretabilityRunner(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    layers_to_capture=[0, 12, 25],  # Which layers to save
)

results = runner.run_study(
    scenario='fishery',
    agent_modules=['theory_of_mind'],
    gm_modules=['social_intelligence'],  # For ground truth labels
    num_trials=50,
    max_rounds=10,
    use_gm=True,  # Enable GM ground truth
)

runner.print_summary(results)
runner.save_dataset('negotiation_data.pt')
```

### On RunPod/Lambda Labs with GPU

```bash
# 1. Deploy a GPU pod on RunPod (H200 or A100 recommended)
# 2. SSH or use web terminal, then:

git clone https://github.com/YOUR_USERNAME/concordia.git
cd concordia
pip install -e ".[interpretability]"

python -c "
from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner

runner = InterpretabilityRunner(
    model_name='google/gemma-2-2b-it',
    device='cuda',
    layers_to_capture=[0, 13, 25],  # Gemma-2-2B has 26 layers
)

results = runner.run_study(
    scenario='fishery',
    agent_modules=['theory_of_mind'],
    num_trials=100,
    use_gm=True,
)

runner.save_dataset('gemma_negotiation_activations.pt')
runner.print_summary(results)
"
```

## Output Format

The saved `.pt` file contains:

```python
data = torch.load('negotiation_data.pt')

# Activations: shape [N, num_layers, d_model]
# Example: [500, 3, 2048] = 500 samples, 3 layers, 2048 dimensions
data['activations']

# Agent Labels (first-person beliefs): shape [N, 4]
data['agent_labels']
data['agent_label_names']  # ['perceived_deception', 'emotion_intensity', 'trust_level', 'cooperation_intent']

# GM Labels (third-person ground truth): shape [N, 4]
data['gm_labels']
data['gm_label_names']  # ['actual_deception', 'commitment_violation', 'manipulation_score', 'consistency_score']

# Legacy format (same as agent_labels)
data['labels']

# Layer names that were captured
data['layers']  # ['blocks.0.hook_resid_post', 'blocks.12.hook_resid_post', ...]

# Metadata for each sample
data['metadata']  # [{'trial_id': 1, 'round_num': 0, 'agent_name': 'Agent_A', ...}, ...]
```

### Agent Labels (First-Person Beliefs)

What the agent **believes** about others:

| Label | Range | Description |
|-------|-------|-------------|
| `perceived_deception` | 0.0 - 1.0 | "I think YOU are deceiving ME" |
| `emotion_intensity` | 0.0 - 1.0 | Detected emotional intensity in the interaction |
| `trust_level` | 0.0 - 1.0 | Agent's trust in counterpart |
| `cooperation_intent` | 0.0 - 1.0 | Agent's own cooperative disposition |

### GM Labels (Third-Person Ground Truth)

What the agent is **actually doing** (from Game Master):

| Label | Range | Description |
|-------|-------|-------------|
| `actual_deception` | 0.0 - 1.0 | "YOU ARE deceiving" (objective truth) |
| `commitment_violation` | 0.0 - 1.0 | Did agent break previous commitments? |
| `manipulation_score` | 0.0 - 1.0 | Is agent using manipulation tactics? |
| `consistency_score` | 0.0 - 1.0 | Are agent's statements consistent? |

### Key Distinction

For interpretability research, use **GM labels** (`actual_deception`) as your probe target because they represent ground truth. Agent labels represent beliefs, which may differ from reality.

## Training Probes

Once you have the data, train a probe to predict labels from activations:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = torch.load('negotiation_data.pt')
X = data['activations']  # [N, num_layers, d_model]
y = data['labels'][:, 0]  # Just deception_risk for this example

# Flatten layers or use specific layer
X_flat = X.reshape(X.shape[0], -1)  # [N, num_layers * d_model]
# Or single layer: X_layer0 = X[:, 0, :]  # [N, d_model]

# Train/test split
split = int(0.8 * len(X_flat))
X_train, X_test = X_flat[:split], X_flat[split:]
y_train, y_test = y[:split], y[split:]

# Simple linear probe
probe = nn.Linear(X_flat.shape[1], 1)
optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    pred = probe(X_train).squeeze()
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            test_pred = probe(X_test).squeeze()
            test_loss = criterion(test_pred, y_test)
        print(f"Epoch {epoch}: train_loss={loss:.4f}, test_loss={test_loss:.4f}")

# Evaluate
with torch.no_grad():
    predictions = probe(X_test).squeeze()
    correlation = torch.corrcoef(torch.stack([predictions, y_test]))[0, 1]
    print(f"Test correlation: {correlation:.4f}")
```

## Scenarios

Three negotiation scenarios are available:

| Scenario | Description | Cooperation Challenge |
|----------|-------------|----------------------|
| `fishery` | Fishing companies share a common pool resource | Limit fishing to sustain stocks |
| `treaty` | Countries negotiate treaty terms | Balance national vs collective interests |
| `gameshow` | Reality show with alliances and elimination | Form alliances vs self-interest |

## Components

### TransformerLensWrapper

A Concordia-compatible language model that captures activations:

```python
from concordia.prefabs.entity.negotiation.evaluation import TransformerLensWrapper

model = TransformerLensWrapper(
    model_name="EleutherAI/gpt-neo-125M",
    device="cpu",
    layers_to_capture=[0, 3, 5],
)

response = model.sample_text("Hello, how are you?")
activations = model.get_activations()  # Dict[layer_name, tensor]
```

### InterpretabilityRunner

Orchestrates the full study:

```python
from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner

runner = InterpretabilityRunner(
    model_name="google/gemma-2-2b-it",
    device="cuda",
)

# Run single negotiation
result = runner.run_single_negotiation(
    scenario_type='fishery',
    modules=['theory_of_mind'],
    max_rounds=10,
)

# Run full study
results = runner.run_study(
    scenario='fishery',
    num_trials=50,
)

# Access samples directly
for sample in runner.activation_samples:
    print(f"Trial {sample.trial_id}, Round {sample.round_num}")
    print(f"  Deception risk: {sample.deception_risk}")
    print(f"  Activation shape: {list(sample.activations.values())[0].shape}")
```

## Model Recommendations

| Model | Parameters | Use Case |
|-------|------------|----------|
| `EleutherAI/gpt-neo-125M` | 125M | Quick testing, CPU-friendly |
| `google/gemma-2-2b-it` | 2B | Good balance for research |
| `google/gemma-2-9b-it` | 9B | Higher quality, needs good GPU |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Alternative, well-studied |

## Hardware Requirements

| Setup | Memory | Speed | RunPod $/hr |
|-------|--------|-------|-------------|
| CPU only | 8GB+ RAM | Very slow (testing only) | - |
| RTX 4090 | 24GB VRAM | Good for 2B models | ~$0.44 |
| A100 40GB | 40GB VRAM | Fast, 2B-9B models | ~$1.09 |
| H200 | 141GB VRAM | Very fast | ~$5-7 |
| B200 | 192GB VRAM | Fastest | ~$8-12 |

**Recommended for 100 trials:** H200 (~35 min, ~$4) or A100 40GB (~1.5 hrs, ~$2)

## Evaluation Metrics (Bonus)

While collecting interpretability data, you also get evaluation metrics:

```python
results = runner.run_study(...)

print(f"Cooperation Rate: {results.cooperation_rate:.2%}")
print(f"Agreement Rate: {results.agreement_rate:.2%}")
print(f"Total LLM Calls: {results.total_llm_calls}")
print(f"Activation Samples: {len(results.activation_samples)}")
```

## Files in This Module

| File | Purpose |
|------|---------|
| `interpretability_evaluation.py` | Main runner with TransformerLens + GM integration |
| `mech_interp_tools.py` | SAE Lens, probing tools, direction extraction |
| `llm_evaluation.py` | LLM-based evaluation without interpretability |
| `contest_scenarios.py` | Negotiation scenario definitions |
| `metrics.py` | Evaluation metrics collection |
| `test_full_integration.py` | Comprehensive integration tests |
| `USAGE_GUIDE.md` | Detailed guide with all inputs/outputs |

## Research Applications

This module supports research into:

1. **Deception Detection**: Can neural patterns predict deceptive behavior?
2. **Trust Dynamics**: How does trust evolve in activation space?
3. **Emotional Reasoning**: Where does the model process emotional cues?
4. **Strategic Planning**: Do planning-related activations predict cooperation?
5. **Theory of Mind**: How does the model represent beliefs about others?

## Citation

If you use this for research, please cite:

```bibtex
@software{concordia_interpretability,
  title = {Concordia Negotiation Interpretability Module},
  author = {GSoC Concordia Project},
  year = {2024},
  url = {https://github.com/google-deepmind/concordia}
}
```

## Troubleshooting

### Out of Memory
- Reduce `layers_to_capture` (fewer layers = less memory)
- Use a smaller model
- Reduce batch size in generation

### Slow Generation
- Use GPU (`device="cuda"`)
- Reduce `max_tokens` in generation
- Use a smaller model for testing

### Missing Labels
- Ensure `theory_of_mind` module is enabled
- Check that agents have observed counterpart actions
- Labels may be 0 at start before interaction builds mental models
