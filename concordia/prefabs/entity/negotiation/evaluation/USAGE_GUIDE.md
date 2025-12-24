# Interpretability Framework Usage Guide

A complete guide to collecting and analyzing neural activation data from negotiation agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Inputs Reference](#inputs-reference)
3. [Configuration Options](#configuration-options)
4. [Output Data Structure](#output-data-structure)
5. [Interpreting Results](#interpreting-results)
6. [Analysis Workflows](#analysis-workflows)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This framework captures what happens inside a language model when it makes negotiation decisions. You get two things paired together:

- **Activations**: The internal state of the model (numbers)
- **Labels**: What the agent was "thinking" (from Theory of Mind module)

By analyzing these pairs, you can answer: *Does the model internally represent concepts like deception, trust, or cooperation?*

---

## Inputs Reference

### Required Inputs

| Input | Type | Description | Example |
|-------|------|-------------|---------|
| `model_name` | string | HuggingFace model identifier | `"google/gemma-2-2b-it"` |
| `scenario` | string | Which negotiation game to run | `"fishery"` |
| `num_trials` | int | How many complete negotiations to run | `50` |

### Optional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | string | `"cuda"` | Where to run the model |
| `layers_to_capture` | list[int] | `[0, mid, last]` | Which layers to save activations from |
| `modules` | list[string] | `["theory_of_mind"]` | Which cognitive modules to enable |
| `max_rounds` | int | `10` | Rounds per negotiation |
| `module_configs` | dict | `{}` | Fine-tune module behavior |

---

## Configuration Options

### 1. Model Selection (`model_name`)

**What it is:** The language model that powers the agents' decisions.

**Options:**

| Model | Size | Speed | Quality | Memory Needed |
|-------|------|-------|---------|---------------|
| `EleutherAI/gpt-neo-125M` | 125M | Very fast | Low | 1GB |
| `google/gemma-2-2b-it` | 2B | Medium | Good | 8GB |
| `google/gemma-2-9b-it` | 9B | Slow | High | 24GB |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Medium | Good | 16GB |

**When to use what:**
- **Testing/debugging**: Use `gpt-neo-125M` (fast, runs on CPU)
- **Real research**: Use `gemma-2-2b-it` (good balance)
- **Best quality**: Use `gemma-2-9b-it` (needs good GPU)

**Influence:** Larger models may have more structured internal representations, making probes work better.

---

### 2. Device (`device`)

**What it is:** Where the model runs.

**Options:**

| Option | When to Use |
|--------|-------------|
| `"cpu"` | Testing only. Very slow but works anywhere. |
| `"cuda"` | You have an NVIDIA GPU. Required for real experiments. |
| `"mps"` | Mac with M1/M2/M3 chip. Faster than CPU. |

**Influence:** Only affects speed, not results.

---

### 3. Layers to Capture (`layers_to_capture`)

**What it is:** Which layers of the model to save activations from.

**What layers mean:**
- **Early layers (0-5)**: Process raw input, syntax, basic patterns
- **Middle layers (6-15)**: Build meaning, relationships, context
- **Late layers (20+)**: Make decisions, plan outputs

**Options:**

| Setting | Captures | Use Case |
|---------|----------|----------|
| `[0, 12, 25]` | Early, mid, late | Standard research (recommended) |
| `[12]` | Middle only | Quick experiments, less storage |
| `list(range(26))` | All layers | Full analysis, lots of storage |

**Influence:**
- More layers = more data = more storage needed
- Different layers encode different types of information
- Middle layers often best for semantic concepts like "deception"

**How to choose:**
```
Model has 26 layers?
  - Capture [0, 13, 25] for overview
  - Capture [10, 11, 12, 13, 14, 15] to zoom in on middle
```

---

### 4. Scenario (`scenario`)

**What it is:** The negotiation game agents play.

**Options:**

| Scenario | Description | Cooperation Challenge |
|----------|-------------|----------------------|
| `"fishery"` | Companies share fishing waters | Limit fishing to sustain fish population |
| `"treaty"` | Countries negotiate treaty terms | Balance national vs. collective interests |
| `"gameshow"` | Reality show with alliances | Form alliances vs. betray for personal gain |

**Influence:**
- Different scenarios create different types of interactions
- Some scenarios produce more deception than others
- `"fishery"` is simplest and most studied

**When to use what:**
- **Starting out**: Use `"fishery"` (well-defined, predictable)
- **Studying deception**: Use `"gameshow"` (more betrayal opportunities)
- **Complex negotiations**: Use `"treaty"` (multi-issue)

---

### 5. Number of Trials (`num_trials`)

**What it is:** How many complete negotiations to run.

**Influence:**

| Trials | Samples* | Use Case | Time (2B model, GPU) |
|--------|----------|----------|----------------------|
| 5 | ~100 | Quick test | ~5 min |
| 20 | ~400 | Development | ~20 min |
| 50 | ~1000 | Small study | ~1 hour |
| 100 | ~2000 | Full study | ~2 hours |
| 200 | ~4000 | Publication | ~4 hours |

*Samples = trials × rounds × agents × LLM calls per turn

**How to choose:**
- **Testing code**: 5 trials
- **Checking if probe works**: 20-50 trials
- **Real research**: 100+ trials
- **Statistical significance**: 200+ trials

---

### 6. Max Rounds (`max_rounds`)

**What it is:** How many turns each negotiation lasts.

**Influence:**
- More rounds = more data per trial
- More rounds = richer mental models (ToM has more to observe)
- More rounds = longer runtime

**Options:**

| Rounds | Effect |
|--------|--------|
| 5 | Quick, but ToM may not develop fully |
| 10 | Standard, good balance (recommended) |
| 20 | Rich data, but slower |

---

### 7. Cognitive Modules (`modules`)

**What it is:** Which cognitive capabilities the agents have.

**Available modules:**

| Module | What It Does | Labels It Provides |
|--------|--------------|-------------------|
| `theory_of_mind` | Models counterpart's beliefs/intentions | `deception_risk`, `trust_level`, `emotion_intensity` |
| `cultural_adaptation` | Adjusts style to context | Cultural scores |
| `temporal_strategy` | Considers time pressure | Urgency scores |
| `uncertainty_aware` | Tracks confidence | Uncertainty scores |
| `strategy_evolution` | Learns from past | Strategy scores |

**Influence:**
- `theory_of_mind` is **required** for deception_risk labels
- Other modules add more label types
- More modules = slower but richer data

**Recommended:**
```python
modules=['theory_of_mind']  # Start here
modules=['theory_of_mind', 'uncertainty_aware']  # Add more later
```

---

### 8. Module Configs (`module_configs`)

**What it is:** Fine-tune how modules behave.

**Theory of Mind options:**

| Setting | Type | Default | Effect |
|---------|------|---------|--------|
| `max_recursion_depth` | int | 2 | How deep "I think you think I think..." |
| `emotion_sensitivity` | float | 0.7 | How much emotions affect judgments |

**Example:**
```python
module_configs={
    'theory_of_mind': {
        'max_recursion_depth': 3,      # Deeper reasoning
        'emotion_sensitivity': 0.9,    # More emotional awareness
    }
}
```

**Influence:**
- Higher recursion = more nuanced mental models, slower
- Higher emotion sensitivity = more emotional labels variation

---

## Output Data Structure

When you call `runner.save_dataset('output.pt')`, you get a PyTorch file with:

### 1. Activations

```python
data['activations']  # Shape: [N, num_layers, d_model]
```

| Dimension | Meaning | Example |
|-----------|---------|---------|
| N | Number of samples | 1000 |
| num_layers | Layers you captured | 3 |
| d_model | Model's hidden dimension | 2048 |

**What the numbers mean:**
- Each number is one neuron's activation
- Higher = that neuron is more "active" for this input
- The pattern across all neurons = the model's "thought"

### 2. Agent Labels (First-Person Beliefs)

```python
data['agent_labels']  # Shape: [N, 4]
```

These are what the agent **believes** about others:

| Column | Name | Range | Meaning |
|--------|------|-------|---------|
| 0 | `perceived_deception` | 0.0-1.0 | "I think YOU are deceiving ME" |
| 1 | `emotion_intensity` | 0.0-1.0 | Detected emotional intensity in interaction |
| 2 | `trust_level` | 0.0-1.0 | Agent's trust in counterpart |
| 3 | `cooperation_intent` | 0.0-1.0 | Agent's willingness to cooperate |

**Source:** Theory of Mind module (agent's internal assessment)

### 3. GM Labels (Third-Person Ground Truth)

```python
data['gm_labels']  # Shape: [N, 4]
```

These are what the agent is **actually doing** (ground truth from Game Master):

| Column | Name | Range | Meaning |
|--------|------|-------|---------|
| 0 | `actual_deception` | 0.0-1.0 | "YOU ARE deceiving" (objective truth) |
| 1 | `commitment_violation` | 0.0-1.0 | Did agent break previous commitments? |
| 2 | `manipulation_score` | 0.0-1.0 | Is agent using manipulation tactics? |
| 3 | `consistency_score` | 0.0-1.0 | Are agent's statements consistent? |

**Source:** Social Intelligence GM module (objective third-party assessment)

### Key Distinction: Agent vs GM Labels

This is **critical** for interpretability research:

| | Agent Labels | GM Labels |
|---|---|---|
| **Perspective** | First-person (what I believe about you) | Third-person (what you're actually doing) |
| **Source** | Agent's Theory of Mind module | Game Master's Social Intelligence module |
| **Use Case** | Studying belief formation | Finding ground truth representations |
| **Example** | "I think you're lying" | "You ARE lying" |

**For interpretability probes**, you typically want **GM labels** (`actual_deception`) as your target because:
1. They represent objective ground truth
2. If the model encodes actual deception differently than perceived deception, that's a finding!
3. You can compare: Does the model represent what it *thinks* vs what it *does*?

### 4. Legacy Labels (Backwards Compatible)

```python
data['labels']  # Shape: [N, 4] - Same as agent_labels
data['label_names']  # ['perceived_deception', 'emotion_intensity', 'trust_level', 'cooperation_intent']
```

### 5. Layer Names

```python
data['layers']  # ['blocks.0.hook_resid_post', 'blocks.13.hook_resid_post', ...]
```

### 6. Metadata

```python
data['metadata']  # List of dicts, one per sample
```

Each entry contains:

| Field | Type | Meaning |
|-------|------|---------|
| `trial_id` | int | Which negotiation (1, 2, 3...) |
| `round_num` | int | Which round in that negotiation (0-9) |
| `agent_name` | string | Which agent ("Agent_A" or "Agent_B") |
| `scenario` | string | Which scenario was running |
| `agent_modules` | list | Agent modules enabled (e.g., ["theory_of_mind"]) |
| `gm_modules` | list | GM modules enabled (e.g., ["social_intelligence"]) |

---

## Interpreting Results

### What Makes a Good Probe?

You train a probe to predict labels from activations. The key metric is **R² score**:

| R² Score | Interpretation |
|----------|----------------|
| < 0.05 | No relationship. Model doesn't encode this concept (or probe is wrong). |
| 0.05 - 0.15 | Weak signal. Something's there but noisy. |
| 0.15 - 0.30 | Moderate signal. Model probably encodes this. |
| 0.30 - 0.50 | Strong signal. Clear encoding of the concept. |
| > 0.50 | Very strong. Concept is well-represented in activations. |

### What Different Results Mean

**High R² on deception_risk:**
> "The model has internal representations that correlate with deception detection. When the ToM module flags something as deceptive, the model's neurons show a consistent pattern."

**Low R² on deception_risk:**
> "Either the model doesn't have a clear 'deception detector', or deception is encoded in a complex/distributed way that linear probes can't find."

**Higher R² in middle layers vs early layers:**
> "The concept is computed/built in middle layers, not just read from input."

**Higher R² in later layers:**
> "The concept is used for decision-making, represented close to output."

### Comparing Across Layers

```
Layer 0:  R² = 0.05  (input processing, no deception concept yet)
Layer 13: R² = 0.35  (middle layer, concept is computed here)
Layer 25: R² = 0.28  (output layer, concept used for decisions)
```

**Interpretation:** Deception detection emerges in middle layers and persists to output.

---

## Analysis Workflows

### Workflow 1: Basic Probe Analysis (GM Ground Truth)

**Goal:** See if the model encodes actual deception (ground truth).

```python
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Load data
data = torch.load('output.pt')
X = data['activations'][:, 1, :].numpy()  # Middle layer
y = data['gm_labels'][:, 0].numpy()       # actual_deception (GROUND TRUTH)

# Train probe with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=5, scoring='r2')
print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

### Workflow 2: Layer Comparison

**Goal:** Find which layer best encodes deception.

```python
for layer_idx, layer_name in enumerate(data['layers']):
    X = data['activations'][:, layer_idx, :].numpy()
    y = data['labels'][:, 0].numpy()

    scores = cross_val_score(Ridge(), X, y, cv=5, scoring='r2')
    print(f"{layer_name}: R² = {scores.mean():.3f}")
```

### Workflow 3: Compare Agent vs GM Labels

**Goal:** Does the model encode perceived deception differently than actual deception?

```python
# This is a key research question!
X = data['activations'][:, 1, :].numpy()  # Middle layer

# Probe for perceived deception (what agent THINKS)
y_perceived = data['agent_labels'][:, 0].numpy()
scores_perceived = cross_val_score(Ridge(), X, y_perceived, cv=5, scoring='r2')

# Probe for actual deception (what agent IS DOING)
y_actual = data['gm_labels'][:, 0].numpy()
scores_actual = cross_val_score(Ridge(), X, y_actual, cv=5, scoring='r2')

print(f"Perceived deception: R² = {scores_perceived.mean():.3f}")
print(f"Actual deception: R² = {scores_actual.mean():.3f}")

# Compare: If R²_actual > R²_perceived, model has better ground truth representation
# If R²_perceived > R²_actual, model represents beliefs better than actions
```

### Workflow 4: All Labels Comparison

**Goal:** Which concepts are best encoded in the model?

```python
print("Agent Labels (First-Person Beliefs):")
for label_idx, label_name in enumerate(data['agent_label_names']):
    y = data['agent_labels'][:, label_idx].numpy()
    scores = cross_val_score(Ridge(), X, y, cv=5, scoring='r2')
    print(f"  {label_name}: R² = {scores.mean():.3f}")

print("\nGM Labels (Ground Truth):")
for label_idx, label_name in enumerate(data['gm_label_names']):
    y = data['gm_labels'][:, label_idx].numpy()
    scores = cross_val_score(Ridge(), X, y, cv=5, scoring='r2')
    print(f"  {label_name}: R² = {scores.mean():.3f}")
```

### Workflow 5: Find Important Neurons

**Goal:** Which neurons matter most for actual deception?

```python
from sklearn.linear_model import Ridge
import numpy as np

X = data['activations'][:, 1, :].numpy()
y = data['gm_labels'][:, 0].numpy()  # actual_deception (ground truth)

probe = Ridge().fit(X, y)
importance = np.abs(probe.coef_)

# Top 10 neurons for actual deception
top_neurons = np.argsort(importance)[-10:][::-1]
print(f"Most important neurons for ACTUAL deception: {top_neurons}")
print(f"Their weights: {importance[top_neurons]}")
```

### Workflow 6: Visualize Activation Space

**Goal:** See if deceptive vs non-deceptive samples cluster.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = data['activations'][:, 1, :].numpy()
y_actual = data['gm_labels'][:, 0].numpy()  # Ground truth

# Reduce to 2D
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

# Plot colored by actual deception
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_actual, cmap='RdYlGn_r', alpha=0.5)
plt.colorbar(label='Actual Deception (GM Ground Truth)')
plt.title('Activation Space by Actual Deception')
plt.savefig('actual_deception_tsne.png')
```

### Workflow 7: SAE Feature Analysis (Advanced)

**Goal:** Find interpretable features related to actual deception.

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    load_gemma_scope_sae,
    extract_sae_features,
)

# Load SAE
sae, cfg = load_gemma_scope_sae(model_size="2b", layer=12)

# Get features for each sample
all_features = []
for i in range(len(data['activations'])):
    act = data['activations'][i, 1, :]  # Middle layer
    features = extract_sae_features(sae, act)
    all_features.append(features.features.squeeze().numpy())

X_sae = np.stack(all_features)
y = data['gm_labels'][:, 0].numpy()  # actual_deception (ground truth)

# Now probe on SAE features (more interpretable!)
scores = cross_val_score(Ridge(), X_sae, y, cv=5, scoring='r2')
print(f"SAE features R² for actual_deception = {scores.mean():.3f}")
```

---

## Troubleshooting

### Problem: Low R² scores on everything

**Possible causes:**
1. Not enough data (try more trials)
2. Labels don't vary much (check `y.std()`)
3. Wrong layer (try different layers)
4. Concept isn't linearly encoded (try non-linear probe)

### Problem: Out of memory

**Solutions:**
1. Reduce `layers_to_capture` to just 1-2 layers
2. Use smaller model
3. Run fewer trials, save incrementally

### Problem: Very slow

**Solutions:**
1. Use GPU (`device="cuda"`)
2. Use smaller model for testing
3. Reduce `max_rounds`

### Problem: All labels are the same value

**Cause:** Not enough interaction for ToM to develop opinions.

**Solution:**
- Increase `max_rounds`
- Make sure agents observe each other's actions
- Check that `theory_of_mind` module is enabled

---

## Quick Reference

### Minimal Example

```python
from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner

runner = InterpretabilityRunner(
    model_name="google/gemma-2-2b-it",
    device="cuda",
)

results = runner.run_study(
    scenario='fishery',
    num_trials=50,
    use_gm=True,  # Enable GM for ground truth labels
)

runner.save_dataset('data.pt')
```

### Full Example with All Options

```python
from concordia.prefabs.entity.negotiation.evaluation import InterpretabilityRunner

runner = InterpretabilityRunner(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    layers_to_capture=[0, 6, 12, 18, 25],  # 5 layers
)

results = runner.run_study(
    scenario='fishery',
    agent_modules=['theory_of_mind', 'uncertainty_aware'],
    gm_modules=['social_intelligence'],  # For ground truth labels
    num_trials=100,
    max_rounds=15,
    use_gm=True,  # Enable GM for ground truth
)

runner.print_summary(results)
runner.save_dataset('full_experiment.pt')

# The saved file now contains BOTH:
# - agent_labels: What agent believes (perceived_deception, trust, etc.)
# - gm_labels: Ground truth (actual_deception, consistency, etc.)
```

### Recommended Settings for Different Goals

| Goal | Trials | Rounds | Layers | Model |
|------|--------|--------|--------|-------|
| Quick test | 5 | 5 | [0, 6, 11] | gpt-neo-125M |
| Development | 20 | 10 | [0, 13, 25] | gemma-2-2b |
| Research | 100 | 10 | [0, 6, 13, 19, 25] | gemma-2-2b |
| Publication | 200 | 15 | All layers | gemma-2-9b |
