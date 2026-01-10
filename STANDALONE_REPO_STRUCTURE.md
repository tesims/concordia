# Standalone Repository Structure

This document outlines the structure for a self-contained open-source repository that includes:
1. Full negotiation agent framework (cognitive modules)
2. Mechanistic interpretability pipeline (deception detection)
3. All required Concordia dependencies

---

## Repository Structure

```
negotiation-deception-detection/
│
├── README.md                          # Project overview, installation, usage
├── LICENSE                            # Apache 2.0 (matching Concordia)
├── CITATION.cff                       # Citation file for academic use
├── pyproject.toml                     # Modern Python packaging
├── requirements.txt                   # Dependencies
├── setup.py                           # Package installation
│
├── docs/
│   ├── METHODOLOGY.md                 # Research methodology
│   ├── SCENARIOS.md                   # Deception scenario descriptions
│   ├── COGNITIVE_MODULES.md           # Negotiation module documentation
│   └── REPLICATION.md                 # How to replicate results
│
├── scripts/
│   ├── run_experiment.sh              # Full experiment runner
│   ├── quick_test.sh                  # Quick validation test
│   └── setup_runpod.sh                # RunPod GPU setup
│
├── examples/
│   ├── basic_negotiation.py           # Simple negotiation demo
│   ├── deception_detection.py         # Deception detection demo
│   └── activation_analysis.py         # Interpretability demo
│
├── tests/
│   ├── test_scenarios.py
│   ├── test_probes.py
│   └── test_agents.py
│
│
├── concordia_mini/                    # Minimal Concordia subset (95 files)
│   ├── __init__.py
│   ├── typing/
│   │   ├── __init__.py
│   │   ├── entity.py
│   │   ├── entity_component.py
│   │   ├── prefab.py
│   │   ├── logging.py
│   │   └── scene.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── entity_agent.py
│   │   └── entity_agent_with_logging.py
│   ├── associative_memory/
│   │   ├── __init__.py
│   │   └── basic_associative_memory.py
│   ├── language_model/
│   │   ├── __init__.py
│   │   ├── language_model.py
│   │   └── together_ai.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── agent/                     # 11 files
│   │   └── game_master/               # 16 files
│   ├── prefabs/
│   │   ├── __init__.py
│   │   ├── entity/                    # 7 files
│   │   └── game_master/               # 13 files
│   ├── document/
│   ├── environment/
│   ├── thought_chains/
│   └── utils/
│
│
├── negotiation/                       # Our negotiation framework (29 files)
│   ├── __init__.py
│   ├── advanced_negotiator.py         # Main advanced agent prefab
│   ├── base_negotiator.py             # Base agent prefab
│   ├── config.py                      # Configuration
│   ├── constants.py                   # Constants
│   │
│   ├── components/                    # 9 cognitive modules
│   │   ├── __init__.py
│   │   ├── theory_of_mind.py          # Recursive belief modeling
│   │   ├── cultural_adaptation.py     # Hofstede dimensions
│   │   ├── temporal_strategy.py       # Multi-horizon planning
│   │   ├── swarm_intelligence.py      # Multi-agent voting
│   │   ├── uncertainty_aware.py       # Bayesian reasoning
│   │   ├── strategy_evolution.py      # Genetic algorithms
│   │   ├── negotiation_instructions.py
│   │   ├── negotiation_memory.py
│   │   └── negotiation_strategy.py
│   │
│   ├── game_master/                   # 12 GM components
│   │   ├── __init__.py
│   │   ├── negotiation.py
│   │   └── components/
│   │       ├── negotiation_state.py
│   │       ├── negotiation_validation.py
│   │       ├── negotiation_modules.py
│   │       ├── gm_social_intelligence.py
│   │       ├── gm_cultural_awareness.py
│   │       ├── gm_temporal_dynamics.py
│   │       ├── gm_collective_intelligence.py
│   │       ├── gm_uncertainty_management.py
│   │       └── gm_strategy_evolution.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── parsing.py
│
│
├── interpretability/                  # Our interpretability pipeline (15 files)
│   ├── __init__.py
│   ├── run_experiment.py              # Main entry point
│   ├── hybrid_model.py                # HybridLanguageModel
│   ├── runner.py                      # InterpretabilityRunner
│   │
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── emergent_prompts.py        # 6 incentive-based scenarios
│   │   ├── deception_scenarios.py     # Apollo-style scenarios
│   │   └── contest_scenarios.py       # Fishery, Treaty, Gameshow
│   │
│   ├── probes/
│   │   ├── __init__.py
│   │   ├── train_probes.py            # Linear probe training
│   │   ├── sanity_checks.py           # Validation checks
│   │   └── mech_interp_tools.py       # TransformerLens/SAE
│   │
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── activation_patching.py
│   │   ├── ablation.py
│   │   └── steering_vectors.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── statistical_analysis.py
│       └── baseline_agents.py
│
│
└── results/                           # Experiment outputs (gitignored)
    └── .gitkeep
```

---

## File Count Summary

| Directory | Files | Description |
|-----------|-------|-------------|
| `concordia_mini/` | 95 | Minimal Concordia framework subset |
| `negotiation/` | 29 | Our cognitive negotiation modules |
| `interpretability/` | 15 | Our interpretability pipeline |
| `docs/` | 4 | Documentation |
| `scripts/` | 3 | Runner scripts |
| `examples/` | 3 | Example usage |
| `tests/` | 3 | Test files |
| Root files | 6 | README, LICENSE, setup, etc. |
| **TOTAL** | **~158** | Complete standalone package |

---

## Installation

```bash
# Clone
git clone https://github.com/USERNAME/negotiation-deception-detection.git
cd negotiation-deception-detection

# Install
pip install -e .

# Or with dependencies
pip install -e ".[dev]"
```

---

## Usage

```python
# Quick deception detection experiment
from interpretability import run_experiment

results = run_experiment(
    model="google/gemma-2-2b-it",
    scenarios=["ultimatum_bluff", "alliance_betrayal"],
    trials=10,
    device="cuda",
)

# Full negotiation simulation
from negotiation import AdvancedNegotiator
from concordia_mini.language_model import TogetherAI

model = TogetherAI(api_key="...")
agent = AdvancedNegotiator(
    model=model,
    modules=["theory_of_mind", "temporal_strategy"],
)
```

---

## pyproject.toml

```toml
[project]
name = "negotiation-deception-detection"
version = "1.0.0"
description = "Mechanistic interpretability for detecting deception in LLM negotiation agents"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "Your Name", email = "your@email.com"}
]
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.35",
    "transformer-lens>=1.0",
    "sae-lens>=0.5",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-xdist",
    "pylint",
]

[project.scripts]
run-deception-experiment = "interpretability.run_experiment:main"
```

---

## Licensing Considerations

Since we're including Concordia code:
1. Keep Apache 2.0 license (same as Concordia)
2. Add attribution in README
3. Keep original copyright headers in Concordia files
4. Add our own copyright for novel code

```
# In README.md
## Acknowledgments

This project builds upon [Concordia](https://github.com/google-deepmind/concordia)
by Google DeepMind. The `concordia_mini/` directory contains a minimal subset of
the Concordia framework, used under the Apache 2.0 license.
```

---

## Next Steps

1. Create the new repo structure
2. Copy and organize files
3. Update imports to use new paths
4. Test that everything works
5. Write comprehensive README
6. Add CITATION.cff for academic citation
