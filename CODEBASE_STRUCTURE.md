# Codebase Structure Documentation

This document categorizes all files in the negotiation/interpretability system.

---

## 1. CORE FILES (Required for Deception Experiment)

These files are **required** to run `run_deception_experiment.py`:

### Entry Point
| File | Purpose |
|------|---------|
| `evaluation/run_deception_experiment.py` | Main CLI entry point - runs full pipeline |

### Interpretability Core
| File | Purpose |
|------|---------|
| `evaluation/interpretability_evaluation.py` | `InterpretabilityRunner`, `HybridLanguageModel`, activation capture |
| `evaluation/mech_interp_tools.py` | TransformerLens/SAE loading, feature extraction |
| `evaluation/train_probes.py` | Linear probe training (Ridge, mass-mean) |
| `evaluation/sanity_checks.py` | Probe validation (random labels, train-test gap) |
| `evaluation/causal_validation.py` | Activation patching, ablation, steering vectors |

### Scenario Definitions
| File | Purpose |
|------|---------|
| `evaluation/emergent_prompts.py` | 6 incentive-based deception scenarios (NO deception words) |
| `evaluation/deception_scenarios.py` | Apollo-style instructed deception scenarios |
| `evaluation/contest_scenarios.py` | Fishery, Treaty, Gameshow base scenarios |

### Package Init
| File | Purpose |
|------|---------|
| `evaluation/__init__.py` | Exports all evaluation classes |

---

## 1B. CONCORDIA CORE FRAMEWORK (Required - DO NOT DELETE)

These are **Concordia framework files** that our negotiation system depends on:

### Typing System
| File | Purpose |
|------|---------|
| `concordia/typing/entity.py` | Entity type definitions |
| `concordia/typing/entity_component.py` | Component base classes, lifecycle |
| `concordia/typing/prefab.py` | Prefab base class |
| `concordia/typing/__init__.py` | Package exports |
| `concordia/typing/deprecated/logging.py` | Logging utilities |

### Agent System
| File | Purpose |
|------|---------|
| `concordia/agents/entity_agent.py` | Core agent implementation |
| `concordia/agents/entity_agent_with_logging.py` | Agent with logging wrapper |
| `concordia/agents/__init__.py` | Package exports |

### Memory System
| File | Purpose |
|------|---------|
| `concordia/associative_memory/basic_associative_memory.py` | Memory bank implementation |
| `concordia/associative_memory/__init__.py` | Package exports |

### Language Model Interface
| File | Purpose |
|------|---------|
| `concordia/language_model/language_model.py` | Base `LanguageModel` abstract class |
| `concordia/language_model/__init__.py` | Package exports |

### Agent Components (used by base_negotiator)
| File | Purpose |
|------|---------|
| `concordia/components/agent/__init__.py` | Exports all agent components |
| `concordia/components/agent/instructions.py` | Instructions component |
| `concordia/components/agent/memory.py` | Memory component |
| `concordia/components/agent/observation.py` | Observation component |
| `concordia/components/agent/plan.py` | Planning component |
| `concordia/components/agent/constant.py` | Constant context |
| `concordia/components/agent/no_op_context_processor.py` | No-op processor |
| `concordia/components/agent/report_function.py` | Report functions |
| `concordia/components/agent/all_similar_memories.py` | Memory retrieval |

### Utilities
| File | Purpose |
|------|---------|
| `concordia/utils/concurrency.py` | Threading utilities |
| `concordia/utils/measurements.py` | Metrics collection |

### Entity Prefabs
| File | Purpose |
|------|---------|
| `concordia/prefabs/entity/minimal.py` | Minimal entity (used as counterpart base) |
| `concordia/prefabs/entity/__init__.py` | Package exports |

---

## 1C. NEGOTIATION FRAMEWORK (Our Code - Required)

### Negotiation Agent Prefabs
| File | Purpose |
|------|---------|
| `prefabs/entity/negotiation/advanced_negotiator.py` | Advanced negotiator with cognitive modules |
| `prefabs/entity/negotiation/base_negotiator.py` | Base negotiator prefab |
| `prefabs/entity/negotiation/config.py` | Strategy thresholds, evaluation config |
| `prefabs/entity/negotiation/constants.py` | Negotiation constants |
| `prefabs/entity/negotiation/__init__.py` | Package exports |

### Cognitive Module Components
| File | Purpose |
|------|---------|
| `components/negotiation_instructions.py` | Base negotiation instructions |
| `components/negotiation_memory.py` | Negotiation-specific memory |
| `components/negotiation_strategy.py` | Strategy component |
| `components/theory_of_mind.py` | Recursive belief modeling |
| `components/cultural_adaptation.py` | Cultural profiles |
| `components/temporal_strategy.py` | Multi-horizon planning |
| `components/swarm_intelligence.py` | Multi-agent voting |
| `components/uncertainty_aware.py` | Bayesian reasoning |
| `components/strategy_evolution.py` | Genetic algorithms |
| `components/__init__.py` | Package exports |

### Utilities
| File | Purpose |
|------|---------|
| `utils/parsing.py` | Response parsing utilities |
| `utils/__init__.py` | Package exports |

---

## 2. OUTDATED / UNUSED FILES (Can Delete)

These files are **not used** by the current system:

### Root-Level Clutter
| File | Status | Reason |
|------|--------|--------|
| `NEGOTIATION_TESTING_GUIDE.md` | Outdated | Superseded by `RUNPOD_EXPERIMENT_GUIDE.md` |
| `NEGOTIATION_COMPONENTS_SUMMARY.md` | Outdated | Generic overview, not actionable |
| `LLM_TESTING_GUIDE.md` | Outdated | Old testing approach |
| `HOW_TO_TEST_WITH_LLM.md` | Outdated | Duplicate of above |
| `TOGETHER_AI_TESTING_GUIDE.md` | Outdated | Old Together AI setup |
| `SETUP_GUIDE.md` | Outdated | Superseded by `RUNPOD_EXPERIMENT_GUIDE.md` |
| `NEGOTIATION_PR_REVIEW.md` | Outdated | PR review notes, not documentation |
| `TRANSFORMERLENS_INTEGRATION.md` | Outdated | Integrated into main code |
| `SETUP_AND_RUN_GUIDE.md` | Duplicate | Same as RUNPOD guide |
| `MATS_RESEARCH_TECHNICAL_ARCHITECTURE_v3.md` | Outdated | Old architecture doc |
| `MATS_APPLICATION_DATA.md` | Personal | Application notes |
| `MATS_TEMPLATE_REFERENCE.md` | Personal | Template reference |
| `METHODOLOGY.md` | Outdated | Old methodology |
| `RUNPOD_COMMANDS.md` | Duplicate | Merged into RUNPOD_EXPERIMENT_GUIDE |
| `EXPERIMENT_RESULTS.md` | Outdated | Old results |
| `MECH_INTERP_ANALYSIS.md` | Outdated | Old analysis |
| `OPEN_SOURCE_CHECKLIST.md` | Completed | Checklist done |
| `RUNPOD_SETUP.sh` | Duplicate | Commands in guide |
| `RUN_EXPERIMENT.sh` | Duplicate | Commands in guide |
| `RUNPOD_QUICK_TEST.sh` | Duplicate | Commands in guide |
| `run_full_experiment.py` | Outdated | Old entry point |
| `test_interpretability_evaluation.py` | Outdated | Old tests |
| `test_interpretability_quick.py` | Outdated | Old tests |
| `test_mech_interp_tools.py` | Outdated | Old tests |
| `test_transformerlens_integration.py` | Outdated | Old tests |

### Evaluation Folder Clutter
| File | Status | Reason |
|------|--------|--------|
| `evaluation/evaluation_harness.py` | Unused | Old harness, replaced by `interpretability_evaluation.py` |
| `evaluation/evaluation_harness_integrated.py` | Unused | Old integrated harness |
| `evaluation/run_experiments.py` | Unused | Old experiment runner |
| `evaluation/llm_evaluation.py` | Partially used | Only `create_*_model()` funcs used |
| `evaluation/runpod_eval.py` | Outdated | Old RunPod evaluation |
| `evaluation/merge_results.py` | Utility | Only for merging parallel results |
| `evaluation/analyze_results.py` | Utility | Post-hoc analysis |
| `evaluation/test_*.py` | Tests | Not part of runtime |
| `evaluation/verify_*.py` | Tests | Not part of runtime |
| `evaluation/USAGE_GUIDE.md` | Outdated | Old guide |
| `evaluation/SETUP_AND_RUN_GUIDE.md` | Duplicate | Superseded |
| `evaluation/INTERPRETABILITY_README.md` | Outdated | Old readme |
| `evaluation/RUNPOD_SETUP.md` | Duplicate | Superseded |
| `evaluation/EXPERIMENT_REFERENCE.md` | Outdated | Old reference |
| `evaluation/RUNPOD_COMMANDS.md` | Duplicate | Superseded |
| `evaluation/VISUALIZATION_REFERENCE.md` | Reference | Keep for viz |
| `evaluation/VALIDATION_REQUIREMENTS.md` | Reference | Keep for validation |

### docs/negotiation/ Folder
| File | Status | Reason |
|------|--------|--------|
| `docs/negotiation/*.md` | All outdated | Old documentation, superseded |

---

## 3. VALUABLE BUT NOT USED IN DECEPTION EXPERIMENT

These files are **part of the negotiation framework** but not used in the current deception detection pipeline:

### Cognitive Modules (Entity)
| File | Purpose | When Used |
|------|---------|-----------|
| `components/theory_of_mind.py` | Recursive belief modeling, emotion tracking | Full agent simulations |
| `components/cultural_adaptation.py` | Hofstede dimensions, cultural profiles | Cross-cultural negotiation |
| `components/temporal_strategy.py` | Multi-horizon planning, relationship tracking | Long-term negotiations |
| `components/swarm_intelligence.py` | Multi-agent voting, sub-agent analysis | Collective decisions |
| `components/uncertainty_aware.py` | Bayesian belief updates, confidence intervals | Probabilistic reasoning |
| `components/strategy_evolution.py` | Genetic algorithms, strategy genomes | Meta-learning |
| `components/negotiation_memory.py` | Negotiation-specific memory | Agent memory |
| `components/negotiation_instructions.py` | Base instructions | Agent setup |
| `components/negotiation_strategy.py` | Strategy component | Agent strategy |
| `components/__init__.py` | Exports | Package |

### Integration Framework
| File | Purpose | When Used |
|------|---------|-----------|
| `integration_framework.py` | Module dependency graph, coordination | Full agent with all modules |

### Game Master Components
| File | Purpose | When Used |
|------|---------|-----------|
| `game_master/negotiation/negotiation.py` | Main GM prefab | Full simulations |
| `game_master/negotiation/components/negotiation_state.py` | State tracking | GM scenarios |
| `game_master/negotiation/components/negotiation_validation.py` | BATNA validation | GM scenarios |
| `game_master/negotiation/components/negotiation_modules.py` | GM modules | GM scenarios |
| `game_master/negotiation/components/gm_social_intelligence.py` | Social tracking | Full simulations |
| `game_master/negotiation/components/gm_cultural_awareness.py` | Cultural context | Full simulations |
| `game_master/negotiation/components/gm_temporal_dynamics.py` | Temporal tracking | Full simulations |
| `game_master/negotiation/components/gm_collective_intelligence.py` | Collective tracking | Full simulations |
| `game_master/negotiation/components/gm_uncertainty_management.py` | Uncertainty tracking | Full simulations |
| `game_master/negotiation/components/gm_strategy_evolution.py` | Strategy tracking | Full simulations |

### Evaluation Utilities
| File | Purpose | When Used |
|------|---------|-----------|
| `evaluation/metrics.py` | Metrics collection | Full evaluation runs |
| `evaluation/statistical_analysis.py` | Stats, effect sizes, ANOVA | Paper analysis |
| `evaluation/baseline_agents.py` | Baseline agent implementations | Comparison studies |

### Examples
| File | Purpose | When Used |
|------|---------|-----------|
| `examples/example_agent_only.py` | Agent-only demo | Learning/demos |
| `examples/example_gm_only.py` | GM-only demo | Learning/demos |
| `examples/example_full_setup.py` | Full setup demo | Learning/demos |

---

## 4. RECOMMENDED CLEANUP

### Delete These (safe to remove):
```bash
# Root level
rm NEGOTIATION_TESTING_GUIDE.md LLM_TESTING_GUIDE.md HOW_TO_TEST_WITH_LLM.md
rm TOGETHER_AI_TESTING_GUIDE.md SETUP_GUIDE.md NEGOTIATION_PR_REVIEW.md
rm TRANSFORMERLENS_INTEGRATION.md SETUP_AND_RUN_GUIDE.md
rm MATS_RESEARCH_TECHNICAL_ARCHITECTURE_v3.md MATS_APPLICATION_DATA.md
rm MATS_TEMPLATE_REFERENCE.md METHODOLOGY.md RUNPOD_COMMANDS.md
rm EXPERIMENT_RESULTS.md MECH_INTERP_ANALYSIS.md OPEN_SOURCE_CHECKLIST.md
rm RUNPOD_SETUP.sh RUN_EXPERIMENT.sh RUNPOD_QUICK_TEST.sh
rm run_full_experiment.py test_*.py

# Evaluation folder
rm concordia/prefabs/entity/negotiation/evaluation/evaluation_harness.py
rm concordia/prefabs/entity/negotiation/evaluation/evaluation_harness_integrated.py
rm concordia/prefabs/entity/negotiation/evaluation/run_experiments.py
rm concordia/prefabs/entity/negotiation/evaluation/runpod_eval.py
rm concordia/prefabs/entity/negotiation/evaluation/USAGE_GUIDE.md
rm concordia/prefabs/entity/negotiation/evaluation/SETUP_AND_RUN_GUIDE.md
rm concordia/prefabs/entity/negotiation/evaluation/INTERPRETABILITY_README.md
rm concordia/prefabs/entity/negotiation/evaluation/RUNPOD_SETUP.md
rm concordia/prefabs/entity/negotiation/evaluation/EXPERIMENT_REFERENCE.md
rm concordia/prefabs/entity/negotiation/evaluation/RUNPOD_COMMANDS.md

# Docs folder
rm -rf docs/negotiation/
```

### Keep These (valuable):
- `RUNPOD_EXPERIMENT_GUIDE.md` - Main experiment guide
- `CLAUDE.md` - Claude Code instructions
- `README.md` - Main readme
- `RESEARCH_QUESTIONS_AND_OUTPUTS.md` - Research questions
- `CODEBASE_STRUCTURE.md` - This file
- `OUTPUT_INTERPRETATION_GUIDE.md` - Output interpretation

---

## 5. MINIMAL FILE SET FOR HACKATHON SUBMISSION

**IMPORTANT:** The full Concordia framework is required. You cannot extract just our files.

For hackathon, submit the **full repo** but highlight these as **our contributions**:

### Our Novel Contributions (Highlight These)
```
concordia/prefabs/entity/negotiation/
├── evaluation/
│   ├── run_deception_experiment.py      # Entry point (NOVEL)
│   ├── interpretability_evaluation.py   # HybridModel + runner (NOVEL)
│   ├── emergent_prompts.py              # 6 deception scenarios (NOVEL)
│   ├── train_probes.py                  # Probe training (NOVEL)
│   ├── causal_validation.py             # Causal tests (NOVEL)
│   ├── sanity_checks.py                 # Validation (NOVEL)
│   └── mech_interp_tools.py             # TransformerLens/SAE (NOVEL)
├── components/
│   ├── theory_of_mind.py                # ToM module (NOVEL)
│   ├── cultural_adaptation.py           # Cultural module (NOVEL)
│   ├── temporal_strategy.py             # Temporal module (NOVEL)
│   ├── swarm_intelligence.py            # Swarm module (NOVEL)
│   ├── uncertainty_aware.py             # Bayesian module (NOVEL)
│   └── strategy_evolution.py            # Evolution module (NOVEL)
├── advanced_negotiator.py               # Agent prefab (NOVEL)
├── base_negotiator.py                   # Base prefab (NOVEL)
└── config.py                            # Config (NOVEL)
```

### Required Concordia Framework (Not Our Code)
```
concordia/
├── typing/           # Entity/component types
├── agents/           # Agent implementation
├── components/agent/ # Base components
├── associative_memory/
├── language_model/
├── utils/
└── prefabs/entity/minimal.py
```

### Summary
| Category | Files | Action |
|----------|-------|--------|
| Our novel code | ~20 | Highlight in submission |
| Concordia framework | ~30 | Required dependency |
| Outdated clutter | ~40 | Delete before submission |
