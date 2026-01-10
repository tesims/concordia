# Codebase Structure Documentation

**VERIFIED BY IMPORT TRACE** - All files listed here were traced by parsing actual Python imports from `run_deception_experiment.py`.

---

## SUMMARY

| Category | Count | Status |
|----------|-------|--------|
| Evaluation (our interpretability code) | 15 | USED |
| Negotiation (our agent code) | 29 | USED |
| Concordia Core (framework dependency) | 95 | USED |
| **TOTAL REQUIRED** | **139** | KEEP |
| Unused evaluation files | 10 | CAN DELETE |
| Root-level MD clutter | ~30 | CAN DELETE |

---

## 1. EVALUATION FILES (15 files - USED)

These are **our interpretability/deception detection code**:

```
concordia/prefabs/entity/negotiation/evaluation/
├── __init__.py                    # Package exports
├── run_deception_experiment.py    # ENTRY POINT - main CLI
├── interpretability_evaluation.py # HybridLanguageModel, InterpretabilityRunner
├── emergent_prompts.py            # 6 incentive-based deception scenarios
├── deception_scenarios.py         # Apollo-style instructed scenarios
├── contest_scenarios.py           # Fishery, Treaty, Gameshow scenarios
├── train_probes.py                # Linear probe training
├── causal_validation.py           # Activation patching, ablation, steering
├── sanity_checks.py               # Probe validation checks
├── mech_interp_tools.py           # TransformerLens/SAE integration
├── metrics.py                     # Metrics collection
├── statistical_analysis.py        # Stats, effect sizes, ANOVA
├── baseline_agents.py             # Baseline agent implementations
├── evaluation_harness.py          # Experiment runner harness
└── llm_evaluation.py              # LLM model creation helpers
```

---

## 2. NEGOTIATION FILES (29 files - USED)

These are **our negotiation agent framework**:

### Agent Prefabs
```
concordia/prefabs/entity/negotiation/
├── __init__.py
├── advanced_negotiator.py         # Advanced agent with cognitive modules
├── base_negotiator.py             # Base negotiation agent
├── config.py                      # Strategy thresholds
├── constants.py                   # Module type constants
└── utils/
    ├── __init__.py
    └── parsing.py                 # Response parsing
```

### Cognitive Modules (Entity)
```
concordia/prefabs/entity/negotiation/components/
├── __init__.py
├── theory_of_mind.py              # Recursive belief modeling
├── cultural_adaptation.py         # Hofstede cultural dimensions
├── temporal_strategy.py           # Multi-horizon planning
├── swarm_intelligence.py          # Multi-agent voting
├── uncertainty_aware.py           # Bayesian belief updates
├── strategy_evolution.py          # Genetic algorithms
├── negotiation_instructions.py    # Base instructions
├── negotiation_memory.py          # Negotiation memory
└── negotiation_strategy.py        # Strategy component
```

### Game Master Modules
```
concordia/prefabs/game_master/negotiation/
├── __init__.py
├── negotiation.py                 # Main GM prefab
└── components/
    ├── __init__.py
    ├── negotiation_state.py       # State tracking
    ├── negotiation_validation.py  # BATNA validation
    ├── negotiation_modules.py     # GM modules
    ├── gm_social_intelligence.py
    ├── gm_cultural_awareness.py
    ├── gm_temporal_dynamics.py
    ├── gm_collective_intelligence.py
    ├── gm_uncertainty_management.py
    └── gm_strategy_evolution.py
```

---

## 3. CONCORDIA CORE (95 files - FRAMEWORK DEPENDENCY)

These are **Google DeepMind's Concordia framework** files we depend on. **DO NOT DELETE**.

### Typing System (10 files)
```
concordia/typing/
├── __init__.py
├── entity.py
├── entity_component.py
├── prefab.py
├── logging.py
├── scene.py
└── deprecated/
    ├── __init__.py
    ├── component.py
    ├── entity.py
    └── logging.py
```

### Agent System (4 files)
```
concordia/agents/
├── __init__.py
├── entity_agent.py
└── entity_agent_with_logging.py
```

### Agent Components (11 files)
```
concordia/components/agent/
├── __init__.py
├── action_spec_ignored.py
├── all_similar_memories.py
├── concat_act_component.py
├── constant.py
├── instructions.py
├── memory.py
├── no_op_context_processor.py
├── observation.py
├── plan.py
├── question_of_recent_memories.py
├── report_function.py
└── scripted_act.py
```

### Game Master Components (16 files)
```
concordia/components/game_master/
├── __init__.py
├── event_resolution.py
├── formative_memories_initializer.py
├── instructions.py
├── inventory.py
├── make_observation.py
├── next_acting.py
├── next_game_master.py
├── open_ended_questionnaire.py
├── payoff_matrix.py
├── questionnaire.py
├── scene_tracker.py
├── script.py
├── switch_act.py
├── terminate.py
└── world_state.py
```

### Entity Prefabs (7 files)
```
concordia/prefabs/entity/
├── __init__.py
├── basic.py
├── basic_scripted.py
├── basic_with_plan.py
├── conversational.py
├── fake_assistant_with_configurable_system_prompt.py
└── minimal.py
```

### Game Master Prefabs (13 files)
```
concordia/prefabs/game_master/
├── __init__.py
├── dialogic.py
├── dialogic_and_dramaturgic.py
├── formative_memories_initializer.py
├── game_theoretic_and_dramaturgic.py
├── generic.py
├── interviewer.py
├── marketplace.py
├── open_ended_interviewer.py
├── psychology_experiment.py
├── scripted.py
├── situated.py
└── situated_in_time_and_place.py
```

### Other Core (34 files)
```
concordia/
├── __init__.py
├── associative_memory/ (2 files)
├── language_model/ (7 files: language_model.py, together_ai.py, etc.)
├── document/ (3 files)
├── environment/ (4 files)
├── thought_chains/ (2 files)
├── utils/ (8 files)
└── contrib/ (4 files)
```

---

## 4. UNUSED FILES (VERIFIED - CAN DELETE)

### Unused Evaluation Files (10 files)
These are **NOT imported** by `run_deception_experiment.py`:

```bash
rm concordia/prefabs/entity/negotiation/evaluation/analyze_results.py
rm concordia/prefabs/entity/negotiation/evaluation/evaluation_harness_integrated.py
rm concordia/prefabs/entity/negotiation/evaluation/merge_results.py
rm concordia/prefabs/entity/negotiation/evaluation/run_experiments.py
rm concordia/prefabs/entity/negotiation/evaluation/runpod_eval.py
rm concordia/prefabs/entity/negotiation/evaluation/test_component_fixes.py
rm concordia/prefabs/entity/negotiation/evaluation/test_full_integration.py
rm concordia/prefabs/entity/negotiation/evaluation/test_multi_agent_enhancements.py
rm concordia/prefabs/entity/negotiation/evaluation/verify_implementation.py
rm concordia/prefabs/entity/negotiation/evaluation/verify_live_test.py
```

### Unused Test Files
```bash
rm concordia/prefabs/entity/negotiation/agent_builders_test.py
rm concordia/prefabs/entity/negotiation/negotiation_modules_test.py
rm concordia/prefabs/game_master/negotiation/*_test.py
rm concordia/prefabs/game_master/negotiation/components/*_test.py
```

### Root-Level Clutter (can delete)
```bash
rm NEGOTIATION_TESTING_GUIDE.md LLM_TESTING_GUIDE.md HOW_TO_TEST_WITH_LLM.md
rm TOGETHER_AI_TESTING_GUIDE.md SETUP_GUIDE.md NEGOTIATION_PR_REVIEW.md
rm TRANSFORMERLENS_INTEGRATION.md SETUP_AND_RUN_GUIDE.md
rm MATS_RESEARCH_TECHNICAL_ARCHITECTURE_v3.md MATS_APPLICATION_DATA.md
rm MATS_TEMPLATE_REFERENCE.md METHODOLOGY.md RUNPOD_COMMANDS.md
rm EXPERIMENT_RESULTS.md MECH_INTERP_ANALYSIS.md OPEN_SOURCE_CHECKLIST.md
rm RUNPOD_SETUP.sh RUN_EXPERIMENT.sh RUNPOD_QUICK_TEST.sh
rm run_full_experiment.py test_*.py
```

### Evaluation Folder MD Clutter
```bash
rm concordia/prefabs/entity/negotiation/evaluation/USAGE_GUIDE.md
rm concordia/prefabs/entity/negotiation/evaluation/SETUP_AND_RUN_GUIDE.md
rm concordia/prefabs/entity/negotiation/evaluation/INTERPRETABILITY_README.md
rm concordia/prefabs/entity/negotiation/evaluation/RUNPOD_SETUP.md
rm concordia/prefabs/entity/negotiation/evaluation/EXPERIMENT_REFERENCE.md
rm concordia/prefabs/entity/negotiation/evaluation/RUNPOD_COMMANDS.md
```

### docs/negotiation/ Folder (outdated)
```bash
rm -rf docs/negotiation/
```

---

## 5. KEEP THESE FILES

- `RUNPOD_EXPERIMENT_GUIDE.md` - Main experiment guide
- `CLAUDE.md` - Claude Code instructions
- `README.md` - Main readme
- `CODEBASE_STRUCTURE.md` - This file
- `RESEARCH_QUESTIONS_AND_OUTPUTS.md` - Research questions
- `OUTPUT_INTERPRETATION_GUIDE.md` - Output guide
- `concordia/prefabs/entity/negotiation/README.md` - Negotiation readme
- `concordia/prefabs/entity/negotiation/evaluation/README.md` - Evaluation readme
- `concordia/prefabs/entity/negotiation/evaluation/VALIDATION_REQUIREMENTS.md` - Keep
- `concordia/prefabs/entity/negotiation/evaluation/VISUALIZATION_REFERENCE.md` - Keep

---

## 6. HACKATHON SUBMISSION

### Our Novel Contributions (44 files)
```
Evaluation (15 files):
  run_deception_experiment.py, interpretability_evaluation.py,
  emergent_prompts.py, deception_scenarios.py, contest_scenarios.py,
  train_probes.py, causal_validation.py, sanity_checks.py,
  mech_interp_tools.py, metrics.py, statistical_analysis.py,
  baseline_agents.py, evaluation_harness.py, llm_evaluation.py, __init__.py

Negotiation Framework (29 files):
  advanced_negotiator.py, base_negotiator.py, config.py, constants.py,
  9 cognitive module components,
  12 game master components,
  parsing.py, __init__.py files
```

### Framework Dependency (95 files)
```
Google DeepMind's Concordia framework - required but not our code
```
