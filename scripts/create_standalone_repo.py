#!/usr/bin/env python3
"""
Script to create a standalone repository from the current codebase.

This script:
1. Creates the new directory structure
2. Copies required Concordia core files
3. Copies our negotiation framework
4. Copies our interpretability pipeline
5. Updates imports to use new paths

Usage:
    python scripts/create_standalone_repo.py --output ../negotiation-deception-detection
"""

import argparse
import os
import re
import shutil
from pathlib import Path

# Source paths (relative to gsoc-concordia root)
SOURCE_ROOT = Path(__file__).parent.parent

# Files to copy - organized by destination
FILES_TO_COPY = {
    # Concordia core files -> concordia_mini/
    "concordia_mini": [
        # Root
        "concordia/__init__.py",

        # Typing
        "concordia/typing/__init__.py",
        "concordia/typing/entity.py",
        "concordia/typing/entity_component.py",
        "concordia/typing/prefab.py",
        "concordia/typing/logging.py",
        "concordia/typing/scene.py",
        "concordia/typing/deprecated/__init__.py",
        "concordia/typing/deprecated/component.py",
        "concordia/typing/deprecated/entity.py",
        "concordia/typing/deprecated/logging.py",

        # Agents
        "concordia/agents/__init__.py",
        "concordia/agents/entity_agent.py",
        "concordia/agents/entity_agent_with_logging.py",

        # Associative memory
        "concordia/associative_memory/__init__.py",
        "concordia/associative_memory/basic_associative_memory.py",

        # Language model
        "concordia/language_model/__init__.py",
        "concordia/language_model/language_model.py",
        "concordia/language_model/together_ai.py",
        "concordia/language_model/ollama_model.py",
        "concordia/language_model/google_aistudio_model.py",
        "concordia/language_model/gpt_model.py",
        "concordia/language_model/base_gpt_model.py",

        # Components - agent
        "concordia/components/__init__.py",
        "concordia/components/agent/__init__.py",
        "concordia/components/agent/action_spec_ignored.py",
        "concordia/components/agent/all_similar_memories.py",
        "concordia/components/agent/concat_act_component.py",
        "concordia/components/agent/constant.py",
        "concordia/components/agent/instructions.py",
        "concordia/components/agent/memory.py",
        "concordia/components/agent/no_op_context_processor.py",
        "concordia/components/agent/observation.py",
        "concordia/components/agent/plan.py",
        "concordia/components/agent/question_of_recent_memories.py",
        "concordia/components/agent/report_function.py",
        "concordia/components/agent/scripted_act.py",

        # Components - game master
        "concordia/components/game_master/__init__.py",
        "concordia/components/game_master/event_resolution.py",
        "concordia/components/game_master/formative_memories_initializer.py",
        "concordia/components/game_master/instructions.py",
        "concordia/components/game_master/inventory.py",
        "concordia/components/game_master/make_observation.py",
        "concordia/components/game_master/next_acting.py",
        "concordia/components/game_master/next_game_master.py",
        "concordia/components/game_master/open_ended_questionnaire.py",
        "concordia/components/game_master/payoff_matrix.py",
        "concordia/components/game_master/questionnaire.py",
        "concordia/components/game_master/scene_tracker.py",
        "concordia/components/game_master/script.py",
        "concordia/components/game_master/switch_act.py",
        "concordia/components/game_master/terminate.py",
        "concordia/components/game_master/world_state.py",

        # Prefabs - entity
        "concordia/prefabs/__init__.py",
        "concordia/prefabs/entity/__init__.py",
        "concordia/prefabs/entity/basic.py",
        "concordia/prefabs/entity/basic_scripted.py",
        "concordia/prefabs/entity/basic_with_plan.py",
        "concordia/prefabs/entity/conversational.py",
        "concordia/prefabs/entity/fake_assistant_with_configurable_system_prompt.py",
        "concordia/prefabs/entity/minimal.py",

        # Prefabs - game master
        "concordia/prefabs/game_master/__init__.py",
        "concordia/prefabs/game_master/dialogic.py",
        "concordia/prefabs/game_master/dialogic_and_dramaturgic.py",
        "concordia/prefabs/game_master/formative_memories_initializer.py",
        "concordia/prefabs/game_master/game_theoretic_and_dramaturgic.py",
        "concordia/prefabs/game_master/generic.py",
        "concordia/prefabs/game_master/interviewer.py",
        "concordia/prefabs/game_master/marketplace.py",
        "concordia/prefabs/game_master/open_ended_interviewer.py",
        "concordia/prefabs/game_master/psychology_experiment.py",
        "concordia/prefabs/game_master/scripted.py",
        "concordia/prefabs/game_master/situated.py",
        "concordia/prefabs/game_master/situated_in_time_and_place.py",

        # Document
        "concordia/document/__init__.py",
        "concordia/document/document.py",
        "concordia/document/interactive_document.py",

        # Environment
        "concordia/environment/__init__.py",
        "concordia/environment/engine.py",
        "concordia/environment/engines/__init__.py",
        "concordia/environment/engines/sequential.py",

        # Thought chains
        "concordia/thought_chains/__init__.py",
        "concordia/thought_chains/thought_chains.py",

        # Utils
        "concordia/utils/__init__.py",
        "concordia/utils/concurrency.py",
        "concordia/utils/helper_functions.py",
        "concordia/utils/measurements.py",
        "concordia/utils/sampling.py",
        "concordia/utils/text.py",
        "concordia/utils/deprecated/__init__.py",
        "concordia/utils/deprecated/measurements.py",

        # Contrib
        "concordia/contrib/__init__.py",
        "concordia/contrib/data/__init__.py",
        "concordia/contrib/data/questionnaires/__init__.py",
        "concordia/contrib/data/questionnaires/base_questionnaire.py",
    ],

    # Our negotiation framework -> negotiation/
    "negotiation": [
        "concordia/prefabs/entity/negotiation/__init__.py",
        "concordia/prefabs/entity/negotiation/advanced_negotiator.py",
        "concordia/prefabs/entity/negotiation/base_negotiator.py",
        "concordia/prefabs/entity/negotiation/config.py",
        "concordia/prefabs/entity/negotiation/constants.py",

        # Components
        "concordia/prefabs/entity/negotiation/components/__init__.py",
        "concordia/prefabs/entity/negotiation/components/theory_of_mind.py",
        "concordia/prefabs/entity/negotiation/components/cultural_adaptation.py",
        "concordia/prefabs/entity/negotiation/components/temporal_strategy.py",
        "concordia/prefabs/entity/negotiation/components/swarm_intelligence.py",
        "concordia/prefabs/entity/negotiation/components/uncertainty_aware.py",
        "concordia/prefabs/entity/negotiation/components/strategy_evolution.py",
        "concordia/prefabs/entity/negotiation/components/negotiation_instructions.py",
        "concordia/prefabs/entity/negotiation/components/negotiation_memory.py",
        "concordia/prefabs/entity/negotiation/components/negotiation_strategy.py",

        # Utils
        "concordia/prefabs/entity/negotiation/utils/__init__.py",
        "concordia/prefabs/entity/negotiation/utils/parsing.py",

        # Game master
        "concordia/prefabs/game_master/negotiation/__init__.py",
        "concordia/prefabs/game_master/negotiation/negotiation.py",
        "concordia/prefabs/game_master/negotiation/components/__init__.py",
        "concordia/prefabs/game_master/negotiation/components/negotiation_state.py",
        "concordia/prefabs/game_master/negotiation/components/negotiation_validation.py",
        "concordia/prefabs/game_master/negotiation/components/negotiation_modules.py",
        "concordia/prefabs/game_master/negotiation/components/gm_social_intelligence.py",
        "concordia/prefabs/game_master/negotiation/components/gm_cultural_awareness.py",
        "concordia/prefabs/game_master/negotiation/components/gm_temporal_dynamics.py",
        "concordia/prefabs/game_master/negotiation/components/gm_collective_intelligence.py",
        "concordia/prefabs/game_master/negotiation/components/gm_uncertainty_management.py",
        "concordia/prefabs/game_master/negotiation/components/gm_strategy_evolution.py",
    ],

    # Our interpretability pipeline -> interpretability/
    "interpretability": [
        "concordia/prefabs/entity/negotiation/evaluation/__init__.py",
        "concordia/prefabs/entity/negotiation/evaluation/run_deception_experiment.py",
        "concordia/prefabs/entity/negotiation/evaluation/interpretability_evaluation.py",
        "concordia/prefabs/entity/negotiation/evaluation/emergent_prompts.py",
        "concordia/prefabs/entity/negotiation/evaluation/deception_scenarios.py",
        "concordia/prefabs/entity/negotiation/evaluation/contest_scenarios.py",
        "concordia/prefabs/entity/negotiation/evaluation/train_probes.py",
        "concordia/prefabs/entity/negotiation/evaluation/sanity_checks.py",
        "concordia/prefabs/entity/negotiation/evaluation/causal_validation.py",
        "concordia/prefabs/entity/negotiation/evaluation/mech_interp_tools.py",
        "concordia/prefabs/entity/negotiation/evaluation/metrics.py",
        "concordia/prefabs/entity/negotiation/evaluation/statistical_analysis.py",
        "concordia/prefabs/entity/negotiation/evaluation/baseline_agents.py",
        "concordia/prefabs/entity/negotiation/evaluation/evaluation_harness.py",
        "concordia/prefabs/entity/negotiation/evaluation/llm_evaluation.py",
    ],
}

# Import path replacements
IMPORT_REPLACEMENTS = [
    # Concordia imports -> concordia_mini
    (r"from concordia\.", "from concordia_mini."),
    (r"import concordia\.", "import concordia_mini."),

    # Negotiation prefab imports -> negotiation
    (r"from concordia\.prefabs\.entity\.negotiation\.", "from negotiation."),
    (r"from concordia\.prefabs\.game_master\.negotiation\.", "from negotiation.game_master."),

    # Evaluation imports -> interpretability
    (r"from concordia\.prefabs\.entity\.negotiation\.evaluation\.", "from interpretability."),
    (r"from \.evaluation\.", "from interpretability."),
]


def create_directory_structure(output_dir: Path):
    """Create the directory structure."""
    dirs = [
        "concordia_mini/typing/deprecated",
        "concordia_mini/agents",
        "concordia_mini/associative_memory",
        "concordia_mini/language_model",
        "concordia_mini/components/agent",
        "concordia_mini/components/game_master",
        "concordia_mini/prefabs/entity",
        "concordia_mini/prefabs/game_master",
        "concordia_mini/document",
        "concordia_mini/environment/engines",
        "concordia_mini/thought_chains",
        "concordia_mini/utils/deprecated",
        "concordia_mini/contrib/data/questionnaires",
        "negotiation/components",
        "negotiation/utils",
        "negotiation/game_master/components",
        "interpretability/scenarios",
        "interpretability/probes",
        "interpretability/causal",
        "interpretability/evaluation",
        "docs",
        "scripts",
        "examples",
        "tests",
        "results",
    ]

    for d in dirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def copy_files(output_dir: Path):
    """Copy all required files."""
    for dest_prefix, files in FILES_TO_COPY.items():
        for src_file in files:
            src_path = SOURCE_ROOT / src_file

            if not src_path.exists():
                print(f"WARNING: Source file not found: {src_file}")
                continue

            # Determine destination path
            if dest_prefix == "concordia_mini":
                # concordia/x/y.py -> concordia_mini/x/y.py
                dest_file = src_file.replace("concordia/", "concordia_mini/")
            elif dest_prefix == "negotiation":
                # concordia/prefabs/entity/negotiation/x.py -> negotiation/x.py
                dest_file = src_file.replace("concordia/prefabs/entity/negotiation/", "negotiation/")
                dest_file = dest_file.replace("concordia/prefabs/game_master/negotiation/", "negotiation/game_master/")
            elif dest_prefix == "interpretability":
                # concordia/prefabs/entity/negotiation/evaluation/x.py -> interpretability/x.py
                dest_file = src_file.replace("concordia/prefabs/entity/negotiation/evaluation/", "interpretability/")

            dest_path = output_dir / dest_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dest_path)
            print(f"Copied: {src_file} -> {dest_file}")


def update_imports(output_dir: Path):
    """Update imports in all Python files."""
    for py_file in output_dir.rglob("*.py"):
        with open(py_file, "r") as f:
            content = f.read()

        original = content
        for pattern, replacement in IMPORT_REPLACEMENTS:
            content = re.sub(pattern, replacement, content)

        if content != original:
            with open(py_file, "w") as f:
                f.write(content)
            print(f"Updated imports: {py_file.relative_to(output_dir)}")


def create_root_files(output_dir: Path):
    """Create root-level files."""

    # README.md
    readme = '''# Negotiation Deception Detection

Mechanistic interpretability for detecting deception in LLM negotiation agents.

## Overview

This project provides:
1. **Negotiation Framework** - Cognitive agent modules for multi-agent negotiation
2. **Interpretability Pipeline** - Tools for detecting deception via activation analysis
3. **Deception Scenarios** - 6 incentive-based scenarios that elicit emergent deception

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from interpretability.run_experiment import main

# Run deception detection experiment
main([
    "--mode", "emergent",
    "--scenarios", "3",
    "--trials", "10",
    "--model", "google/gemma-2-2b-it",
    "--device", "cuda",
])
```

## Project Structure

- `concordia_mini/` - Minimal subset of Google DeepMind's Concordia framework
- `negotiation/` - Our cognitive negotiation agent modules
- `interpretability/` - Deception detection and analysis pipeline

## Citation

```bibtex
@software{negotiation_deception_detection,
  title = {Mechanistic Interpretability for Detecting Deception in LLM Negotiation Agents},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/USERNAME/negotiation-deception-detection}
}
```

## License

Apache 2.0 - See LICENSE file.

## Acknowledgments

This project builds upon [Concordia](https://github.com/google-deepmind/concordia)
by Google DeepMind, included under Apache 2.0 license.
'''
    (output_dir / "README.md").write_text(readme)

    # LICENSE
    license_text = '''Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

... (full Apache 2.0 license text)
'''
    (output_dir / "LICENSE").write_text(license_text)

    # requirements.txt
    requirements = '''torch>=2.0
transformers>=4.35
transformer-lens>=1.0
sae-lens>=0.5
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.65
sentence-transformers>=2.2
'''
    (output_dir / "requirements.txt").write_text(requirements)

    # pyproject.toml
    pyproject = '''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "negotiation-deception-detection"
version = "1.0.0"
description = "Mechanistic interpretability for detecting deception in LLM negotiation agents"
readme = "README.md"
license = {text = "Apache-2.0"}
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
    "sentence-transformers>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pylint"]

[project.scripts]
run-deception-experiment = "interpretability.run_deception_experiment:main"

[tool.setuptools.packages.find]
include = ["concordia_mini*", "negotiation*", "interpretability*"]
'''
    (output_dir / "pyproject.toml").write_text(pyproject)

    # setup.py
    setup = '''from setuptools import setup, find_packages
setup(
    name="negotiation-deception-detection",
    packages=find_packages(),
)
'''
    (output_dir / "setup.py").write_text(setup)

    # results/.gitkeep
    (output_dir / "results" / ".gitkeep").touch()

    print("Created root files: README.md, LICENSE, requirements.txt, pyproject.toml, setup.py")


def main():
    parser = argparse.ArgumentParser(description="Create standalone repository")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually copy files")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()

    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}")
        return 1

    print(f"Creating standalone repo at: {output_dir}")
    print()

    if args.dry_run:
        print("DRY RUN - not copying files")
        return 0

    # Create structure
    print("Creating directory structure...")
    create_directory_structure(output_dir)

    # Copy files
    print("\nCopying files...")
    copy_files(output_dir)

    # Update imports
    print("\nUpdating imports...")
    update_imports(output_dir)

    # Create root files
    print("\nCreating root files...")
    create_root_files(output_dir)

    print("\nDone!")
    print(f"\nNext steps:")
    print(f"  cd {output_dir}")
    print(f"  pip install -e .")
    print(f"  python -m interpretability.run_deception_experiment --help")

    return 0


if __name__ == "__main__":
    exit(main())
'''
