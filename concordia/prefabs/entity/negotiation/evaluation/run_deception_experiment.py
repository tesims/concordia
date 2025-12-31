#!/usr/bin/env python3
"""
Unified Deception Detection Experiment Runner

This script runs the complete deception detection pipeline:
1. Runs negotiation scenarios through Concordia agents (GM + Entity)
2. Captures activations via TransformerLens
3. Gets ground truth labels from GM modules AND emergent scenario rules
4. Trains linear probes to detect deception
5. Validates with sanity checks

Supports both:
- INSTRUCTED mode: Apollo Research style explicit deception instructions
- EMERGENT mode: Incentive-based, no deception words (novel contribution)

Usage:
    # Quick test
    python run_deception_experiment.py --mode emergent --scenarios 2 --trials 10

    # Full experiment
    python run_deception_experiment.py --mode emergent --scenarios 6 --trials 100 --model google/gemma-2-9b-it

    # With GPU
    python run_deception_experiment.py --device cuda --dtype bfloat16

    # Train probes on existing data
    python run_deception_experiment.py --train-only --data activations.pt
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from concordia.prefabs.entity.negotiation.evaluation import (
    # Core evaluation
    InterpretabilityRunner,
    # Emergent scenarios
    EMERGENT_SCENARIOS,
    IncentiveCondition,
    get_emergent_scenarios,
    generate_scenario_params,
    compute_emergent_ground_truth,
    # Instructed scenarios
    INSTRUCTED_SCENARIOS,
    Condition,
    ExperimentMode,
    get_instructed_scenarios,
    # Probe training
    run_full_analysis,
    train_ridge_probe,
    compute_generalization_auc,
    # Sanity checks
    run_all_sanity_checks,
    print_limitations,
)


def run_emergent_experiment(
    runner: "InterpretabilityRunner",
    scenarios: List[str],
    trials_per_scenario: int = 50,
    conditions: List[IncentiveCondition] = None,
) -> Dict[str, Any]:
    """
    Run emergent deception experiment through Concordia framework.

    Args:
        runner: InterpretabilityRunner with TransformerLens model
        scenarios: List of scenario names to run
        trials_per_scenario: Trials per scenario per condition
        conditions: IncentiveCondition values to test

    Returns:
        Dict with all results
    """
    if conditions is None:
        conditions = [IncentiveCondition.HIGH_INCENTIVE, IncentiveCondition.LOW_INCENTIVE]

    print(f"\n{'='*60}")
    print("EMERGENT DECEPTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Scenarios: {scenarios}")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"Trials per condition: {trials_per_scenario}")
    print(f"Total trials: {len(scenarios) * len(conditions) * trials_per_scenario}")

    # Use the integrated run_all_emergent_scenarios method
    results = runner.run_all_emergent_scenarios(
        scenarios=scenarios,
        trials_per_scenario=trials_per_scenario,
        conditions=conditions,
    )

    return results


def run_instructed_experiment(
    runner: "InterpretabilityRunner",
    scenarios: List[str],
    trials_per_scenario: int = 50,
    conditions: List[Condition] = None,
) -> Dict[str, Any]:
    """
    Run instructed deception experiment (Apollo Research style).

    Args:
        runner: InterpretabilityRunner with TransformerLens model
        scenarios: List of scenario names to run
        trials_per_scenario: Trials per scenario per condition
        conditions: Condition values to test

    Returns:
        Dict with all results
    """
    if conditions is None:
        conditions = [Condition.DECEPTIVE, Condition.HONEST]

    print(f"\n{'='*60}")
    print("INSTRUCTED DECEPTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Scenarios: {scenarios}")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"Trials per condition: {trials_per_scenario}")
    print(f"Total trials: {len(scenarios) * len(conditions) * trials_per_scenario}")

    # Use the integrated run_study method for each scenario
    all_samples = []
    for scenario in scenarios:
        for condition in conditions:
            print(f"\nRunning {scenario} / {condition.value}...")
            result = runner.run_study(
                scenario=scenario,
                num_trials=trials_per_scenario,
                condition=condition.value,
                use_gm=True,
            )
            all_samples.extend(result.activation_samples)

    return {"samples": all_samples, "mode": "instructed"}


def train_probes_on_data(data_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Train probes on captured activation data.

    Args:
        data_path: Path to activations.pt file
        output_dir: Directory for output files

    Returns:
        Dict with probe results
    """
    print(f"\n{'='*60}")
    print("PROBE TRAINING")
    print(f"{'='*60}")
    print(f"Loading data from: {data_path}")

    # Run full analysis
    results = run_full_analysis(data_path)

    # Save results
    if output_dir:
        output_path = Path(output_dir) / "probe_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run deception detection experiment with Concordia agents"
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="emergent",
        choices=["emergent", "instructed", "both"],
        help="Experiment mode: emergent (incentive-based) or instructed (explicit)"
    )

    # Model configuration
    parser.add_argument(
        "--model", type=str, default="google/gemma-2-9b-it",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )

    # Experiment configuration
    parser.add_argument(
        "--scenarios", type=int, default=6,
        help="Number of scenarios to run (max 6)"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Trials per scenario per condition"
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated list of layers to capture (default: auto)"
    )

    # Training mode
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train probes on existing data"
    )
    parser.add_argument(
        "--data", type=str,
        help="Path to activations.pt file for training"
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="./experiment_output",
        help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training-only mode
    if args.train_only:
        if not args.data:
            parser.error("--data is required when using --train-only")
        results = train_probes_on_data(args.data, str(output_dir))
        return

    # Get scenarios
    all_emergent = get_emergent_scenarios()
    all_instructed = get_instructed_scenarios()

    n_scenarios = min(args.scenarios, 6)
    emergent_scenarios = all_emergent[:n_scenarios]
    instructed_scenarios = all_instructed[:n_scenarios]

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"\n{'='*60}")
    print("DECEPTION DETECTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Scenarios: {n_scenarios}")
    print(f"Trials per condition: {args.trials}")
    print(f"Output directory: {output_dir}")

    # Initialize runner
    print(f"\nInitializing InterpretabilityRunner...")
    start_time = time.time()

    runner = InterpretabilityRunner(
        model_name=args.model,
        device=args.device,
        torch_dtype=dtype,
        layers_to_capture=layers,
    )

    init_time = time.time() - start_time
    print(f"Initialization complete in {init_time:.1f}s")

    # Run experiments
    all_results = {}

    if args.mode in ["emergent", "both"]:
        results = run_emergent_experiment(
            runner=runner,
            scenarios=emergent_scenarios,
            trials_per_scenario=args.trials,
        )
        all_results["emergent"] = results

    if args.mode in ["instructed", "both"]:
        results = run_instructed_experiment(
            runner=runner,
            scenarios=instructed_scenarios,
            trials_per_scenario=args.trials,
        )
        all_results["instructed"] = results

    # Save activations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    activations_path = output_dir / f"activations_{args.mode}_{timestamp}.pt"
    runner.save_dataset(str(activations_path))
    print(f"\nActivations saved to: {activations_path}")

    # Run sanity checks and train probes
    print(f"\n{'='*60}")
    print("POST-EXPERIMENT ANALYSIS")
    print(f"{'='*60}")

    probe_results = train_probes_on_data(str(activations_path), str(output_dir))

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {len(runner.activation_samples)}")
    print(f"Activations saved: {activations_path}")
    print(f"Output directory: {output_dir}")

    if probe_results.get("best_probe"):
        print(f"\nBest probe performance:")
        print(f"  Layer: {probe_results['best_probe']['layer']}")
        print(f"  R²: {probe_results['best_probe']['r2']:.3f}")

    if probe_results.get("gm_vs_agent"):
        gm_vs_agent = probe_results["gm_vs_agent"]
        print(f"\nGM vs Agent comparison:")
        print(f"  GM R²: {gm_vs_agent['gm_ridge_r2']:.3f}")
        print(f"  Agent R²: {gm_vs_agent['agent_ridge_r2']:.3f}")
        if gm_vs_agent["gm_wins"]:
            print(f"  >> GM labels more predictable (implicit deception encoding)")

    # Print limitations
    print_limitations(
        n_samples=len(runner.activation_samples),
        model_name=args.model,
        causal_validated=False,
    )

    print(f"\nTotal experiment time: {(time.time() - start_time):.1f}s")


if __name__ == "__main__":
    main()
