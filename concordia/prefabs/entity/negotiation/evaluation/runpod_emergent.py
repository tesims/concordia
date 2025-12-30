#!/usr/bin/env python3
"""
RunPod script for Emergent Concordia Evaluation

This runs emergent deception experiments through the FULL Concordia
negotiation framework with TransformerLens activation capture.

Usage on RunPod:
    # Set environment first
    export HF_HOME=/workspace/.cache/huggingface
    export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
    export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

    # Login to HuggingFace (Gemma is gated)
    python -c "from huggingface_hub import login; login()"

    # Clone and install
    cd /workspace
    git clone -b emergent-deception-v2 https://github.com/tesims/concordia.git
    cd concordia
    pip install -e .

    # Run experiment
    cd concordia/prefabs/entity/negotiation/evaluation
    python runpod_emergent.py --quick        # Quick test (5 trials)
    python runpod_emergent.py --trials 50    # Production run

Configuration:
    --model: Model to use (default: google/gemma-2-9b-it)
    --trials: Trials per condition per scenario (default: 50)
    --scenarios: Comma-separated scenarios (default: all 6)
    --quick: Quick test mode (5 trials, 2 scenarios)
"""

import os
import sys
import time
import argparse
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'model_name': 'google/gemma-2-9b-it',
    'device': 'cuda',
    'trials_per_condition': 50,
    'max_rounds': 5,
    'agent_modules': ['theory_of_mind'],
    'output_dir': 'results',
    'scenarios': [
        'ultimatum_bluff',
        'capability_bluff',
        'hidden_value',
        'info_withholding',
        'promise_break',
        'alliance_betrayal',
    ],
}


def check_gpu():
    """Check GPU availability and return device."""
    import torch

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Running on CPU will be very slow.")
        return 'cpu'

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Check for 27B model requirements
    if gpu_mem < 70 and '27b' in DEFAULT_CONFIG['model_name'].lower():
        print("WARNING: 27B model needs ~70GB VRAM")
        print("  Switching to 9B model instead...")
        DEFAULT_CONFIG['model_name'] = 'google/gemma-2-9b-it'

    return 'cuda'


def main():
    parser = argparse.ArgumentParser(description="Emergent Concordia Evaluation")
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model_name'],
                        help='Model name (default: google/gemma-2-9b-it)')
    parser.add_argument('--trials', type=int, default=DEFAULT_CONFIG['trials_per_condition'],
                        help='Trials per condition per scenario (default: 50)')
    parser.add_argument('--scenarios', type=str, default=None,
                        help='Comma-separated scenarios (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (5 trials, 2 scenarios)')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Output directory (default: results)')

    args = parser.parse_args()

    print("=" * 70)
    print("EMERGENT CONCORDIA EVALUATION")
    print("Full Concordia Framework + TransformerLens + Emergent Deception")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Check GPU
    device = check_gpu()

    # Configure
    model_name = args.model
    num_trials = 5 if args.quick else args.trials
    scenarios = DEFAULT_CONFIG['scenarios']

    if args.quick:
        scenarios = ['ultimatum_bluff', 'promise_break']
    elif args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(',')]

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Trials per condition: {num_trials}")
    print(f"  Total trials: {len(scenarios) * num_trials * 2}")
    print(f"  Agent modules: {DEFAULT_CONFIG['agent_modules']}")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Import (slow due to TransformerLens)
    print("Loading TransformerLens (this may take a minute)...")
    from emergent_concordia_evaluation import EmergentConcordiaRunner

    # Create runner
    print(f"\nLoading model: {model_name}")
    runner = EmergentConcordiaRunner(
        model_name=model_name,
        device=device,
    )

    total_start = time.time()
    all_results = {}

    # Run each scenario
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*70}")

        start_time = time.time()

        results = runner.run_study(
            scenario=scenario,
            num_trials=num_trials,
            agent_modules=DEFAULT_CONFIG['agent_modules'],
            max_rounds=DEFAULT_CONFIG['max_rounds'],
        )

        elapsed = time.time() - start_time
        all_results[scenario] = results

        print(f"\n{scenario} complete: {elapsed/60:.1f} min")

        # Save checkpoint
        checkpoint_file = f"{args.output}/{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        runner.save_dataset(checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

    # Final save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_file = f"{args.output}/emergent_concordia_{timestamp}.pt"
    runner.save_dataset(final_file)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)

    print(f"\nResults by scenario:")
    for scenario, results in all_results.items():
        for condition, data in results.get('conditions', {}).items():
            print(f"  {scenario} ({condition}): "
                  f"deception_rate={data['deception_rate']:.1%}")

    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Total samples: {len(runner.activation_samples)}")
    print(f"\nOutput: {final_file}")

    print("\n" + "-" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("""
To analyze results:
    from concordia.prefabs.entity.negotiation.evaluation.mech_interp_tools import (
        mass_mean_probe, train_linear_probe
    )

    import torch
    data = torch.load('emergent_concordia_*.pt')

    # Train probe on emergent ground truth
    X = data['activations'][:, 1, :].numpy()  # Middle layer
    y = data['emergent_labels'][:, 0].numpy()  # Binary deception

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, roc_auc_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred):.3f}")
""")


if __name__ == '__main__':
    main()
