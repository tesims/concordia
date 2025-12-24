#!/usr/bin/env python3
"""Ready-to-run evaluation script for RunPod/Lambda Labs.

Usage on RunPod:
    1. Deploy a GPU pod (H200 or A100 recommended)
    2. Open web terminal or SSH
    3. Run:
        git clone https://github.com/YOUR_USERNAME/concordia.git
        cd concordia
        pip install -e ".[interpretability]"
        python concordia/prefabs/entity/negotiation/evaluation/runpod_eval.py

Configuration:
    Edit the CONFIG section below to customize your run.
"""

import os
import sys
import time
from datetime import datetime

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

CONFIG = {
    # Model settings
    'model_name': 'google/gemma-2-9b-it',
    'device': 'cuda',
    'layers_to_capture': None,  # None = auto (first, middle, last)

    # Experiment settings
    'scenario': 'fishery',
    'num_trials': 5,
    'max_rounds': 10,
    'use_gm': True,

    # Module settings
    'agent_modules': ['theory_of_mind'],
    'gm_modules': ['social_intelligence'],

    # Output
    'output_file': 'activations_{timestamp}.pt',
}

# =============================================================================
# MAIN SCRIPT - No need to edit below
# =============================================================================

def check_gpu():
    """Check GPU availability."""
    import torch
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Running on CPU will be very slow.")
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        time.sleep(5)
        return 'cpu'

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    return 'cuda'


def main():
    print("=" * 70)
    print("NEGOTIATION INTERPRETABILITY EVALUATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Check GPU
    device = check_gpu()
    if CONFIG['device'] == 'cuda' and device == 'cpu':
        CONFIG['device'] = 'cpu'

    # Import after GPU check (TransformerLens import is slow)
    print("Loading TransformerLens (this may take a minute)...")
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        InterpretabilityRunner
    )

    # Create runner
    print(f"\nLoading model: {CONFIG['model_name']}")
    runner = InterpretabilityRunner(
        model_name=CONFIG['model_name'],
        device=CONFIG['device'],
        layers_to_capture=CONFIG['layers_to_capture'],
    )

    # Run study
    print(f"\nRunning {CONFIG['num_trials']} trials...")
    print(f"  Scenario: {CONFIG['scenario']}")
    print(f"  Agent modules: {CONFIG['agent_modules']}")
    print(f"  GM modules: {CONFIG['gm_modules']}")
    print(f"  Max rounds: {CONFIG['max_rounds']}")
    print("-" * 70)

    start_time = time.time()

    results = runner.run_study(
        scenario=CONFIG['scenario'],
        agent_modules=CONFIG['agent_modules'],
        gm_modules=CONFIG['gm_modules'],
        num_trials=CONFIG['num_trials'],
        max_rounds=CONFIG['max_rounds'],
        use_gm=CONFIG['use_gm'],
    )

    elapsed = time.time() - start_time

    # Print summary
    runner.print_summary(results)

    # Save dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = CONFIG['output_file'].format(timestamp=timestamp)
    runner.save_dataset(output_file)

    # Final stats
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Output file: {output_file}")
    print(f"Samples collected: {len(results.activation_samples)}")
    print()
    print("To download your file:")
    print(f"  runpodctl send {output_file}")
    print("  # or use SCP/SFTP from your local machine")


if __name__ == '__main__':
    main()
