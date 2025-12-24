#!/usr/bin/env python3
"""
Run all 3 scenarios in parallel on a multi-GPU pod.

Usage:
    python run_parallel.py          # Uses all available GPUs
    python run_parallel.py --quick  # Quick test mode

Requires: 3 GPUs (each scenario runs on 1 GPU)
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import torch


def get_gpu_count():
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def run_scenario_on_gpu(scenario: str, gpu_id: int, quick: bool = False):
    """Run a scenario on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "run_experiment.py",
        "--tier", "2",
        "--scenario", scenario,
        "--output-dir", f"./outputs_{scenario}",
    ]

    if quick:
        cmd.append("--quick")

    print(f"[GPU {gpu_id}] Starting {scenario}...")
    return subprocess.Popen(cmd, env=env)


def merge_results(scenarios: list, output_file: str = "merged_activations.pt"):
    """Merge results from all scenarios."""
    print("\nMerging results...")

    all_activations = {}
    all_labels = {
        "gm_labels": [],
        "agent_labels": [],
        "is_deceptive": [],
        "scenario": [],
        "condition": [],
        "trial_id": [],
    }

    for scenario in scenarios:
        output_dir = Path(f"./outputs_{scenario}")

        # Find the run directory
        run_dirs = list(output_dir.glob("run_*"))
        if not run_dirs:
            print(f"  WARNING: No results found for {scenario}")
            continue

        latest_run = sorted(run_dirs)[-1]
        activation_file = latest_run / "activations.pt"

        if not activation_file.exists():
            print(f"  WARNING: No activations.pt for {scenario}")
            continue

        print(f"  Loading {scenario} from {activation_file}")
        data = torch.load(activation_file)

        # Merge activations
        for layer, acts in data["activations"].items():
            if layer not in all_activations:
                all_activations[layer] = []
            all_activations[layer].append(acts)

        # Merge labels
        for key in all_labels:
            if key in data["labels"]:
                all_labels[key].extend(data["labels"][key])

    # Stack activations
    merged_activations = {}
    for layer, acts_list in all_activations.items():
        merged_activations[layer] = torch.cat(acts_list, dim=0)

    # Save merged
    merged = {
        "activations": merged_activations,
        "labels": all_labels,
        "config": {"scenarios": scenarios, "merged": True},
    }

    torch.save(merged, output_file)
    print(f"\nMerged results saved to: {output_file}")
    print(f"Total samples: {len(all_labels['gm_labels'])}")

    return output_file


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    scenarios = ["ultimatum_bluff", "hidden_value", "promise_break"]

    print("=" * 60)
    print("PARALLEL DECEPTION PROBING EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Check GPUs
    gpu_count = get_gpu_count()
    print(f"\nDetected {gpu_count} GPU(s)")

    if gpu_count < 3:
        print(f"\nWARNING: Only {gpu_count} GPU(s) available.")
        print("Will run scenarios sequentially on available GPUs.")

        # Run sequentially if not enough GPUs
        for i, scenario in enumerate(scenarios):
            gpu_id = i % max(1, gpu_count)
            proc = run_scenario_on_gpu(scenario, gpu_id, args.quick)
            proc.wait()  # Wait for completion before next
    else:
        # Run all 3 in parallel
        print("\nRunning 3 scenarios in parallel (1 GPU each)...")

        processes = []
        for i, scenario in enumerate(scenarios):
            proc = run_scenario_on_gpu(scenario, i, args.quick)
            processes.append((scenario, proc))
            time.sleep(2)  # Small delay between starts

        # Wait for all to complete
        print("\nWaiting for all scenarios to complete...")
        for scenario, proc in processes:
            proc.wait()
            print(f"  {scenario} completed with return code {proc.returncode}")

    # Merge results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = merge_results(scenarios, f"merged_activations_{timestamp}.pt")

    print("\n" + "=" * 60)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Train probes on merged results:")
    print(f"  python train_probes.py --data {merged_file} --plot")


if __name__ == "__main__":
    main()
