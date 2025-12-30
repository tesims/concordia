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


def run_scenario_on_gpu(scenario: str, gpu_id: int, quick: bool = False, mode: str = "emergent",
                        trials: int = None, model: str = None, script_dir: str = None):
    """Run a scenario on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Use the directory where this script is located
    if script_dir is None:
        script_dir = Path(__file__).parent.resolve()

    run_experiment_path = script_dir / "run_experiment.py"
    output_dir = script_dir / f"outputs_{scenario}"

    cmd = [
        sys.executable,
        str(run_experiment_path),
        "--tier", "2",
        "--scenario", scenario,
        "--output-dir", str(output_dir),
        "--mode", mode,
    ]

    if quick:
        cmd.append("--quick")

    if trials:
        cmd.extend(["--trials", str(trials)])

    if model:
        cmd.extend(["--model", model])

    model_short = model.split("/")[-1] if model else "default"
    print(f"[GPU {gpu_id}] Starting {scenario} (mode={mode}, trials={trials or 50}, model={model_short})...")
    return subprocess.Popen(cmd, env=env, cwd=str(script_dir))


def merge_results(scenarios: list, output_file: str = "merged_activations.pt", script_dir: Path = None):
    """Merge results from all scenarios."""
    print("\nMerging results...")

    if script_dir is None:
        script_dir = Path(__file__).parent.resolve()

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
        output_dir = script_dir / f"outputs_{scenario}"

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

    output_path = script_dir / output_file
    torch.save(merged, output_path)
    print(f"\nMerged results saved to: {output_path}")
    print(f"Total samples: {len(all_labels['gm_labels'])}")

    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test mode (2 scenarios)")
    parser.add_argument("--mode", type=str, default="emergent",
                        choices=["instructed", "emergent"],
                        help="Experiment mode: 'instructed' or 'emergent' (default: emergent)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Trials per condition (default: 50). Recommended: 50-100")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it",
                        choices=["google/gemma-2-9b-it", "google/gemma-2-27b-it", "google/gemma-2-2b-it"],
                        help="Model to use (default: gemma-2-9b-it)")
    args = parser.parse_args()

    # All 6 scenarios
    if args.quick:
        scenarios = ["ultimatum_bluff", "hidden_value"]
    else:
        scenarios = [
            "ultimatum_bluff", "capability_bluff", "hidden_value",
            "info_withholding", "promise_break", "alliance_betrayal"
        ]

    script_dir = Path(__file__).parent.resolve()

    mode_desc = "EMERGENT (incentive-based)" if args.mode == "emergent" else "INSTRUCTED (explicit)"
    model_short = args.model.split("/")[-1]

    print("=" * 60)
    print("PARALLEL DECEPTION PROBING EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {mode_desc}")
    print(f"Model: {model_short}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Trials per condition: {args.trials}")
    print(f"Total trials: {args.trials * 2 * len(scenarios)}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Script directory: {script_dir}")

    # Check GPUs
    gpu_count = get_gpu_count()
    print(f"\nDetected {gpu_count} GPU(s)")

    if gpu_count < len(scenarios):
        print(f"\nWARNING: Only {gpu_count} GPU(s) available for {len(scenarios)} scenarios.")
        print("Will run scenarios with GPU cycling.")

        # Run with GPU cycling
        processes = []
        for i, scenario in enumerate(scenarios):
            gpu_id = i % max(1, gpu_count)
            # Wait for previous job on same GPU to finish
            for prev_scenario, prev_proc in processes:
                if processes.index((prev_scenario, prev_proc)) % max(1, gpu_count) == gpu_id:
                    prev_proc.wait()
            proc = run_scenario_on_gpu(scenario, gpu_id, args.quick, args.mode, args.trials, args.model, script_dir)
            processes.append((scenario, proc))
            time.sleep(1)

        # Wait for remaining
        for scenario, proc in processes:
            proc.wait()
            print(f"  {scenario} completed with return code {proc.returncode}")
    else:
        # Run all scenarios in parallel
        print(f"\nRunning {len(scenarios)} scenarios in parallel...")

        processes = []
        for i, scenario in enumerate(scenarios):
            proc = run_scenario_on_gpu(scenario, i, args.quick, args.mode, args.trials, args.model, script_dir)
            processes.append((scenario, proc))
            time.sleep(2)  # Small delay between starts

        # Wait for all to complete
        print("\nWaiting for all scenarios to complete...")
        for scenario, proc in processes:
            proc.wait()
            print(f"  {scenario} completed with return code {proc.returncode}")

    # Merge results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = merge_results(scenarios, f"merged_activations_{timestamp}.pt", script_dir)

    print("\n" + "=" * 60)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 60)
    print(f"\nNext step: Train probes on merged results:")
    print(f"  python train_probes.py --data {merged_file} --plot")


if __name__ == "__main__":
    main()
