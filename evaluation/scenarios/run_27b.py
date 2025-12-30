#!/usr/bin/env python3
"""
Wrapper for running Gemma 27B experiments with memory safety.

Usage:
    python run_27b.py                    # Default: 50 trials per condition
    python run_27b.py --trials 100       # Publication quality
    python run_27b.py --quick            # Quick test (5 trials)

Requirements:
    - GPU with >= 70GB VRAM (A100-80GB or H100)
    - ~100GB disk space for model + activations
"""

import os
import sys
import argparse
from pathlib import Path


def check_requirements():
    """Check GPU memory and disk space before running."""
    print("=" * 60)
    print("GEMMA 27B EXPERIMENT - PRE-FLIGHT CHECK")
    print("=" * 60)

    # Check PyTorch and CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return False

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")
        print(f"GPU memory: {gpu_mem:.1f} GB")

        if gpu_mem < 70:
            print("\nWARNING: 27B model needs ~70GB VRAM with activation caching")
            print("  Your GPU has only {:.1f}GB".format(gpu_mem))
            print("  Options:")
            print("    1. Use --trials 25 to reduce memory")
            print("    2. Switch to google/gemma-2-9b-it")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                return False
        else:
            print("  Memory check PASSED")

    except ImportError:
        print("ERROR: PyTorch not installed")
        return False

    # Check disk space
    script_dir = Path(__file__).parent.resolve()
    import shutil
    total, used, free = shutil.disk_usage(script_dir)
    free_gb = free / 1e9
    print(f"\nDisk space available: {free_gb:.1f} GB")

    if free_gb < 100:
        print("WARNING: Recommend 100GB+ free for model + activations")
        print(f"  You have {free_gb:.1f}GB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma 27B deception probing experiment"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Trials per condition (default: 50)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode (5 trials, 2 scenarios)"
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip pre-flight checks"
    )
    parser.add_argument(
        "--mode", type=str, default="emergent",
        choices=["instructed", "emergent"],
        help="Experiment mode (default: emergent)"
    )

    args = parser.parse_args()

    # Pre-flight checks
    if not args.skip_checks:
        if not check_requirements():
            print("\nPre-flight checks failed. Use --skip-checks to override.")
            sys.exit(1)

    # Set memory optimization environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Build arguments for run_parallel.py
    run_args = [
        "--model", "google/gemma-2-27b-it",
        "--mode", args.mode,
    ]

    if args.quick:
        run_args.append("--quick")
    else:
        run_args.extend(["--trials", str(args.trials)])

    print("\n" + "=" * 60)
    print("STARTING EXPERIMENT")
    print("=" * 60)
    print(f"Model: google/gemma-2-27b-it")
    print(f"Mode: {args.mode}")
    print(f"Trials: {5 if args.quick else args.trials} per condition")
    print(f"Total: {(5 if args.quick else args.trials) * 2 * (2 if args.quick else 6)} samples")
    print("=" * 60 + "\n")

    # Import and run
    sys.argv = [sys.argv[0]] + run_args
    from run_parallel import main as run_main
    run_main()


if __name__ == "__main__":
    main()
