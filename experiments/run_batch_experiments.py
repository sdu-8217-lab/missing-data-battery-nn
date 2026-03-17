#!/usr/bin/env python
"""
Batch experiment runner for XJTU datasets.

Runs experiments across all 6 XJTU batches with consistent timestamp.

Usage:
    # Run all batches with all configurations
    python experiments/run_batch_experiments.py --mode mar --method mim
    
    # Run specific batches
    python experiments/run_batch_experiments.py --batches 2C 3C --mode mcar
    
    # Dry run (show what would be executed)
    python experiments/run_batch_experiments.py --dry-run
"""

import sys
sys.path.insert(0, '.')

import argparse
import subprocess
from pathlib import Path
from typing import List

from src.utils.paths import get_timestamp


# XJTU batch configurations
XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]

# Model types
MODELS = ["mlp", "lstm", "cnn"]

# Missing modes
MISSING_MODES = ["mcar", "mar", "mnar"]

# Methods
METHODS = ["baseline", "mim"]


def run_experiment(
    batch_id: str,
    model: str,
    mode: str,
    method: str,
    missing_rate: float,
    seeds: List[int],
    timestamp: str,
    dry_run: bool = False
) -> bool:
    """Run a single experiment configuration."""
    
    # Map config names
    config_map = {
        ("mar", "baseline"): "youth_mar_baseline",
        ("mar", "mim"): "youth_mar_mim",
        ("mcar", "baseline"): "youth_mcar_baseline",
        ("mcar", "mim"): "youth_mcar_mim",
    }
    
    config_name = config_map.get((mode, method))
    if config_name is None:
        print(f"⚠️  No config for mode={mode}, method={method}, skipping...")
        return True
    
    cmd = [
        "python", "experiments/run_experiment.py",
        f"experiment={config_name}",
        f"data.batch_id={batch_id}",
        f"model.type={model}",
        f"missing.rate_eval={missing_rate}",
        f"training.seeds={seeds}",
        f"timestamp={timestamp}",
    ]
    
    print(f"\n{'='*60}")
    print(f"Batch: {batch_id} | Model: {model} | Mode: {mode} | Method: {method} | MR: {missing_rate}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    if dry_run:
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--batches', nargs='+', choices=XJTU_BATCHES + ['all'],
                       default=['all'], help='Which batches to run')
    parser.add_argument('--models', nargs='+', choices=MODELS + ['all'],
                       default=['all'], help='Which models to run')
    parser.add_argument('--mode', choices=MISSING_MODES, required=True,
                       help='Missing data mode')
    parser.add_argument('--method', choices=METHODS, required=True,
                       help='Imputation method')
    parser.add_argument('--missing-rates', nargs='+', type=float,
                       default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Missing rates to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds')
    parser.add_argument('--timestamp', help='Custom timestamp (default: auto-generated)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing')
    
    args = parser.parse_args()
    
    # Determine batches and models
    batches = XJTU_BATCHES if 'all' in args.batches else args.batches
    models = MODELS if 'all' in args.models else args.models
    
    # Generate timestamp for this batch run
    timestamp = args.timestamp or get_timestamp()
    
    print(f"\n{'#'*70}")
    print(f"# Batch Experiment Run")
    print(f"# Timestamp: {timestamp}")
    print(f"# Batches: {batches}")
    print(f"# Models: {models}")
    print(f"# Mode: {args.mode}")
    print(f"# Method: {args.method}")
    print(f"# Missing Rates: {args.missing_rates}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Dry Run: {args.dry_run}")
    print(f"{'#'*70}\n")
    
    # Calculate total experiments
    total = len(batches) * len(models) * len(args.missing_rates)
    completed = 0
    failed = 0
    
    # Run experiments
    for batch_id in batches:
        for model in models:
            for mr in args.missing_rates:
                success = run_experiment(
                    batch_id=batch_id,
                    model=model,
                    mode=args.mode,
                    method=args.method,
                    missing_rate=mr,
                    seeds=args.seeds,
                    timestamp=timestamp,
                    dry_run=args.dry_run
                )
                
                if success:
                    completed += 1
                else:
                    failed += 1
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# Run Complete")
    print(f"# Total: {total}")
    print(f"# Completed: {completed}")
    print(f"# Failed: {failed}")
    print(f"# Timestamp: {timestamp}")
    print(f"{'#'*70}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
