#!/usr/bin/env python
"""
Batch experiment runner for XJTU datasets.

Runs experiments across all 6 XJTU batches with consistent timestamp.

Usage:
    # Run all batches with MAR mode and mean imputation
    python experiments/run_batch_experiments.py --mode mar --method mean
    
    # Run full matrix (all modes, methods, models)
    python experiments/run_batch_experiments.py --full-matrix --seeds 42 123
    
    # Run specific batches
    python experiments/run_batch_experiments.py --batches 2C 3C --mode mcar --method mim
    
    # Dry run (show what would be executed)
    python experiments/run_batch_experiments.py --mode mar --method mean --dry-run
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

# Imputation methods (baseline methods)
IMPUTATION_METHODS = ["mean", "knn", "iterative", "zero"]

# All methods including MIM
ALL_METHODS = IMPUTATION_METHODS + ["mim"]

# Full experiment matrix
FULL_MATRIX = {
    "modes": ["mcar", "mar", "mnar"],
    "methods": ["mean", "knn", "iterative", "zero", "mim"],
    "missing_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


def get_config_name(mode: str, method: str) -> str:
    """Get the config name for a mode-method combination."""
    # Map method names to config names
    if method == "mim":
        # MIM configs
        config_map = {
            "mcar": "youth_mcar_mim",
            "mar": "youth_mar_mim",
            "mnar": "full_test_mnar_mim",
        }
    else:
        # Baseline configs with specific imputation methods
        config_map = {
            ("mcar", "mean"): "youth_mcar_baseline",
            ("mcar", "knn"): "full_test_mcar_knn",
            ("mcar", "iterative"): "full_test_mcar_iterative",
            ("mcar", "zero"): "full_test_mcar_zero",
            ("mar", "mean"): "full_test_mar_mean",
            ("mar", "knn"): "full_test_mar_knn",
            ("mar", "iterative"): "full_test_mar_iterative",
            ("mar", "zero"): "full_test_mar_zero",
            ("mnar", "mean"): "full_test_mnar_mean",
            ("mnar", "knn"): "full_test_mnar_knn",
            ("mnar", "iterative"): "full_test_mnar_iterative",
            ("mnar", "zero"): "full_test_mnar_zero",
        }
    
    if method == "mim":
        return config_map.get(mode)
    else:
        return config_map.get((mode, method))


def run_experiment(
    batch_id: str,
    model: str,
    mode: str,
    method: str,
    missing_rate: float,
    seeds: List[int],
    timestamp: str,
    epochs: int = 100,
    dry_run: bool = False
) -> bool:
    """Run a single experiment configuration."""
    
    config_name = get_config_name(mode, method)
    if config_name is None:
        print(f"⚠️  No config for mode={mode}, method={method}, skipping...")
        return True
    
    cmd = [
        "python", "experiments/run_experiment.py",
        f"--config-name={config_name}",
        f"data.batch_id={batch_id}",
        f"model.type={model}",
        f"missing.rate_eval={missing_rate}",
        f"training.seeds={seeds}",
        f"training.epochs={epochs}",
        f"+timestamp={timestamp}",
    ]
    
    # Override method for baseline configs
    if method != "mim":
        cmd.append(f"method={method}")
    
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


def run_full_matrix(args) -> int:
    """Run full experiment matrix."""
    batches = XJTU_BATCHES if 'all' in args.batches else args.batches
    models = MODELS if 'all' in args.models else args.models
    timestamp = args.timestamp or get_timestamp()
    
    total = len(batches) * len(models) * len(FULL_MATRIX["modes"]) * len(FULL_MATRIX["methods"]) * len(FULL_MATRIX["missing_rates"])
    completed = 0
    failed = 0
    
    print(f"\n{'#'*70}")
    print(f"# Full Matrix Experiment Run")
    print(f"# Timestamp: {timestamp}")
    print(f"# Batches: {batches}")
    print(f"# Models: {models}")
    print(f"# Modes: {FULL_MATRIX['modes']}")
    print(f"# Methods: {FULL_MATRIX['methods']}")
    print(f"# Missing Rates: {FULL_MATRIX['missing_rates']}")
    print(f"# Seeds: {args.seeds}")
    print(f"# Epochs: {args.epochs}")
    print(f"# Total Experiments: {total}")
    print(f"# Dry Run: {args.dry_run}")
    print(f"{'#'*70}\n")
    
    for batch_id in batches:
        for model in models:
            for mode in FULL_MATRIX["modes"]:
                for method in FULL_MATRIX["methods"]:
                    for mr in FULL_MATRIX["missing_rates"]:
                        success = run_experiment(
                            batch_id=batch_id,
                            model=model,
                            mode=mode,
                            method=method,
                            missing_rate=mr,
                            seeds=args.seeds,
                            timestamp=timestamp,
                            epochs=args.epochs,
                            dry_run=args.dry_run
                        )
                        
                        if success:
                            completed += 1
                        else:
                            failed += 1
    
    print(f"\n{'#'*70}")
    print(f"# Run Complete")
    print(f"# Total: {total}")
    print(f"# Completed: {completed}")
    print(f"# Failed: {failed}")
    print(f"# Timestamp: {timestamp}")
    print(f"{'#'*70}")
    
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--full-matrix', action='store_true',
                       help='Run full experiment matrix (all modes, methods)')
    parser.add_argument('--batches', nargs='+', choices=XJTU_BATCHES + ['all'],
                       default=['all'], help='Which batches to run')
    parser.add_argument('--models', nargs='+', choices=MODELS + ['all'],
                       default=['all'], help='Which models to run')
    parser.add_argument('--mode', choices=MISSING_MODES,
                       help='Missing data mode (required if not --full-matrix)')
    parser.add_argument('--method', choices=ALL_METHODS,
                       help='Imputation method (required if not --full-matrix)')
    parser.add_argument('--missing-rates', nargs='+', type=float,
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Missing rates to evaluate (default: 0.0-0.9)')
    parser.add_argument('--seeds', nargs='+', type=int,
                       default=[42, 123, 456, 789, 1011],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--timestamp', help='Custom timestamp (default: auto-generated)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing')
    
    args = parser.parse_args()
    
    if args.full_matrix:
        return run_full_matrix(args)
    
    if not args.mode or not args.method:
        parser.error("--mode and --method are required (or use --full-matrix)")
    
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
    print(f"# Epochs: {args.epochs}")
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
                    epochs=args.epochs,
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
