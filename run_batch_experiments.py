#!/usr/bin/env python3
"""Run batch experiments following 9-Level Architecture."""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from battery_soh.experiments import ExperimentRunner, ExperimentConfig, EvalConfig
from battery_soh.core.constants import XJTU_BATCHES, MODEL_TYPES, MISSING_MODES, IMPUTATION_METHODS


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    
    # Experiment ranges
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                       help="Random seeds to run")
    parser.add_argument("--batches", type=str, nargs="+", default=["2C"],
                       choices=XJTU_BATCHES, help="Battery batches")
    parser.add_argument("--models", type=str, nargs="+", default=["mlp"],
                       choices=MODEL_TYPES, help="Model types")
    parser.add_argument("--mim", action="store_true", help="Run with MIM")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no MIM)")
    
    # Evaluation conditions (L7-L9)
    parser.add_argument("--missing-modes", type=str, nargs="+", default=["MCAR"],
                       choices=MISSING_MODES, help="Missing modes to test")
    parser.add_argument("--missing-rates", type=float, nargs="+", 
                       default=[0.0, 0.1, 0.3, 0.5],
                       help="Missing rates to test")
    parser.add_argument("--imputations", type=str, nargs="+", default=["mean"],
                       choices=IMPUTATION_METHODS, help="Imputation methods")
    
    # Other
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--output-file", type=str, default="batch_results.csv",
                       help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Determine MIM configurations
    use_mim_list = []
    if args.baseline:
        use_mim_list.append(False)
    if args.mim:
        use_mim_list.append(True)
    if not use_mim_list:
        use_mim_list = [False]  # Default to baseline
    
    # Build evaluation configs (L7-L9)
    eval_configs = []
    for mode in args.missing_modes:
        for mr in args.missing_rates:
            for imp in args.imputations:
                eval_configs.append(EvalConfig(mode, mr, imp))
    
    print("=" * 60)
    print("Batch Experiment Runner")
    print("=" * 60)
    print(f"\nL1-L6 (Training) combinations:")
    print(f"  Seeds: {args.seeds}")
    print(f"  Batches: {args.batches}")
    print(f"  Models: {args.models}")
    print(f"  MIM: {use_mim_list}")
    print(f"\nL7-L9 (Evaluation) combinations: {len(eval_configs)}")
    for cfg in eval_configs[:5]:
        print(f"  - {cfg.missing_mode}, MR={cfg.missing_rate}, {cfg.imputation_method}")
    if len(eval_configs) > 5:
        print(f"  ... and {len(eval_configs) - 5} more")
    
    total = len(args.seeds) * len(args.batches) * len(args.models) * len(use_mim_list) * len(eval_configs)
    print(f"\nTotal experiments: {total}")
    print("=" * 60)
    
    confirm = input("\nContinue? [Y/n]: ")
    if confirm.lower() not in ('', 'y', 'yes'):
        print("Aborted.")
        return 1
    
    # Run batch
    runner = ExperimentRunner(output_dir=args.output)
    results_file = runner.run_batch(
        seeds=args.seeds,
        batches=args.batches,
        model_types=args.models,
        use_mim_list=use_mim_list,
        eval_configs=eval_configs,
        output_file=args.output_file
    )
    
    print(f"\nResults saved to: {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
