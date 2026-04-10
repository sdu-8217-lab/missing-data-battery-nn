#!/usr/bin/env python3
"""Command-line interface for running experiments."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from battery_soh.experiments import run_experiment, ExperimentRunner, ExperimentConfig, EvalConfig
from battery_soh.core.constants import XJTU_BATCHES, MODEL_TYPES, MISSING_MODES, IMPUTATION_METHODS


def main():
    parser = argparse.ArgumentParser(
        description="Run Battery SOH experiments with missing data"
    )
    
    # L1-L6 arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed (L1)")
    parser.add_argument("--batch", type=str, default="2C", 
                       choices=XJTU_BATCHES, help="Battery batch (L3)")
    parser.add_argument("--model", type=str, default="mlp",
                       choices=MODEL_TYPES, help="Model type (L4)")
    parser.add_argument("--mim", action="store_true", help="Use MIM (L5)")
    parser.add_argument("--train-mr", type=float, default=0.0,
                       help="Training missing rate for MIM (L6)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    
    # L7-L9 arguments
    parser.add_argument("--missing-mode", type=str, default="MCAR",
                       choices=MISSING_MODES, help="Missing mode (L7)")
    parser.add_argument("--missing-rate", type=float, default=0.3,
                       help="Test missing rate (L8)")
    parser.add_argument("--imputation", type=str, default="mean",
                       choices=IMPUTATION_METHODS, help="Imputation method (L9)")
    
    # Output
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Battery SOH Experiment Runner")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  L1 (Seed): {args.seed}")
    print(f"  L2 (Dataset): XJTU")
    print(f"  L3 (Batch): {args.batch}")
    print(f"  L4 (Model): {args.model}")
    print(f"  L5 (MIM): {args.mim}")
    print(f"  L6 (Train MR): {args.train_mr}")
    print(f"  L7 (Missing Mode): {args.missing_mode}")
    print(f"  L8 (Test MR): {args.missing_rate}")
    print(f"  L9 (Imputation): {args.imputation}")
    print(f"  Epochs: {args.epochs}")
    print()
    
    # Run experiment
    result = run_experiment(
        seed=args.seed,
        batch=args.batch,
        model_type=args.model,
        use_mim=args.mim,
        test_missing_mode=args.missing_mode,
        test_missing_rate=args.missing_rate,
        imputation_method=args.imputation,
        epochs=args.epochs
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  MAE:  {result.mae:.4f}")
    print(f"  RMSE: {result.rmse:.4f}")
    print(f"  R²:   {result.r2:.4f}")
    print(f"  Training epochs: {result.training_epochs}")
    print(f"  Model parameters: {result.n_params:,}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
