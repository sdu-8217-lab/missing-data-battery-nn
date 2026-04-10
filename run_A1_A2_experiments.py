#!/usr/bin/env python3
"""
A1 & A2 实验脚本

A1: MIM 增益来源研究
  G1 (MIM):      z = [x̂ | m]           - 标准MIM
  G2 (Random):   z = [x̂ | r], r~Bernoulli(0.4) - 随机指示器
  G3 (Shuffled): z = [x̂ | m_shuffled]  - 打乱缺失指示
  G4 (Copy):     z = [x̂ | x̂]          - 复制插补值

A2: 块状缺失鲁棒性
  B1 (MCAR):  随机均匀缺失
  B2 (Block): 连续5-cycle缺失 (模拟传感器故障)

Usage:
    # A1 - G1 (MIM baseline)
    python run_A1_A2_experiments.py --group G1 --seed 42
    
    # A1 - G2 (Random indicator)
    python run_A1_A2_experiments.py --group G2 --seed 42
    
    # A2 - B2 (Block missing)
    python run_A1_A2_experiments.py --group B2 --seed 42
    
    # Run all groups with 3 seeds
    python run_A1_A2_experiments.py --run-all
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from battery_soh import Seed
from battery_soh.core.constants import set_seed
from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model, count_parameters
from battery_soh.training import LightningTrainer, TrainingConfig
from battery_soh.evaluation import Evaluator
from battery_soh.evaluation.metrics import Metrics


# Experiment configuration for each group
GROUP_CONFIGS = {
    # A1: MIM ablation study
    "G1": {
        "name": "MIM (Standard)",
        "description": "z = [x̂ | m] - Standard MIM",
        "missing_mode": "MCAR",
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "standard",
    },
    "G2": {
        "name": "Random Indicator",
        "description": "z = [x̂ | r], r~Bernoulli(0.4) - Test regularization hypothesis",
        "missing_mode": "MCAR",
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "random",
    },
    "G3": {
        "name": "Shuffled Mask",
        "description": "z = [x̂ | m_shuffled] - Test information hypothesis",
        "missing_mode": "MCAR",
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "shuffled",
    },
    "G4": {
        "name": "Copy Features",
        "description": "z = [x̂ | x̂] - Test dimensionality hypothesis",
        "missing_mode": "MCAR",
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "copy",
    },
    # A2: Block missing study
    "B1": {
        "name": "MCAR (Baseline)",
        "description": "Random uniform missing - Baseline for A2",
        "missing_mode": "MCAR",
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "standard",
    },
    "B2": {
        "name": "Block Missing",
        "description": "Continuous 5-cycle missing - Simulate sensor failure",
        "missing_mode": "block",
        "block_size": 5,
        "missing_rate": 0.4,
        "use_mim": True,
        "mim_variant": "standard",
    },
}


def run_single_experiment(
    group: str,
    seed: int,
    batch: str = "2C",
    model_type: str = "mlp",
    epochs: int = 200,
    output_dir: str = "results/A1_A2"
) -> dict:
    """Run a single experiment for one group with one seed."""
    
    config = GROUP_CONFIGS[group]
    set_seed(Seed(seed))
    
    print(f"\n{'='*60}")
    print(f"Group {group}: {config['name']}")
    print(f"  {config['description']}")
    print(f"  Seed: {seed}, Batch: {batch}, Model: {model_type}")
    print(f"{'='*60}")
    
    # Load data
    loader = XJTULoader()
    X, y, battery_ids = loader.load_batch(batch)
    
    # Split
    splitter = BatteryWiseSplit(seed=Seed(seed))
    train, val, test = splitter.split(X, y, battery_ids)
    
    print(f"Data: {len(train)} train, {len(val)} val, {len(test)} test samples")
    
    # Create model
    model = create_model(model_type, use_mim=True)
    n_params = count_parameters(model)
    print(f"Model: {model_type}, {n_params:,} params")
    
    # Prepare training data with MIM mask (all zeros since training data is complete)
    train_X = train.X
    val_X = val.X
    if True:  # Always add MIM mask for A1/A2 experiments
        # Add zero mask (no missing in training data)
        train_mask = np.zeros_like(train_X, dtype=np.float32)
        val_mask = np.zeros_like(val_X, dtype=np.float32)
        train_X = np.concatenate([train_X, train_mask], axis=1)
        val_X = np.concatenate([val_X, val_mask], axis=1)
    
    # Train
    train_config = TrainingConfig(epochs=epochs, patience=30)
    trainer = LightningTrainer(train_config)
    
    result = trainer.fit(
        model,
        train_data=(train_X, train.y),
        val_data=(val_X, val.y)
    )
    
    print(f"Training complete: {result.total_epochs} epochs, best_val_loss={result.best_val_loss:.4f}")
    
    # Evaluate with group-specific configuration
    evaluator_kwargs = {
        "missing_mode": config.get("missing_mode", "MCAR"),
        "missing_rate": config.get("missing_rate", 0.4),
        "imputation_method": "mean",
        "use_mim": True,
        "mim_variant": config.get("mim_variant", "standard"),
    }
    
    # Add block_size if present
    if "block_size" in config:
        evaluator_kwargs["block_size"] = config["block_size"]
    
    evaluator = Evaluator(**evaluator_kwargs)
    metrics = evaluator.evaluate(model, test.X, test.y, Seed(seed))
    
    print(f"Test Results: MAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}, R²={metrics.r2:.4f}")
    
    # Save result
    result_dict = {
        "group": group,
        "group_name": config["name"],
        "seed": seed,
        "batch": batch,
        "model_type": model_type,
        "mim_variant": config.get("mim_variant", "N/A"),
        "missing_mode": config.get("missing_mode", "MCAR"),
        "missing_rate": config.get("missing_rate", 0.4),
        "block_size": config.get("block_size", "N/A"),
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "epochs": result.total_epochs,
        "best_val_loss": result.best_val_loss,
        "n_params": n_params,
    }
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "A1_A2_results.csv"
    
    file_exists = csv_file.exists()
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)
    
    print(f"Result saved to {csv_file}")
    
    return result_dict


def run_all_groups(seeds: list[int], **kwargs):
    """Run all 6 groups with specified seeds."""
    groups = ["G1", "G2", "G3", "G4", "B1", "B2"]
    
    total = len(groups) * len(seeds)
    completed = 0
    
    print(f"\n{'#'*60}")
    print(f"# Running A1 & A2 Experiments")
    print(f"# Groups: {groups}")
    print(f"# Seeds: {seeds}")
    print(f"# Total experiments: {total}")
    print(f"{'#'*60}\n")
    
    for group in groups:
        for seed in seeds:
            completed += 1
            print(f"\n[{completed}/{total}] ", end="")
            try:
                run_single_experiment(group=group, seed=seed, **kwargs)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: results/A1_A2/A1_A2_results.csv")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run A1 & A2 experiments")
    
    parser.add_argument("--group", type=str, choices=list(GROUP_CONFIGS.keys()),
                       help="Experiment group to run")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--batch", type=str, default="2C",
                       help="Battery batch")
    parser.add_argument("--model", type=str, default="mlp",
                       help="Model type")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Training epochs")
    parser.add_argument("--run-all", action="store_true",
                       help="Run all groups with multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Seeds for --run-all")
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_groups(
            seeds=args.seeds,
            batch=args.batch,
            model_type=args.model,
            epochs=args.epochs
        )
    elif args.group:
        run_single_experiment(
            group=args.group,
            seed=args.seed,
            batch=args.batch,
            model_type=args.model,
            epochs=args.epochs
        )
    else:
        parser.print_help()
        print("\n\nExperiment Groups:")
        for g, cfg in GROUP_CONFIGS.items():
            print(f"  {g}: {cfg['name']}")
            print(f"      {cfg['description']}")


if __name__ == "__main__":
    main()
