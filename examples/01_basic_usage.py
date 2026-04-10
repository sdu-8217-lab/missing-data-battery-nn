"""
Basic usage example for Battery SOH Framework.

This example demonstrates:
1. Loading battery data
2. Creating and training a model
3. Evaluating under missing data scenarios
"""

import numpy as np

from battery_soh import Seed
from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model, count_parameters
from battery_soh.training import LightningTrainer, TrainingConfig
from battery_soh.evaluation import Evaluator
from battery_soh.core.constants import set_seed


def main():
    """Run basic example."""
    # Set random seed for reproducibility
    set_seed(Seed(42))
    
    print("=" * 60)
    print("Battery SOH Framework - Basic Usage Example")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    loader = XJTULoader()
    
    # Check available batches
    available = loader.list_available_batches()
    print(f"Available batches: {available}")
    
    if not available:
        print("No data found. Please download XJTU dataset.")
        print("Expected location: data/XJTU data/")
        return
    
    batch_id = available[0]
    print(f"Loading batch: {batch_id}")
    
    try:
        X, y, battery_ids = loader.load_batch(batch_id)
        print(f"Loaded {len(X)} samples from {len(np.unique(battery_ids))} batteries")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Split data
    print("\n[2/4] Splitting data...")
    splitter = BatteryWiseSplit(seed=Seed(42))
    train_split, val_split, test_split = splitter.split(X, y, battery_ids)
    
    print(f"Train: {len(train_split)} samples, {train_split.n_batteries} batteries")
    print(f"Val:   {len(val_split)} samples, {val_split.n_batteries} batteries")
    print(f"Test:  {len(test_split)} samples, {test_split.n_batteries} batteries")
    
    # 3. Create and train model (without MIM)
    print("\n[3/4] Training model (without MIM)...")
    model = create_model("mlp", use_mim=False)
    n_params = count_parameters(model)
    print(f"Model: MLP with {n_params:,} parameters")
    
    # Train
    config = TrainingConfig(
        epochs=10,  # Small for demo
        batch_size=64,
        patience=5
    )
    trainer = LightningTrainer(config)
    
    result = trainer.fit(
        model,
        train_data=(train_split.X, train_split.y),
        val_data=(val_split.X, val_split.y)
    )
    
    print(f"Training complete!")
    print(f"  Best val loss: {result.best_val_loss:.4f}")
    print(f"  Epochs: {result.total_epochs}")
    
    # 4. Evaluate under missing data
    print("\n[4/4] Evaluating under missing data (MCAR, 30% missing)...")
    evaluator = Evaluator(
        missing_mode="MCAR",
        missing_rate=0.3,
        imputation_method="mean",
        use_mim=False
    )
    
    metrics = evaluator.evaluate(
        model=model,
        X=test_split.X,
        y=test_split.y,
        seed=Seed(42)
    )
    
    print(f"Test metrics:")
    print(f"  MAE:  {metrics.mae:.4f}")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  R²:   {metrics.r2:.4f}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
