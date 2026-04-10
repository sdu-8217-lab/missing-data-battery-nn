"""Experiment runner implementing the 9-Level Architecture."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from battery_soh import Seed, MissingRate
from battery_soh.core.constants import set_seed, XJTU_BATCHES, MISSING_MODES, IMPUTATION_METHODS
from battery_soh.data import XJTULoader, BatteryWiseSplit
from battery_soh.models import create_model, count_parameters
from battery_soh.training import LightningTrainer, TrainingConfig
from battery_soh.evaluation import Evaluator


@dataclass
class ExperimentConfig:
    """L1-L6 configuration (Above the Divide - Training)."""
    seed: int
    batch: str
    model_type: str  # mlp, lstm, cnn
    use_mim: bool
    train_missing_rate: float = 0.0  # For MIM training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass  
class EvalConfig:
    """L7-L9 configuration (Below the Divide - Testing)."""
    missing_mode: str
    missing_rate: float
    imputation_method: str


@dataclass
class ExperimentResult:
    """Complete experiment result (L1-L9)."""
    # L1-L6
    seed: int
    batch: str
    model_type: str
    use_mim: bool
    train_missing_rate: float
    # L7-L9
    test_missing_mode: str
    test_missing_rate: float
    imputation_method: str
    # Metrics
    mae: float
    rmse: float
    r2: float
    # Model info
    n_params: int
    training_epochs: int
    best_val_loss: float


class ExperimentRunner:
    """Run experiments following the 9-Level Architecture."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run(
        self,
        config: ExperimentConfig,
        eval_config: Optional[EvalConfig] = None,
        verbose: bool = True
    ) -> ExperimentResult:
        """Run a single experiment (L1-L6 train + L7-L9 evaluate).
        
        Args:
            config: L1-L6 training configuration
            eval_config: L7-L9 evaluation configuration. If None, uses test MR=0.0
            verbose: Print progress
            
        Returns:
            ExperimentResult with all metrics
        """
        # Set seed (L1)
        set_seed(Seed(config.seed))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: seed={config.seed}, batch={config.batch}, "
                  f"model={config.model_type}, MIM={config.use_mim}")
            print(f"{'='*60}")
        
        # Load data (L2-L3)
        loader = XJTULoader()
        X, y, battery_ids = loader.load_batch(config.batch)
        
        # Split data
        splitter = BatteryWiseSplit(seed=Seed(config.seed))
        train, val, test = splitter.split(X, y, battery_ids)
        
        if verbose:
            print(f"Data: {len(train)} train, {len(val)} val, {len(test)} test samples")
        
        # Create model (L4-L5)
        model = create_model(config.model_type, use_mim=config.use_mim)
        n_params = count_parameters(model)
        
        if verbose:
            print(f"Model: {config.model_type}, {n_params:,} params, MIM={config.use_mim}")
        
        # Train (L6 - if MIM with missing data)
        train_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=30
        )
        trainer = LightningTrainer(train_config)
        
        # Apply training missing rate if MIM
        train_X, train_y = train.X, train.y
        if config.use_mim and config.train_missing_rate > 0:
            from battery_soh.missing.generators import MCARGenerator
            gen = MCARGenerator()
            train_X, _ = gen.generate(train_X, MissingRate(config.train_missing_rate), Seed(config.seed))
        
        result = trainer.fit(model, (train_X, train_y), (val.X, val.y))
        
        if verbose:
            print(f"Training complete: {result.total_epochs} epochs, "
                  f"best_val_loss={result.best_val_loss:.4f}")
        
        # Evaluate (L7-L9)
        if eval_config is None:
            eval_config = EvalConfig("MCAR", 0.0, "mean")
        
        evaluator = Evaluator(
            missing_mode=eval_config.missing_mode,
            missing_rate=eval_config.missing_rate,
            imputation_method=eval_config.imputation_method,
            use_mim=config.use_mim
        )
        
        metrics = evaluator.evaluate(model, test.X, test.y, Seed(config.seed))
        
        if verbose:
            print(f"Test: MAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}, R²={metrics.r2:.4f}")
        
        return ExperimentResult(
            seed=config.seed,
            batch=config.batch,
            model_type=config.model_type,
            use_mim=config.use_mim,
            train_missing_rate=config.train_missing_rate,
            test_missing_mode=eval_config.missing_mode,
            test_missing_rate=eval_config.missing_rate,
            imputation_method=eval_config.imputation_method,
            mae=metrics.mae,
            rmse=metrics.rmse,
            r2=metrics.r2,
            n_params=n_params,
            training_epochs=result.total_epochs,
            best_val_loss=result.best_val_loss
        )
    
    def run_batch(
        self,
        seeds: list[int],
        batches: list[str],
        model_types: list[str],
        use_mim_list: list[bool],
        eval_configs: list[EvalConfig],
        output_file: str = "results.csv"
    ) -> Path:
        """Run batch experiments (Cartesian product of L1-L6 × L7-L9).
        
        This implements the efficient evaluation strategy from meta.md:
        - Train models for L1-L6 combinations
        - Evaluate each trained model across all L7-L9 combinations
        """
        results_file = self.output_dir / output_file
        
        # Write header
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'seed', 'batch', 'model_type', 'use_mim', 'train_missing_rate',
                'test_missing_mode', 'test_missing_rate', 'imputation_method',
                'mae', 'rmse', 'r2', 'n_params', 'training_epochs', 'best_val_loss'
            ])
        
        total = len(seeds) * len(batches) * len(model_types) * len(use_mim_list)
        completed = 0
        
        # L1-L6: Training combinations
        for seed in seeds:
            for batch in batches:
                for model_type in model_types:
                    for use_mim in use_mim_list:
                        completed += 1
                        print(f"\n[{completed}/{total}] Training L1-L6 configuration...")
                        
                        # Train once
                        train_config = ExperimentConfig(
                            seed=seed,
                            batch=batch,
                            model_type=model_type,
                            use_mim=use_mim
                        )
                        
                        try:
                            model_result = self.run(train_config, verbose=False)
                            
                            # L7-L9: Evaluate trained model across all test conditions
                            for eval_cfg in eval_configs:
                                result = self.run(train_config, eval_cfg, verbose=False)
                                
                                # Append to CSV
                                with open(results_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        result.seed, result.batch, result.model_type,
                                        result.use_mim, result.train_missing_rate,
                                        result.test_missing_mode, result.test_missing_rate,
                                        result.imputation_method, result.mae, result.rmse,
                                        result.r2, result.n_params, result.training_epochs,
                                        result.best_val_loss
                                    ])
                                
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
        
        print(f"\n{'='*60}")
        print(f"Batch complete! Results saved to: {results_file}")
        print(f"Total experiments: {completed * len(eval_configs)}")
        print(f"{'='*60}")
        
        return results_file


def run_experiment(
    seed: int = 42,
    batch: str = "2C",
    model_type: str = "mlp",
    use_mim: bool = False,
    test_missing_mode: str = "MCAR",
    test_missing_rate: float = 0.3,
    imputation_method: str = "mean",
    epochs: int = 200
) -> ExperimentResult:
    """Quick function to run a single experiment.
    
    Example:
        >>> result = run_experiment(seed=42, model_type="mlp", use_mim=True)
        >>> print(f"MAE: {result.mae:.4f}")
    """
    runner = ExperimentRunner()
    config = ExperimentConfig(
        seed=seed,
        batch=batch,
        model_type=model_type,
        use_mim=use_mim,
        epochs=epochs
    )
    eval_config = EvalConfig(
        missing_mode=test_missing_mode,
        missing_rate=test_missing_rate,
        imputation_method=imputation_method
    )
    return runner.run(config, eval_config)
