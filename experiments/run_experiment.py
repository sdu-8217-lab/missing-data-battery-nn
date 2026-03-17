"""
Unified Experiment Runner

Supports the full experiment matrix:
- Methods: baseline, mim
- Models: mlp, lstm, cnn (with configurable sizes)
- Missing: mcar, mar, mnar (future)
- Data: xjtu batches (2c, 3c, etc.)
- Seeds: configurable list

Usage:
    python src/experiments/run_experiment.py \
        experiment=youth_mar_mim \
        data.batch_id=2c \
        model.type=cnn \
        missing.rate_eval=0.3 \
        experiments.training.seeds=[42]
"""

import sys
sys.path.insert(0, '.')

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import pandas as pd
from typing import List, Dict

from src.data.loader import load_dataset
from src.utils.seed_manager import set_seed
from src.utils.logger import setup_logger
from src.utils.paths import get_results_dir, ensure_results_structure, get_timestamp


def create_model(cfg: DictConfig, input_dim: int):
    """根据配置创建模型"""
    model_type = cfg.model.type
    batch_id = cfg.data.batch_id
    
    # 从架构配置中加载
    arch_cfg = OmegaConf.load("configs/model_architectures.yaml")
    arch_key = f"{model_type}_small_best_{batch_id.lower().replace('c', 'c')}"
    
    if arch_key not in arch_cfg.small_architectures:
        raise ValueError(f"Architecture {arch_key} not found in config")
    
    arch = arch_cfg.small_architectures[arch_key]
    
    if model_type == "mlp":
        from src.models.mlp import MLP
        return MLP(
            input_dim=input_dim,
            hidden_dims=arch.hidden_dims,
            dropout=arch.get("dropout", 0.1),
        )
    
    elif model_type == "lstm":
        from src.models.lstm import LSTM
        return LSTM(
            input_dim=input_dim,
            hidden_dim=arch.hidden_dim,
            num_layers=arch.num_layers,
            dropout=arch.get("dropout", 0.1),
        )
    
    elif model_type == "cnn":
        from src.models.cnn1d import CNN1D
        return CNN1D(
            input_dim=input_dim,
            channels=arch.channels,
            kernel_sizes=arch.kernel_sizes,
            fc_hidden_dims=arch.get("fc_hidden_dims", [32]),
            dropout=arch.get("dropout", 0.1),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_missing_data_mim(X: torch.Tensor, y: torch.Tensor, cfg: DictConfig, seed: int):
    """
    Create multi-MR training data for MIM method
    
    Returns augmented training set with missing rates: 0.0, 0.1, ..., 0.9
    """
    from src.missing_data.mcar import simulate_mcar
    
    rate_list = cfg.missing.get("rate_train_list", [i/10.0 for i in range(10)])
    
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(rate_list):
        mr_seed = seed + i * 100
        X_imp, mask, mim_input = simulate_mcar(X, mr, mr_seed)
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    combined_input = torch.cat(all_inputs, dim=0)
    combined_target = torch.cat(all_targets, dim=0)
    
    return combined_input, combined_target


def create_missing_data_eval(X: torch.Tensor, y: torch.Tensor, cfg: DictConfig, seed: int):
    """Create evaluation data with specified missing mechanism and rate"""
    mode = cfg.missing.mode
    rate = cfg.missing.rate_eval
    
    if mode == "mar":
        from src.missing_data.mar import simulate_mar
        beta = cfg.missing.get("beta", 2.0)
        gamma = cfg.missing.get("gamma", 0.05)
        alpha = cfg.missing.get("alpha", None)
        X_imp, mask, mim_input = simulate_mar(X, y, rate, alpha, beta, gamma, seed)
    
    elif mode == "mcar":
        from src.missing_data.mcar import simulate_mcar
        X_imp, mask, mim_input = simulate_mcar(X, rate, seed)
    
    else:
        raise ValueError(f"Unknown missing mode: {mode}")
    
    return X_imp, mask, mim_input


def train_model(model, train_loader, val_loader, cfg: DictConfig):
    """训练模型"""
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    device = torch.device("cpu")
    model = model.to(device)
    
    epochs = cfg.training.epochs
    lr = cfg.training.learning_rate
    patience = cfg.training.early_stopping.patience
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                val_loss += criterion(outputs, batch_y).item() * len(batch_x)
        
        val_loss /= len(val_loader.dataset)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, test_loader):
    """评估模型"""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    return {
        "test_mae": mean_absolute_error(targets, preds),
        "test_rmse": np.sqrt(mean_squared_error(targets, preds)),
        "test_r2": r2_score(targets, preds),
    }


@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="youth_mar_baseline")
def main(cfg: DictConfig):
    """主函数"""
    
    # Setup logging
    logger = setup_logger("experiment", log_file=Path("logs/experiment.log"))
    
    method = cfg.method
    model_type = cfg.model.type
    batch_id = cfg.data.batch_id
    mr_eval = cfg.missing.rate_eval
    seeds = cfg.training.seeds
    
    logger.info(f"Starting experiment: method={method}, model={model_type}, batch={batch_id}, mr={mr_eval}")
    
    results = []
    
    for seed in seeds:
        logger.info(f"Running with seed={seed}")
        set_seed(seed)
        
        # Load data
        data_cfg = OmegaConf.create({
            "data": {
                "dataset": "xjtu",
                "data_dir": cfg.data.data_dir,
                "batch": batch_id,
                "features": [
                    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
                    "current mean", "current std", "current kurtosis", "current skewness",
                    "CC Q", "CC charge time", "CV Q", "CV charge time",
                    "voltage slope", "current slope", "voltage entropy", "current entropy"
                ],
                "target": "capacity",
                "preprocessing": {
                    "remove_outliers": True,
                    "outlier_method": "3sigma",
                    "standardize": True,
                    "fill_na": True,
                    "fill_method": "mean",
                },
                "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            }
        })
        
        data = load_dataset(data_cfg)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Determine input dimension
        if method == "mim":
            input_dim = 32  # 16 features + 16 masks
            train_input, train_target = create_missing_data_mim(X_train, y_train, cfg, seed)
            _, _, val_input = create_missing_data_eval(X_val, y_val, cfg, seed + 999)
            _, _, test_input = create_missing_data_eval(X_test, y_test, cfg, seed + 1000)
        else:  # baseline
            input_dim = 16
            X_train_imp, _, _ = create_missing_data_eval(X_train, y_train, cfg, seed)
            X_val_imp, _, _ = create_missing_data_eval(X_val, y_val, cfg, seed + 999)
            X_test_imp, _, _ = create_missing_data_eval(X_test, y_test, cfg, seed + 1000)
            train_input, train_target = X_train_imp, y_train
            val_input = X_val_imp
            test_input = X_test_imp
        
        # Create model
        model = create_model(cfg, input_dim)
        
        # Create data loaders
        batch_size = cfg.training.batch_size
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_input, train_target),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_input, y_val),
            batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_input, y_test),
            batch_size=batch_size, shuffle=False
        )
        
        # Train and evaluate
        model = train_model(model, train_loader, val_loader, cfg)
        metrics = evaluate_model(model, test_loader)
        
        # Record result
        result = {
            "dataset": "xjtu",
            "batch_id": batch_id,
            "missing_mode": cfg.missing.mode,
            "missing_rate": mr_eval,
            "method": method,
            "model": model_type,
            "model_size": "small",
            "seed": seed,
            **metrics,
        }
        results.append(result)
        
        logger.info(f"Seed {seed}: MAE={metrics['test_mae']:.4f}, R2={metrics['test_r2']:.4f}")
    
    # Save results to batch-specific directory
    df = pd.DataFrame(results)
    
    # Get timestamp from config or generate new one
    timestamp = cfg.get("timestamp", None)
    if timestamp is None:
        timestamp = get_timestamp()
    
    # Create results directory: results/{timestamp}_{batch}/
    results_dir = get_results_dir(batch_id, timestamp)
    paths = ensure_results_structure(results_dir)
    output_csv = paths["csv_path"]
    
    # Append or create new
    if output_csv.exists():
        df_existing = pd.read_csv(output_csv)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"Completed {len(seeds)} experiments")
    logger.info(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()
