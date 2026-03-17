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
    mode = cfg.missing.mode
    rate_list = cfg.missing.get("rate_train_list", [i/10.0 for i in range(10)])
    
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(rate_list):
        mr_seed = seed + i * 100
        
        if mode == "mar":
            from src.missing_data.mar import simulate_mar
            beta = cfg.missing.get("beta", 2.0)
            gamma = cfg.missing.get("gamma", 0.05)
            alpha = cfg.missing.get("alpha", None)
            X_imp, mask, mim_input = simulate_mar(X, y, mr, alpha, beta, gamma, mr_seed)
        elif mode == "mnar":
            from src.missing_data.mnar import simulate_mnar
            alpha = cfg.missing.get("alpha", None)
            beta = cfg.missing.get("beta", 0.05)
            feature_idx = cfg.missing.get("feature_index", 0)
            X_imp, mask, mim_input = simulate_mnar(X, y, mr, alpha, beta, feature_idx=feature_idx, seed=mr_seed)
        else:  # mcar
            from src.missing_data.mcar import simulate_mcar
            X_imp, mask, mim_input = simulate_mcar(X, mr, mr_seed)
        
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    combined_input = torch.cat(all_inputs, dim=0)
    combined_target = torch.cat(all_targets, dim=0)
    
    return combined_input, combined_target


def create_missing_data_eval(X: torch.Tensor, y: torch.Tensor, cfg: DictConfig, seed: int, rate: float = None):
    """Create evaluation data with specified missing mechanism and rate
    
    Args:
        X: Input features
        y: Target values
        cfg: Configuration
        seed: Random seed
        rate: Missing rate (if None, uses cfg.missing.rate_eval)
    """
    mode = cfg.missing.mode
    if rate is None:
        rate = cfg.missing.rate_eval
    
    if mode == "mar":
        from src.missing_data.mar import simulate_mar
        beta = cfg.missing.get("beta", 2.0)
        gamma = cfg.missing.get("gamma", 0.05)
        alpha = cfg.missing.get("alpha", None)
        X_imp, mask, mim_input = simulate_mar(X, y, rate, alpha, beta, gamma, seed)
    
    elif mode == "mnar":
        from src.missing_data.mnar import simulate_mnar
        alpha = cfg.missing.get("alpha", None)
        beta = cfg.missing.get("beta", 0.05)
        feature_idx = cfg.missing.get("feature_index", 0)
        X_imp, mask, mim_input = simulate_mnar(X, y, rate, alpha, beta, feature_idx=feature_idx, seed=seed)
    
    elif mode == "mcar":
        from src.missing_data.mcar import simulate_mcar
        X_imp, mask, mim_input = simulate_mcar(X, rate, seed)
    
    else:
        raise ValueError(f"Unknown missing mode: {mode}")
    
    return X_imp, mask, mim_input


def validate_multi_mr(model, X_val, y_val, cfg: DictConfig, seed: int, batch_size: int, method: str = "mim"):
    """
    在多个缺失率下验证模型，返回平均损失
    
    Args:
        model: 模型
        X_val: 验证特征 (原始特征，16维)
        y_val: 验证标签
        cfg: 配置
        seed: 随机种子
        batch_size: 批次大小
        method: "mim" 或 "baseline"
    
    Returns:
        avg_val_loss: 所有缺失率下的平均验证损失
        per_mr_losses: 每个缺失率的具体损失 {mr: loss}
    """
    import torch.nn as nn
    
    model.eval()
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    
    # 验证缺失率列表（可配置，默认使用0.0-0.9）
    val_missing_rates = cfg.missing.get(
        'validation_missing_rates', 
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    
    per_mr_losses = {}
    
    with torch.no_grad():
        for mr in val_missing_rates:
            # 创建当前缺失率的验证数据
            X_imp, mask, mim_input = create_missing_data_eval(
                X_val, y_val, cfg, seed=seed, rate=mr
            )
            
            # 根据方法选择正确的输入格式
            # baseline: 使用 X_imp (16维填充数据)
            # mim: 使用 mim_input (32维 = 16特征 + 16掩码)
            if method == "mim":
                val_input = mim_input  # [N, 32]
            else:
                val_input = X_imp  # [N, 16]
            
            # 创建DataLoader
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(val_input, y_val),
                batch_size=batch_size,
                shuffle=False
            )
            
            # 计算该MR下的损失
            mr_loss = 0.0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze(-1)
                loss = criterion(outputs, batch_y)
                mr_loss += loss.item() * len(batch_x)
            
            mr_loss /= len(val_loader.dataset)
            per_mr_losses[mr] = mr_loss
    
    # 使用平均损失作为早停指标
    avg_val_loss = sum(per_mr_losses.values()) / len(per_mr_losses)
    
    return avg_val_loss, per_mr_losses


def train_model(model, train_loader, val_loader, cfg: DictConfig, 
                X_val=None, y_val=None, multi_mr_val: bool = False, method: str = "mim"):
    """训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（单MR验证时使用）
        cfg: 配置
        X_val: 验证特征（多MR验证时使用）
        y_val: 验证标签（多MR验证时使用）
        multi_mr_val: 是否使用多MR验证
        method: "mim" 或 "baseline"
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    device = torch.device("cpu")
    model = model.to(device)
    
    epochs = cfg.training.epochs
    lr = cfg.training.learning_rate
    patience = cfg.training.early_stopping.patience
    batch_size = cfg.training.batch_size
    
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
            outputs = model(batch_x).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        
        # Validation
        if multi_mr_val and X_val is not None and y_val is not None:
            # 多缺失率验证
            val_loss, per_mr_losses = validate_multi_mr(
                model, X_val, y_val, cfg, seed=epoch + 999, batch_size=batch_size, method=method
            )
            
            # 每10个epoch打印各MR的损失
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Avg Val Loss = {val_loss:.4f}")
                for mr, loss in sorted(per_mr_losses.items()):
                    print(f"  MR={mr:.1f}: {loss:.4f}")
        else:
            # 单一缺失率验证（原逻辑）
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze(-1)
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
                if multi_mr_val:
                    print(f"Early stopping at epoch {epoch}")
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
            outputs = model(batch_x).squeeze(-1)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    return {
        "test_mae": mean_absolute_error(targets, preds),
        "test_rmse": np.sqrt(mean_squared_error(targets, preds)),
        "test_r2": r2_score(targets, preds),
    }


@hydra.main(version_base=None, config_path="../configs/experiments", config_name="youth_mar_baseline")
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
                "batch_id": batch_id,
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
        # Check if multi-MR validation is enabled
        multi_mr_val = cfg.get("multi_mr_validation", False)
        if multi_mr_val:
            logger.info("Using multi-MR validation (0.0-0.9)")
            # 对于多MR验证，需要传递原始特征，以便在每个epoch生成不同MR的验证数据
            # MIM方法使用原始X_val (16维)，baseline方法也使用原始X_val (16维)
            # create_missing_data_eval会根据方法类型生成正确的输入格式
            model = train_model(model, train_loader, val_loader, cfg, 
                               X_val=X_val, y_val=y_val, multi_mr_val=True, method=method)
        else:
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
