"""数据加载和 DataLoader 创建 - 支持 Baseline(插补) 和 MIM."""
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from .features import build_features
from .preprocessing import clean_dataframe
from .splits import train_val_test_split


def load_dataset(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """根据配置加载数据集."""
    dataset = cfg.data.get("dataset", None)
    if dataset is None:
        raise ValueError("cfg.data.dataset must be specified")

    loaders = {
        "xjtu": _load_xjtu,
        "tju": _load_tju,
        "hust": _load_hust,
        "mit": _load_mit,
    }
    
    if dataset not in loaders:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")
    
    return loaders[dataset](cfg)


def _load_csv_files(data_dir: Path, pattern: str, cfg: DictConfig, recursive: bool = False):
    """加载匹配模式的CSV文件."""
    if recursive:
        files = sorted(data_dir.rglob(pattern))
    else:
        files = sorted(data_dir.glob(pattern))
    
    if not files:
        available = list(data_dir.rglob("*.csv"))[:10]
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")
    
    X_list, y_list = [], []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = clean_dataframe(df, cfg)
            if df.empty:
                continue
            X, y = build_features(df, cfg)
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Warning: Failed to load {fp.name}: {e}")
            continue
    
    if not X_list:
        raise ValueError("No valid data loaded")
    
    return np.vstack(X_list), np.concatenate(y_list)


def _load_xjtu(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "2C")
    pattern = cfg.data.get("file_pattern", f"{batch}_battery-*.csv").format(batch_id=batch)
    X, y = _load_csv_files(data_dir, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_tju(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "Dataset_1_NCA_battery")
    X, y = _load_csv_files(data_dir / batch, "*.csv", cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_hust(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    X, y = _load_csv_files(data_dir, "*.csv", cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_mit(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "2017-05-12")
    X, y = _load_csv_files(data_dir / batch, "*battery*.csv", cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def apply_missing_and_impute(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rate: float,
    impute_method: str,
    mode: str,
    seed: int
) -> torch.Tensor:
    """
    Baseline: 施加缺失 + 传统插补
    
    Args:
        X: 原始特征 [N, D]
        y: 目标值（用于 MAR）[N]
        missing_rate: 缺失率
        impute_method: "mean", "median", "knn", "zero"
        mode: "mcar" 或 "mar"
        seed: 随机种子
    """
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    from ..missing_data.imputation import mean_imputation, median_imputation, knn_imputation, zero_imputation
    
    # 生成缺失
    if mode == 'mar':
        _, missing_mask, _ = simulate_mar(X, y, missing_rate, seed=seed)
    else:
        _, missing_mask, _ = simulate_mcar(X, missing_rate, seed=seed)
    
    # 应用插补
    if impute_method == "mean":
        return mean_imputation(X, missing_mask)
    elif impute_method == "median":
        return median_imputation(X, missing_mask)
    elif impute_method == "knn":
        return knn_imputation(X, missing_mask)
    elif impute_method == "zero":
        return zero_imputation(X, missing_mask)
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")


def create_mim_training_data(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rates: List[float],
    cfg: DictConfig,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MIM 训练：将训练集复制多份，每份施加不同缺失率，构造 MIM 输入
    """
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(missing_rates):
        mr_seed = seed + i * 100
        
        if cfg.missing.mode == 'mar':
            _, _, mim_input = simulate_mar(X, y, mr, seed=mr_seed)
        else:
            _, _, mim_input = simulate_mcar(X, mr, seed=mr_seed)
        
        all_inputs.append(mim_input)
        all_targets.append(y)
    
    return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)


def create_dataloaders(
    data_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    mode: str = 'train',
    method: str = 'mim',  # 'mim', 'mean', 'median', 'knn', 'zero'
    missing_rate: float = 0.0,
) -> Tuple:
    """
    创建 DataLoader
    
    Args:
        data_dict: 数据集
        cfg: 配置
        mode: 'train', 'val', 'eval'
        method: 
            - 'mim': MIM 方法（输入维度 32）
            - 'mean'/'median'/'knn'/'zero': 插补 baseline（输入维度 16）
        missing_rate: 缺失率（用于 MIM 评估或插补测试）
    """
    seq_len = cfg.data.get('seq_len', 5)
    batch_size = cfg.training.batch_size
    model_type = cfg.model.type
    
    def to_sequence(X):
        if X.dim() == 2 and model_type in ['lstm', 'gru', 'cnn1d']:
            X = X.unsqueeze(1).repeat(1, seq_len, 1)
        return X
    
    if mode == 'train':
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        if method == 'mim':
            # MIM: 混合多种缺失率
            mim_rates = cfg.missing.get('mim_train_rates', [i/10.0 for i in range(10)])
            X_train, y_train = create_mim_training_data(X_train, y_train, mim_rates, cfg)
        else:
            # Baseline: 训练时使用完整数据（不施加缺失）
            pass  # X_train 保持原样
        
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        X_train = to_sequence(X_train)
        X_val = to_sequence(X_val)
        
        return (
            DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        )
    
    elif mode == 'val':
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        X_val = to_sequence(X_val)
        return DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    else:  # eval
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        if method == 'mim':
            # MIM: 施加缺失，构造 MIM 输入
            from ..missing_data.mcar import simulate_mcar
            from ..missing_data.mar import simulate_mar
            
            if cfg.missing.mode == 'mar':
                _, _, X_test = simulate_mar(X_test, y_test, missing_rate, seed=42)
            else:
                _, _, X_test = simulate_mcar(X_test, missing_rate, seed=42)
        else:
            # Baseline: 施加缺失 + 插补
            X_test = apply_missing_and_impute(X_test, y_test, missing_rate, method, cfg.missing.mode, seed=42)
        
        X_test = to_sequence(X_test)
        return None, None, DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
