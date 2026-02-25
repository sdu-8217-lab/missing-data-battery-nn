"""数据加载和 DataLoader 创建."""
from pathlib import Path
from typing import Dict, Tuple

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
        raise NotImplementedError(f"Dataset '{dataset}' not supported. Choose from {list(loaders.keys())}")
    
    return loaders[dataset](cfg)


def _load_csv_files(data_dir: Path, pattern: str, cfg: DictConfig, recursive: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """加载匹配模式的CSV文件."""
    if recursive:
        files = sorted(data_dir.rglob(pattern))
    else:
        files = sorted(data_dir.glob(pattern))
    
    if not files:
        available = list(data_dir.rglob("*.csv"))[:10]
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {data_dir}\n"
            f"Available: {[str(f.relative_to(data_dir)) for f in available]}"
        )
    
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


def _load_xjtu(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 XJTU 数据集."""
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "3C")
    pattern = cfg.data.get("file_pattern", f"{batch}_battery-*.csv").format(batch_id=batch)
    X, y = _load_csv_files(data_dir, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_tju(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 TJU 数据集."""
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "Dataset_1_NCA_battery")
    pattern = cfg.data.get("file_pattern", "*.csv")
    # TJU 数据在子目录中
    X, y = _load_csv_files(data_dir / batch, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_hust(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 HUST 数据集."""
    data_dir = Path(cfg.data.data_dir)
    pattern = cfg.data.get("file_pattern", "*.csv")
    X, y = _load_csv_files(data_dir, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_mit(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 MIT 数据集."""
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "2017-05-12")
    pattern = cfg.data.get("file_pattern", "*battery*.csv")
    # MIT 数据在子目录中
    X, y = _load_csv_files(data_dir / batch, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def create_dataloaders(
    data_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    mode: str = 'train',
    missing_rate: float = 0.0
) -> Tuple[DataLoader, ...]:
    """创建 DataLoader，可选应用缺失机制."""
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    
    seq_len = cfg.data.get('seq_len', 5)
    batch_size = cfg.training.batch_size
    use_mim = cfg.model.use_mim
    
    def apply_missing(X, y, mr):
        """应用缺失机制."""
        if mr == 0:
            if use_mim:
                mask = torch.zeros_like(X)
                return torch.cat([X, mask], dim=1), y
            return X, y
        
        # X is already a tensor
        if cfg.missing.mode == 'mar':
            X_imp, mask, mim_input = simulate_mar(X, mr, seed=42)
        else:
            X_imp, mask, mim_input = simulate_mcar(X, mr, seed=42)
        
        if use_mim:
            return mim_input, y
        return X_imp, y
    
    def to_sequence(X):
        """Convert to sequence format for LSTM/GRU/CNN."""
        if X.dim() == 2:
            X = X.unsqueeze(1).repeat(1, seq_len, 1)
        return X
    
    if mode == 'train':
        X_train, y_train = apply_missing(data_dict['X_train'], data_dict['y_train'], missing_rate)
        X_val, y_val = apply_missing(data_dict['X_val'], data_dict['y_val'], 0)
        
        if cfg.model.type in ['lstm', 'gru', 'cnn1d']:
            X_train = to_sequence(X_train)
            X_val = to_sequence(X_val)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    else:
        X_test, y_test = apply_missing(data_dict['X_test'], data_dict['y_test'], missing_rate)
        
        if cfg.model.type in ['lstm', 'gru', 'cnn1d']:
            X_test = to_sequence(X_test)
        
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        return None, None, test_loader
