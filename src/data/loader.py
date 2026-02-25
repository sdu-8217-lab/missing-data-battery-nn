"""数据加载和 DataLoader 创建."""
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
    batch = cfg.data.get("batch_id", "2C")
    pattern = cfg.data.get("file_pattern", f"{batch}_battery-*.csv").format(batch_id=batch)
    X, y = _load_csv_files(data_dir, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def _load_tju(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """加载 TJU 数据集."""
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "Dataset_1_NCA_battery")
    pattern = cfg.data.get("file_pattern", "*.csv")
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
    X, y = _load_csv_files(data_dir / batch, pattern, cfg, recursive=False)
    return train_val_test_split(X, y, cfg)


def create_mim_training_data(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rates: List[float],
    cfg: DictConfig,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建 MIM 训练数据：将训练集复制多份，每份施加不同缺失率
    
    Args:
        X: 原始特征 [N, D]
        y: 目标 [N]
        missing_rates: 缺失率列表，如 [0.0, 0.1, ..., 0.9]
        cfg: 配置
        seed: 随机种子
        
    Returns:
        X_mim: MIM 输入 [N * len(missing_rates), 2D]
        y_mim: 目标 [N * len(missing_rates)]
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
    
    X_mim = torch.cat(all_inputs, dim=0)
    y_mim = torch.cat(all_targets, dim=0)
    
    return X_mim, y_mim


def create_dataloaders(
    data_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    mode: str = 'train',
    missing_rate: float = 0.0,
    use_mim: bool = False,
    mim_train_rates: List[float] = None
) -> Tuple[DataLoader, ...]:
    """创建 DataLoader，支持 MIM 训练.
    
    Args:
        data_dict: 包含 X_train, y_train, X_val, y_val, X_test, y_test
        cfg: 配置
        mode: 'train', 'val', 或 'eval'
        missing_rate: 应用的缺失率（非 MIM 模式）
        use_mim: 是否使用 MIM 训练
        mim_train_rates: MIM 训练时的缺失率列表
    
    Returns:
        train_loader, val_loader (mode='train')
        或 val_loader (mode='val')
        或 None, None, test_loader (mode='eval')
    """
    from ..missing_data.mcar import simulate_mcar
    from ..missing_data.mar import simulate_mar
    
    seq_len = cfg.data.get('seq_len', 5)
    batch_size = cfg.training.batch_size
    model_type = cfg.model.type
    
    def to_sequence(X):
        """Convert to sequence format for LSTM/GRU/CNN using sliding window."""
        if X.dim() == 2 and model_type in ['lstm', 'gru', 'cnn1d']:
            # TODO: Implement sliding window
            # For now, use simple repeat
            X = X.unsqueeze(1).repeat(1, seq_len, 1)
        return X
    
    if mode == 'train':
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        if use_mim and mim_train_rates:
            # MIM 训练：混合多种缺失率
            X_train, y_train = create_mim_training_data(
                X_train, y_train, mim_train_rates, cfg, seed=42
            )
        elif missing_rate > 0:
            # 非 MIM，单一缺失率
            if cfg.missing.mode == 'mar':
                _, _, X_train = simulate_mar(X_train, y_train, missing_rate, seed=42)
            else:
                _, _, X_train = simulate_mcar(X_train, missing_rate, seed=42)
        else:
            # 无缺失，添加零掩码
            mask = torch.zeros_like(X_train)
            X_train = torch.cat([X_train, mask], dim=1)
        
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        # 验证集始终使用完整数据（添加零掩码）
        mask_val = torch.zeros_like(X_val)
        X_val = torch.cat([X_val, mask_val], dim=1)
        
        # 转换为序列
        X_train = to_sequence(X_train)
        X_val = to_sequence(X_val)
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader
    
    elif mode == 'val':
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        mask_val = torch.zeros_like(X_val)
        X_val = torch.cat([X_val, mask_val], dim=1)
        X_val = to_sequence(X_val)
        
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        return val_loader
    
    else:  # eval mode
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        if missing_rate > 0:
            if cfg.missing.mode == 'mar':
                _, _, X_test = simulate_mar(X_test, y_test, missing_rate, seed=42)
            else:
                _, _, X_test = simulate_mcar(X_test, missing_rate, seed=42)
        else:
            mask_test = torch.zeros_like(X_test)
            X_test = torch.cat([X_test, mask_test], dim=1)
        
        X_test = to_sequence(X_test)
        
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False
        )
        return None, None, test_loader
