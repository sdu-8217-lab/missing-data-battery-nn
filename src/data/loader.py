"""数据加载和 DataLoader 创建 - 支持按电池划分."""
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from .features import build_features
from .preprocessing import clean_dataframe
from .splits import train_val_test_split_by_battery


def extract_battery_id(filepath: Path, dataset_type: str) -> str:
    """
    从文件路径提取电池ID
    
    XJTU: 2C_battery-1.csv -> "2C_battery-1"
    TJU:  Dataset_1_NCA_battery/CY25-025_1-#1.csv -> "CY25-025_1-#1"
    HUST: 1-1.csv -> "1" (电池编号)
    MIT:  2017-05-12/2017-05-12_battery-1.csv -> "2017-05-12_battery-1"
    """
    if dataset_type == 'hust':
        # HUST: 文件名格式 "{电池编号}-{循环}.csv"
        return filepath.stem.split('-')[0]
    else:
        # 其他: 使用文件名（不含扩展名）
        return filepath.stem


def load_batteries(
    data_dir: Path,
    pattern: str,
    cfg: DictConfig,
    dataset_type: str,
    recursive: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    按电池加载数据，返回 {battery_id: (X, y)} 字典
    
    Args:
        data_dir: 数据目录
        pattern: 文件匹配模式
        cfg: 配置
        dataset_type: 'xjtu', 'tju', 'hust', 'mit'
        recursive: 是否递归搜索
        
    Returns:
        {battery_id: (X, y)} 字典
    """
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
    
    battery_data = {}
    
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = clean_dataframe(df, cfg)
            if df.empty:
                continue
            
            X, y = build_features(df, cfg)
            battery_id = extract_battery_id(fp, dataset_type)
            
            if battery_id in battery_data:
                # 同一个电池的多个文件，合并数据
                X_existing, y_existing = battery_data[battery_id]
                battery_data[battery_id] = (
                    np.vstack([X_existing, X]),
                    np.concatenate([y_existing, y])
                )
            else:
                battery_data[battery_id] = (X, y)
                
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
            continue
    
    if not battery_data:
        raise ValueError("No valid data loaded")
    
    print(f"Loaded {len(battery_data)} batteries from {data_dir}")
    for bid, (X, y) in battery_data.items():
        print(f"  {bid}: {len(y)} samples")
    
    return battery_data


def load_dataset(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """根据配置加载数据集."""
    dataset = cfg.data.get("dataset", None)
    if dataset is None:
        raise ValueError("cfg.data.dataset must be specified")
    
    data_dir = Path(cfg.data.data_dir)
    batch = cfg.data.get("batch_id", "")
    pattern = cfg.data.get("file_pattern", "*.csv")
    
    if dataset == "xjtu":
        # XJTU: data/XJTU data/2C_battery-*.csv
        pattern = pattern.format(batch_id=batch) if '{batch_id}' in pattern else f"{batch}_battery-*.csv"
        battery_data = load_batteries(data_dir, pattern, cfg, dataset, recursive=False)
        
    elif dataset == "tju":
        # TJU: data/TJU data/Dataset_1_NCA_battery/*.csv
        battery_data = load_batteries(data_dir / batch, "*.csv", cfg, dataset, recursive=False)
        
    elif dataset == "hust":
        # HUST: data/HUST data/*.csv
        battery_data = load_batteries(data_dir, "*.csv", cfg, dataset, recursive=False)
        
    elif dataset == "mit":
        # MIT: data/MIT data/2017-05-12/*battery*.csv
        battery_data = load_batteries(data_dir / batch, "*battery*.csv", cfg, dataset, recursive=False)
        
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported")
    
    # 按电池划分
    return train_val_test_split_by_battery(battery_data, cfg)


# ... (保持 create_dataloaders 不变)
from .loader_dataloaders import create_dataloaders
