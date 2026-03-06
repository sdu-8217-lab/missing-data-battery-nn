"""数据集划分模块 - 支持按电池划分"""
from typing import Dict, List, Tuple
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .preprocessing import standardize_features


def split_batteries(
    battery_ids: List[str],
    test_size: float = 0.25,
    val_size: float = 0.25,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    将电池ID列表划分为训练/验证/测试集
    
    Args:
        battery_ids: 电池ID列表
        test_size: 测试集比例
        val_size: 验证集比例（占训练+验证的比例）
        random_state: 随机种子
        
    Returns:
        train_ids, val_ids, test_ids
    """
    # 第一步：划分出测试集
    train_val_ids, test_ids = train_test_split(
        battery_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # 第二步：划分训练集和验证集
    val_ratio = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    return train_ids, val_ids, test_ids


def train_val_test_split_by_battery(
    battery_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg: DictConfig
) -> Dict[str, torch.Tensor]:
    """
    按电池划分训练/验证/测试集
    
    Args:
        battery_data: {battery_id: (X, y)} 字典
        cfg: 配置
        
    Returns:
        包含划分后数据的字典
    """
    battery_ids = list(battery_data.keys())
    
    if len(battery_ids) == 0:
        raise ValueError("No batteries loaded")
    
    if len(battery_ids) < 3:
        raise ValueError(
            f"Need at least 3 batteries for train/val/test split, got {len(battery_ids)}. "
            f"Consider using random split instead."
        )
    
    # 划分电池ID
    split_config = cfg.data.get('split', {})
    test_size = split_config.get('test_size', 0.25)
    val_size = split_config.get('val_size', 0.25)
    random_state = split_config.get('random_state', 42)
    
    train_ids, val_ids, test_ids = split_batteries(
        battery_ids, test_size, val_size, random_state
    )
    
    print(f"Battery split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"  Train: {train_ids}")
    print(f"  Val:   {val_ids}")
    print(f"  Test:  {test_ids}")
    
    # 合并各组的电池数据
    X_train = np.vstack([battery_data[bid][0] for bid in train_ids])
    y_train = np.concatenate([battery_data[bid][1] for bid in train_ids])
    
    X_val = np.vstack([battery_data[bid][0] for bid in val_ids])
    y_val = np.concatenate([battery_data[bid][1] for bid in val_ids])
    
    X_test = np.vstack([battery_data[bid][0] for bid in test_ids])
    y_test = np.concatenate([battery_data[bid][1] for bid in test_ids])
    
    # 标准化（使用训练集统计量）
    mean = None
    std = None
    preprocessing = cfg.data.get('preprocessing', {})
    if preprocessing.get('standardize', False):
        X_train, mean, std = standardize_features(X_train)
        if mean is not None and std is not None:
            std_safe = np.where(std == 0, 1.0, std)
            X_val = (X_val - mean) / std_safe
            X_test = (X_test - mean) / std_safe
    
    result = {
        'X_train': torch.tensor(X_train, dtype=torch.float32),
        'y_train': torch.tensor(y_train, dtype=torch.float32),
        'X_val': torch.tensor(X_val, dtype=torch.float32),
        'y_val': torch.tensor(y_val, dtype=torch.float32),
        'X_test': torch.tensor(X_test, dtype=torch.float32),
        'y_test': torch.tensor(y_test, dtype=torch.float32),
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
    }
    
    if mean is not None:
        result['mean'] = torch.tensor(mean, dtype=torch.float32)
    if std is not None:
        result['std'] = torch.tensor(std, dtype=torch.float32)
    
    return result
