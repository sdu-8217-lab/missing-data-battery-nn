"""数据集划分模块

提供训练/验证/测试集划分功能。
"""

from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .preprocessing import standardize_features


def train_val_test_split(
    X: np.ndarray, y: np.ndarray, cfg: DictConfig
) -> Dict[str, torch.Tensor]:
    """划分训练/验证/测试集

    首先划分出测试集，然后在剩余数据中划分训练集和验证集。
    如果配置了标准化，则使用训练集的统计量标准化所有数据。

    Args:
        X: 特征矩阵 [N, D]
        y: 目标向量 [N]
        cfg: 配置对象，包含 split 相关配置

    Returns:
        字典包含以下键值对（均为 torch.Tensor）：
            - X_train: 训练集特征 [N_train, D]
            - y_train: 训练集目标 [N_train]
            - X_val: 验证集特征 [N_val, D]
            - y_val: 验证集目标 [N_val]
            - X_test: 测试集特征 [N_test, D]
            - y_test: 测试集目标 [N_test]
            - mean: 训练集均值（如果标准化）[D]
            - std: 训练集标准差（如果标准化）[D]

    Raises:
        ValueError: 如果数据为空或划分比例无效
    """
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Cannot split empty dataset")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different number of samples: X={X.shape[0]}, y={y.shape[0]}"
        )

    # 获取划分参数
    test_size = cfg.data.split.get("test_size", 0.2)
    val_size = cfg.data.split.get("val_size", 0.2)
    random_state = cfg.data.split.get("random_state", 42)

    # 验证比例
    if test_size < 0 or val_size < 0:
        raise ValueError("test_size and val_size must be non-negative")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be less than 1.0, got {test_size + val_size}")

    # 第一步：划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 第二步：在剩余数据中划分训练集和验证集
    # 计算验证集占剩余数据的比例
    val_ratio = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )

    # 标准化（可选）
    mean = None
    std = None
    if cfg.data.preprocessing.get("standardize", False):
        X_train, mean, std = standardize_features(X_train)
        if mean is not None and std is not None:
            # 使用训练集的统计量标准化验证集和测试集
            std_safe = np.where(std == 0, 1.0, std)
            X_val = (X_val - mean) / std_safe
            X_test = (X_test - mean) / std_safe

    # 转换为 torch.Tensor
    result = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
    }

    # 添加标准化统计量（如果有）
    if mean is not None:
        result["mean"] = torch.tensor(mean, dtype=torch.float32)
    if std is not None:
        result["std"] = torch.tensor(std, dtype=torch.float32)

    return result
