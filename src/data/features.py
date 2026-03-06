"""特征工程模块

提供从清洗后的数据构建特征和目标变量的功能。
"""

from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def build_features(df: pd.DataFrame, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """从清洗后的数据框构建特征X和目标y

    步骤：
        1. 从 cfg.data.features 读取特征列名
        2. 从 cfg.data.target 读取目标列
        3. 计算 SOH = capacity / nominal_capacity

    Args:
        df: 清洗后的数据框
        cfg: 配置对象，包含 features 和 target 相关配置

    Returns:
        元组包含:
            - X: 特征矩阵 [N, D]
            - y: SOH 向量 [N]

    Raises:
        ValueError: 如果特征列或目标列不存在，或数据为空
        KeyError: 如果配置的列名在数据框中不存在
    """
    if df.empty:
        raise ValueError("Cannot build features from empty DataFrame")

    # 获取特征列名
    feature_cols = cfg.data.get("features", None)
    if feature_cols is None:
        raise ValueError("cfg.data.features must be specified")

    # 获取目标列
    target_col = cfg.data.get("target", None)
    if target_col is None:
        raise ValueError("cfg.data.target must be specified")

    # 验证列存在
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise KeyError(f"Feature columns not found: {missing_features}")

    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    # 提取特征
    X = df[feature_cols].values.astype(np.float32)

    # 计算 SOH
    capacity = df[target_col].values.astype(np.float32)
    nominal_capacity = cfg.data.get("nominal_capacity", None)

    if nominal_capacity is None:
        # 使用第一个容量值作为标称容量
        nominal_capacity = capacity[0] if len(capacity) > 0 else 1.0

    y = capacity / nominal_capacity

    # 验证输出
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature and target have different number of samples: "
            f"X={X.shape[0]}, y={y.shape[0]}"
        )

    return X, y
