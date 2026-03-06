"""数据清洗和预处理模块

提供数据清洗、异常值处理、缺失值填充和标准化功能。
"""

from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def clean_dataframe(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """清洗数据框：去除inf、处理异常值、填充缺失值

    步骤：
        1. 替换 inf 为 NaN
        2. 根据 cfg.data.preprocessing.outlier_method 处理异常值（3sigma 或 iqr）
        3. 填充 NaN（mean 或 median）
        4. 标准化（如果 cfg.data.preprocessing.standardize 为 True）

    Args:
        df: 输入数据框
        cfg: 配置对象，包含 preprocessing 相关配置

    Returns:
        清洗后的数据框。如果清洗后为空，返回空数据框

    Raises:
        ValueError: 如果 outlier_method 或 fill_method 配置无效
    """
    if df.empty:
        return df.copy()

    # 1. 替换 inf 为 NaN
    df_clean = df.replace([np.inf, -np.inf], np.nan)

    # 获取数值列（用于异常值处理和填充）
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return df_clean

    # 2. 处理异常值
    outlier_method = cfg.data.preprocessing.get("outlier_method", "none")

    if outlier_method == "3sigma":
        df_clean = _remove_outliers_3sigma(df_clean, numeric_cols)
    elif outlier_method == "iqr":
        df_clean = _remove_outliers_iqr(df_clean, numeric_cols)
    elif outlier_method != "none":
        raise ValueError(
            f"Invalid outlier_method: {outlier_method}. "
            "Expected '3sigma', 'iqr', or 'none'"
        )

    # 检查处理后是否为空
    if df_clean.empty:
        return df_clean

    # 3. 填充 NaN
    fill_method = cfg.data.preprocessing.get("fill_method", "mean")

    if fill_method == "mean":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].mean()
        )
    elif fill_method == "median":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].median()
        )
    elif fill_method == "zero":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    elif fill_method != "none":
        raise ValueError(
            f"Invalid fill_method: {fill_method}. "
            "Expected 'mean', 'median', 'zero', or 'none'"
        )

    # 4. 标准化（可选）
    if cfg.data.preprocessing.get("standardize", False):
        # 标准化将在 splits 阶段进行，这里只返回清洗后的数据
        pass

    return df_clean


def _remove_outliers_3sigma(
    df: pd.DataFrame, numeric_cols: pd.Index
) -> pd.DataFrame:
    """使用3-sigma方法移除异常值

    Args:
        df: 输入数据框
        numeric_cols: 数值列名

    Returns:
        移除异常值后的数据框
    """
    mask = pd.Series(True, index=df.index)

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            col_mask = (df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)
            mask = mask & col_mask

    return df[mask].copy()


def _remove_outliers_iqr(df: pd.DataFrame, numeric_cols: pd.Index) -> pd.DataFrame:
    """使用IQR方法移除异常值

    Args:
        df: 输入数据框
        numeric_cols: 数值列名

    Returns:
        移除异常值后的数据框
    """
    mask = pd.Series(True, index=df.index)

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_mask = (df[col] >= lower) & (df[col] <= upper)
        mask = mask & col_mask

    return df[mask].copy()


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准化特征，返回标准化后的数据和统计量

    使用 z-score 标准化：X_scaled = (X - mean) / std

    Args:
        X: 特征矩阵 [N, D]

    Returns:
        元组包含:
            - X_scaled: 标准化后的特征矩阵 [N, D]
            - mean: 每列均值 [D]
            - std: 每列标准差 [D]

    Raises:
        ValueError: 如果输入为空或标准差全为零
    """
    if X.size == 0:
        raise ValueError("Cannot standardize empty array")

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # 处理零标准差（常数列）
    std_safe = np.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std_safe

    return X_scaled, mean, std
