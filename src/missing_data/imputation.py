"""
传统插补方法 (Traditional Imputation Baselines)
"""
import torch
import numpy as np
from typing import Tuple
from sklearn.impute import KNNImputer, SimpleImputer


def mean_imputation(X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """
    均值插补：用每列的均值填充缺失值
    
    Args:
        X: 原始特征 [N, D]
        missing_mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        
    Returns:
        X_imputed: 插补后的特征 [N, D]
    """
    X_np = X.numpy()
    mask_np = missing_mask.numpy()
    
    X_imputed = X_np.copy()
    for j in range(X_np.shape[1]):
        col = X_np[:, j]
        mask_col = mask_np[:, j]
        
        if mask_col.sum() > 0:
            mean_val = col[mask_col == 1].mean()
            X_imputed[mask_col == 0, j] = mean_val
    
    return torch.from_numpy(X_imputed).float()


def median_imputation(X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """
    中位数插补：用每列的中位数填充缺失值
    """
    X_np = X.numpy()
    mask_np = missing_mask.numpy()
    
    X_imputed = X_np.copy()
    for j in range(X_np.shape[1]):
        col = X_np[:, j]
        mask_col = mask_np[:, j]
        
        if mask_col.sum() > 0:
            median_val = np.median(col[mask_col == 1])
            X_imputed[mask_col == 0, j] = median_val
    
    return torch.from_numpy(X_imputed).float()


def knn_imputation(X: torch.Tensor, missing_mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    KNN 插补：用 K 近邻均值填充缺失值
    
    Args:
        X: 原始特征 [N, D]
        missing_mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        k: 近邻数量
        
    Returns:
        X_imputed: 插补后的特征 [N, D]
    """
    X_np = X.numpy()
    mask_np = missing_mask.numpy()
    
    # 先用均值填充，KNNImputer 需要完整矩阵
    X_filled = X_np.copy()
    for j in range(X_np.shape[1]):
        col = X_np[:, j]
        mask_col = mask_np[:, j]
        if mask_col.sum() > 0:
            mean_val = col[mask_col == 1].mean()
            X_filled[mask_col == 0, j] = mean_val
    
    # 标记缺失位置（用 nan）
    X_for_knn = X_np.copy()
    X_for_knn[mask_np == 0] = np.nan
    
    # KNN 插补
    imputer = KNNImputer(n_neighbors=min(k, len(X_np)-1))
    X_imputed = imputer.fit_transform(X_for_knn)
    
    return torch.from_numpy(X_imputed).float()


def zero_imputation(X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """
    零值插补：用 0 填充缺失值（最简单 baseline）
    """
    X_imputed = X.clone()
    X_imputed[missing_mask == 0] = 0
    return X_imputed


def create_imputed_input(
    X: torch.Tensor,
    missing_rate: float,
    impute_method: str = "mean",
    mode: str = "mcar",
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建插补后的输入（用于传统 baseline）
    
    Args:
        X: 原始特征 [N, D]
        missing_rate: 缺失率
        impute_method: "mean", "median", "knn", "zero"
        mode: "mcar" 或 "mar"
        seed: 随机种子
        
    Returns:
        X_imputed: 插补后的特征 [N, D]
        y: 目标值（SOH）[N]
    """
    from .mcar import simulate_mcar
    from .mar import simulate_mar
    
    # 生成缺失
    if mode == "mar":
        # MAR 需要 SOH 值，这里假设 X 的最后一列是 SOH 的代理
        # 或者从外部传入 SOH，这里简化处理
        sohs = torch.ones(X.shape[0]) * 0.7  # 默认 SOH
        _, missing_mask, _ = simulate_mar(X, sohs, missing_rate, seed=seed)
    else:
        _, missing_mask, _ = simulate_mcar(X, missing_rate, seed=seed)
    
    # 应用插补
    if impute_method == "mean":
        X_imputed = mean_imputation(X, missing_mask)
    elif impute_method == "median":
        X_imputed = median_imputation(X, missing_mask)
    elif impute_method == "knn":
        X_imputed = knn_imputation(X, missing_mask)
    elif impute_method == "zero":
        X_imputed = zero_imputation(X, missing_mask)
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")
    
    return X_imputed, missing_mask
