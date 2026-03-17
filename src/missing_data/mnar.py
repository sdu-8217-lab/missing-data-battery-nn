"""
MNAR (Missing Not At Random) 缺失模拟

MNAR 机制：缺失概率依赖于特征值本身（极端值更容易缺失）
用于模拟传感器在极端测量条件下失效的场景
"""

import torch
import numpy as np
from typing import Tuple, Optional


def _impute_column(
    X_np: np.ndarray,
    mask_np: np.ndarray,
    col_idx: int,
    impute_method: str = 'mean'
) -> np.ndarray:
    """对单特征列进行插补"""
    col = X_np[:, col_idx].copy()
    col_mask = mask_np[:, col_idx]
    
    observed = col[col_mask == 1]
    if len(observed) == 0:
        return col
    
    if impute_method == 'mean':
        fill_value = observed.mean()
    elif impute_method == 'median':
        fill_value = np.median(observed)
    elif impute_method == 'zero':
        fill_value = 0.0
    else:
        fill_value = observed.mean()
    
    col[col_mask == 0] = fill_value
    return col


def _compute_alpha_for_target_rate(
    normalized_features: torch.Tensor,
    target_missing_rate: float,
    beta: float,
    gamma: float
) -> float:
    """
    根据目标缺失率反推alpha参数
    
    E[p_missing] = E[alpha * (1 - x) + beta * x^gamma] ≈ target_missing_rate
    简化：假设x服从均匀分布，E[1-x] = 0.5
    alpha * 0.5 + beta * E[x^gamma] ≈ target_missing_rate
    """
    # 计算E[x^gamma]
    x_gamma_mean = (normalized_features ** gamma).mean().item()
    
    # 求解alpha
    # alpha * 0.5 ≈ target_missing_rate - beta * x_gamma_mean
    expected_term = 0.5  # E[1-x] for normalized features
    
    numerator = target_missing_rate - beta * x_gamma_mean
    if expected_term < 1e-10 or numerator < 0:
        return 0.0
    
    alpha = numerator / expected_term
    return min(alpha, 2.0)  # 限制alpha上限避免过度缺失


def simulate_mnar(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rate: float,
    alpha: Optional[float] = None,
    beta: float = 0.05,
    gamma: float = 1.0,
    feature_index: int = 0,
    seed: int = 42,
    impute_method: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 MNAR 缺失模式（基于特征值）
    
    缺失机制：极端特征值更容易缺失
    p_missing(i,j) = alpha * (1 - normalized_X[i,j]) + beta * normalized_X[i,j]^gamma
    
    其中：
    - alpha: 缺失强度系数（控制整体缺失率）
    - beta: 基础缺失率（所有样本都有）
    - gamma: 非线性参数（>1时极端值影响更大）
    - feature_index: 驱动缺失的特征列索引（默认第0列）
    
    物理意义：
    - 当特征值接近最小值（normalized→0）时：(1-0)=1，缺失概率≈alpha+beta
    - 当特征值接近最大值（normalized→1）时：(1-1)=0，缺失概率≈beta
    - 适用于：低电压/电流时传感器更容易故障的场景
    
    参数:
        X: 原始特征张量 [N, D]
        y: 目标值（SOH/Capacity）[N]，用于保持一致性接口，实际可能不用
        missing_rate: 目标整体缺失率 (0-1)
        alpha: 缺失强度系数，None则自动计算
        beta: 基础缺失率（所有样本）
        gamma: 非线性参数（默认1.0线性）
        feature_index: 驱动缺失的特征列索引
        seed: 随机种子
        impute_method: 插补方法 ('mean', 'median', 'zero')
        
    返回:
        X_imputed: 插补后的特征 [N, D]
        mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        mim_input: MIM 输入 [N, 2D]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N, D = X.shape
    
    # 获取驱动缺失的特征列
    feature_col = X[:, feature_index]
    
    # 归一化特征到[0, 1]（用于计算缺失概率）
    f_min = feature_col.min()
    f_max = feature_col.max()
    if f_max - f_min < 1e-10:
        # 特征值几乎恒定，使用MCAR退化为均匀缺失
        normalized_feature = torch.ones_like(feature_col) * 0.5
    else:
        normalized_feature = (feature_col - f_min) / (f_max - f_min)
    
    # 计算alpha（如果需要）
    if alpha is None:
        alpha = _compute_alpha_for_target_rate(
            normalized_feature, missing_rate, beta, gamma
        )
    
    # 计算每个样本的缺失概率
    # p = alpha * (1 - x) + beta * x^gamma
    p_missing = alpha * (1 - normalized_feature) + beta * (normalized_feature ** gamma)
    
    # 限制概率在[0, 1]范围内
    p_missing = torch.clamp(p_missing, 0.0, 1.0)
    
    # 为每个特征复制概率 [N, D]（所有特征列使用相同的样本级缺失概率）
    p_matrix = p_missing.unsqueeze(1).expand(N, D)
    
    # 生成掩码: rand > p则观测(1)，否则缺失(0)
    mask = (torch.rand(N, D, device=X.device) > p_matrix).float()
    
    # 插补
    X_np = X.cpu().numpy()
    mask_np = mask.cpu().numpy()
    X_imputed_np = np.zeros_like(X_np)
    
    for j in range(D):
        X_imputed_np[:, j] = _impute_column(X_np, mask_np, j, impute_method)
    
    X_imputed = torch.from_numpy(X_imputed_np).to(X.device, dtype=X.dtype)
    
    # 构造MIM输入 [X_imputed, mask_indicator]
    mask_indicator = 1.0 - mask  # 1表示缺失，0表示观测
    mim_input = torch.cat([X_imputed, mask_indicator], dim=1)
    
    return X_imputed, mask, mim_input


def simulate_mnar_by_soh(
    X: torch.Tensor,
    y: torch.Tensor,
    missing_rate: float,
    alpha: Optional[float] = None,
    beta: float = 0.05,
    seed: int = 42,
    impute_method: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MNAR变体：基于SOH的极端值缺失（老化电池极端测量）
    
    适用于：老化电池（低SOH）在极端工况下更容易数据丢失
    
    参数:
        X: 原始特征 [N, D]
        y: SOH值 [N]（作为缺失驱动因素）
        missing_rate: 目标缺失率
        alpha: 缺失强度
        beta: 基础缺失率
        seed: 随机种子
        impute_method: 插补方法
        
    返回:
        X_imputed, mask, mim_input
    """
    torch.manual_seed(seed)
    N, D = X.shape
    
    # 归一化SOH到[0, 1]
    soh_min = y.min()
    soh_max = y.max()
    if soh_max - soh_min < 1e-10:
        normalized_soh = torch.ones_like(y) * 0.5
    else:
        normalized_soh = (y - soh_min) / (soh_max - soh_min)
    
    # 低SOH样本更容易缺失（老化电池）
    # p = alpha * (1 - SOH) + beta
    if alpha is None:
        # 反推alpha: E[p] = alpha * E[1-SOH] + beta = missing_rate
        expected_1_minus_soh = (1 - normalized_soh).mean().item()
        if expected_1_minus_soh > 1e-10:
            alpha = (missing_rate - beta) / expected_1_minus_soh
            alpha = max(0.0, min(alpha, 2.0))
        else:
            alpha = 0.0
    
    p_missing = alpha * (1 - normalized_soh) + beta
    p_missing = torch.clamp(p_missing, 0.0, 1.0)
    
    # 扩展到所有特征列
    p_matrix = p_missing.unsqueeze(1).expand(N, D)
    mask = (torch.rand(N, D, device=X.device) > p_matrix).float()
    
    # 插补
    X_np = X.cpu().numpy()
    mask_np = mask.cpu().numpy()
    X_imputed_np = np.zeros_like(X_np)
    
    for j in range(D):
        X_imputed_np[:, j] = _impute_column(X_np, mask_np, j, impute_method)
    
    X_imputed = torch.from_numpy(X_imputed_np).to(X.device, dtype=X.dtype)
    mim_input = torch.cat([X_imputed, 1.0 - mask], dim=1)
    
    return X_imputed, mask, mim_input
