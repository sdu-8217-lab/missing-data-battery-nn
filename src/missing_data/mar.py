"""
MAR (Missing At Random) 缺失模拟 - 修正版

MAR 机制：缺失概率依赖于 SOH
修正公式: p_i = alpha * SOH_i^beta + gamma
"""

import numpy as np
import torch
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


def simulate_mar(
    X: torch.Tensor,
    sohs: torch.Tensor,
    missing_rate: float,
    alpha: Optional[float] = None,
    beta: float = 2.0,
    gamma: float = 0.05,
    seed: int = 42,
    impute_method: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 MAR 缺失模式（依赖 SOH）- 修正版
    
    修正说明:
        原公式: p_i = alpha * (1-SOH)^beta + gamma (beta<0)
        新公式: p_i = alpha * SOH^beta + gamma (beta>0)
        
        新公式更符合直觉：
        - SOH 高（新电池）→ 缺失率低
        - SOH 低（老电池）→ 缺失率高
    
    参数:
        X: 原始特征张量 [N, D]
        sohs: SOH 值 [N]，范围 [0, 1]
        missing_rate: 目标整体缺失率
        alpha: 缺失强度，None 则自动计算
        beta: 非线性参数 (>0)
        gamma: 基础缺失率
        seed: 随机种子
        impute_method: 插补方法 ('mean', 'median', 'zero')
        
    返回:
        X_imputed: 插补后的特征 [N, D]
        mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        mim_input: MIM 输入 [N, 2D]
    """
    torch.manual_seed(seed)
    N, D = X.shape
    
    # 确保 SOH 在 [0, 1] 范围内
    sohs = torch.clamp(sohs, 0.0, 1.0)
    
    # 计算 alpha
    if alpha is None:
        # 计算 E[SOH^beta]
        expected_term = (sohs ** beta).mean()
        
        # 处理除零情况
        if expected_term < 1e-10:
            alpha = 0.0
        else:
            # 反推 alpha: E[p] = alpha * E[SOH^beta] + gamma = missing_rate
            alpha = (missing_rate - gamma) / expected_term
            alpha = max(alpha, 0.0)  # 确保非负
            alpha = min(alpha, 1.0)  # 确保不超过1
    
    # 计算每个样本的缺失概率: p = alpha * SOH^beta + gamma
    p = alpha * (sohs ** beta) + gamma
    
    # 限制概率在 [0, 1] 范围内
    p = torch.clamp(p, 0.0, 1.0)
    
    # 为每个特征复制概率 [N, D]
    p_matrix = p.unsqueeze(1).expand(N, D)
    
    # 生成掩码: rand > p 则观测(1)，否则缺失(0)
    mask = (torch.rand(N, D, device=X.device) > p_matrix).float()
    
    # 插补
    X_np = X.cpu().numpy()
    mask_np = mask.cpu().numpy()
    X_imputed_np = np.zeros_like(X_np)
    
    for j in range(D):
        X_imputed_np[:, j] = _impute_column(X_np, mask_np, j, impute_method)
    
    X_imputed = torch.from_numpy(X_imputed_np).to(X.device, dtype=X.dtype)
    
    # 构造 MIM 输入
    mim_input = torch.cat([X_imputed, 1.0 - mask], dim=1)
    
    return X_imputed, mask, mim_input
