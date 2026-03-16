"""
MCAR (Missing Completely At Random) 缺失模拟

MCAR 机制：缺失完全随机，与观测值和未观测值均无关
"""

import torch
import numpy as np
from typing import Tuple


def _impute_column(X_np, mask_np, col_idx, method='mean'):
    """对单列进行插补"""
    col = X_np[:, col_idx].copy()
    mask_col = mask_np[:, col_idx]
    
    if mask_col.sum() == 0:
        return col  # 全缺失，保持原样
    
    observed = col[mask_col == 1]
    
    if method == 'mean':
        fill_val = observed.mean()
    elif method == 'median':
        fill_val = np.median(observed)
    elif method == 'zero':
        fill_val = 0.0
    else:
        fill_val = observed.mean()  # 默认均值
    
    col[mask_col == 0] = fill_val
    return col


def simulate_mcar(
    X: torch.Tensor,
    missing_rate: float,
    seed: int,
    impute_method: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 MCAR 缺失模式
    
    参数:
        X: 原始特征张量 [N, D]
        missing_rate: 缺失率 (0-1)
        seed: 随机种子
        impute_method: 插补方法 ('mean', 'median', 'zero')
        
    返回:
        X_imputed: 插补后的特征 [N, D]
        mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        mim_input: MIM 输入 [N, 2D] = [X_imputed, 1-mask]
    """
    torch.manual_seed(seed)
    N, D = X.shape
    
    # 生成掩码
    mask = (torch.rand(N, D, device=X.device) > missing_rate).float()
    
    # 转换为numpy进行插补
    X_np = X.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    # 按指定方法插补
    X_imputed_np = np.zeros_like(X_np)
    for j in range(D):
        X_imputed_np[:, j] = _impute_column(X_np, mask_np, j, impute_method)
    
    # 转回torch
    X_imputed = torch.from_numpy(X_imputed_np).to(X.device, dtype=X.dtype)
    
    # MIM 输入
    mim_input = torch.cat([X_imputed, 1.0 - mask], dim=1)
    
    return X_imputed, mask, mim_input
