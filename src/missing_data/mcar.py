"""
MCAR (Missing Completely At Random) 缺失模拟

MCAR 机制：缺失完全随机，与观测值和未观测值均无关
"""

import torch
from typing import Tuple


def simulate_mcar(
    X: torch.Tensor,
    missing_rate: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 MCAR 缺失模式
    
    参数:
        X: 原始特征张量 [N, D]
        missing_rate: 缺失率 (0-1)
        seed: 随机种子
        
    返回:
        X_imputed: 均值插补后的特征 [N, D]
        mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        mim_input: MIM 输入 [N, 2D] = [X_imputed, 1-mask]
    """
    torch.manual_seed(seed)
    N, D = X.shape
    
    # 生成掩码
    mask = (torch.rand(N, D, device=X.device) > missing_rate).float()
    
    # 均值插补
    X_imputed = X.clone()
    for j in range(D):
        feature = X[:, j]
        mask_j = mask[:, j]
        if mask_j.sum() > 0:
            mean_val = feature[mask_j == 1].mean()
            X_imputed[mask_j == 0, j] = mean_val
    
    # MIM 输入
    mim_input = torch.cat([X_imputed, 1.0 - mask], dim=1)
    
    return X_imputed, mask, mim_input
