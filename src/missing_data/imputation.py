"""
传统插补方法 (Traditional Imputation Baselines)
"""
import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.impute import KNNImputer, SimpleImputer

# IterativeImputer 在 sklearn.experimental 中
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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


def iterative_imputation(
    X: torch.Tensor,
    missing_mask: torch.Tensor,
    estimator: str = 'bayesian_ridge',
    max_iter: int = 10,
    random_state: int = 42
) -> torch.Tensor:
    """
    迭代插补 (MICE-style)：使用迭代回归预测缺失值
    
    每轮迭代中，将每个含缺失特征作为目标变量，
    利用其他特征训练回归模型进行预测，多轮迭代直至收敛。
    
    这是经典的迭代插补方法，与MIM的端到端神经网络形成对比：
    - Iterative: 显式建模特征间条件依赖，迭代优化
    - MIM: 隐式学习缺失模式，端到端训练
    
    Args:
        X: 原始特征 [N, D]
        missing_mask: 缺失掩码 [N, D], 1=观测, 0=缺失
        estimator: 回归器类型 ('bayesian_ridge' 或 'random_forest')
        max_iter: 最大迭代次数
        random_state: 随机种子
        
    Returns:
        X_imputed: 插补后的特征 [N, D]
    """
    X_np = X.numpy()
    mask_np = missing_mask.numpy()
    
    # 标记缺失位置（用nan）
    X_for_imputer = X_np.copy()
    X_for_imputer[mask_np == 0] = np.nan
    
    # 选择回归器
    if estimator == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        estimator_obj = RandomForestRegressor(
            n_estimators=10,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    else:  # bayesian_ridge (default)
        from sklearn.linear_model import BayesianRidge
        estimator_obj = BayesianRidge()
    
    # 迭代插补
    imputer = IterativeImputer(
        estimator=estimator_obj,
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=False,  # 确定性插补（非随机采样）
        verbose=0
    )
    
    X_imputed = imputer.fit_transform(X_for_imputer)
    
    return torch.from_numpy(X_imputed).float()


def forward_fill_imputation(X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """
    前向填充插补：时序数据标准方法
    用前一个观测值填充当前缺失值
    """
    X_np = X.numpy()
    mask_np = missing_mask.numpy()
    X_filled = X_np.copy()
    N, D = X_np.shape
    
    for j in range(D):
        last_observed = None
        for i in range(N):
            if mask_np[i, j] == 1:  # 观测到
                last_observed = X_np[i, j]
            elif last_observed is not None:  # 缺失，但有前值
                X_filled[i, j] = last_observed
            else:  # 开头缺失，找后值
                for k in range(i+1, N):
                    if mask_np[k, j] == 1:
                        X_filled[i, j] = X_np[k, j]
                        break
                else:
                    X_filled[i, j] = 0  # 整列缺失
    
    return torch.from_numpy(X_filled).float()


def create_imputed_input(
    X: torch.Tensor,
    missing_rate: float,
    impute_method: str = "mean",
    mode: str = "mcar",
    seed: int = 42,
    y: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建插补后的输入（用于传统 baseline）
    
    Args:
        X: 原始特征 [N, D]
        missing_rate: 缺失率
        impute_method: "mean", "median", "knn", "zero", "forward_fill", "iterative"
        mode: "mcar", "mar", "mnar"
        seed: 随机种子
        y: 目标值SOH [N]，用于MAR/MNAR模式
        **kwargs: 额外参数（如iterative的estimator, max_iter）
        
    Returns:
        X_imputed: 插补后的特征 [N, D]
        missing_mask: 缺失掩码
    """
    from .mcar import simulate_mcar
    from .mar import simulate_mar
    from .mnar import simulate_mnar
    
    # 生成缺失
    if mode == "mar":
        if y is None:
            y = torch.ones(X.shape[0]) * 0.7
        _, missing_mask, _ = simulate_mar(X, y, missing_rate, seed=seed)
    elif mode == "mnar":
        if y is None:
            y = torch.ones(X.shape[0]) * 0.7
        _, missing_mask, _ = simulate_mnar(X, y, missing_rate, seed=seed)
    else:  # mcar
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
    elif impute_method == "forward_fill":
        X_imputed = forward_fill_imputation(X, missing_mask)
    elif impute_method == "iterative":
        estimator = kwargs.get('estimator', 'bayesian_ridge')
        max_iter = kwargs.get('max_iter', 10)
        X_imputed = iterative_imputation(
            X, missing_mask,
            estimator=estimator,
            max_iter=max_iter,
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown impute method: {impute_method}")
    
    return X_imputed, missing_mask
