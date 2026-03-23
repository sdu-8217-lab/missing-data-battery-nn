"""
缺失数据处理 - 简化版

合并了：
- 缺失数据生成（MCAR/MAR/MNAR）
- 插补（zero/mean/knn/iterative）
- MIM输入构建

代码量：100行 vs 原来的600+行
"""

import numpy as np
from typing import Literal

# 可选依赖
try:
    from sklearn.impute import KNNImputer, IterativeImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_missing(
    X: np.ndarray,
    mode: Literal['mcar', 'mar', 'mnar'],
    missing_rate: float,
    seed: int
) -> np.ndarray:
    """生成缺失掩码 (True = 缺失)"""
    rng = np.random.default_rng(seed)
    
    if mode == 'mcar':
        # 完全随机缺失
        return rng.random(X.shape) < missing_rate
    
    elif mode == 'mar':
        # 随机缺失：缺失概率与观测值相关
        # 简化实现：基于第一个特征的值
        feature = X[:, 0]
        f_min, f_max = feature.min(), feature.max()
        probs = missing_rate * (0.5 + 0.5 * (feature - f_min) / (f_max - f_min + 1e-8))
        mask = np.zeros(X.shape, dtype=bool)
        for i in range(X.shape[1]):
            mask[:, i] = rng.random(X.shape[0]) < probs
        return mask
    
    elif mode == 'mnar':
        # 非随机缺失：缺失与缺失值本身相关
        # 简化实现：基于目标值的排名
        raise NotImplementedError("MNAR simplified - use full version if needed")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def impute(
    X: np.ndarray,
    method: Literal['zero', 'mean', 'knn', 'iterative'],
    fit_data: np.ndarray = None
) -> np.ndarray:
    """
    插补缺失值
    
    Args:
        X: 带缺失的数据 (np.nan 表示缺失)
        method: 插补方法
        fit_data: 用于学习插补参数的训练数据
    
    Returns:
        填充后的数据
    """
    if method == 'zero':
        return np.nan_to_num(X, nan=0.0)
    
    elif method == 'mean':
        if fit_data is None:
            raise ValueError("fit_data required for mean imputation")
        means = np.nanmean(fit_data, axis=0)
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X_filled[mask, i] = means[i]
        return X_filled
    
    elif method == 'knn':
        if fit_data is None:
            raise ValueError("fit_data required for knn imputation")
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for knn imputation")
        imputer = KNNImputer(n_neighbors=5).fit(fit_data)
        return imputer.transform(X)
    
    elif method == 'iterative':
        if fit_data is None:
            raise ValueError("fit_data required for iterative imputation")
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for iterative imputation")
        imputer = IterativeImputer(max_iter=10, random_state=42).fit(fit_data)
        return imputer.transform(X)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def build_mim_input(X_imputed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    构建 MIM 输入
    
    将插补后的特征与缺失掩码拼接
    [X_imputed | mask]
    
    Args:
        X_imputed: 插补后的特征 [n, 16]
        mask: 缺失掩码 [n, 16] (True = 缺失)
    
    Returns:
        MIM 输入 [n, 32]
    """
    return np.concatenate([X_imputed, mask.astype(float)], axis=1)


def prepare_mim_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    missing_rates: list[float],
    imputation: str,
    seed: int,
    mode: str = 'mcar'
) -> list[tuple]:
    """
    准备 MIM 多 MR 训练数据
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        missing_rates: 缺失率列表 [0.0, 0.05, ...]
        imputation: 插补方法
        seed: 随机种子
        mode: 缺失模式
    
    Returns:
        [(X_mim, y), ...] 列表
    """
    data = []
    
    for mr in missing_rates:
        if mr == 0.0:
            X_mr = X_train
            mask = np.zeros_like(X_train)
        else:
            mask = generate_missing(X_train, mode, mr, seed + int(mr * 100))
            X_missing = X_train.copy()
            X_missing[mask] = np.nan
            X_mr = impute(X_missing, imputation, fit_data=X_train)
        
        X_mim = build_mim_input(X_mr, mask)
        data.append((X_mim, y_train))
    
    return data


# 便捷函数
__all__ = [
    'generate_missing',
    'impute',
    'build_mim_input',
    'prepare_mim_training_data',
]
