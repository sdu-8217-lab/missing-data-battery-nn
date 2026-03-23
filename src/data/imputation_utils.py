"""
插补工具 - 支持多种插补方法

用于训练和测试阶段的缺失值填充
"""

import numpy as np
from typing import Literal, Optional
import warnings

# 可选依赖
try:
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def impute_missing_values(
    X_missing: np.ndarray,
    method: Literal['mean', 'knn', 'iterative', 'zero'],
    train_stats: Optional[dict] = None,
    seed: int = 42
) -> np.ndarray:
    """
    对缺失数据进行插补
    
    Args:
        X_missing: 带缺失值的数据 (np.nan 表示缺失)
        method: 插补方法
            - 'mean': 均值插补（使用训练集统计量）
            - 'knn': K近邻插补
            - 'iterative': 迭代插补（MICE）
            - 'zero': 零值填充
        train_stats: 训练集统计量（用于 mean 方法）
        seed: 随机种子（用于 iterative 方法）
        
    Returns:
        插补后的数据
        
    Example:
        >>> X = np.array([[1, np.nan], [2, 3], [np.nan, 4]])
        >>> X_imp = impute_missing_values(X, 'mean')
    """
    if method == 'zero':
        return np.nan_to_num(X_missing, nan=0.0)
    
    elif method == 'mean':
        if train_stats is not None and 'mean' in train_stats:
            # 使用训练集均值
            mean_vals = train_stats['mean']
            X_imputed = X_missing.copy()
            mask = np.isnan(X_missing)
            for i in range(X_missing.shape[1]):
                X_imputed[mask[:, i], i] = mean_vals[i]
            return X_imputed
        else:
            # 使用当前数据的均值
            imputer = SimpleImputer(strategy='mean')
            return imputer.fit_transform(X_missing)
    
    elif method == 'knn':
        if not HAS_SKLEARN:
            raise ImportError("需要 scikit-learn 库。安装: pip install scikit-learn")
        # KNN 插补
        imputer = KNNImputer(n_neighbors=5)
        return imputer.fit_transform(X_missing)
    
    elif method == 'iterative':
        if not HAS_SKLEARN:
            raise ImportError("需要 scikit-learn 库。安装: pip install scikit-learn")
        # 迭代插补（MICE）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = IterativeImputer(max_iter=10, random_state=seed)
            return imputer.fit_transform(X_missing)
    
    else:
        raise ValueError(f"未知的插补方法: {method}")


def compute_train_statistics(X_train: np.ndarray) -> dict:
    """
    计算训练集统计量，用于后续的插补
    
    Args:
        X_train: 训练数据（完整数据，无缺失）
        
    Returns:
        包含 mean, std 等统计量的字典
    """
    return {
        'mean': X_train.mean(axis=0),
        'std': X_train.std(axis=0),
        'median': np.median(X_train, axis=0),
        'min': X_train.min(axis=0),
        'max': X_train.max(axis=0),
    }


def generate_mcar_missing(X: np.ndarray, missing_rate: float, seed: int) -> tuple:
    """
    生成MCAR缺失数据
    
    Args:
        X: 原始数据
        missing_rate: 缺失率
        seed: 随机种子
        
    Returns:
        (X_missing, mask) - 缺失数据和掩码
    """
    np.random.seed(seed)
    mask = np.random.rand(*X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def prepare_mim_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_stats: dict,
    imputation_method: Literal['mean', 'knn', 'iterative', 'zero'],
    missing_rates: list,
    seed: int,
    model_type: str = 'mlp'
) -> list:
    """
    为MIM训练准备多MR数据（使用指定插补方法）
    
    Args:
        X_train: 标准化后的训练数据
        y_train: 训练标签
        train_stats: 训练集统计量
        imputation_method: 插补方法
        missing_rates: 缺失率列表
        seed: 随机种子
        model_type: 模型类型
        
    Returns:
        [(X_combined, y), ...] 列表
    """
    import torch
    
    multi_mr_data = []
    
    for mr in missing_rates:
        if mr == 0.0:
            # MR=0时使用完整数据
            X_mr = X_train.copy()
            mask = np.zeros_like(X_train)
        else:
            # 生成缺失数据
            X_missing, mask = generate_mcar_missing(X_train, mr, seed + int(mr * 100))
            # 使用指定插补方法填充
            X_mr = impute_missing_values(X_missing, imputation_method, train_stats, seed)
        
        # 转换为tensor并拼接掩码
        mask_tensor = torch.FloatTensor(mask.astype(float))
        X_mr_tensor = torch.FloatTensor(X_mr)
        X_combined = torch.cat([X_mr_tensor, mask_tensor], dim=1)
        
        # 对于CNN/LSTM模型，添加序列维度
        if model_type in ['cnn', 'lstm']:
            X_combined = X_combined.unsqueeze(1)
        
        multi_mr_data.append((X_combined, torch.FloatTensor(y_train)))
    
    return multi_mr_data


def prepare_validation_data(
    X_val: np.ndarray,
    train_stats: dict,
    imputation_method: Literal['mean', 'knn', 'iterative', 'zero'],
    missing_rates: list,
    seed: int,
    model_type: str = 'mlp'
) -> list:
    """
    为验证准备多MR数据
    
    Returns:
        [(X_val_combined, mr), ...] 列表
    """
    import torch
    
    val_data_list = []
    
    for mr in missing_rates:
        if mr == 0.0:
            X_val_mr = X_val.copy()
            mask_val = np.zeros_like(X_val)
        else:
            np.random.seed(seed + 999 + int(mr * 100))
            mask = np.random.rand(*X_val.shape) < mr
            X_val_missing = X_val.copy()
            X_val_missing[mask] = np.nan
            X_val_mr = impute_missing_values(X_val_missing, imputation_method, train_stats, seed)
            mask_val = mask.astype(float)
        
        # 转换为tensor并拼接
        mask_tensor = torch.FloatTensor(mask_val)
        X_val_tensor = torch.FloatTensor(X_val_mr)
        X_combined = torch.cat([X_val_tensor, mask_tensor], dim=1)
        
        if model_type in ['cnn', 'lstm']:
            X_combined = X_combined.unsqueeze(1)
        
        val_data_list.append((X_combined, mr))
    
    return val_data_list
