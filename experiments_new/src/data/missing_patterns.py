"""
缺失数据生成模块
支持MCAR、MAR、MNAR三种缺失模式
"""
import numpy as np
from typing import Optional


def generate_mcar_mask(
    X: np.ndarray,
    missing_rate: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成MCAR（完全随机缺失）掩码
    
    参数:
        X: 数据矩阵 [n_samples, n_features]
        missing_rate: 缺失率
        seed: 随机种子
    
    返回:
        缺失掩码（1表示缺失，0表示观测）
    """
    rng = np.random.RandomState(seed)
    # 元素级独立随机缺失 (META §4.2)
    mask = rng.rand(*X.shape) < missing_rate
    return mask.astype(np.float32)


def generate_mar_mask(
    X: np.ndarray,
    missing_rate: float,
    seed: Optional[int] = None,
    dependent_features: Optional[list] = None
) -> np.ndarray:
    """
    生成MAR（随机缺失）掩码
    
    缺失概率依赖于其他观测特征的值（但缺失机制不依赖于缺失值本身）
    
    参数:
        X: 数据矩阵
        missing_rate: 缺失率
        seed: 随机种子
        dependent_features: 用于决定缺失的特征索引列表
    
    返回:
        缺失掩码（1表示缺失，0表示观测）
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    
    # 如果没有指定依赖特征，使用前一半特征
    if dependent_features is None:
        dependent_features = list(range(n_features // 2))
    
    mask = np.zeros_like(X, dtype=np.float32)
    
    # 对每个目标特征，基于依赖特征的值决定缺失概率
    # 所有特征都可能缺失，只是缺失概率由其他特征决定（MAR的定义）
    for target_idx in range(n_features):
        # 选择除当前特征外的其他特征作为依赖
        other_features = [f for f in range(n_features) if f != target_idx]
        if len(other_features) == 0:
            # 如果只有一个特征，退化为MCAR
            missing_prob = np.ones(n_samples) * missing_rate
        else:
            # 基于其他特征的均值决定缺失概率
            dep_values = X[:, other_features].mean(axis=1)
            
            # 归一化到[0, 1]范围
            dep_min, dep_max = dep_values.min(), dep_values.max()
            if dep_max > dep_min:
                dep_normalized = (dep_values - dep_min) / (dep_max - dep_min)
            else:
                dep_normalized = np.ones_like(dep_values) * 0.5
            
            # 缺失概率与依赖特征值相关
            # 值越大，缺失概率越高
            # 系数设计：使平均缺失概率接近目标missing_rate
            missing_prob = missing_rate * (0.2 + 1.6 * dep_normalized)
        
        # 生成缺失
        random_vals = rng.rand(n_samples)
        mask[:, target_idx] = (random_vals < missing_prob).astype(np.float32)
    
    return mask


def generate_mnar_mask(
    X: np.ndarray,
    missing_rate: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成MNAR（非随机缺失）掩码
    
    缺失概率依赖于缺失特征本身的值
    
    参数:
        X: 数据矩阵
        missing_rate: 缺失率
        seed: 随机种子
    
    返回:
        缺失掩码
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    
    mask = np.zeros_like(X, dtype=np.float32)
    
    # 对每个特征，基于其自身的值决定缺失概率
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # 归一化到[0, 1]范围
        f_min, f_max = feature_values.min(), feature_values.max()
        if f_max > f_min:
            f_normalized = (feature_values - f_min) / (f_max - f_min)
        else:
            f_normalized = np.ones_like(feature_values) * 0.5
        
        # 缺失概率与特征值本身相关
        # 例如：值越大，缺失概率越高（高值更容易缺失）
        # 系数设计：使平均缺失概率接近目标missing_rate
        missing_prob = missing_rate * (0.2 + 1.6 * f_normalized)
        
        # 生成缺失
        random_vals = rng.rand(n_samples)
        mask[:, feature_idx] = (random_vals < missing_prob).astype(np.float32)
    
    return mask


def generate_missing_mask(
    X: np.ndarray,
    missing_rate: float,
    mode: str = "MCAR",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    统一接口：生成缺失掩码
    
    参数:
        X: 数据矩阵
        missing_rate: 缺失率
        mode: 缺失模式 ("MCAR", "MAR", "MNAR")
        seed: 随机种子
    
    返回:
        缺失掩码（1表示缺失，0表示观测）
    """
    if mode.upper() == "MCAR":
        return generate_mcar_mask(X, missing_rate, seed)
    elif mode.upper() == "MAR":
        return generate_mar_mask(X, missing_rate, seed)
    elif mode.upper() == "MNAR":
        return generate_mnar_mask(X, missing_rate, seed)
    else:
        raise ValueError(f"未知的缺失模式: {mode}")


class MissingPatternGenerator:
    """
    缺失模式生成器
    
    用于测试阶段生成各种缺失条件的组合
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def generate_test_conditions(self) -> list:
        """
        生成所有测试条件组合
        
        返回:
            条件列表，每项为 (mode, missing_rate) 元组
        """
        modes = ["MCAR", "MAR", "MNAR"]
        missing_rates = [i * 0.05 for i in range(20)]  # 0.0, 0.05, ..., 0.95
        
        conditions = []
        for mode in modes:
            for mr in missing_rates:
                conditions.append((mode, mr))
        
        return conditions
    
    def apply_missing(
        self,
        X: np.ndarray,
        mode: str,
        missing_rate: float,
        offset: int = 0
    ) -> tuple:
        """
        对数据应用缺失
        
        参数:
            X: 原始数据
            mode: 缺失模式
            missing_rate: 缺失率
            offset: 种子偏移（用于不同模型/批次）
        
        返回:
            (X_with_missing, mask)
        """
        seed = self.seed + offset + hash(f"{mode}_{missing_rate}") % 10000
        mask = generate_missing_mask(X, missing_rate, mode, seed)
        X_missing = X.copy()
        X_missing[mask == 1] = np.nan
        return X_missing, mask
