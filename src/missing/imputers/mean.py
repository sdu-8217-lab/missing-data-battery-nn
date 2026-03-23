"""
均值插补器

使用训练集各特征的均值填充缺失值。
"""

import numpy as np
from src.core import IMPUTERS
from .base import BaseImputer


@IMPUTERS.register("mean")
class MeanImputer(BaseImputer):
    """
    均值填充插补器
    
    对每个特征，使用该特征在训练集中的均值填充缺失值。
    需要 fit 阶段计算各特征的均值。
    """
    
    def __init__(self):
        self._means = None
        self._n_features = None
    
    def fit(self, X: np.ndarray) -> "MeanImputer":
        """
        计算训练集各特征的均值
        
        Args:
            X: 训练数据 [n_samples, n_features]
            
        Returns:
            self
        """
        # 沿样本轴计算均值，得到每个特征的均值
        self._means = np.nanmean(X, axis=0)
        self._n_features = X.shape[1]
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """
        用训练集均值填充缺失值
        
        Args:
            X_missing: 带缺失的数据 [n_samples, n_features]
            
        Returns:
            X_imputed: 填充后的数据
            
        Raises:
            RuntimeError: 如果先调用 transform 而没有先调用 fit
        """
        if self._means is None:
            raise RuntimeError("Must call fit() before transform()")
        
        X_imputed = X_missing.copy()
        
        # 对每个特征，用对应的均值填充
        for i in range(X_missing.shape[1]):
            mask = np.isnan(X_missing[:, i])
            if mask.any():
                X_imputed[mask, i] = self._means[i]
        
        return X_imputed
    
    @property
    def name(self) -> str:
        return "mean"
    
    @property
    def means(self) -> np.ndarray:
        """获取学习到的均值（用于调试）"""
        return self._means
    
    def __repr__(self) -> str:
        if self._means is not None:
            return f"MeanImputer(means={self._means[:3]}...)"  # 只显示前3个
        return f"MeanImputer(not fitted)"
