"""
KNN 插补器

使用 K-Nearest Neighbors 算法填充缺失值。
"""

import numpy as np
from src.core import IMPUTERS
from .base import BaseImputer

# 可选依赖 sklearn
try:
    from sklearn.impute import KNNImputer as SklearnKNNImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    SklearnKNNImputer = None


@IMPUTERS.register("knn")
class KNNImputer(BaseImputer):
    """
    K-近邻插补器
    
    使用样本的 K 个最近邻的加权均值填充缺失值。
    基于 scikit-learn 的 KNNImputer。
    
    Args:
        n_neighbors: 近邻数量，默认 5
        weights: 权重方式，'uniform' 或 'distance'
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        if not HAS_SKLEARN:
            raise ImportError(
                "KNNImputer requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = None
    
    def fit(self, X: np.ndarray) -> "KNNImputer":
        """
        拟合 KNN 插补器
        
        Args:
            X: 训练数据（用于学习样本间的距离关系）
            
        Returns:
            self
        """
        self._imputer = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """
        使用 KNN 填充缺失值
        
        Args:
            X_missing: 带缺失的数据
            
        Returns:
            X_imputed: 填充后的数据
        """
        if self._imputer is None:
            raise RuntimeError("Must call fit() before transform()")
        
        return self._imputer.transform(X_missing)
    
    @property
    def name(self) -> str:
        return "knn"
    
    def __repr__(self) -> str:
        return f"KNNImputer(n_neighbors={self.n_neighbors}, weights='{self.weights}')"
