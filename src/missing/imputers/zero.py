"""
零值插补器

最简单的插补策略：用 0 填充所有缺失值。
适用于已标准化的数据（均值为0的情况）。
"""

import numpy as np
from src.core import IMPUTERS
from .base import BaseImputer


@IMPUTERS.register("zero")
class ZeroImputer(BaseImputer):
    """
    零值填充插补器
    
    对标准化后的数据效果较好（均值为0）。
    不需要 fit，因为填充值固定为0。
    """
    
    def __init__(self):
        pass
    
    def fit(self, X: np.ndarray) -> "ZeroImputer":
        """
        零值插补不需要学习参数
        
        Args:
            X: 训练数据（忽略）
            
        Returns:
            self
        """
        # 零值插补不需要任何学习
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """
        用 0 填充缺失值
        
        Args:
            X_missing: 带缺失的数据 (np.nan 表示缺失)
            
        Returns:
            X_imputed: 填充后的数据
        """
        X_imputed = X_missing.copy()
        X_imputed = np.nan_to_num(X_imputed, nan=0.0)
        return X_imputed
    
    @property
    def name(self) -> str:
        return "zero"
    
    def __repr__(self) -> str:
        return f"ZeroImputer()"
