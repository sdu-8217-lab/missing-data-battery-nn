"""
插补器基类

提供通用的 fit_transform 实现。
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseImputer(ABC):
    """
    插补器抽象基类
    
    所有具体插补器应继承此类，只需实现 fit 和 transform。
    fit_transform 由基类提供。
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseImputer":
        """拟合插补器"""
        ...
    
    @abstractmethod
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """执行插补"""
        ...
    
    def fit_transform(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        便捷方法：先 fit 再 transform
        
        Args:
            X: 数据（如果 mask 为 None，则 X 应包含 np.nan）
            mask: 缺失掩码（可选）
            
        Returns:
            X_imputed: 插补后的数据
        """
        if mask is not None:
            X_missing = X.copy()
            X_missing[mask] = np.nan
        else:
            X_missing = X
        return self.fit(X).transform(X_missing)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插补方法名称"""
        ...
