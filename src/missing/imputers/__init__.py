"""
插补策略模块

提供多种缺失数据填充方法，均实现 IImputer 接口。

使用示例:
    >>> from src.missing.imputers import create_imputer
    >>> 
    >>> # 创建零值插补器
    >>> imputer = create_imputer("zero")
    >>> 
    >>> # 创建均值插补器
    >>> imputer = create_imputer("mean")
    >>> X_filled = imputer.fit(X_train).transform(X_missing)
"""

# 导入基类
from .base import BaseImputer

# 导入以触发注册
from .zero import ZeroImputer
from .mean import MeanImputer
from .knn import KNNImputer
from .iterative import IterativeImputer

# 导出便捷函数
from src.core import create_imputer, IMPUTERS

__all__ = [
    'BaseImputer',
    'ZeroImputer',
    'MeanImputer',
    'KNNImputer',
    'IterativeImputer',
    'create_imputer',
    'IMPUTERS',
]


def list_available_imputers() -> list[str]:
    """列出所有可用的插补方法"""
    return IMPUTERS.list_available()
