"""
迭代插补器（MICE）

使用迭代方法填充缺失值，每次迭代将每个特征作为目标，
其他特征作为预测器训练模型预测缺失值。
"""

import numpy as np
import warnings
from src.core import IMPUTERS
from .base import BaseImputer

# 可选依赖 sklearn
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer as SklearnIterativeImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    SklearnIterativeImputer = None


@IMPUTERS.register("iterative")
class IterativeImputer(BaseImputer):
    """
    迭代插补器（MICE - Multivariate Imputation by Chained Equations）
    
    通过迭代建模每个特征与其他特征的关系来填充缺失值。
    比简单方法更精确，但计算成本更高。
    
    Args:
        max_iter: 最大迭代次数，默认 10
        random_state: 随机种子，用于可复现性
        initial_strategy: 初始填充策略，'mean' 或 'median'
    """
    
    def __init__(
        self, 
        max_iter: int = 10,
        random_state: int = 42,
        initial_strategy: str = "mean"
    ):
        if not HAS_SKLEARN:
            raise ImportError(
                "IterativeImputer requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
        
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self._imputer = None
    
    def fit(self, X: np.ndarray) -> "IterativeImputer":
        """
        拟合迭代插补器
        
        Args:
            X: 训练数据
            
        Returns:
            self
        """
        # 忽略 sklearn 的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self._imputer = SklearnIterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                initial_strategy=self.initial_strategy,
                sample_posterior=False,  # 确定性结果
                skip_complete=True  # 跳过完整特征
            )
            self._imputer.fit(X)
        
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """
        使用迭代方法填充缺失值
        
        Args:
            X_missing: 带缺失的数据
            
        Returns:
            X_imputed: 填充后的数据
        """
        if self._imputer is None:
            raise RuntimeError("Must call fit() before transform()")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._imputer.transform(X_missing)
    
    @property
    def name(self) -> str:
        return "iterative"
    
    @property
    def n_iter_(self) -> int:
        """实际迭代次数（用于调试）"""
        if self._imputer is not None:
            return self._imputer.n_iter_
        return 0
    
    def __repr__(self) -> str:
        return (
            f"IterativeImputer("
            f"max_iter={self.max_iter}, "
            f"random_state={self.random_state})"
        )
