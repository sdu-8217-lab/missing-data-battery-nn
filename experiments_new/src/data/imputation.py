"""
插补方法模块
支持多种插补策略：zero, mean, knn, iterative
"""
import numpy as np
from typing import Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Imputer:
    """
    插补器
    
    支持方法:
    - zero: 零填充
    - mean: 均值填充
    - knn: K近邻填充
    - iterative: 迭代填充（MICE）
    """
    
    def __init__(self, method: str = "zero"):
        """
        参数:
            method: 插补方法 ("zero", "mean", "knn", "iterative")
        """
        self.method = method
        self.imputer = None
        self.fitted = False
    
    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        拟合插补器
        
        参数:
            X: 完整数据（用于学习统计信息）
            mask: 缺失掩码（1表示缺失）
        """
        if self.method == "zero":
            # zero不需要拟合
            self.fitted = True
            return self
        
        elif self.method == "mean":
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(X)
        
        elif self.method == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
            self.imputer.fit(X)
        
        elif self.method == "iterative":
            self.imputer = IterativeImputer(max_iter=10, random_state=42)
            self.imputer.fit(X)
        
        else:
            raise ValueError(f"未知的插补方法: {self.method}")
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        对缺失数据进行插补
        
        参数:
            X: 原始数据（可能包含缺失值）
            mask: 缺失掩码（1表示缺失，0表示观测）
        
        返回:
            插补后的数据
        """
        if not self.fitted:
            raise RuntimeError("插补器尚未拟合")
        
        if self.method == "zero":
            # 缺失位置填0
            return np.where(mask == 0, X, 0.0)
        
        else:
            # 使用sklearn插补器
            # 先将缺失位置设为nan
            X_missing = X.copy()
            X_missing[mask == 1] = np.nan
            
            # 插补
            X_imputed = self.imputer.transform(X_missing)
            
            return X_imputed
    
    def fit_transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        return self.fit(X).transform(X, mask)


class MIMDatasetWithImputation:
    """
    MIM数据集（带插补）
    
    流程:
    1. 生成MCAR缺失
    2. 应用插补方法
    3. 拼接缺失指示器
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_rates: list,
        imputation_method: str = "zero",
        use_mim: bool = True,
        base_seed: int = 42,
        fit_imputer: bool = True
    ):
        """
        参数:
            X: 特征数组
            y: 标签数组
            missing_rates: 缺失率列表
            imputation_method: 插补方法
            use_mim: 是否使用MIM
            base_seed: 基础随机种子
            fit_imputer: 是否使用训练数据拟合插补器
        """
        self.use_mim = use_mim
        self.imputation_method = imputation_method
        
        # 创建插补器
        imputer = Imputer(method=imputation_method)
        
        if fit_imputer and imputation_method != "zero":
            # 使用完整数据拟合插补器
            imputer.fit(X)
        else:
            imputer.fitted = True  # zero方法不需要拟合
        
        # 为每个缺失率生成数据
        X_expanded = []
        y_expanded = []
        
        for i, mr in enumerate(missing_rates):
            # 使用不同种子确保不同MR的缺失模式不同
            rng = np.random.RandomState(base_seed + i * 100)
            
            # 生成MCAR缺失掩码
            mask = rng.rand(*X.shape) < mr  # True表示缺失
            
            # 应用插补
            X_imputed = imputer.transform(X, mask)
            
            if use_mim:
                # 拼接特征和缺失指示器 [n_samples, features*2]
                missing_indicators = mask.astype(np.float32)
                X_combined = np.concatenate([X_imputed, missing_indicators], axis=1)
            else:
                X_combined = X_imputed
            
            X_expanded.append(X_combined)
            y_expanded.append(y)
        
        self.X = np.vstack(X_expanded)
        self.y = np.concatenate(y_expanded)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        import torch
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


def apply_imputation(
    X: np.ndarray,
    mask: np.ndarray,
    method: str = "zero",
    imputer: Optional[Imputer] = None
) -> np.ndarray:
    """
    对缺失数据应用插补
    
    参数:
        X: 原始数据
        mask: 缺失掩码（1表示缺失）
        method: 插补方法
        imputer: 预拟合的插补器（可选）
    
    返回:
        插补后的数据
    """
    if imputer is None:
        imputer = Imputer(method=method)
        imputer.fit(X)
    
    return imputer.transform(X, mask)
