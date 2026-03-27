"""
PyTorch Dataset定义
支持MIM和Baseline两种模式
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List


class BatteryDataset(Dataset):
    """
    基础电池数据集（非序列模型）
    
    参数:
        X: 特征数组 [n_samples, n_features]
        y: 标签数组 [n_samples]
        missing_rate: 缺失率
        use_mim: 是否使用缺失指示器
        seed: 随机种子
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_rate: float = 0.0,
        use_mim: bool = False,
        seed: Optional[int] = None
    ):
        self.X = X
        self.y = y
        self.missing_rate = missing_rate
        self.use_mim = use_mim
        
        # 生成缺失掩码
        if seed is not None:
            np.random.seed(seed)
        self.mask = np.random.rand(*X.shape) > missing_rate
        
        # 应用缺失（缺失位置置0）
        self.X_missing = np.where(self.mask, X, 0.0)
        
        # 缺失指示器（1表示缺失）
        self.missing_indicators = (~self.mask).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        x = self.X_missing[idx]
        
        if self.use_mim:
            # 拼接特征和缺失指示器 [features + indicators]
            x = np.concatenate([x, self.missing_indicators[idx]])
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class SequenceDataset(Dataset):
    """
    序列电池数据集（用于LSTM/GRU/CNN）
    
    参数:
        X: 特征数组 [n_samples, n_features]
        y: 标签数组 [n_samples]
        seq_len: 序列长度
        missing_rate: 缺失率
        use_mim: 是否使用缺失指示器
        seed: 随机种子
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 5,
        missing_rate: float = 0.0,
        use_mim: bool = False,
        seed: Optional[int] = None
    ):
        self.seq_len = seq_len
        self.use_mim = use_mim
        
        # 生成缺失掩码
        if seed is not None:
            np.random.seed(seed)
        mask = np.random.rand(*X.shape) > missing_rate
        
        # 应用缺失
        X_missing = np.where(mask, X, 0.0)
        missing_indicators = (~mask).astype(np.float32)
        
        # 构造序列
        sequences = []
        labels = []
        
        for i in range(len(X) - seq_len + 1):
            seq_x = X_missing[i:i+seq_len]
            seq_m = missing_indicators[i:i+seq_len]
            
            if use_mim:
                # 拼接特征和指示器 [seq_len, features*2]
                seq = np.concatenate([seq_x, seq_m], axis=1)
            else:
                seq = seq_x
            
            sequences.append(seq)
            # 标签是序列最后一个时间步的SOH
            labels.append(y[i+seq_len-1])
        
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class MIMDataset(Dataset):
    """
    MIM训练数据集 - 支持多缺失率混合和插补方法
    
    参数:
        X: 特征数组
        y: 标签数组
        missing_rates: 缺失率列表
        use_mim: 是否使用MIM
        base_seed: 基础随机种子
        imputation_method: 插补方法 ("zero", "mean", "knn", "iterative")
        fit_imputer: 是否使用X拟合插补器
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        missing_rates: List[float],
        use_mim: bool = True,
        base_seed: int = 42,
        imputation_method: str = "zero",
        fit_imputer: bool = True
    ):
        self.use_mim = use_mim
        self.imputation_method = imputation_method
        
        # 导入插补器
        from .imputation import Imputer
        
        # 创建并拟合插补器
        imputer = Imputer(method=imputation_method)
        if fit_imputer and imputation_method != "zero":
            imputer.fit(X)
        else:
            imputer.fitted = True
        
        # 为每个缺失率生成数据副本
        X_expanded = []
        y_expanded = []
        
        for i, mr in enumerate(missing_rates):
            # 使用独立随机数生成器 (META §4.3)
            rng = np.random.RandomState(base_seed + i * 100)
            
            # 生成MCAR缺失掩码 (元素级独立随机缺失)
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
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class SequenceMIMDataset(Dataset):
    """
    序列MIM训练数据集 - 支持多缺失率和插补方法
    
    参数:
        X: 特征数组
        y: 标签数组
        seq_len: 序列长度
        missing_rates: 缺失率列表
        use_mim: 是否使用MIM
        base_seed: 基础随机种子
        imputation_method: 插补方法 ("zero", "mean", "knn", "iterative")
        fit_imputer: 是否使用X拟合插补器
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 5,
        missing_rates: Optional[List[float]] = None,
        use_mim: bool = True,
        base_seed: int = 42,
        imputation_method: str = "zero",
        fit_imputer: bool = True
    ):
        self.seq_len = seq_len
        self.use_mim = use_mim
        self.imputation_method = imputation_method
        
        if missing_rates is None:
            missing_rates = [0.0]
        
        # 导入插补器
        from .imputation import Imputer
        
        # 创建并拟合插补器
        imputer = Imputer(method=imputation_method)
        if fit_imputer and imputation_method != "zero":
            imputer.fit(X)
        else:
            imputer.fitted = True
        
        sequences = []
        labels = []
        
        # 为每个缺失率生成序列
        for rate_idx, mr in enumerate(missing_rates):
            rng = np.random.RandomState(base_seed + rate_idx)
            
            # 生成MCAR缺失掩码 (True表示缺失)
            mask = rng.rand(*X.shape) < mr
            
            # 应用插补
            X_imputed = imputer.transform(X, mask)
            missing_indicators = mask.astype(np.float32)
            
            # 构造序列
            for i in range(len(X) - seq_len + 1):
                seq_x = X_imputed[i:i+seq_len]
                seq_m = missing_indicators[i:i+seq_len]
                
                if use_mim:
                    # 拼接 [seq_len, features*2]
                    seq = np.concatenate([seq_x, seq_m], axis=1)
                else:
                    seq = seq_x
                
                sequences.append(seq)
                labels.append(y[i+seq_len-1])
        
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )
