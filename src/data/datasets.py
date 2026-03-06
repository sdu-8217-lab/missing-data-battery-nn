"""PyTorch Dataset定义"""
import numpy as np
import torch
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    """基础数据集，返回单循环特征"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 missing_rate: float = 0.0, 
                 use_mim: bool = False,
                 seed: int = None):
        """
        Args:
            X: 特征数组 [n_samples, n_features]
            y: 标签数组 [n_samples]
            missing_rate: 缺失率
            use_mim: 是否使用缺失指示器
            seed: 随机种子
        """
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
        
        # 缺失指示器
        self.missing_indicators = (~self.mask).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X_missing[idx]
        
        if self.use_mim:
            # 拼接特征和缺失指示器
            x = np.concatenate([x, self.missing_indicators[idx]])
        
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)


class SequenceDataset(Dataset):
    """序列数据集，用于LSTM/GRU/1D-CNN"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = 5,
                 missing_rate: float = 0.0,
                 use_mim: bool = False,
                 seed: int = None):
        """
        Args:
            X: 特征数组 [n_samples, n_features]
            y: 标签数组 [n_samples]
            seq_len: 序列长度
            missing_rate: 缺失率
            use_mim: 是否使用缺失指示器
            seed: 随机种子
        """
        self.seq_len = seq_len
        self.missing_rate = missing_rate
        self.use_mim = use_mim
        
        # 生成缺失掩码（一次性生成，不随epoch变化）
        if seed is not None:
            np.random.seed(seed)
        mask = np.random.rand(*X.shape) > missing_rate
        
        # 应用缺失
        X_missing = np.where(mask, X, 0.0)
        missing_indicators = (~mask).astype(np.float32)
        
        # 构造序列
        self.sequences = []
        self.labels = []
        
        for i in range(len(X) - seq_len + 1):
            seq_x = X_missing[i:i+seq_len]
            seq_m = missing_indicators[i:i+seq_len]
            
            if use_mim:
                # 拼接特征和指示器 [seq_len, features*2]
                seq = np.concatenate([seq_x, seq_m], axis=1)
            else:
                seq = seq_x
            
            self.sequences.append(seq)
            # 标签是序列最后一个时间步的SOH
            self.labels.append(y[i+seq_len-1])
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)


class MIMDataset(Dataset):
    """MIM训练数据集，支持多缺失率混合"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 missing_rates: list,
                 use_mim: bool = True,
                 base_seed: int = 42):
        """
        Args:
            X: 特征数组
            y: 标签数组
            missing_rates: 缺失率列表
            use_mim: 是否使用MIM
            base_seed: 基础随机种子
        """
        self.use_mim = use_mim
        
        # 为每个缺失率生成数据副本
        X_expanded = []
        y_expanded = []
        
        for i, mr in enumerate(missing_rates):
            np.random.seed(base_seed + i)
            
            # 生成缺失掩码
            mask = np.random.rand(*X.shape) > mr
            
            # 应用缺失
            X_missing = np.where(mask, X, 0.0)
            missing_indicators = (~mask).astype(np.float32)
            
            if use_mim:
                # 拼接 [n_samples, features*2]
                X_combined = np.concatenate([X_missing, missing_indicators], axis=1)
            else:
                X_combined = X_missing
            
            X_expanded.append(X_combined)
            y_expanded.append(y)
        
        self.X = np.vstack(X_expanded)
        self.y = np.concatenate(y_expanded)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)


class SequenceMIMDataset(Dataset):
    """序列MIM训练数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = 5,
                 missing_rates: list = None,
                 use_mim: bool = True,
                 base_seed: int = 42):
        """
        Args:
            X: 特征数组
            y: 标签数组
            seq_len: 序列长度
            missing_rates: 缺失率列表
            use_mim: 是否使用MIM
            base_seed: 基础随机种子
        """
        self.seq_len = seq_len
        self.use_mim = use_mim
        
        if missing_rates is None:
            missing_rates = [0.0]
        
        sequences = []
        labels = []
        
        # 为每个缺失率生成序列
        for rate_idx, mr in enumerate(missing_rates):
            np.random.seed(base_seed + rate_idx)
            
            # 生成缺失掩码
            mask = np.random.rand(*X.shape) > mr
            X_missing = np.where(mask, X, 0.0)
            missing_indicators = (~mask).astype(np.float32)
            
            # 构造序列
            for i in range(len(X) - seq_len + 1):
                seq_x = X_missing[i:i+seq_len]
                seq_m = missing_indicators[i:i+seq_len]
                
                if use_mim:
                    seq = np.concatenate([seq_x, seq_m], axis=1)
                else:
                    seq = seq_x
                
                sequences.append(seq)
                labels.append(y[i+seq_len-1])
        
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)
