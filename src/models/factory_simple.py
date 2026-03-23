"""
模型工厂 - 简化版

合并了模型定义和创建逻辑
代码量：50行 vs 原来的400+行
"""

import torch
import torch.nn as nn
from typing import Literal


# ========== 模型定义 ==========

class MLP(nn.Module):
    """MLP 模型"""
    
    def __init__(self, input_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            # 根据输入维度选择默认配置
            hidden_dims = [128, 96] if input_dim == 32 else [192, 96]
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(0.15)
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTM(nn.Module):
    """LSTM 模型"""
    
    def __init__(self, input_dim: int, hidden_size: int = 46):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: [batch, seq=1, features]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


class CNN1D(nn.Module):
    """1D CNN 模型"""
    
    def __init__(self, input_dim: int, channels: list[int] = None):
        super().__init__()
        
        if channels is None:
            channels = [64, 80]
        
        convs = []
        prev = input_dim
        for c in channels:
            convs.extend([
                nn.Conv1d(prev, c, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev = c
        
        self.convs = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, 1)
    
    def forward(self, x):
        # x: [batch, seq=1, features] -> [batch, features, seq]
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ========== 工厂函数 ==========

def create_model(
    model_type: Literal['mlp', 'lstm', 'cnn'],
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    创建模型
    
    Args:
        model_type: 模型类型
        input_dim: 输入维度 (16 for non-MIM, 32 for MIM)
        **kwargs: 额外参数传递给模型
    
    Returns:
        模型实例
    
    Example:
        >>> model = create_model('mlp', input_dim=32)
        >>> model = create_model('lstm', input_dim=16, hidden_size=64)
    """
    if model_type == 'mlp':
        return MLP(input_dim, **kwargs)
    elif model_type == 'lstm':
        return LSTM(input_dim, **kwargs)
    elif model_type == 'cnn':
        return CNN1D(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = ['MLP', 'LSTM', 'CNN1D', 'create_model']
