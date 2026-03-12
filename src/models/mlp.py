"""MLP (Multi-Layer Perceptron) 模型"""
import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """多层感知机模型
    
    适用于电池SOH预测的简单前馈神经网络。
    
    Args:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [192, 96, 48, 24],
        dropout: float = 0.15
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            预测值 [batch_size]
        """
        return self.network(x).squeeze(-1)
