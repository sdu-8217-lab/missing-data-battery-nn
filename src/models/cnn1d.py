"""1D-CNN (One-Dimensional Convolutional Neural Network) 模型"""
import torch
import torch.nn as nn
from typing import List


class CNN1D(nn.Module):
    """一维卷积神经网络模型
    
    利用卷积核提取时间序列局部特征进行电池SOH预测。
    
    Args:
        input_dim: 输入特征维度
        channels: 卷积通道数列表
        kernel_size: 卷积核大小
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        input_dim: int,
        channels: List[int] = [72, 32],
        kernel_size: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 确保input_dim是序列长度
        self.input_dim = input_dim
        
        # 构建卷积层
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Linear(in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            预测值 [batch_size]
        """
        # 转置为 [batch_size, input_dim, seq_len] 以适应Conv1d
        x = x.transpose(1, 2)
        
        # 卷积层
        conv_out = self.conv_layers(x)
        
        # 自适应池化
        pooled = self.adaptive_pool(conv_out).squeeze(-1)
        
        # 输出层
        output = self.fc(pooled)
        
        return output.squeeze(-1)
