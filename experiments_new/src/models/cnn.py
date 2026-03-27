"""
1D CNN模型
"""
import torch
import torch.nn as nn
from typing import List, Optional


class CNN1D(nn.Module):
    """
    一维卷积神经网络
    
    参数:
        input_dim: 输入特征维度
        channels: 卷积通道列表，如[72, 32]
        kernel_size: 卷积核大小
        dropout: dropout比率
        seq_len: 序列长度
        use_mim: 是否使用MIM
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        channels: Optional[List[int]] = None,
        kernel_size: int = 4,
        dropout: float = 0.0,
        seq_len: int = 5,
        use_mim: bool = False
    ):
        super().__init__()
        
        self.use_mim = use_mim
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # MIM模式下输入维度翻倍
        actual_input_dim = input_dim * 2 if use_mim else input_dim
        
        # 默认通道配置
        if channels is None:
            channels = [64, 32]
        
        # 构建卷积层
        conv_layers = []
        in_channels = actual_input_dim
        
        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 计算卷积后的特征维度
        # 假设输入是 (batch, seq_len, features)，需要转置为 (batch, features, seq_len)
        # 卷积后维度保持不变（因为有padding），然后全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(channels[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (batch_size, seq_len, features) 或 (batch_size, features)
        输出: (batch_size,)
        """
        # 如果输入是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # 转置为 (batch, features, seq_len) 以适应Conv1d
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # 卷积
        features = self.conv_layers(x)  # (batch, channels[-1], seq_len)
        
        # 全局池化
        pooled = self.global_pool(features).squeeze(-1)  # (batch, channels[-1])
        
        # 输出层
        output = self.fc(pooled).squeeze()
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'type': 'CNN1D',
            'input_dim': self.input_dim,
            'actual_input_dim': self.input_dim * 2 if self.use_mim else self.input_dim,
            'use_mim': self.use_mim,
            'channels': self._get_channels(),
            'kernel_size': self._get_kernel_size(),
            'seq_len': self.seq_len,
            'parameters': sum(p.numel() for p in self.parameters())
        }
    
    def _get_channels(self) -> List[int]:
        """提取通道配置"""
        channels = []
        for module in self.conv_layers:
            if isinstance(module, nn.Conv1d):
                channels.append(module.out_channels)
        return channels
    
    def _get_kernel_size(self) -> int:
        """提取卷积核大小"""
        for module in self.conv_layers:
            if isinstance(module, nn.Conv1d):
                return module.kernel_size[0]
        return 3
