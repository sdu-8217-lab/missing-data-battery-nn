"""
多层感知机模型
"""
import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    多层感知机
    
    参数:
        input_dim: 输入维度
        hidden_layers: 隐藏层结构，如[192, 96, 48, 24]
        dropout: dropout比率
        use_mim: 是否使用MIM（影响输入维度处理）
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
        use_mim: bool = False
    ):
        super().__init__()
        
        self.use_mim = use_mim
        self.input_dim = input_dim
        
        # MIM模式下输入维度翻倍（特征+缺失指示器）
        actual_input_dim = input_dim * 2 if use_mim else input_dim
        
        # 默认隐藏层结构
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        # 构建网络
        layers = []
        prev_size = actual_input_dim
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # 在隐藏层之间添加dropout（除了最后一层）
            if dropout > 0 and i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x).squeeze()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'type': 'MLP',
            'input_dim': self.input_dim,
            'actual_input_dim': self.input_dim * 2 if self.use_mim else self.input_dim,
            'use_mim': self.use_mim,
            'hidden_layers': self._get_hidden_layers(),
            'parameters': sum(p.numel() for p in self.parameters())
        }
    
    def _get_hidden_layers(self) -> List[int]:
        """提取隐藏层结构"""
        layers = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                layers.append(module.out_features)
        return layers[:-1]  # 排除输出层
