"""
LSTM模型
"""
import torch
import torch.nn as nn
from typing import Optional


class LSTMModel(nn.Module):
    """
    长短期记忆网络
    
    参数:
        input_dim: 输入特征维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: dropout比率
        seq_len: 序列长度
        use_mim: 是否使用MIM
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_size: int = 48,
        num_layers: int = 2,
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
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=actual_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (batch_size, seq_len, features) 或 (batch_size, features)
        输出: (batch_size,)
        """
        # 如果输入是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # LSTM前向
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后时刻的隐藏状态
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # 输出层
        output = self.fc(last_hidden).squeeze()
        return output
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'type': 'LSTM',
            'input_dim': self.input_dim,
            'actual_input_dim': self.input_dim * 2 if self.use_mim else self.input_dim,
            'use_mim': self.use_mim,
            'hidden_size': self.lstm.hidden_size,
            'num_layers': self.lstm.num_layers,
            'seq_len': self.seq_len,
            'parameters': sum(p.numel() for p in self.parameters())
        }
