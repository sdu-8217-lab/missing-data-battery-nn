"""GRU (Gated Recurrent Unit) 模型"""
import torch
import torch.nn as nn


class GRU(nn.Module):
    """GRU循环神经网络模型
    
    LSTM的轻量级替代，利用门控机制进行电池SOH预测。
    
    Args:
        input_dim: 输入特征维度
        hidden_size: GRU隐藏层大小
        num_layers: GRU层数
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            
        Returns:
            预测值 [batch_size]
        """
        # GRU输出
        gru_out, h_n = self.gru(x)
        
        # 使用最后一个时间步的隐藏状态
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # 输出层
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)
