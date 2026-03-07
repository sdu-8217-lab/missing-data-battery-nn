"""LSTM (Long Short-Term Memory) 模型"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM循环神经网络模型
    
    利用时间序列信息进行电池SOH预测。
    
    Args:
        input_dim: 输入特征维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
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
        # LSTM输出
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的隐藏状态
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # 输出层
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)
