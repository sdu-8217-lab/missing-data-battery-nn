"""LSTM model for SOH prediction."""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM for sequential SOH prediction.
    
    Uses the last hidden state for prediction.
    Suitable for capturing temporal patterns in battery cycling.
    
    Args:
        input_dim: Input feature dimension per timestep
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied between LSTM layers)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 46,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [BATCH_SIZE, SEQ_LEN, INPUT_DIM]
            
        Returns:
            Predictions of shape [BATCH_SIZE]
        """
        # LSTM output
        _, (h_n, _) = self.lstm(x)
        
        # Use last layer's hidden state
        last_hidden = h_n[-1]
        
        # Output layer
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)
