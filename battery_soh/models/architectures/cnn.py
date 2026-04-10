"""1D CNN model for SOH prediction."""

from typing import List
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for SOH prediction.
    
    Uses 1D convolutions to capture local patterns in the input sequence.
    Suitable for detecting patterns in battery cycle sequences.
    
    Args:
        input_dim: Input feature dimension (treated as channels)
        channels: List of channel dimensions for convolutional layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if channels is None:
            channels = [64, 80]
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_dim
        
        for out_channels in channels:
            conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling + output layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [BATCH_SIZE, SEQ_LEN, INPUT_DIM]
            
        Returns:
            Predictions of shape [BATCH_SIZE]
        """
        # Transpose for Conv1d: [BATCH, INPUT_DIM, SEQ_LEN]
        x = x.transpose(1, 2)
        
        # Convolutional layers
        conv_out = self.conv_layers(x)
        
        # Global pooling
        pooled = self.pool(conv_out).squeeze(-1)
        
        # Output layer
        output = self.fc(pooled)
        
        return output.squeeze(-1)
