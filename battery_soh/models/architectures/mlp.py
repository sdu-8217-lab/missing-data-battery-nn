"""
Multi-Layer Perceptron (MLP) for SOH prediction.

Simple feedforward network with configurable hidden layers.
Suitable for non-sequential battery data.
"""

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for battery SOH prediction.
    
    Architecture follows the parameter budget from meta.md:
    - input_dim=16, hidden_dims=[192,96] → ~21,889 params
    - input_dim=32, hidden_dims=[128,96] → ~16,705 params
    
    Args:
        input_dim: Input feature dimension (16 or 32)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        
    Example:
        >>> model = MLP(input_dim=16, hidden_dims=[192, 96], dropout=0.15)
        >>> x = torch.randn(32, 16)  # batch_size=32
        >>> y = model(x)
        >>> y.shape
        torch.Size([32])
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.15
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [192, 96]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [BATCH_SIZE, INPUT_DIM]
            
        Returns:
            Predictions of shape [BATCH_SIZE]
        """
        return self.network(x).squeeze(-1)
