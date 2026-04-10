"""
Model factory for creating neural network architectures.

All models are constrained to have parameter counts within the budget
defined in meta.md: 16,384 to 32,768 parameters.
"""

import torch.nn as nn

from battery_soh.core.types import ModelType
from battery_soh.core.constants import (
    MIN_PARAMS, 
    MAX_PARAMS, 
    N_FEATURES,
    N_FEATURES_WITH_MIM
)


def create_model(
    model_type: ModelType,
    use_mim: bool = False,
    **kwargs
) -> nn.Module:
    """Create a neural network model.
    
    Factory function that creates models with architecture-specific
    configurations that respect the parameter budget.
    
    Args:
        model_type: One of "mlp", "lstm", "cnn"
        use_mim: Whether to use MIM (affects input dimension)
        **kwargs: Override default configuration
        
    Returns:
        PyTorch model instance
        
    Raises:
        ValueError: If model_type is invalid
        
    Example:
        >>> model = create_model("mlp", use_mim=False)
        >>> print(f"Parameters: {count_parameters(model):,}")
        Parameters: 21,889
    """
    input_dim = N_FEATURES_WITH_MIM if use_mim else N_FEATURES
    
    if model_type == "mlp":
        from battery_soh.models.architectures.mlp import MLP
        
        # Default hidden dims based on input size
        if "hidden_dims" not in kwargs:
            kwargs["hidden_dims"] = [128, 96] if use_mim else [192, 96]
        
        model = MLP(input_dim=input_dim, **kwargs)
        
    elif model_type == "lstm":
        from battery_soh.models.architectures.lstm import LSTM
        
        # LSTM uses same config for both input sizes
        if "hidden_size" not in kwargs:
            kwargs["hidden_size"] = 46
        if "num_layers" not in kwargs:
            kwargs["num_layers"] = 2
            
        model = LSTM(input_dim=input_dim, **kwargs)
        
    elif model_type == "cnn":
        from battery_soh.models.architectures.cnn import CNN1D
        
        if "channels" not in kwargs:
            kwargs["channels"] = [64, 80]
        if "kernel_size" not in kwargs:
            kwargs["kernel_size"] = 3
            
        model = CNN1D(input_dim=input_dim, **kwargs)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_parameter_budget(
    model: nn.Module,
    verbose: bool = False
) -> bool:
    """Check if model parameter count is within budget.
    
    Args:
        model: PyTorch model
        verbose: If True, print parameter information
        
    Returns:
        True if within budget, False otherwise
    """
    n_params = count_parameters(model)
    in_budget = MIN_PARAMS <= n_params <= MAX_PARAMS
    
    if verbose:
        status = "✓" if in_budget else "✗"
        print(f"{status} {model.__class__.__name__}: {n_params:,} parameters")
        print(f"  Budget: {MIN_PARAMS:,} - {MAX_PARAMS:,}")
        
    return in_budget
