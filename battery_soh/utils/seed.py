"""
Random seed management for reproducibility.

Controls all random number generators to ensure experiment reproducibility.
"""

import random
import numpy as np
import torch

from battery_soh.core.types import Seed


def set_seed(seed: Seed) -> None:
    """Set random seed for all random number generators.
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch backends (deterministic mode)
    
    Args:
        seed: Random seed value
        
    Example:
        >>> from battery_soh.utils import set_seed
        >>> from battery_soh.core.types import Seed
        >>> set_seed(Seed(42))
        >>> # All subsequent random operations are reproducible
    """
    # Python random
    random.seed(int(seed))
    
    # NumPy
    np.random.seed(int(seed))
    
    # PyTorch
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_seed(seed: Seed, worker_id: int) -> int:
    """Generate unique seed for DataLoader workers.
    
    Args:
        seed: Base random seed
        worker_id: Worker process ID
        
    Returns:
        Unique seed for this worker
    """
    return int(seed) + worker_id
