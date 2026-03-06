"""随机种子管理模块"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    设置全局随机种子，确保实验可复现
    
    设置:
        - Python random
        - NumPy
        - PyTorch (CPU & CUDA)
        - PyTorch cudnn
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
