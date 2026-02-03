"""随机种子管理模块"""
import random
import numpy as np
import torch


def set_seeds(seed: int = 42):
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_seeds(n: int, base_seed: int = 42) -> list:
    """
    生成n个不同的随机种子
    
    Args:
        n: 种子数量
        base_seed: 基础种子
        
    Returns:
        list: 种子列表
    """
    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(1, 100000, size=n).tolist()
    return seeds
