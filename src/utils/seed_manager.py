"""随机种子管理模块 - 确保实验可复现"""
import random
from typing import Optional, Callable
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
    
    Args:
        seed: 随机种子值 (>= 0)
    """
    if seed < 0:
        raise ValueError(f"随机种子必须 >= 0，当前: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_worker_init_fn(seed: int) -> Callable:
    """
    创建 DataLoader 的 worker_init_fn，确保多进程数据加载的确定性
    
    Usage:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=create_worker_init_fn(seed)
        ... )
    
    Args:
        seed: 基础随机种子
        
    Returns:
        worker_init_fn 函数
    """
    def worker_init_fn(worker_id: int):
        # 每个 worker 使用不同的种子，但基于基础种子确定性地生成
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return worker_init_fn


def create_dataloader_generator(seed: int) -> torch.Generator:
    """
    创建 DataLoader 使用的 torch.Generator
    
    用于确保 shuffle=True 时的确定性行为
    
    Usage:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     generator=create_dataloader_generator(seed)
        ... )
    
    Args:
        seed: 随机种子
        
    Returns:
        配置好的 torch.Generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def set_benchmark_mode(enabled: bool = True):
    """
    设置 cuDNN benchmark 模式
    
    警告: 启用 benchmark 会牺牲确定性以换取性能提升
    仅在确信不需要严格可复现性时使用
    
    Args:
        enabled: 是否启用 benchmark 模式
    """
    torch.backends.cudnn.benchmark = enabled
    if enabled:
        torch.backends.cudnn.deterministic = False


def get_random_state() -> dict:
    """
    获取当前的随机状态快照
    
    Returns:
        包含各随机源状态的字典
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_random_state(state: dict):
    """
    恢复随机状态
    
    Args:
        state: 由 get_random_state() 返回的状态字典
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class DeterministicContext:
    """
    确定性上下文管理器
    
    在上下文中临时设置特定的随机种子，退出后恢复原始状态
    
    Usage:
        >>> with DeterministicContext(seed=42):
        ...     # 这里的随机操作使用种子42
        ...     result = model.generate()
        >>> # 退出后自动恢复原随机状态
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.previous_state = None
    
    def __enter__(self):
        self.previous_state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_state is not None:
            set_random_state(self.previous_state)
        return False


def verify_seed_setting(seed: int = 42) -> bool:
    """
    验证种子设置是否正确工作
    
    执行两次相同种子的随机操作，检查结果是否一致
    
    Args:
        seed: 测试用的随机种子
        
    Returns:
        True if deterministic, False otherwise
    """
    results = []
    
    for _ in range(2):
        set_seed(seed)
        
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        results.append({
            'python': python_rand,
            'numpy': numpy_rand,
            'torch': torch_rand
        })
    
    return (
        results[0]['python'] == results[1]['python'] and
        results[0]['numpy'] == results[1]['numpy'] and
        results[0]['torch'] == results[1]['torch']
    )
