"""
模型工厂 - 统一模型创建和配置管理

基于 meta.md 的架构要求:
- 所有模型参数量统一在 (2^14, 2^15) = (16384, 32768) 范围内
- 支持 MIM (32维输入) 和 non-MIM (16维输入) 两种模式
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass

from .mlp import MLP
from .lstm import LSTM
from .cnn1d import CNN1D


@dataclass(frozen=True)
class ModelSpec:
    """模型规格定义"""
    model_type: Literal['mlp', 'lstm', 'cnn']
    input_dim: int
    use_mim: bool
    
    def __post_init__(self):
        assert self.input_dim in [16, 32], f"input_dim 必须是 16 或 32，当前: {self.input_dim}"
        if self.use_mim:
            assert self.input_dim == 32, "MIM 模式下 input_dim 必须是 32"
        else:
            assert self.input_dim == 16, "non-MIM 模式下 input_dim 必须是 16"


# ============================================================================
# 模型配置常量 (基于 meta.md 的参数量要求)
# ============================================================================

MODEL_CONFIGS = {
    'mlp': {
        'non_mim': {
            'hidden_dims': [192, 96],
            'dropout': 0.15,
            # 参数量: input=16, hidden=[192,96] -> ~21,889 params
        },
        'mim': {
            'hidden_dims': [128, 96],
            'dropout': 0.15,
            # 参数量: input=32, hidden=[128,96] -> ~16,705 params
        }
    },
    'lstm': {
        'non_mim': {
            'hidden_size': 46,
            'num_layers': 2,
            'dropout': 0.2,
            # 参数量: input=16, hidden=46, layers=2 -> ~29,119 params
        },
        'mim': {
            'hidden_size': 46,
            'num_layers': 2,
            'dropout': 0.2,
            # 参数量: input=32, hidden=46, layers=2 -> ~32,063 params
        }
    },
    'cnn': {
        'non_mim': {
            'channels': [64, 80],
            'kernel_size': 3,
            'dropout': 0.1,
            # 参数量: input=16, channels=[64,80] -> ~18,657 params
        },
        'mim': {
            'channels': [64, 80],
            'kernel_size': 3,
            'dropout': 0.1,
            # 参数量: input=32, channels=[64,80] -> ~21,729 params
        }
    }
}


# ============================================================================
# 模型工厂函数
# ============================================================================

def create_model(
    model_type: Literal['mlp', 'lstm', 'cnn'],
    input_dim: int,
    use_mim: bool = False,
    **override_kwargs
) -> nn.Module:
    """
    创建模型实例
    
    工厂函数根据 model_type 和 use_mim 自动选择合适的配置，
    确保参数量在 (16384, 32768) 范围内。
    
    Args:
        model_type: 模型类型 ('mlp', 'lstm', 'cnn')
        input_dim: 输入维度 (16 for non-MIM, 32 for MIM)
        use_mim: 是否使用MIM (影响配置选择)
        **override_kwargs: 覆盖默认配置的参数
        
    Returns:
        创建的模型实例
        
    Example:
        >>> # Non-MIM mode
        >>> model = create_model('mlp', input_dim=16, use_mim=False)
        >>> 
        >>> # MIM mode
        >>> model = create_model('mlp', input_dim=32, use_mim=True)
        >>> 
        >>> # Override dropout
        >>> model = create_model('lstm', input_dim=16, use_mim=False, dropout=0.3)
    """
    spec = ModelSpec(model_type=model_type, input_dim=input_dim, use_mim=use_mim)
    
    # 获取基础配置
    config_key = 'mim' if use_mim else 'non_mim'
    base_config = MODEL_CONFIGS[model_type][config_key].copy()
    
    # 应用覆盖参数
    base_config.update(override_kwargs)
    
    # 创建模型
    if model_type == 'mlp':
        return MLP(
            input_dim=input_dim,
            hidden_dims=base_config['hidden_dims'],
            dropout=base_config['dropout']
        )
    elif model_type == 'lstm':
        return LSTM(
            input_dim=input_dim,
            hidden_size=base_config['hidden_size'],
            num_layers=base_config['num_layers'],
            dropout=base_config['dropout']
        )
    elif model_type == 'cnn':
        return CNN1D(
            input_dim=input_dim,
            channels=base_config['channels'],
            kernel_size=base_config['kernel_size'],
            dropout=base_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_config(config) -> nn.Module:
    """
    从 ExperimentConfig 创建模型
    
    Args:
        config: ExperimentConfig 对象 (来自 src.config)
        
    Returns:
        创建的模型实例
    """
    input_dim = 32 if config.mim.use_mim else 16
    
    # 获取模型特定配置
    model_type = config.model.model_type
    
    if model_type == 'mlp':
        override_kwargs = {
            'hidden_dims': config.model.mlp_hidden_dims,
            'dropout': config.model.mlp_dropout
        }
    elif model_type == 'lstm':
        override_kwargs = {
            'hidden_size': config.model.lstm_hidden_size,
            'num_layers': config.model.lstm_num_layers,
            'dropout': config.model.lstm_dropout
        }
    elif model_type == 'cnn':
        override_kwargs = {
            'channels': config.model.cnn_channels,
            'kernel_size': config.model.cnn_kernel_size,
            'dropout': config.model.cnn_dropout
        }
    else:
        override_kwargs = {}
    
    return create_model(
        model_type=model_type,
        input_dim=input_dim,
        use_mim=config.mim.use_mim,
        **override_kwargs
    )


# ============================================================================
# 参数量计算工具
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数量
    
    Args:
        model: PyTorch 模型
        trainable_only: 是否只计算可训练参数
        
    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型详细信息
    
    Args:
        model: PyTorch 模型
        
    Returns:
        包含模型信息的字典
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # 计算模型大小 (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb,
        'model_class': model.__class__.__name__,
    }


def verify_parameter_budget(
    model: nn.Module,
    min_params: int = 16384,  # 2^14
    max_params: int = 32768,  # 2^15
    verbose: bool = True
) -> bool:
    """
    验证模型参数量是否在预算范围内
    
    Args:
        model: PyTorch 模型
        min_params: 最小参数量
        max_params: 最大参数量
        verbose: 是否打印信息
        
    Returns:
        是否在预算范围内
    """
    params = count_parameters(model)
    in_budget = min_params <= params <= max_params
    
    if verbose:
        status = "✅" if in_budget else "⚠️"
        print(f"{status} 模型: {model.__class__.__name__}")
        print(f"   参数量: {params:,} ({params / 1000:.1f}K)")
        print(f"   预算范围: [{min_params:,}, {max_params:,}]")
        if not in_budget:
            if params < min_params:
                print(f"   警告: 参数量不足，建议增加模型容量")
            else:
                print(f"   警告: 参数量超限，建议减少模型容量")
    
    return in_budget


# ============================================================================
# 预创建模型（用于快速验证）
# ============================================================================

def get_all_model_variants() -> Dict[str, nn.Module]:
    """
    获取所有模型变体 (3 model × 2 mim = 6 个模型)
    
    Returns:
        字典，键为 "{model_type}_{mim}"，值为模型实例
    """
    variants = {}
    
    for model_type in ['mlp', 'lstm', 'cnn']:
        for use_mim in [False, True]:
            input_dim = 32 if use_mim else 16
            key = f"{model_type}_{'mim' if use_mim else 'no_mim'}"
            variants[key] = create_model(model_type, input_dim, use_mim)
    
    return variants


def print_model_summary():
    """打印所有模型变体的摘要"""
    print("=" * 70)
    print("模型规格摘要 (基于 meta.md 参数量要求)")
    print("=" * 70)
    print(f"{'模型':<15} {'模式':<10} {'输入维度':<10} {'参数量':<12} {'状态'}")
    print("-" * 70)
    
    variants = get_all_model_variants()
    
    for key, model in variants.items():
        parts = key.split('_')
        model_type = parts[0]
        mim_status = '_'.join(parts[1:])  # 处理 'no_mim' 的情况
        input_dim = 32 if mim_status == 'mim' else 16
        params = count_parameters(model)
        in_budget = 16384 <= params <= 32768
        status = "✅" if in_budget else "⚠️"
        
        print(f"{model_type.upper():<15} {mim_status:<10} {input_dim:<10} {params:>10,}   {status}")
    
    print("=" * 70)


# ============================================================================
# 兼容性接口
# ============================================================================

def build_model(
    model_type: str,
    use_mim: bool = False,
    **kwargs
) -> nn.Module:
    """
    兼容性接口 (与 run_experiment.py 兼容)
    
    Args:
        model_type: 模型类型 ('mlp', 'lstm', 'cnn')
        use_mim: 是否使用MIM
        **kwargs: 额外参数
        
    Returns:
        创建的模型实例
    """
    input_dim = 32 if use_mim else 16
    return create_model(model_type, input_dim, use_mim, **kwargs)


# 向后兼容的别名
make_model = create_model
