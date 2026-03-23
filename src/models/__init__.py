"""
模型模块

提供电池SOH预测的神经网络模型实现。

可用模型:
- MLP: 多层感知机
- LSTM: 长短期记忆网络
- CNN1D: 一维卷积网络

工厂函数:
- create_model: 根据类型和配置创建模型
- create_model_from_config: 从 ExperimentConfig 创建模型
- count_parameters: 计算模型参数量
- verify_parameter_budget: 验证参数量是否符合要求

使用示例:
    >>> from src.models import create_model, get_model_info
    >>> 
    >>> # 创建 non-MIM MLP
    >>> model = create_model('mlp', input_dim=16, use_mim=False)
    >>> 
    >>> # 创建 MIM LSTM
    >>> model = create_model('lstm', input_dim=32, use_mim=True)
    >>> 
    >>> # 获取模型信息
    >>> info = get_model_info(model)
    >>> print(f"参数量: {info['total_params']:,}")
"""

# 基础模型类
from .mlp import MLP
from .lstm import LSTM
from .cnn1d import CNN1D

# 工厂函数
from .factory import (
    create_model,
    create_model_from_config,
    make_model,
    build_model,
    count_parameters,
    get_model_info,
    verify_parameter_budget,
    get_all_model_variants,
    print_model_summary,
    ModelSpec,
    MODEL_CONFIGS,
)

__all__ = [
    # 模型类
    'MLP',
    'LSTM',
    'CNN1D',
    # 工厂函数
    'create_model',
    'create_model_from_config',
    'make_model',
    'build_model',
    # 工具函数
    'count_parameters',
    'get_model_info',
    'verify_parameter_budget',
    'get_all_model_variants',
    'print_model_summary',
    # 配置
    'ModelSpec',
    'MODEL_CONFIGS',
]
