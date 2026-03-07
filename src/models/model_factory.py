"""模型工厂 - 统一创建和管理神经网络模型"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .mlp import MLP
from .lstm import LSTM
from .gru import GRU
from .cnn1d import CNN1D


class ModelFactory:
    """模型工厂类
    
    提供统一接口创建和配置各种神经网络模型。
    """
    
    # 模型注册表
    _models = {
        'mlp': MLP,
        'lstm': LSTM,
        'gru': GRU,
        'cnn1d': CNN1D,
        'CNN1D': CNN1D,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        input_dim: int,
        device: str = 'cpu',
        **kwargs
    ) -> nn.Module:
        """创建模型实例
        
        Args:
            model_type: 模型类型 ('mlp', 'lstm', 'gru', 'cnn1d')
            input_dim: 输入维度
            device: 计算设备
            **kwargs: 模型特定参数
            
        Returns:
            创建的模型实例
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        model = model_class(input_dim=input_dim, **kwargs)
        
        return model.to(device)
    
    @classmethod
    def count_parameters(cls, model: nn.Module) -> int:
        """计算模型可训练参数数量
        
        Args:
            model: 模型实例
            
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @classmethod
    def get_model_info(cls, model: nn.Module) -> Dict[str, Any]:
        """获取模型信息
        
        Args:
            model: 模型实例
            
        Returns:
            模型信息字典
        """
        return {
            'name': model.__class__.__name__,
            'total_params': cls.count_parameters(model),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """注册新模型类型
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        cls._models[name.lower()] = model_class


# 便捷函数
def create_model(model_type: str, input_dim: int, device: str = 'cpu', **kwargs) -> nn.Module:
    """创建模型的便捷函数
    
    Args:
        model_type: 模型类型
        input_dim: 输入维度
        device: 计算设备
        **kwargs: 模型特定参数
        
    Returns:
        创建的模型实例
    """
    return ModelFactory.create_model(model_type, input_dim, device, **kwargs)
