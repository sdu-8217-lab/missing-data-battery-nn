"""
模型工厂
根据配置创建对应的模型实例
"""
import torch
from typing import Dict, Any, Optional

from .mlp import MLP
from .lstm import LSTMModel
from .gru import GRUModel
from .cnn import CNN1D


def create_model(
    model_type: str,
    input_dim: int = 16,
    use_mim: bool = False,
    device: str = "auto",
    **kwargs
) -> torch.nn.Module:
    """
    根据配置创建模型
    
    参数:
        model_type: 模型类型，可选 'mlp', 'lstm', 'gru', 'cnn1d'
        input_dim: 输入维度
        use_mim: 是否使用MIM
        device: 计算设备，'auto', 'cuda', 'cpu'
        **kwargs: 模型特定参数
    
    返回:
        创建好的模型实例
    """
    # 自动检测设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 根据类型创建模型
    if model_type.lower() == 'mlp':
        model = MLP(
            input_dim=input_dim,
            hidden_layers=kwargs.get('hidden_layers'),
            dropout=kwargs.get('dropout', 0.0),
            use_mim=use_mim
        )
    elif model_type.lower() == 'lstm':
        model = LSTMModel(
            input_dim=input_dim,
            hidden_size=kwargs.get('hidden_size', 48),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.0),
            seq_len=kwargs.get('seq_len', 5),
            use_mim=use_mim
        )
    elif model_type.lower() == 'gru':
        model = GRUModel(
            input_dim=input_dim,
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.0),
            seq_len=kwargs.get('seq_len', 5),
            use_mim=use_mim
        )
    elif model_type.lower() in ['cnn', 'cnn1d']:
        model = CNN1D(
            input_dim=input_dim,
            channels=kwargs.get('channels'),
            kernel_size=kwargs.get('kernel_size', 4),
            dropout=kwargs.get('dropout', 0.0),
            seq_len=kwargs.get('seq_len', 5),
            use_mim=use_mim
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 移动到设备
    model = model.to(device)
    
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """获取模型摘要信息"""
    if hasattr(model, 'get_model_info'):
        return model.get_model_info()
    else:
        return {
            'type': model.__class__.__name__,
            'parameters': count_parameters(model)
        }


def list_available_models() -> list:
    """列出可用的模型类型"""
    return ['mlp', 'lstm', 'gru', 'cnn1d']
