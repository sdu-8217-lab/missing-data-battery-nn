"""
模型参数量计算工具 - 精简版

使用 torchinfo 进行参数统计，保留候选架构生成的业务逻辑。
"""
from typing import Dict, List
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数总量
    
    使用 torchinfo 或 PyTorch 原生方法。
    
    Args:
        model: PyTorch模型
        trainable_only: 是否只计算可训练参数
        
    Returns:
        参数总数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
    """格式化参数数量显示"""
    if n < 1000:
        return f"{n}"
    elif n < 1_000_000:
        return f"{n/1000:.1f}K"
    else:
        return f"{n/1_000_000:.2f}M"


def check_param_budget(model: nn.Module, target: int = 8192, tolerance: float = 0.2) -> Dict:
    """
    检查模型参数量是否在目标范围内
    
    Args:
        model: 模型
        target: 目标参数量 (默认8192 = 2^13)
        tolerance: 容忍误差比例 (默认±20%)
        
    Returns:
        包含参数统计的字典
    """
    n_params = count_parameters(model)
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)
    
    return {
        "n_params": n_params,
        "formatted": format_param_count(n_params),
        "target": target,
        "lower": lower,
        "upper": upper,
        "in_budget": lower <= n_params <= upper,
        "ratio": n_params / target,
    }


# 预实验用的模型配置生成器
def generate_mlp_candidates(input_dim: int = 16) -> List[Dict]:
    """
    生成MLP候选结构 (目标: ~8192参数)
    """
    candidates = []
    
    configs = [
        [64, 2], [96, 2], [128, 2],
        [48, 3], [64, 3], [80, 3],
        [32, 4], [48, 4], [56, 4],
    ]
    
    for hidden, layers in configs:
        cfg = {
            "name": f"mlp_h{hidden}_l{layers}",
            "type": "mlp",
            "input_dim": input_dim,
            "hidden_dim": hidden,
            "n_layers": layers,
            "dropout": 0.1,
        }
        candidates.append(cfg)
    
    return candidates


def generate_lstm_candidates(input_dim: int = 16) -> List[Dict]:
    """
    生成LSTM候选结构 (目标: ~8192参数)
    """
    candidates = []
    
    configs = [
        [32, 1], [48, 1],
        [24, 2], [32, 2],
        [40, 1], [28, 2], [36, 1],
    ]
    
    for hidden, layers in configs:
        cfg = {
            "name": f"lstm_h{hidden}_l{layers}",
            "type": "lstm",
            "input_dim": input_dim,
            "hidden_dim": hidden,
            "n_layers": layers,
            "dropout": 0.1,
        }
        candidates.append(cfg)
    
    return candidates


def generate_cnn_candidates(input_dim: int = 16) -> List[Dict]:
    """
    生成CNN候选结构 (目标: ~8192参数)
    """
    candidates = []
    
    configs = [
        [[16, 8], 3, 32],
        [[24, 12], 3, 32],
        [[16, 16], 3, 32],
        [[24, 16], 3, 32],
        [[32, 16], 3, 32],
        [[24, 24], 3, 32],
        [[16, 8, 4], 3, 32],
        [[24, 12, 6], 3, 32],
    ]
    
    for channels, kernel, fc_hidden in configs:
        cfg = {
            "name": f"cnn_c{'_'.join(map(str, channels))}_k{kernel}_fc{fc_hidden}",
            "type": "cnn",
            "input_dim": input_dim,
            "channels": channels,
            "kernel_size": kernel,
            "fc_hidden": fc_hidden,
            "dropout": 0.1,
        }
        candidates.append(cfg)
    
    return candidates
