"""
模型参数量计算工具
用于估算神经网络模型的参数总量
"""
import torch
import torch.nn as nn
from typing import Dict, List


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数总量
    
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
    
    MLP参数量 ≈ (input_dim × hidden) + (hidden × hidden) × (layers-1) + hidden × 1
    """
    candidates = []
    
    # 搜索空间设计
    configs = [
        # [hidden_dim, n_layers]
        [64, 2],   # ~4K
        [96, 2],   # ~6K
        [128, 2],  # ~10K (稍超)
        [48, 3],   # ~5K
        [64, 3],   # ~8K (目标)
        [80, 3],   # ~12K (稍超)
        [32, 4],   # ~4K
        [48, 4],   # ~7K
        [56, 4],   # ~9K
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
    
    LSTM参数量 ≈ 4 × (input_dim × hidden + hidden × hidden + hidden) × n_layers
    """
    candidates = []
    
    configs = [
        # [hidden, n_layers]
        [32, 1],   # ~6K
        [48, 1],   # ~13K (超)
        [24, 2],   # ~7K
        [32, 2],   # ~12K (超)
        [40, 1],   # ~11K (超)
        [28, 2],   # ~9K
        [36, 1],   # ~9K
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
    
    CNN参数量 ≈ sum(channels[i] × channels[i+1] × kernel × 1) + FC layers
    输入为 [N, 1, 16]，经过卷积后flatten，再经过FC层
    """
    candidates = []
    
    # 重新设计搜索空间，考虑FC层参数量
    configs = [
        # [channels_list, kernel_size, fc_hidden]
        [[16, 8], 3, 32],      # 估计 ~3K
        [[24, 12], 3, 32],     # 估计 ~5K
        [[16, 16], 3, 32],     # 估计 ~6K
        [[24, 16], 3, 32],     # 估计 ~7K
        [[32, 16], 3, 32],     # 估计 ~8K (目标)
        [[24, 24], 3, 32],     # 估计 ~9K
        [[16, 8, 4], 3, 32],   # 估计 ~2K
        [[24, 12, 6], 3, 32],  # 估计 ~4K
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
