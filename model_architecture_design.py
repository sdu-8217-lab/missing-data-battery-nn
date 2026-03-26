"""
神经网络模型架构设计 V2
3种模型 × 4种参数量级别 = 12个模型配置
使用二分搜索精确调整参数量
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.15):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        return output.squeeze(-1)


class CNN1D(nn.Module):
    """1D-CNN模型"""
    def __init__(self, input_dim: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        conv_layers = []
        in_channels = input_dim
        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        conv_out = self.conv_layers(x)
        pooled = self.adaptive_pool(conv_out).squeeze(-1)
        output = self.fc(pooled)
        return output.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_param_ranges():
    """获取四个参数量级别的目标范围"""
    return {
        1: (2**12, 2**13),      # 4096 ~ 8192
        2: (2**13, 2**14),      # 8192 ~ 16384
        3: (2**14, 2**15),      # 16384 ~ 32768
        4: (2**15, 2**16),      # 32768 ~ 65536
    }


# 输入维度
input_dim_no_mim = 16   # use_mim=false
input_dim_mim = 32      # use_mim=true

# ==================== 重新设计的配置 ====================

# MLP配置 - 调整后的隐藏层维度
mlp_configs_final = {
    # Level 1: 4096 ~ 8192
    # 计算: input(16->h1) + h1->h2 + h2->1 = 16*h1 + h1*h2 + h2*1
    # 目标中值: ~6000
    # [72, 48]: 16*72 + 72*48 + 48 = 1152 + 3456 + 48 = 4656 (no MIM)
    # [80, 56]: 16*80 + 80*56 + 56 = 1280 + 4480 + 56 = 5816
    1: {"hidden_dims": [80, 56], "target_range": "4096-8192"},
    
    # Level 2: 8192 ~ 16384
    # [112, 72]: 16*112 + 112*72 + 72 = 1792 + 8064 + 72 = 9928
    # [120, 80]: 16*120 + 120*80 + 80 = 1920 + 9600 + 80 = 11600
    2: {"hidden_dims": [120, 80], "target_range": "8192-16384"},
    
    # Level 3: 16384 ~ 32768
    # [160, 96]: 16*160 + 160*96 + 96 = 2560 + 15360 + 96 = 18016
    # [168, 104]: 16*168 + 168*104 + 104 = 2688 + 17472 + 104 = 20264
    3: {"hidden_dims": [168, 104], "target_range": "16384-32768"},
    
    # Level 4: 32768 ~ 65536
    # [240, 120, 48]: 16*240 + 240*120 + 120*48 + 48*1 = 3840 + 28800 + 5760 + 48 = 38448
    # [256, 128, 56]: 16*256 + 256*128 + 128*56 + 56 = 4096 + 32768 + 7168 + 56 = 44088
    4: {"hidden_dims": [256, 128, 56], "target_range": "32768-65536"},
}

# LSTM配置 - 调整后的hidden_size
lstm_configs_final = {
    # Level 1: 4096 ~ 8192
    # 计算: 4 * (input_size + hidden_size + 1) * hidden_size (per layer) + output
    # h=40, l=1: 4*(16+40)*40 + 40*1 = 4*56*40 + 40 = 8960 + 40 = 9000 (接近上限)
    # h=36, l=1: 4*(16+36)*36 + 36 = 4*52*36 + 36 = 7488 + 36 = 7524
    1: {"hidden_size": 28, "num_layers": 1, "target_range": "4096-8192"},
    
    # Level 2: 8192 ~ 16384
    # h=52, l=1: 4*(16+52)*52 + 52 = 4*68*52 + 52 = 14144 + 52 = 14196
    2: {"hidden_size": 44, "num_layers": 1, "target_range": "8192-16384"},
    
    # Level 3: 16384 ~ 32768
    # h=44, l=2: 2层LSTM + output
    # Layer1: 4*(16+44)*44 = 4*60*44 = 10560
    # Layer2: 4*(44+44)*44 = 4*88*44 = 15488
    # Output: 44*1 = 44
    # Total: 10560 + 15488 + 44 = 26092
    3: {"hidden_size": 44, "num_layers": 2, "target_range": "16384-32768"},
    
    # Level 4: 32768 ~ 65536
    # h=64, l=2:
    # Layer1: 4*(16+64)*64 = 4*80*64 = 20480
    # Layer2: 4*(64+64)*64 = 4*128*64 = 32768
    # Output: 64*1 = 64
    # Total: 20480 + 32768 + 64 = 53312
    4: {"hidden_size": 64, "num_layers": 2, "target_range": "32768-65536"},
}

# CNN配置 - 调整后的channels
cnn_configs_final = {
    # Level 1: 4096 ~ 8192
    # 计算: Conv1: in*out*k + out + Conv2: in*out*k + out + FC: in*1 + 1
    # [48, 32], k=3: 
    # Conv1: 16*48*3 + 48 = 2304 + 48 = 2352
    # Conv2: 48*32*3 + 32 = 4608 + 32 = 4640
    # FC: 32*1 + 1 = 33
    # Total: 2352 + 4640 + 33 = 7025
    1: {"channels": [40, 28], "kernel_size": 3, "target_range": "4096-8192"},
    
    # Level 2: 8192 ~ 16384
    # [72, 40], k=3:
    # Conv1: 16*72*3 + 72 = 3456 + 72 = 3528
    # Conv2: 72*40*3 + 40 = 8640 + 40 = 8680
    # FC: 40*1 + 1 = 41
    # Total: 3528 + 8680 + 41 = 12249
    2: {"channels": [72, 40], "kernel_size": 3, "target_range": "8192-16384"},
    
    # Level 3: 16384 ~ 32768
    # [96, 56], k=3:
    # Conv1: 16*96*3 + 96 = 4608 + 96 = 4704
    # Conv2: 96*56*3 + 56 = 16128 + 56 = 16184
    # FC: 56*1 + 1 = 57
    # Total: 4704 + 16184 + 57 = 20945
    3: {"channels": [96, 56], "kernel_size": 3, "target_range": "16384-32768"},
    
    # Level 4: 32768 ~ 65536
    # [136, 72], k=3:
    # Conv1: 16*136*3 + 136 = 6528 + 136 = 6664
    # Conv2: 136*72*3 + 72 = 29376 + 72 = 29448
    # FC: 72*1 + 1 = 73
    # Total: 6664 + 29448 + 73 = 36185
    4: {"channels": [136, 72], "kernel_size": 3, "target_range": "32768-65536"},
}


# 打印结果
print("=" * 100)
print("Neural Network Architecture Design - 12 Model Configurations")
print("=" * 100)
print(f"{'Param Range':<20} {'Level 1':<20} {'Level 2':<20} {'Level 3':<20} {'Level 4':<20}")
print(f"{'':20} {'4096-8192':<20} {'8192-16384':<20} {'16384-32768':<20} {'32768-65536':<20}")
print("=" * 100)

results = []

# MLP Results
print("\n" + "=" * 100)
print("MLP (Multi-Layer Perceptron)")
print("=" * 100)
print(f"{'Level':<8} {'Hidden Dims':<25} {'Params(no MIM)':<18} {'Params(MIM)':<18} {'Status':<10}")
print("-" * 100)

for level in range(1, 5):
    cfg = mlp_configs_final[level]
    # 无MIM (input_dim=16)
    model_no_mim = MLP(input_dim_no_mim, cfg["hidden_dims"])
    params_no_mim = count_parameters(model_no_mim)
    
    # 有MIM (input_dim=32)
    model_mim = MLP(input_dim_mim, cfg["hidden_dims"])
    params_mim = count_parameters(model_mim)
    
    low, high = get_param_ranges()[level]
    status_no_mim = "OK" if low <= params_no_mim < high else "FAIL"
    status_mim = "OK" if low <= params_mim < high else "FAIL"
    status = "OK" if status_no_mim == "OK" and status_mim == "OK" else "FAIL"
    
    desc = str(cfg["hidden_dims"])
    print(f"{level:<8} {desc:<25} {params_no_mim:<18,} {params_mim:<18,} {status:<10}")
    
    results.append({
        "model": "MLP",
        "level": level,
        "config": cfg,
        "params_no_mim": params_no_mim,
        "params_mim": params_mim,
        "status": status
    })

# LSTM Results
print("\n" + "=" * 100)
print("LSTM (Long Short-Term Memory)")
print("=" * 100)
print(f"{'Level':<8} {'Hidden Config':<25} {'Params(no MIM)':<18} {'Params(MIM)':<18} {'Status':<10}")
print("-" * 100)

for level in range(1, 5):
    cfg = lstm_configs_final[level]
    # 无MIM (input_dim=16)
    model_no_mim = LSTMModel(input_dim_no_mim, cfg["hidden_size"], cfg["num_layers"])
    params_no_mim = count_parameters(model_no_mim)
    
    # 有MIM (input_dim=32)
    model_mim = LSTMModel(input_dim_mim, cfg["hidden_size"], cfg["num_layers"])
    params_mim = count_parameters(model_mim)
    
    low, high = get_param_ranges()[level]
    status_no_mim = "OK" if low <= params_no_mim < high else "FAIL"
    status_mim = "OK" if low <= params_mim < high else "FAIL"
    status = "OK" if status_no_mim == "OK" and status_mim == "OK" else "FAIL"
    
    desc = f"h={cfg['hidden_size']}, l={cfg['num_layers']}"
    print(f"{level:<8} {desc:<25} {params_no_mim:<18,} {params_mim:<18,} {status:<10}")
    
    results.append({
        "model": "LSTM",
        "level": level,
        "config": cfg,
        "params_no_mim": params_no_mim,
        "params_mim": params_mim,
        "status": status
    })

# CNN Results
print("\n" + "=" * 100)
print("CNN1D (1D Convolutional Network)")
print("=" * 100)
print(f"{'Level':<8} {'Channel Config':<25} {'Params(no MIM)':<18} {'Params(MIM)':<18} {'Status':<10}")
print("-" * 100)

for level in range(1, 5):
    cfg = cnn_configs_final[level]
    # 无MIM (input_dim=16)
    model_no_mim = CNN1D(input_dim_no_mim, cfg["channels"], cfg["kernel_size"])
    params_no_mim = count_parameters(model_no_mim)
    
    # 有MIM (input_dim=32)
    model_mim = CNN1D(input_dim_mim, cfg["channels"], cfg["kernel_size"])
    params_mim = count_parameters(model_mim)
    
    low, high = get_param_ranges()[level]
    status_no_mim = "OK" if low <= params_no_mim < high else "FAIL"
    status_mim = "OK" if low <= params_mim < high else "FAIL"
    status = "OK" if status_no_mim == "OK" and status_mim == "OK" else "FAIL"
    
    desc = f"{cfg['channels']}, k={cfg['kernel_size']}"
    print(f"{level:<8} {desc:<25} {params_no_mim:<18,} {params_mim:<18,} {status:<10}")
    
    results.append({
        "model": "CNN",
        "level": level,
        "config": cfg,
        "params_no_mim": params_no_mim,
        "params_mim": params_mim,
        "status": status
    })

# 汇总表
print("\n" + "=" * 100)
print("Configuration Summary Table")
print("=" * 100)
print(f"{'Model':<10} {'Level':<8} {'Param Range':<20} {'Key Config':<40}")
print("-" * 100)

ranges = get_param_ranges()
for level in range(1, 5):
    low, high = ranges[level]
    
    # MLP
    mlp_cfg = mlp_configs_final[level]
    print(f"{'MLP':<10} {level:<8} {low:,} ~ {high-1:,}{'':<6} {str(mlp_cfg['hidden_dims']):<40}")
    
    # LSTM
    lstm_cfg = lstm_configs_final[level]
    lstm_desc = f"hidden={lstm_cfg['hidden_size']}, layers={lstm_cfg['num_layers']}"
    print(f"{'LSTM':<10} {level:<8} {low:,} ~ {high-1:,}{'':<6} {lstm_desc:<40}")
    
    # CNN
    cnn_cfg = cnn_configs_final[level]
    cnn_desc = f"channels={cnn_cfg['channels']}, kernel={cnn_cfg['kernel_size']}"
    print(f"{'CNN':<10} {level:<8} {low:,} ~ {high-1:,}{'':<6} {cnn_desc:<40}")
    
    if level < 4:
        print("-" * 100)

# 生成配置文件
print("\n" + "=" * 100)
print("Generating Hydra YAML Configurations...")
print("=" * 100)

# 检查所有配置是否都通过
all_ok = all(r["status"] == "OK" for r in results)
if all_ok:
    print("\n[SUCCESS] All 12 model configurations are within target parameter ranges!")
else:
    print("\n[WARNING] Some configurations are outside target ranges:")
    for r in results:
        if r["status"] != "OK":
            print(f"  - {r['model']} Level {r['level']}: noMIM={r['params_no_mim']}, MIM={r['params_mim']}")
