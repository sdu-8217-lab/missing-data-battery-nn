# 神经网络模型架构设计文档

> **12个模型配置 (3种模型 × 4种参数量级别)**

## 概述

本项目设计了12个神经网络模型，涵盖3种架构（MLP、LSTM、CNN）和4种参数量级别。

### 参数量级别定义

| 级别 | 参数范围 | 数值范围 |
|------|----------|----------|
| Level 1 | 2^12 ~ 2^13 | 4,096 ~ 8,191 |
| Level 2 | 2^13 ~ 2^14 | 8,192 ~ 16,383 |
| Level 3 | 2^14 ~ 2^15 | 16,384 ~ 32,767 |
| Level 4 | 2^15 ~ 2^16 | 32,768 ~ 65,535 |

### 输入维度

- **use_mim=false**: 16维输入
- **use_mim=true**: 32维输入（16特征 + 16掩码）

---

## MLP (多层感知机)

### 架构设计

| 级别 | 隐藏层维度 | Dropout | 参数量(无MIM) | 参数量(有MIM) | 状态 |
|------|-----------|---------|---------------|---------------|------|
| Level 1 | [80, 56] | 0.15 | 5,953 | 7,233 | ✓ |
| Level 2 | [120, 80] | 0.15 | 11,801 | 13,721 | ✓ |
| Level 3 | [168, 104] | 0.15 | 20,537 | 23,225 | ✓ |
| Level 4 | [256, 128, 56] | 0.15 | 44,529 | 48,625 | ✓ |

### 计算公式
```
参数量 = input_dim × h1 + h1 × h2 + h2 × 1 (两层)
参数量 = input_dim × h1 + h1 × h2 + h2 × h3 + h3 × 1 (三层)
```

---

## LSTM (长短期记忆网络)

### 架构设计

| 级别 | 隐藏层大小 | 层数 | Dropout | 参数量(无MIM) | 参数量(有MIM) | 状态 |
|------|-----------|------|---------|---------------|---------------|------|
| Level 1 | 28 | 1 | 0.2 | 5,181 | 6,973 | ✓ |
| Level 2 | 44 | 1 | 0.2 | 10,957 | 13,773 | ✓ |
| Level 3 | 44 | 2 | 0.2 | 26,797 | 29,613 | ✓ |
| Level 4 | 64 | 2 | 0.2 | 54,337 | 58,433 | ✓ |

### 计算公式
```
单层LSTM: 4 × (input_dim + hidden_size) × hidden_size + hidden_size × 1
多层LSTM: 各层参数之和 + 输出层
```

---

## CNN1D (一维卷积网络)

### 架构设计

| 级别 | 通道数 | 卷积核 | Dropout | 参数量(无MIM) | 参数量(有MIM) | 状态 |
|------|--------|--------|---------|---------------|---------------|------|
| Level 1 | [40, 28] | 3 | 0.1 | 5,377 | 7,297 | ✓ |
| Level 2 | [72, 40] | 3 | 0.1 | 12,249 | 15,705 | ✓ |
| Level 3 | [96, 56] | 3 | 0.1 | 20,945 | 25,553 | ✓ |
| Level 4 | [136, 72] | 3 | 0.1 | 36,185 | 42,713 | ✓ |

### 计算公式
```
Conv1: input_dim × out_channels × kernel_size + out_channels
Conv2: in_channels × out_channels × kernel_size + out_channels
FC: last_channels × 1 + 1
```

---

## 配置汇总表

| 模型 | Level 1 (4K-8K) | Level 2 (8K-16K) | Level 3 (16K-32K) | Level 4 (32K-64K) |
|------|-----------------|------------------|-------------------|-------------------|
| **MLP** | [80, 56] | [120, 80] | [168, 104] | [256, 128, 56] |
| **LSTM** | h=28, l=1 | h=44, l=1 | h=44, l=2 | h=64, l=2 |
| **CNN** | [40, 28] | [72, 40] | [96, 56] | [136, 72] |

---

## 使用示例

### Python代码

```python
from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.cnn1d import CNN1D

# MLP Level 3, use_mim=true (input_dim=32)
model = MLP(input_dim=32, hidden_dims=[168, 104], dropout=0.15)

# LSTM Level 2, use_mim=false (input_dim=16)
model = LSTM(input_dim=16, hidden_size=44, num_layers=1, dropout=0.2)

# CNN Level 4, use_mim=true (input_dim=32)
model = CNN1D(input_dim=32, channels=[136, 72], kernel_size=3, dropout=0.1)
```

### Hydra配置

```yaml
# config.yaml
model:
  name: "mlp"  # 或 "lstm", "cnn"
  level: 3     # 1, 2, 3, 4
  
  # MLP参数
  hidden_dims: [168, 104]
  dropout: 0.15
  
  # LSTM参数 (当name="lstm"时)
  # hidden_size: 44
  # num_layers: 2
  
  # CNN参数 (当name="cnn"时)
  # channels: [96, 56]
  # kernel_size: 3
```

---

## 设计考虑

### 1. 参数量分布
- 所有模型在两种输入维度（16/32）下都落在目标范围内
- 从Level 1到Level 4，参数量呈2倍递增趋势
- 考虑到了MIM模式（32维输入）下的参数量增长

### 2. 架构选择
- **MLP**: 简单前馈网络，适合基线对比
- **LSTM**: 捕捉时序依赖，适合电池容量衰减建模
- **CNN**: 提取局部特征，对噪声有一定鲁棒性

### 3. 超参数固定
- Dropout: MLP(0.15), LSTM(0.2), CNN(0.1)
- 激活函数: ReLU
- 输出层: 单个神经元（回归任务）

---

## 验证方法

```python
import torch
from src.models.mlp import MLP

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 验证MLP Level 3
model = MLP(input_dim=32, hidden_dims=[168, 104], dropout=0.15)
params = count_parameters(model)
print(f"MLP Level 3 (MIM): {params} parameters")  # 应输出约23,225

# 检查是否在范围内
assert 16384 <= params < 32768, "参数量不在目标范围内!"
```

---

## 文件说明

- `model_configs.yaml`: 纯配置文件，可直接用于Hydra
- `model_architecture_design_v2.py`: 设计脚本，用于验证参数量
- `MODEL_ARCHITECTURE_DESIGN.md`: 本文档，设计说明和汇总

---

*生成日期: 2026-03-27*
