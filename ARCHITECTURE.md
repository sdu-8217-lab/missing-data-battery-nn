# Architecture Documentation

## 项目概述

基于 **Hydra + PyTorch Lightning** 的电池 SOH 预测框架，对比 **MIM (Missing Indicator Method)** 与 **传统插补方法**。

## 技术栈

| 组件 | 用途 |
|------|------|
| **Hydra** | 配置管理，分层配置和命令行覆盖 |
| **PyTorch Lightning** | 训练框架，自动处理 GPU/早停/日志 |
| **Weights & Biases** | 实验追踪和可视化 (可选) |
| **OmegaConf** | 结构化配置 |
| **pandas/numpy** | 数据处理 |
| **scikit-learn** | 评估指标和数据划分 |

## 项目结构

```
missing-data-battery-nn/
├── configs/                    # Hydra 配置
│   ├── config.yaml            # 主配置 (实验名称、种子列表)
│   ├── data/                  # 数据集配置 (xjtu/tju/hust/mit)
│   ├── model/                 # 模型配置 (mlp/lstm/gru/cnn1d)
│   └── missing/               # 缺失机制配置 (mcar/mar)
│
├── src/
│   ├── main.py                # 唯一入口
│   ├── data/                  # 数据模块
│   │   ├── loader.py         # 电池级数据加载
│   │   ├── loader_dataloaders.py  # DataLoader创建 (滑动窗口)
│   │   ├── splits.py         # 电池划分逻辑
│   │   ├── features.py       # 特征工程
│   │   └── preprocessing.py  # 数据清洗 + 标准化
│   │
│   ├── models/                # 模型模块
│   │   ├── __init__.py       # MLP, LSTM, GRU, CNN1D 定义
│   │   └── lightning_module.py  # PyTorch Lightning 包装
│   │
│   ├── missing_data/          # 缺失数据模块
│   │   ├── mcar.py           # MCAR 缺失模拟
│   │   ├── mar.py            # MAR 缺失模拟
│   │   └── imputation.py     # Mean/Median/KNN/Zero 插补
│   │
│   └── utils/                 # 工具模块
│       └── seed_manager.py   # 随机种子管理
│
├── data/                      # 数据目录
│   ├── XJTU data/            # 西安交通大学数据集
│   ├── TJU data/             # 天津大学数据集
│   ├── HUST data/            # 华中科技大学数据集
│   └── MIT data/             # 麻省理工大学数据集
│
└── results/                   # 实验结果 (gitignored)
```

## 核心特性

### 1. 缺失机制

#### MCAR (Missing Completely At Random)
- 完全随机缺失
- 每个特征独立以概率 `p` 缺失

#### MAR (Missing At Random) - 修正版
- 依赖于 SOH 的缺失
- **修正公式**: `p_i = alpha * SOH_i^beta + gamma` (beta > 0)
- 物理意义: SOH 越高(新电池) → 缺失率越低

```python
# MAR 实现 (mar.py)
def simulate_mar(X, sohs, missing_rate, alpha=None, beta=2.0, gamma=0.05):
    # 计算 E[SOH^beta]
    expected_term = (sohs ** beta).mean()
    
    # 反推 alpha
    alpha = (missing_rate - gamma) / expected_term
    
    # 每个样本的缺失概率
    p = alpha * (sohs ** beta) + gamma
    
    # 生成掩码并返回 MIM 输入
    mask = (torch.rand(N, D) > p.unsqueeze(1)).float()
    mim_input = torch.cat([X_imputed, 1.0 - mask], dim=1)  # [N, 32]
```

### 2. 方法对比

| 方法 | 训练数据 | 测试处理 | 输入维度 | 特点 |
|------|---------|---------|---------|------|
| **MIM** | 混合缺失率 0.0-0.9 | 直接 MIM 格式 | 32 | 端到端，不插补 |
| **Mean** | 完整数据 | 缺失→均值填充 | 16 | 简单 baseline |
| **KNN** | 完整数据 | 缺失→KNN填充 | 16 | 局部相关性 |

### 3. 支持的模型

| 模型 | dim=16 配置 | dim=32 配置 | 参数量 |
|------|------------|-------------|--------|
| MLP | [100,64,32] | [84,56,28] | ~10K |
| LSTM | hidden=48×2 | hidden=48×2 | ~10K |
| GRU | hidden=64×2 | hidden=64×2 | ~9K |
| CNN1D | [72,32] | [64,32] | ~10K |

### 4. 数据划分 (按电池)

```python
# 原则: 同一电池只属于一个集合
battery_ids = ['2C_battery-1', '2C_battery-2', ..., '2C_battery-8']
train_ids = ['2C_battery-3', '2C_battery-4', '2C_battery-5', '2C_battery-7']
val_ids = ['2C_battery-1', '2C_battery-8']
test_ids = ['2C_battery-2', '2C_battery-6']
```

### 5. 滑动窗口序列

```python
# LSTM/GRU/CNN: 滑动窗口构造真实序列
def create_sliding_windows(X, y, seq_len=5):
    for i in range(N - seq_len + 1):
        window = X[i:i+seq_len]      # [seq_len, D]
        target = y[i+seq_len-1]      # 窗口最后一个时间步
    return X_seq, y_seq  # [N-seq_len+1, seq_len, D]
```

## 使用方法

### 运行实验

```bash
# MIM 方法
python src/main.py data=xjtu model=mlp method=mim

# Baselines
python src/main.py data=xjtu model=mlp method=mean
python src/main.py data=xjtu model=mlp method=knn

# 组合配置
python src/main.py data=xjtu model=lstm method=mim missing=mar

# 覆盖参数
python src/main.py training.epochs=200 training.batch_size=32
```

### 结果可视化

```bash
python plot_results.py
# 生成: results/comparison_plot.png
```

## 关键接口

### 数据加载

```python
from src.data import load_dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
data = load_dataset(cfg)
# Returns: dict with X_train, y_train, X_val, y_val, X_test, y_test
```

### 缺失数据模拟

```python
from src.missing_data import simulate_mcar, simulate_mar

# MCAR
X_imputed, mask, mim_input = simulate_mcar(X, missing_rate=0.3, seed=42)

# MAR (修正公式)
X_imputed, mask, mim_input = simulate_mar(
    X, sohs, missing_rate=0.3, alpha=None, beta=2.0, gamma=0.05, seed=42
)
```

### 模型创建

```python
from src.models import create_model

model = create_model('mlp', input_dim=32, hidden_dims=[84,56,28], dropout=0.15)
```

## 结果格式

CSV 结果文件 (`results/battery_soh_{dataset}_{model}_{method}.csv`):
```csv
seed,missing_rate,model,method,test_mae,test_rmse,test_r2
42,0.1,mlp,mim,0.0104,0.0132,0.9989
42,0.3,mlp,mim,0.0111,0.0141,0.9987
```

## 扩展指南

### 添加新数据集
1. 创建 `configs/data/new_dataset.yaml`
2. 在 `src/data/loader.py` 的 `load_dataset()` 中添加加载逻辑

### 添加新模型
1. 在 `src/models/__init__.py` 中定义模型类
2. 创建 `configs/model/new_model.yaml`
3. 更新 `src/main.py` 的 `get_model_config()`

### 添加新的缺失机制
1. 创建 `src/missing_data/new_mechanism.py`
2. 实现 `simulate_new(X, ...)` 函数
3. 创建 `configs/missing/new_mechanism.yaml`

---

*Last updated: 2026-02-26*  
*Note: MAR formula corrected from `(1-SOH)^beta` to `SOH^beta`*
