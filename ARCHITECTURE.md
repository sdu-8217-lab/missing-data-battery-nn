# Missing Data Battery SOH Prediction - 架构文档

## 项目概述

基于 **Hydra + PyTorch Lightning + Weights & Biases** 的现代深度学习实验框架，用于电池 SOH（State of Health）预测中的缺失数据机制研究。

## 技术栈

| 组件 | 用途 |
|------|------|
| **Hydra** | 配置管理，支持分层配置和命令行覆盖 |
| **PyTorch Lightning** | 训练框架，自动处理 GPU/早停/日志 |
| **Weights & Biases** | 实验追踪和可视化 |
| **OmegaConf** | 结构化配置 |
| **pandas/numpy** | 数据处理 |
| **scikit-learn** | 评估指标和数据划分 |

## 项目结构

```
missing-data-battery-nn/
├── configs/                    # Hydra 配置
│   ├── config.yaml            # 主配置
│   ├── data/                  # 数据集配置
│   │   ├── xjtu.yaml
│   │   ├── hust.yaml
│   │   └── mit.yaml
│   ├── models/                # 模型配置
│   │   ├── cnn1d.yaml
│   │   ├── mlp.yaml
│   │   ├── lstm.yaml
│   │   └── gru.yaml
│   ├── missing/               # 缺失机制配置
│   │   ├── mcar.yaml
│   │   └── mar.yaml
│   └── experiments/           # 实验配置
│       ├── baseline_mcar.yaml
│       ├── mim_mcar.yaml
│       ├── mim_mar_0.3.yaml
│       ├── mim_mar_0.6.yaml
│       └── mim_mar_0.9.yaml
├── src/
│   ├── main.py                # 唯一入口
│   ├── data/                  # 数据模块
│   │   ├── loader.py          # 数据加载
│   │   ├── preprocessing.py   # 数据清洗
│   │   ├── features.py        # 特征工程
│   │   └── splits.py          # 数据集划分
│   ├── missing_data/          # 缺失数据模块
│   │   ├── mcar.py            # MCAR 缺失模拟
│   │   └── mar.py             # MAR 缺失模拟
│   ├── models/                # 模型模块
│   │   ├── base_model.py      # Lightning 基类
│   │   ├── cnn1d.py           # 1D-CNN
│   │   ├── mlp.py             # MLP
│   │   ├── lstm.py            # LSTM
│   │   ├── gru.py             # GRU
│   │   └── model_factory.py   # 模型工厂
│   ├── trainers/              # 训练模块
│   │   ├── lightning_trainer.py
│   │   └── neural_network_trainer.py
│   ├── evaluation/            # 评估模块
│   │   ├── metrics.py
│   │   └── result_writer.py
│   ├── visualization/         # 可视化模块
│   │   ├── missing_rate_curves.py
│   │   └── heatmaps.py
│   └── utils/                 # 工具模块
│       ├── seed_manager.py
│       ├── logger.py
│       └── wandb_utils.py
├── tests/                     # 测试
│   ├── test_missing_generators.py
│   ├── test_data_loader.py
│   └── test_full_pipeline.py
├── data/                      # 数据目录
│   ├── raw/
│   └── processed/
├── results/                   # 结果目录
│   ├── csv/
│   └── images/
└── logs/                      # 日志目录
```

## 核心特性

### 1. 缺失机制

#### MCAR (Missing Completely At Random)
- 完全随机缺失
- 每个特征独立以概率 `p` 缺失

#### MAR (Missing At Random)
- 依赖于 SOH 的缺失
- 公式：`p_i = alpha * (1 - SOH_i)^beta + gamma`
- SOH 越低，缺失概率越高

### 2. MIM (Missing Indicator Method)
- 将缺失掩码作为额外输入特征
- 输入维度：32 (16 原始特征 + 16 缺失指示器)

### 3. 支持的模型

| 模型 | 输入维度 | 特点 |
|------|----------|------|
| CNN1D | 16/32 | 1D 卷积，适合序列特征 |
| MLP | 16/32 | 全连接，简单高效 |
| LSTM | 16/32 | 时序建模 |
| GRU | 16/32 | LSTM 变体，参数更少 |

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 默认实验 (configs/config.yaml 中指定)
python src/main.py

# 指定实验配置
python src/main.py experiment=mim_mar_0.3

# 指定模型
python src/main.py model=cnn1d

# 组合配置
python src/main.py experiment=mim_mar_0.6 model=lstm data=xjtu

# 覆盖参数
python src/main.py training.epochs=100 training.batch_size=32
```

### 运行测试

```bash
# 测试缺失数据生成器
python tests/test_missing_generators.py

# 测试数据加载
python tests/test_data_loader.py

# 测试完整流程
python tests/test_full_pipeline.py
```

### 可视化结果

```python
from src.visualization import plot_missing_rate_curves, plot_mim_improvement_heatmap

# 绘制性能曲线
plot_missing_rate_curves("results/csv/mim_mar_0.3.csv")

# 绘制 MIM 改善热力图
plot_mim_improvement_heatmap("baseline.csv", "mim.csv", metric="mae")
```

## 配置说明

### 实验配置示例 (mim_mar_0.3.yaml)

```yaml
name: "mim_mar_0.3"
description: "MIM with MAR, target MR=0.3"

type: "mim"
use_mim: true
missing_mode: "mar"
missing_rate: 0.3

mar_params:
  alpha: null      # 自动反推
  beta: 2.0
  gamma: 0.05

training:
  epochs: 200
  batch_size: 64
  learning_rate: 1e-3
  seeds: [42, 101, 102, ...]

output:
  result_csv: "results/csv/mim_mar_0.3.csv"
```

## 扩展指南

### 添加新数据集

1. 创建 `configs/data/new_dataset.yaml`
2. 实现 `src/data/loader.py` 中的 `_load_new_dataset()` 函数

### 添加新模型

1. 创建 `src/models/new_model.py`
2. 创建 `configs/models/new_model.yaml`
3. 在 `src/models/model_factory.py` 中注册

### 添加新的缺失机制

1. 创建 `src/missing_data/new_mechanism.py`
2. 实现 `simulate_new(X, ...)` 函数
3. 创建 `configs/missing/new_mechanism.yaml`

## 关键接口

### 缺失数据模拟

```python
from src.missing_data import simulate_mcar, simulate_mar

# MCAR
X_imputed, mask, mim_input = simulate_mcar(X, missing_rate=0.3, seed=42)

# MAR
X_imputed, mask, mim_input = simulate_mar(
    X, sohs, missing_rate=0.3, alpha=None, beta=2.0, gamma=0.05, seed=42
)
```

### 数据加载

```python
from src.data import load_dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
data = load_dataset(cfg)
# Returns: dict with X_train, y_train, X_val, y_val, X_test, y_test
```

### 模型创建

```python
from src.models import create_model

model = create_model(cfg)  # Returns LightningModule
```

## 结果格式

CSV 结果文件包含以下列：
- `seed`: 随机种子
- `missing_rate`: 缺失率
- `model`: 模型名称
- `missing_mode`: 缺失机制 (mcar/mar)
- `use_mim`: 是否使用 MIM
- `mae`: 平均绝对误差
- `rmse`: 均方根误差
- `r2`: R² 决定系数

## 开发计划

### Day 1: 基础架构
- [x] 目录结构
- [x] Hydra 配置体系
- [x] 数据模块
- [x] 缺失数据模块
- [x] 模型模块

### Day 2: 训练与评估
- [x] Lightning 训练器
- [x] 主实验入口
- [x] 评估指标
- [x] 结果写入
- [x] 可视化
- [x] 测试

### 后续: 实验运行
- [ ] 运行 MCAR Baseline
- [ ] 运行 MCAR MIM
- [ ] 运行 MAR MIM (0.3/0.6/0.9)
- [ ] 生成论文图表
