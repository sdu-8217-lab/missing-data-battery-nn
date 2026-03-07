# 代码架构指南

> 了解项目代码结构和设计原则

---

## 架构设计原则

- **单一职责**：每个模块只负责一个明确的功能
- **接口统一**：不同模型实现统一的抽象接口，便于扩展
- **配置驱动**：实验参数通过配置文件管理，代码与配置分离
- **可复用性**：核心组件可在不同实验场景下复用
- **数据简化**：直接使用预处理好的CSV特征数据，无需原始数据解析

---

## 文件结构

```
src/
├── config/                          # 配置管理
│   └── pydantic_config.py           # 实验参数配置类
│
├── data/                            # 数据层
│   ├── loader.py                    # 数据加载
│   ├── loader_dataloaders.py        # DataLoader创建
│   ├── datasets.py                  # PyTorch Dataset类
│   ├── features.py                  # 特征工程
│   └── preprocessing.py             # 数据预处理
│
├── models/                          # 模型层
│   ├── mlp.py                       # MLP实现
│   ├── lstm.py                      # LSTM实现
│   ├── gru.py                       # GRU实现
│   ├── cnn1d.py                     # 1D-CNN实现
│   ├── model_factory.py             # 模型工厂
│   └── lightning_module.py          # Lightning模块
│
├── trainers/                        # 训练层
│   ├── lightning_module.py          # Lightning训练模块
│   ├── lightning_trainer.py         # Lightning训练器
│   ├── neural_network_trainer.py    # 传统训练器
│   └── xgboost_trainer.py           # XGBoost训练器
│
├── evaluators/                      # 评估层
│   ├── metrics.py                   # MAE/RMSE/R²计算
│   └── model_evaluator.py           # 模型评估器
│
├── experiments/                     # 实验层
│   ├── batch_experiment_runner_v2.py # 批量实验运行器
│   ├── experiment_runner.py         # 单次实验运行器
│   └── single_experiment.py         # 单实验配置
│
├── visualization/                   # 可视化层
│   ├── batch_plots.py               # 批量实验图表
│   ├── single_plots.py              # 单实验图表
│   └── missing_rate_curves.py       # 缺失率曲线
│
├── utils/                           # 工具层
│   ├── seed_manager.py              # 随机种子管理
│   └── logger.py                    # 日志配置
│
└── main.py                          # 主入口
```

---

## 核心模块说明

### 1. 数据层（data/）

| 文件 | 职责 |
|------|------|
| `loader.py` | 加载数据集，按电池划分训练/验证/测试集 |
| `loader_dataloaders.py` | 创建PyTorch DataLoader，支持MIM训练策略 |
| `datasets.py` | Dataset类定义：BatteryDataset, SequenceDataset, MIMDataset |

**Dataset类设计**：

```python
class BatteryDataset(Dataset):
    """基础数据集，返回单循环特征"""
    def __init__(self, X, y, missing_rate=0.0, use_mim=False)

class SequenceDataset(Dataset):
    """序列数据集，用于LSTM/GRU/1D-CNN"""
    def __init__(self, X, y, seq_len=5, missing_rate=0.0, use_mim=False)

class MIMDataset(Dataset):
    """MIM训练数据集，支持多缺失率混合"""
    def __init__(self, X, y, missing_rates=[0.0, 0.1, ..., 0.9])
```

### 2. 模型层（models/）

| 文件 | 职责 |
|------|------|
| `mlp.py` | MLP，输入: [batch, 16] 或 [batch, 32] (MIM) |
| `lstm.py` | LSTM，输入: [batch, seq_len, 16/32] |
| `gru.py` | GRU，输入: [batch, seq_len, 16/32] |
| `cnn1d.py` | 1D-CNN，输入: [batch, seq_len, 16/32] |
| `model_factory.py` | 工厂方法创建模型 |

**模型输入维度**：

| 模型 | Baseline输入 | MIM输入 |
|------|-------------|---------|
| MLP | [batch, 16] | [batch, 32] |
| LSTM/GRU/1D-CNN | [batch, seq_len, 16] | [batch, seq_len, 32] |

### 3. 训练层（trainers/）

| 文件 | 职责 |
|------|------|
| `lightning_module.py` | PyTorch Lightning模块 |
| `neural_network_trainer.py` | 传统训练循环，支持早停 |
| `xgboost_trainer.py` | XGBoost训练封装 |

**MIM训练策略**（在 `loader_dataloaders.py` 中实现）：

```python
# 训练时: 10个缺失率版本混合
for mr in [0.0, 0.1, ..., 0.9]:
    X_missing = apply_missing(X_train, mr)
    X_mim = concat([X_missing, missing_indicator])
    train(X_mim, y_train)

# 验证时: 固定缺失率
X_val_mim = apply_missing(X_val, mr=0.5)
validate(X_val_mim, y_val)
```

### 4. 缺失数据处理（missing_data/）

| 文件 | 职责 |
|------|------|
| `mcar.py` | MCAR（完全随机缺失）模拟 |
| `mar.py` | MAR（随机缺失）模拟 |
| `imputation.py` | 插补方法（均值、中位数、KNN等） |

### 5. 评估层（evaluators/）

| 文件 | 职责 |
|------|------|
| `metrics.py` | MAE、RMSE、R²计算 |
| `model_evaluator.py` | 批量评估模型在各缺失率下的表现 |

### 6. 可视化层（visualization/）

| 文件 | 职责 |
|------|------|
| `batch_plots.py` | 批量实验结果可视化 |
| `single_plots.py` | 单实验结果可视化 |
| `missing_rate_curves.py` | MAE随缺失率变化曲线 |

---

## 数据流时序图

```
单次实验流程:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Experiment │───→│  XJTU Data  │───→│   Model     │───→│  Trainer    │
│   Runner    │    │   Loader    │    │   Factory   │    │             │
└─────────────┘    └──────┬──────┘    └─────────────┘    └──────┬──────┘
                          │                                      │
                          │ (CSV直接加载)                        │
                          ↓                                      │
                   ┌─────────────┐                               │
                   │    Split    │                               │
                   │ Train/Val/Test                              │
                   └──────┬──────┘                               │
                          │                                      │
                          └──────────────────────────────────────┘
                                                                 │
                     ┌──────────────────────────────────────────┘
                     │
                     ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Visualizer │←───│  Evaluator  │←───│   Trained   │←───│   MIM       │
│             │    │             │    │    Model    │    │  Strategy   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 扩展性设计

- **新增模型**：继承`nn.Module`，在`model_factory.py`注册
- **新增数据集**：修改`loader.py`添加新数据格式支持
- **新增可视化**：在`visualization/`添加新模块

---

*最后更新: 2026-03-07*
