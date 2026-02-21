# AGENTS.md - AI Coding Agent Guide

> 本文件为AI编码代理提供项目背景、架构说明和开发指南。
> 项目语言：中文（代码注释和文档主要使用中文）

## 项目概述

**项目名称**: 缺参自适 - 基于神经网络的电池状态鲁棒预测框架

**研究目标**: 在输入数据缺失条件下实现电池状态（SOH/SOC）的自适应估计。核心创新是"缺失指示器方法"（Missing Indicator Method, MIM）：不填充缺失值，而是将缺失事实（掩码）作为额外输入特征传递给神经网络，使模型明确知道哪些信息缺失。

**核心思想**: 
- 不掩盖数据缺失事实（不用均值/KNN等填充）
- 通过缺失指示器（1表示缺失，0表示存在）显式告知网络
- 输入维度从16（原始特征）扩展到32（特征+缺失指示器）

## 技术栈

| 类别 | 技术 | 版本/说明 |
|------|------|-----------|
| 编程语言 | Python | 3.8+ |
| 深度学习 | PyTorch | >=2.5.1 |
| 增强训练 | PyTorch Lightning | V2版本使用 |
| 数据科学 | NumPy, Pandas, Scikit-learn | 标准版本 |
| 可视化 | Matplotlib, Seaborn | 学术论文风格 |
| 配置验证 | Pydantic v2 | V2版本使用 |
| 日志 | Loguru (V2) / logging (V1) | UTF-8编码 |

## 项目结构

```
missing-data-battery-nn/
├── src/                          # 核心源代码
│   ├── config/                   # 配置管理
│   │   ├── experiment_config.py  # dataclass配置 (V1)
│   │   └── pydantic_config.py    # Pydantic配置 (V2)
│   ├── data/                     # 数据加载与处理
│   │   ├── dataset_loader.py     # XJTU数据集加载器
│   │   └── datasets.py           # PyTorch Dataset定义
│   ├── models/                   # 神经网络模型
│   │   ├── base_model.py         # 抽象基类
│   │   ├── mlp.py                # 多层感知机
│   │   ├── lstm.py               # 长短期记忆网络
│   │   ├── gru.py                # 门控循环单元
│   │   ├── cnn1d.py              # 1D卷积神经网络
│   │   └── model_factory.py      # 模型工厂
│   ├── trainers/                 # 训练器
│   │   ├── neural_network_trainer.py  # 基础训练器
│   │   ├── lightning_module.py   # Lightning模块 (V2)
│   │   └── lightning_trainer.py  # Lightning训练器 (V2)
│   ├── evaluators/               # 评估模块
│   │   ├── metrics.py            # 评估指标 (MAE, RMSE, R²)
│   │   └── model_evaluator.py    # 模型评估器
│   ├── experiments/              # 实验运行
│   │   ├── experiment_runner.py  # 主实验运行器
│   │   ├── batch_experiment_runner_v2.py  # 批量实验V2
│   │   └── single_experiment.py  # 单实验
│   ├── visualization/            # 可视化
│   │   ├── batch_plots.py        # 批量实验绘图
│   │   ├── single_plots.py       # 单实验绘图
│   │   └── missing_rate_curves.py # 缺失率曲线
│   └── utils/                    # 工具函数
│       ├── logger.py             # 日志配置
│       ├── logger_v2.py          # Loguru日志 (V2)
│       └── seed_manager.py       # 随机种子管理
├── data/                         # 数据目录 (原始数据)
│   └── XJTU data/                # XJTU电池数据集
├── configs/                      # 架构搜索配置
├── configs_v2/                   # V2架构搜索配置
├── configs_v3/                   # V3最佳参数配置 (当前使用)
├── experiments/                  # 实验结果输出
├── experiments_v2/               # V2实验结果
├── results/                      # 结果目录
├── figures_no_xgb/               # 论文图表
├── paper/                        # 论文相关材料
├── docs/                         # 文档 (中文)
├── test_*.py                     # 测试脚本
├── run_*.py                      # 运行脚本
└── *.bat                         # Windows批处理脚本
```

## 核心架构说明

### 1. 缺失指示器机制 (MIM)

```python
# 核心逻辑: 特征与缺失指示器拼接
X_with_missing = np.where(mask == 1, X, 0.0)       # 缺失位置置0
missing_indicators = (~mask).astype(np.float32)     # 1表示缺失
X_combined = np.concatenate([X_with_missing, missing_indicators], axis=1)
# 输入维度: 16 (原始) -> 32 (带指示器)
```

### 2. 模型架构 (configs_v3最佳参数)

| 模型 | 架构 | Dropout | 预期MAE |
|------|------|---------|---------|
| MLP | [192,96,48,24] 4层 | 0.15 | 0.0128 |
| LSTM | h=48, l=2 | 0.20 | 0.0077 |
| GRU | h=64, l=2 | 0.20 | 0.0069 |
| CNN1D | [72,32], kernel=4 | 0.10 | **0.0057** |

### 3. 数据流

```
原始数据 (XJTU CSV)
    ↓
XJTUDatasetLoader.load_battery()  # 加载+清洗+计算SOH
    ↓
按电池划分 Train/Val/Test (电池级别划分，非样本级别)
    ↓
StandardScaler 标准化
    ↓
Dataset (BatteryDataset / SequenceDataset / MIMDataset)
    ↓
DataLoader
    ↓
Model (BaseModel派生类)
    ↓
Metrics (MAE, RMSE, R²)
```

### 4. 实验类型

1. **基线实验 (Baseline)**: 在完整数据上训练，测试不同缺失率下的性能
2. **MIM实验**: 在混合缺失率数据上训练（0.0-0.9），测试不同缺失率下的性能
3. **对比**: Baseline vs MIM 在同一缺失率下的性能差异即为改进率

## 运行与测试

### 快速测试

```bash
# 最小化测试 (2轮训练，1次重复)
python test_quick.py

# V2版本快速测试
python run_big_experiment_v2.py --n_repeats 2 --epochs 5
```

### 完整实验

```bash
# V1版本 (使用dataclass配置)
python src/main.py --batch 3C --n_repeats 100 --epochs 100

# V2版本 (推荐，使用Pydantic + Lightning)
python run_big_experiment_v2.py --batch 3C --n_repeats 100 --epochs 200

# 批量实验V2
python run_batch_experiment_v2.py --batch 2C --n-repeats 10 --epochs 250
```

### Windows批处理

```bash
# 运行 2C 批次 10 次重复实验
run_2C_10repeats_v2.bat

# 运行完整100次重复实验
run_full_100_repeats.bat
```

## 开发约定

### 代码风格

1. **文档语言**: 代码注释和docstring使用中文
2. **命名规范**:
   - 类名: `PascalCase` (如 `ExperimentRunner`)
   - 函数/变量: `snake_case` (如 `missing_rate`, `train_model`)
   - 常量: `UPPER_CASE`
3. **类型注解**: 鼓励使用Python类型注解
4. **导入顺序**: 标准库 -> 第三方库 -> 本地模块

### 配置管理

V2版本使用Pydantic进行配置验证:

```python
from src.config.pydantic_config import ExperimentConfig

config = ExperimentConfig(
    batch="3C",           # 必须在预设列表中
    n_repeats=100,        # 范围 [1, 1000]
    epochs=200,           # 范围 [1, 1000]
    lr=0.001              # 范围 (0, 1]
)
```

### 模型开发规范

新增模型需要:

1. 继承 `BaseModel`:
```python
class NewModel(BaseModel):
    def __init__(self, input_dim: int = 16, use_mim: bool = False, ...):
        super().__init__(input_dim, use_mim)  # actual_input_dim自动计算
        # ...
```

2. 在 `ModelFactory.create_model()` 中注册

3. 支持序列模型（LSTM/GRU/CNN1D）需在 `experiment_runner.py` 中处理数据准备

### 日志规范

```python
# V1
from src.utils.logger import setup_logger
logger = setup_logger('name', 'path/to/log.log')

# V2
from src.utils.logger_v2 import setup_logger
logger = setup_logger('name', 'path/to/log.log')
logger.info("中文日志信息")  # 确保UTF-8编码
```

## 测试策略

### 测试文件

- `test_quick.py`: 快速功能测试（训练+评估流程）
- `test_data_loader.py`: 数据加载测试
- `test_new_architecture.py`: 新架构验证
- `test_full.py`: 完整流程测试

### 数据要求

项目使用**XJTU电池数据集**，数据文件命名格式:
```
{batch}_battery-{id}.csv
# 例如: 3C_battery-1.csv, 2C_battery-5.csv
```

数据目录结构:
```
data/XJTU data/
├── 2C_battery-1.csv
├── 2C_battery-2.csv
├── 3C_battery-1.csv
├── ...
└── Sim_satellite_battery-1.csv
```

**注意**: 如果数据文件缺失，程序会抛出 `FileNotFoundError`。

## 输出结构

实验结果保存在 `experiments_v2/{batch}/{timestamp}/`:

```
experiments_v2/
└── 3C/
    └── 20260115_143022/
        ├── config.json              # 实验配置
        ├── results_all.csv          # 完整结果 (每个模型×每个缺失率×每次重复)
        ├── results_partial.csv      # 中间结果
        ├── experiment_*.log         # 日志文件
        └── figures/                 # 可视化图表
            ├── mae_curves.png
            ├── improvement_rate.png
            └── seed_stability.png
```

## 关键注意事项

1. **随机种子**: 使用 `src/utils/seed_manager.py` 统一设置，确保可复现性
2. **数据划分**: 按电池划分（而非样本划分），避免数据泄漏
3. **缺失率范围**: 训练和测试缺失率范围为 [0.0, 0.9]，步长0.1
4. **设备管理**: V2版本自动检测CPU/CUDA，V1版本需手动指定
5. **内存管理**: 大数据集实验可能需要减小 `batch_size`

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| Pydantic验证错误 | 检查配置参数类型和范围 |
| CUDA内存不足 | 添加 `--batch_size 16` 或 `--device cpu` |
| 找不到数据文件 | 确认 `data/XJTU data/` 目录存在且文件命名正确 |
| 日志中文乱码 | 使用 `logger_v2.py` (Loguru) 替代标准logging |

## 相关文档

- `README.md`: 项目详细说明（Q&A形式）
- `BIG_EXPERIMENT_V2_GUIDE.md`: V2版本使用指南
- `CODE_IMPROVEMENT_REPORT.md`: 代码改进报告
- `analysis_summary.md`: 模型筛选结果分析
- `docs/`: 详细开发文档（中文）
