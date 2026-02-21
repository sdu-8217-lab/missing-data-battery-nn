# AGENTS.md - 缺参自适项目指南

本文件为AI编程助手提供项目背景、架构和使用指南，帮助快速理解和操作本项目。

---

## 项目概述

**缺参自适**是基于神经网络的电池状态（SOH - State of Health）鲁棒预测框架。核心创新是在输入数据存在缺失的条件下，实现自适应状态估计，而无需数据插补。

### 核心概念

- **MIM（Missing Indicator Method）**: 缺失指示器方法，通过将原始特征与缺失指示器拼接（16维→32维），让神经网络明确知道哪些特征缺失
- **Baseline**: 基线方法，仅使用完整数据进行训练，不处理缺失情况
- **SOH**: 电池健康状态（State of Health），本项目的预测目标

### 实验设计

- **4种神经网络模型**: MLP、LSTM、GRU、1D-CNN（每种都有Baseline和MIM版本）
- **9种缺失率**: 0.1 ~ 0.9（测试时）
- **100次重复实验**: 使用不同随机种子（42~141）确保统计显著性
- **MIM训练策略**: 将训练集复制10份，分别施加0.0~0.9的缺失率后拼接训练

---

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 编程语言 |
| PyTorch | 2.5.1+ | 深度学习框架 |
| NumPy | 1.24+ | 数值计算 |
| Pandas | 2.0+ | 数据处理 |
| scikit-learn | 1.3+ | 机器学习工具 |
| Matplotlib/Seaborn | 3.7+/0.12+ | 可视化 |
| XGBoost | 2.0+ | 梯度提升树（可选） |

---

## 项目结构

```
.
├── src/                          # 源代码目录
│   ├── config/                   # 配置管理
│   │   └── experiment_config.py  # 实验配置类
│   ├── data/                     # 数据层
│   │   ├── dataset_loader.py     # XJTU数据集加载
│   │   └── datasets.py           # PyTorch Dataset类
│   ├── models/                   # 模型层（引用外部工厂）
│   ├── trainers/                 # 训练层
│   │   └── neural_network_trainer.py
│   ├── evaluators/               # 评估层
│   │   ├── metrics.py            # 评估指标
│   │   └── model_evaluator.py    # 模型评估器
│   ├── experiments/              # 实验层
│   │   ├── single_experiment.py  # 小实验运行器
│   │   ├── batch_experiment.py   # 大实验运行器
│   │   └── checkpoint.py         # 检查点管理
│   ├── visualization/            # 可视化层
│   │   ├── single_plots.py       # 单实验绘图
│   │   ├── batch_plots.py        # 批量实验统计图
│   │   └── missing_rate_curves.py
│   ├── utils/                    # 工具层
│   │   ├── seed_manager.py       # 随机种子管理
│   │   └── logger_v2.py          # 日志配置
│   └── main.py                   # 主入口
├── data/                         # 数据集目录
│   ├── XJTU data/               # XJTU电池数据集
│   ├── MIT data/                # MIT电池数据集
│   ├── TJU data/                # TJU电池数据集
│   └── HUST data/               # HUST电池数据集
├── docs/                         # 文档目录
│   └── 文档索引.txt              # 文档索引与摘要
├── experiments/                  # 实验结果目录
├── experiments_v2/              # 新架构实验结果
├── run_single.py                # 运行单个小实验脚本
├── run_batch.py                 # 运行大实验脚本
├── plot_single.py               # 单实验绘图脚本
├── plot_batch.py                # 批量绘图脚本
└── requirements.txt             # 依赖列表
```

---

## 核心模块说明

### 1. 数据层 (src/data/)

**XJTUDatasetLoader**: 加载XJTU数据集的CSV文件，支持按电池划分训练/验证/测试集

**Dataset类**:
- `BatteryDataset`: 基础数据集，返回单循环特征 [batch, features]
- `SequenceDataset`: 序列数据集，用于LSTM/GRU/CNN [batch, seq_len, features]
- `MIMDataset`: MIM训练数据集，支持多缺失率混合

**关键参数**:
- `missing_rate`: 缺失率（0.0~1.0）
- `use_mim`: 是否使用缺失指示器（True时输入维度翻倍）
- `seq_len`: 序列长度（默认为5）

### 2. 实验层 (src/experiments/)

**分层架构**:
- **小实验 (SingleExperiment)**: 固定随机种子，训练8种模型配置，评估9种缺失率
- **大实验 (BatchExperiment)**: 批量运行100个小实验，支持中断恢复

**检查点机制**: 实验中断后可使用 `--resume` 参数恢复

### 3. 训练策略

**Baseline训练**:
```
输入: 完整训练集 (missing_rate=0.0)
输出: 训练好的模型
```

**MIM训练**:
```
输入: 训练集复制10份
处理:
  - copy_0: missing_rate=0.0 (完整)
  - copy_1~9: missing_rate=0.1~0.9
  - 每份数据：原始特征 + 缺失指示器（拼接为32维）
  - 合并所有副本（10×原大小）
输出: 训练好的模型
```

### 4. 特征说明

**16个输入特征**:
- 电压统计: mean, std, kurtosis, skewness
- 电流统计: mean, std, kurtosis, skewness
- 充电特征: CC Q, CC charge time, CV Q, CV charge time
- 变化率: voltage slope, current slope
- 熵值: voltage entropy, current entropy

**目标变量**: capacity（电池容量，用于计算SOH）

---

## 常用命令

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行实验

**完整大实验（100次重复）**:
```bash
python run_batch.py --batch 3C --n_repeats 100 --epochs 50
```

**单个小实验（调试用）**:
```bash
python run_single.py --batch 3C --seed 42 --epochs 50
```

**中断恢复**:
```bash
python run_batch.py --batch 3C --n_repeats 100 --resume
```

### 生成图表

**批量绘图（基于已完成结果）**:
```bash
python plot_batch.py --input experiments/3C/20240203_120000/
```

**单实验绘图**:
```bash
python plot_single.py --input experiments/3C/20240203_120000/seed_42/results.csv
```

### 快速测试
```bash
python test_quick.py
```

---

## 实验结果目录结构

```
experiments/{batch_name}/{timestamp}/
├── seed_42/                     # 各小实验结果
│   ├── results.csv             # 该种子的所有结果
│   ├── figures/                # 该种子的图表
│   ├── models/                 # 该种子的模型文件
│   └── seed_42.log             # 日志
├── seed_43/
│   └── ...
├── aggregate/                   # 聚合结果
│   ├── results_all.csv         # 所有种子合并结果
│   └── figures/                # 统计性图表
├── logs/
│   └── batch.log               # 大实验日志
└── checkpoint.json             # 检查点文件
```

---

## 代码规范

### 命名约定
- 类名: PascalCase (如 `BatchExperimentRunner`)
- 函数/变量: snake_case (如 `run_single_experiment`)
- 常量: UPPER_SNAKE_CASE
- 私有方法: `_leading_underscore`

### 文档字符串
所有模块、类和函数都应包含文档字符串，使用三重双引号：
```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    函数简短描述
    
    Args:
        arg1: 参数1说明
        arg2: 参数2说明
        
    Returns:
        返回值说明
    """
```

### 类型注解
鼓励使用类型注解提高代码可读性：
```python
from typing import List, Dict, Optional

def process_data(data: np.ndarray, missing_rate: float = 0.0) -> Dict[str, float]:
    ...
```

### 日志记录
使用项目提供的日志工具：
```python
from src.utils import setup_logger

logger = setup_logger("ModuleName", log_file_path)
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

---

## 模型配置

模型配置位于 `src/config/experiment_config.py`:

| 模型 | 架构 | 参数量(无MIM) | dropout |
|------|------|---------------|---------|
| MLP | [192, 96, 48, 24] | ~12K | 0.15 |
| LSTM | hidden=48, layer=2 | ~10K | 0.2 |
| GRU | hidden=64, layer=2 | ~11K | 0.2 |
| CNN1D | [72, 32], kernel=4 | ~10K | 0.1 |

**输入维度**:
- 无MIM: 16维（原始特征）
- 有MIM: 32维（原始特征 + 缺失指示器）

---

## 注意事项

### 数据要求
- 数据文件命名格式: `{batch}_battery-{id}.csv`
- 每个批次至少需要4个电池才能进行训练/验证/测试划分
- 数据文件应位于 `data/XJTU data/` 目录下

### 内存管理
- MIM训练会将训练集扩大10倍，注意内存使用
- 批量实验（100次重复）可能需要较长时间，建议使用检查点机制

### 随机种子
- 使用 `src.utils.set_seeds()` 统一设置PyTorch、NumPy和Python随机种子
- 不同缺失率使用不同种子（`seed + i`）确保独立性

### 评估指标
- MAE (Mean Absolute Error): 平均绝对误差
- RMSE (Root Mean Squared Error): 均方根误差
- R² (R-squared): 决定系数

---

## 扩展指南

### 添加新模型
1. 创建模型类，继承基类并实现 `fit`, `predict`, `save`, `load` 方法
2. 在 `ModelFactory.create_model` 中注册新模型类型
3. 在 `ExperimentConfig.get_model_configs()` 中添加配置

### 添加新数据集
1. 在 `src/data/` 下创建新的数据集加载器
2. 实现 `prepare_data()` 方法，返回统一格式的数据字典
3. 修改配置中的 `data_dir` 和 `batch` 参数

### 添加新可视化
1. 在 `src/visualization/` 下创建新的绘图模块
2. 在 `SingleExperimentPlots` 或 `BatchExperimentPlots` 中添加新方法
3. 在 `plot_all()` 中调用新方法

---

## 故障排查

### 内存不足
- 减小 `batch_size`
- 减少 `n_repeats`
- 使用更小的模型配置

### 实验中断
- 使用 `--resume` 参数恢复实验
- 或使用 `plot_batch.py` 手动生成已有结果的图表

### 数据加载错误
- 检查数据文件命名格式是否正确
- 确认 `data_dir` 路径设置正确
- 验证CSV文件包含所需的特征列

---

## 文档索引

详细文档位于 `docs/` 目录：

| 文档 | 内容 |
|------|------|
| 01-分层实验架构设计.md | 核心架构设计说明 |
| 03-代码架构与环境配置.md | 环境配置和架构详情 |
| 04-完整实验操作指南.md | 实验操作详细指南 |
| 08-src目录说明.md | 源代码结构说明 |
| 14-快速参考卡片.md | 常用命令速查 |
| 15-模型参数调优指南.md | 模型配置详情 |

---

*最后更新: 2026-02-08*
