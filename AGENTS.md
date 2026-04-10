# AGENTS.md - AI Coding Agent Guide

> 本文件为AI编程助手提供项目背景、开发规范和关键信息。
> 项目语言：中文文档 + Python代码

---

## 项目概述

**项目名称**: `missing-data-battery-nn`  
**版本**: 0.4.0  
**核心目标**: 基于神经网络的电池健康状态(SOH)预测，系统对比 **MIM (Missing Indicator Method)** 与传统插补方法在缺失数据场景下的性能。

**核心研究问题**: 在相同插补策略下，添加MIM指示器是否能提升电池SOH预测性能？

---

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| **语言** | Python | >=3.12 |
| **深度学习** | PyTorch | >=2.10.0 |
| **训练框架** | PyTorch Lightning | >=2.6.0 |
| **数据处理** | pandas, numpy | >=2.0.0, >=1.26.0 |
| **机器学习** | scikit-learn | >=1.5.0 |
| **可视化** | matplotlib, seaborn | >=3.9.0, >=0.13.0 |
| **配置管理** | OmegaConf, Hydra | >=2.3.0, >=1.3.0 |
| **超参优化** | Optuna | >=4.0.0 |
| **实验追踪** | Weights & Biases | >=0.19.0 |
| **代码质量** | Black, isort, pre-commit | - |

---

## 9层实验架构

项目遵循 **meta.md** 中定义的9层层次化实验架构，核心是"分界线原则"：

```
╔══════════════════════════════════════════════════════════════╗
║  分界线以上（影响模型训练）                                   ║
╠══════════════════════════════════════════════════════════════╣
L1: Seed (随机种子)          - 控制可复现性
L2: Dataset (数据集)         - 固定使用XJTU数据集
L3: Batch (电池批次)         - 2C, 3C, R2.5, R3, RW, Sim_satellite (6个)
L4: Model (模型架构)         - MLP, LSTM, CNN (3种) × 4种参数量级别 = 12种配置
L5: use_mim (MIM使用)        - false/true (是否使用缺失指示器)
L6: Train MR (训练缺失率)    - use_mim=false时为0.0; use_mim=true时为0.0-0.95
╠══════════════════════════════════════════════════════════════╣
║  分界线：训练完成，模型参数固定                              ║
╠══════════════════════════════════════════════════════════════╣
L7: Mode (缺失模式)          - MCAR, MAR, MNAR (仅测试阶段)
L8: Test MR (测试缺失率)     - 0.0-0.95 (仅测试阶段)
L9: Imputation (插补方法)    - mean, knn, iterative, zero (仅测试阶段)
╚══════════════════════════════════════════════════════════════╝
```

**关键理解**:
- **分界线以上** (L1-L6): 每个组合需独立训练模型
- **分界线以下** (L7-L9): 同一模型可复用测试所有组合
- **总实验规模**: 100种子 × 6批次 × 12模型 × 2 MIM × 3模式 × 20 MR × 4插补

---

## 项目结构

```
.
├── meta.md                     # ⭐ 9层架构权威定义（只读）
├── README.md                   # 项目主文档
├── pyproject.toml              # Python项目配置
├── environment.yml             # Conda环境配置
├── model_configs.yaml          # 12种神经网络架构配置
│
├── src/                        # 源代码
│   ├── core/                   # 核心接口定义
│   │   ├── interfaces.py       # 抽象接口 (IDataLoader, IModel, etc.)
│   │   └── registry.py         # 组件注册中心
│   ├── data/                   # 数据层
│   │   ├── xjtu_loader.py      # XJTU数据加载器
│   │   ├── preprocessing.py    # 数据预处理
│   │   └── splits.py           # 电池划分 (Battery-wise Split)
│   ├── models/                 # 模型层
│   │   ├── mlp.py              # MLP模型
│   │   ├── lstm.py             # LSTM模型
│   │   ├── cnn1d.py            # 1D-CNN模型
│   │   └── factory.py          # 模型工厂
│   ├── missing_data/           # 缺失数据处理
│   │   ├── mcar.py             # MCAR缺失模拟
│   │   ├── mar.py              # MAR缺失模拟
│   │   ├── mnar.py             # MNAR缺失模拟
│   │   └── imputation.py       # 插补方法实现
│   ├── trainers/               # 训练器
│   │   ├── lightning_trainer.py # PyTorch Lightning训练器
│   │   └── lightning_module.py  # Lightning模块定义
│   ├── utils/                  # 工具函数
│   │   ├── seed_manager.py     # 随机种子管理
│   │   └── logger.py           # 日志工具
│   └── experiments/            # 实验执行模块
│       ├── runner.py           # 实验运行器
│       └── database.py         # 结果数据库
│
├── experiments/                # 实验入口脚本
│   ├── run_batch_experiments.py    # 批量实验主入口
│   ├── run_experiment.py           # 单次实验执行
│   ├── run_single.py               # 快速单次测试
│   ├── evaluate.py                 # 结果评估
│   └── visualization_*.py          # 可视化脚本
│
├── tests/                      # 测试
│   └── test_critical_path.py   # 关键路径测试
│
├── configs/                    # 配置文件
├── data/                       # 数据集
│   └── XJTU data/              # XJTU电池数据集
├── models/                     # 保存的训练模型
├── results/                    # 实验结果 (JSON/CSV)
├── scripts/                    # 辅助脚本
└── docs/                       # 文档
    ├── ARCHITECTURE.md         # 代码架构说明
    └── EXPERIMENTS.md          # 实验指南
```

---

## 构建与运行

### 环境配置

```bash
# 方法1: Conda (推荐)
conda env create -f environment.yml
conda activate battery-nn

# 方法2: pip
pip install -r requirements.txt
```

### 运行实验

#### 1. 单次实验 (快速测试)

```bash
# 训练
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 50

# 测试
python experiments/run_experiment.py \
    --phase test \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --mode MCAR \
    --test-mr 0.3 \
    --imputation mean
```

#### 2. 批量实验

```bash
# 完整实验 (训练+测试)
python experiments/run_batch_experiments.py \
    --phase full \
    --epochs 50

# 指定子集
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 42 43 \
    --batches 2C 3C \
    --epochs 50
```

#### 3. 大规模实验 (100种子)

```bash
nohup python experiments/run_batch_experiments.py \
    --phase full \
    --seeds $(seq 0 99) \
    --epochs 50 \
    --model-dir models/100seeds \
    --results-dir results/100seeds \
    > experiment_100seeds.log 2>&1 &
```

---

## 测试

```bash
# 运行关键路径测试
python -m pytest tests/test_critical_path.py -v

# 运行所有测试
python -m pytest tests/ -v

# 排除慢测试
python -m pytest tests/ -v -m "not slow"
```

### 测试标记

| 标记 | 说明 |
|------|------|
| `slow` | 耗时较长的测试 |
| `gpu` | 需要GPU的测试 |
| `experiment` | 实验集成测试 |

---

## 代码风格规范

### 格式化工具

| 工具 | 用途 | 配置 |
|------|------|------|
| **Black** | Python代码格式化 | line-length=100 |
| **isort** | Import排序 | profile=black |
| **pre-commit** | 提交前检查 | 见 `.pre-commit-config.yaml` |

### 提交前检查

```bash
# 安装pre-commit钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

### 提交消息规范 (Conventional Commits)

```
<type>[optional scope]: <简短描述（中文）>

[optional body]

[optional footer]
```

**常用类型**:

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug修复 |
| `refactor` | 代码重构 |
| `docs` | 文档更新 |
| `test` | 测试相关 |
| `chore` | 构建/工具 |
| `experiments` | 实验结果 |
| `paper` | 论文相关 |

**作用域(Scope)**: `data`, `models`, `experiments`, `configs`, `scripts`, `docs`, `paper`, `root`

**示例**:
```
feat(models): 添加GRU模型支持

fix(data): 修复XJTU数据加载器内存泄漏问题

experiments(xjtu-2c): 添加100种子实验结果
```

---

## 关键设计原则

### 1. Battery-wise Split

同一电池的所有循环只属于train/val/test中的一个，避免数据泄漏。

```python
# ✅ 正确：整个电池只属于一个集合
Train batteries: [电池3, 4, 5, 7]
Val batteries:   [电池1, 8]
Test batteries:  [电池2, 6]
```

实现: `src/data/splits.py`

### 2. MIM维度处理

```python
# 传统方法 (16维输入)
input = [feature_values]  # shape: [batch, 16]

# MIM方法 (32维输入)
input = [feature_values] + [missing_mask]  # shape: [batch, 32]
# missing_mask: 1表示缺失，0表示存在
```

### 3. 模型参数量控制

所有模型参数量控制在 **2^12 ~ 2^16** (约4K-64K) 范围内，按4个级别配置:

| 级别 | 参数范围 | 配置键 |
|------|----------|--------|
| Level 1 | 4,096 ~ 8,192 | `level_1` |
| Level 2 | 8,192 ~ 16,384 | `level_2` |
| Level 3 | 16,384 ~ 32,768 | `level_3` |
| Level 4 | 32,768 ~ 65,536 | `level_4` |

配置见: `model_configs.yaml`

---

## 扩展指南

### 添加新模型

1. 创建模型文件 `src/models/new_model.py`:
```python
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        # 实现模型结构
        
    def forward(self, x):
        # 实现前向传播
        return output
```

2. 在 `src/models/factory.py` 注册模型

3. 更新 `model_configs.yaml` 添加配置

### 添加新缺失模式

1. 创建模拟函数 `src/missing_data/new_mode.py`:
```python
def simulate_new_mode(X, y, missing_rate, seed):
    """生成新缺失模式
    
    Args:
        X: 特征矩阵 [N, 16]
        y: 目标值 [N]
        missing_rate: 缺失率 0.0-1.0
        seed: 随机种子
    
    Returns:
        X_missing: 缺失后的特征
        mask: 缺失掩码 [N, 16]
    """
    # 实现缺失生成逻辑
    return X_missing, mask
```

2. 在实验脚本中添加模式处理分支

---

## 重要文件参考

| 文件 | 用途 | 重要性 |
|------|------|--------|
| `meta.md` | 9层架构权威定义 | ⭐⭐⭐ 核心参考 |
| `docs/ARCHITECTURE.md` | 代码架构说明 | ⭐⭐⭐ 开发必读 |
| `README.md` | 项目主文档 | ⭐⭐⭐ 入门必读 |
| `model_configs.yaml` | 模型架构配置 | ⭐⭐ 重要参考 |
| `CONTRIBUTING.md` | 贡献指南 | ⭐⭐ 开发必读 |

---

## 故障排查

### 显存不足
- 减小batch size: `--batch-size 16`
- 减小模型隐藏层维度

### 实验中断
- 支持断点续传，重新运行相同命令会自动跳过已完成实验

### 依赖问题
```bash
pip install -r requirements.txt
```

---

## 许可证

MIT License

---

*本文档基于项目实际内容生成，最后更新: 2026-04-10*
