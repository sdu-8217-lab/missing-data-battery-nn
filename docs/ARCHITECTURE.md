# 代码架构

<!-- 
  confidence: A
  reviewed: 2026-03-19
  reviewer: chen
  ai-assisted: true
  commit: c87f6b7
-->

> **架构说明**
>
> 本文档描述**9层实验架构**的代码实现。
> 与旧版Hydra架构的区别参见[与旧架构的区别](#与旧架构的区别)。

---

## 1. 系统概览

### 1.1 架构核心

基于 **meta.md** 定义的9层实验架构（L1-L9）：

```
分界线以上（影响训练）：L1(Seed) → L2(Dataset) → L3(Batch) → L4(Model) → L5(use_mim) → L6(Train MR)
═══════════════════════════════════════════════════════════════════════════════════════════════════════
分界线以下（仅测试）：L7(Mode) → L8(Test MR) → L9(Imputation)
```

### 1.2 执行入口

| 入口脚本 | 用途 | 对应层级 |
|---------|------|---------|
| `experiments/run_batch_experiments.py` | 批量实验（训练+测试） | L1-L9 |
| `experiments/run_experiment.py` | 单次实验 | L1-L9（单点） |
| `experiments/run_single.py` | 快速单次测试 | 简化版 |

**配置方式**：命令行参数（argparse），非Hydra。

```bash
# 批量实验示例
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 0 1 2 ... 99 \
    --epochs 50

# 单次实验示例
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 50
```

---

## 2. 目录结构

### 2.1 实验层（experiments/）

```
experiments/
├── run_batch_experiments.py    # 批量实验主入口（L1-L9完整循环）
├── run_experiment.py           # 单次实验执行（训练或测试）
├── run_single.py               # 快速单次测试工具
├── train.py                    # 训练逻辑封装
├── evaluate.py                 # 评估工具
├── architecture_search.py      # 架构搜索（预实验）
├── fine_grained_search.py      # 细粒度搜索
├── run_preexperiment_no_missing.py  # 预实验
└── run_youth_exp.py            # 青年基金实验
```

### 2.2 源代码层（src/）

```
src/
├── data/                       # 数据层
│   ├── xjtu_loader.py         # XJTU数据加载
│   ├── preprocessing.py       # 数据预处理
│   └── splits.py              # 电池划分（train/val/test）
├── models/                     # 模型层
│   ├── mlp.py                 # MLP模型
│   ├── lstm.py                # LSTM模型
│   ├── cnn1d.py               # 1D-CNN模型
│   ├── gru.py                 # GRU模型（预留）
│   └── model_factory.py       # 模型工厂
├── missing_data/               # 缺失数据处理
│   ├── mcar.py                # MCAR缺失模拟
│   ├── mar.py                 # MAR缺失模拟
│   ├── mnar.py                # MNAR缺失模拟
│   ├── imputation.py          # 插补方法（mean/knn/iterative/zero）
│   └── fixed_feature_missing.py  # 固定特征缺失
├── trainers/                   # 训练器
│   ├── lightning_module.py    # PyTorch Lightning模块
│   ├── lightning_trainer.py   # Lightning训练器
│   ├── neural_network_trainer.py  # 神经网络训练器
│   └── xgboost_trainer.py     # XGBoost训练器（预留）
├── utils/                      # 工具函数
│   ├── seed_manager.py        # 随机种子管理
│   ├── logger.py              # 日志工具
│   ├── param_counter.py       # 参数量统计
│   ├── paths.py               # 路径管理
│   └── wandb_utils.py         # WandB集成
└── config/                     # 配置（预留）
    └── __init__.py

├── preexperiment/              # 预实验模块
│   ├── search_space.py        # 搜索空间定义
│   ├── runner.py              # 搜索运行器
│   └── objective.py           # 优化目标
```

### 2.3 配置与文档

```
configs/                       # 配置文件（当前实验不使用，保留参考）
├── legacy/                    # 旧Hydra配置
├── schemas/                   # 配置schema（预留）
└── experiments-v2/            # 新实验配置（预留）

docs/                          # 文档
├── CONFIDENCE.md              # 文档置信度体系 🟢
├── HISTORY.md                 # 历史文档索引
├── QUICKSTART.md              # 快速入门 🟡
├── ARCHITECTURE.md            # 本文档
├── EXPERIMENTS.md             # 实验指南（待更新）
├── DATASETS.md                # 数据集说明
└── development/               # 开发文档

meta.md                        # 9层架构定义（权威）🟢
```

---

## 3. 核心流程

### 3.1 批量实验流程（run_batch_experiments.py）

```python
# 分界线以上：训练阶段（L1-L6）
for seed in seeds:                    # L1
    for batch in batches:             # L3
        for model in models:          # L4
            for use_mim in [false, true]:  # L5
                # L6由use_mim决定
                train_model(...)      # 训练并保存
                
# 分界线：模型已固定

# 分界线以下：测试阶段（L7-L9）
for trained_model in saved_models:
    for mode in [MCAR, MAR, MNAR]:    # L7
        for test_mr in [0.0, 0.1, ..., 0.9]:  # L8
            for imputation in [mean, knn, iterative, zero]:  # L9
                test_model(...)       # 测试并记录结果
```

### 3.2 单次实验流程（run_experiment.py）

```
命令行参数解析（argparse）
    ↓
加载数据（src/data/xjtu_loader.py）
    ↓
if --phase == train:
    准备训练数据（MIM多MR或完整数据）
    创建模型（src/models/model_factory.py）
    训练（src/trainers/lightning_trainer.py）
    保存模型（models/）
elif --phase == test:
    加载已训练模型
    生成缺失数据（src/missing_data/）
    插补（src/missing_data/imputation.py）
    评估（计算MAE/RMSE）
    保存结果（results/）
```

---

## 4. 关键设计

### 4.1 MIM（Missing Indicator Method）

```python
# 传统方法（16维输入）
input = [feature_values]  # shape: [batch, 16]

# MIM方法（32维输入）
input = [feature_values] + [missing_mask]  # shape: [batch, 32]
# missing_mask: 1表示缺失，0表示存在
```

**MIM训练策略**：
- `use_mim=false`：使用完整数据训练（MR=0.0）
- `use_mim=true`：混合10种缺失率（0.0-0.9）训练

### 4.2 分界线原则

| 分界线 | 左侧（训练） | 右侧（测试） |
|--------|-------------|-------------|
| 影响 | 影响模型参数 | 不影响模型参数 |
| 组合数 | 100×6×3×2 = 3,600 | 3×10×4 = 120（每模型） |
| 复用性 | 每个组合独立训练 | 同一模型测试所有组合 |

### 4.3 电池划分（Battery-wise Split）

```python
# ❌ 错误：随机划分样本
# 同一电池的不同循环可能分到train/test

# ✅ 正确：整个电池只属于一个集合
Train batteries: [电池3, 4, 5, 7]
Val batteries:   [电池1, 8]
Test batteries:  [电池2, 6]
```

实现：`src/data/splits.py`

### 4.4 缺失模式生成

| 模式 | 实现文件 | 缺失概率依赖 |
|------|---------|-------------|
| MCAR | `src/missing_data/mcar.py` | 均匀随机 |
| MAR  | `src/missing_data/mar.py` | 依赖观测电压 |
| MNAR | `src/missing_data/mnar.py` | 依赖目标SOH |

---

## 5. 数据流

### 5.1 训练阶段数据流

```
原始数据（XJTU CSV）
    ↓
XJTULoader（按电池加载）
    ↓
电池划分（splits.py）→ train/val/test电池组
    ↓
标准化（preprocessing.py）
    ↓
if use_mim:
    复制10份，各赋予不同MR（0.0-0.9）
    每份应用MCAR缺失
    插补 + MIM掩码拼接
    合并为大数据集
else:
    保持完整数据
    ↓
DataLoader（带随机种子）
    ↓
模型训练（PyTorch Lightning）
    ↓
保存模型（models/seed{batch}_{model}_{use_mim}.pt）
```

### 5.2 测试阶段数据流

```
加载已训练模型
    ↓
测试数据（同一批次的test电池组）
    ↓
应用缺失模式（MCAR/MAR/MNAR）@ 指定MR
    ↓
插补（mean/knn/iterative/zero）
    ↓
if use_mim:
    拼接缺失掩码 → 32维输入
else:
    保持16维输入
    ↓
模型推理
    ↓
计算指标（MAE, RMSE）
    ↓
保存结果（results/seed{batch}_{model}_{use_mim}_{mode}_{mr}_{imp}.json）
```

---

## 6. 与旧架构的区别

| 特性 | 旧架构（已废弃） | 当前架构（9层） |
|------|----------------|----------------|
| 入口 | `src/main.py` | `experiments/run_batch_experiments.py` |
| 配置系统 | Hydra（YAML） | argparse（命令行） |
| 实验架构 | 5层循环 | 9层架构（明确分界线） |
| 架构核心 | 方法对比（MIM vs Baseline） | MIM指示器效应（use_mim × imputation） |
| 训练数据 | 固定MR | MIM多MR混合（0.0-0.9） |
| 测试策略 | 单一模式 | 三种模式（MCAR/MAR/MNAR） |

---

## 7. 扩展指南

### 7.1 添加新模型

1. 创建模型文件 `src/models/new_model.py`：
```python
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        # 实现模型结构
        
    def forward(self, x):
        # 实现前向传播
        return output
```

2. 在 `src/models/model_factory.py` 注册：
```python
elif model_name == "new_model":
    from .new_model import NewModel
    return NewModel(input_dim=input_dim, ...)
```

3. 在 `experiments/run_batch_experiments.py` 添加模型名到 `MODELS` 列表

### 7.2 添加新缺失模式

1. 创建模拟函数 `src/missing_data/new_mode.py`：
```python
def simulate_new_mode(X, y, missing_rate, seed):
    """
    生成新缺失模式
    
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

2. 在 `experiments/run_experiment.py` 添加模式处理分支

### 7.3 修改实验参数

编辑 `experiments/run_batch_experiments.py` 中的常量：
```python
SEEDS = [0, 1, 2, ..., 99]  # 修改种子范围
BATCHES = ["2C", "3C", ...]  # 修改批次
MODELS = ["mlp", "lstm", "cnn"]  # 修改模型
# ...
```

---

## 8. 调试与开发

### 8.1 本地快速测试

```bash
# 小规模实验（2种子 × 2批次）
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 42 43 \
    --batches 2C 3C \
    --epochs 10
```

### 8.2 单元测试

```bash
# 运行关键路径测试
python -m pytest tests/test_critical_path.py -v
```

### 8.3 日志查看

```bash
# 实时查看实验进度
tail -f logs/experiments/100seeds_*.log

# 查看特定模型训练
ls -lh models/100seeds/
```

---

## 9. 相关文档

| 文档 | 内容 | 置信度 |
|------|------|--------|
| [../meta.md](../meta.md) | 9层架构定义（权威来源） | 🟢 S |
| [CONFIDENCE.md](CONFIDENCE.md) | 文档置信度体系 | 🟢 S |
| [QUICKSTART.md](QUICKSTART.md) | 快速入门指南 | 🟡 A |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 详细实验指南（待更新） | 🟠 B |
| [HISTORY.md](HISTORY.md) | 历史文档索引 | 🟠 B |

---

*本文档版本: v2.0*
*架构版本: 9层实验架构*
*最后更新: 2026-03-19*
*对应代码: c87f6b7*
