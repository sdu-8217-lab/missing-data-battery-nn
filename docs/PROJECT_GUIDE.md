# 项目指南

> 了解项目架构、设计和使用方式

---

## 📋 文档索引

| 文档 | 用途 | 阅读时机 |
|------|------|---------|
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | 环境配置 | 首次设置 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始 | 首次运行 |
| **本文档** | 项目总览 | 了解项目架构 |
| [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) | 实验指南 | 准备跑实验 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 代码架构 | 深入了解代码 |

---

## 🎯 项目概述

### 核心问题

**如何在电池数据存在缺失的情况下，准确预测电池健康状态(SOH)？**

### 核心创新

| 传统方法 | 我们的方法(MIM) |
|---------|----------------|
| 先插补缺失值，再预测 | 直接告诉网络哪些数据缺失 |
| 16维输入(仅特征) | 32维输入(特征+缺失指示器) |
| 单缺失率训练 | 混合缺失率(0.0-0.9)训练 |

### 实验规模

```
4数据集 × 2缺失机制 × 5方法 × 4模型 × 100种子 × 9缺失率
= 144,000 次评估
```

---

## 🏗️ 技术架构

### 技术栈

```
Hydra (配置管理)
    ↓
PyTorch Lightning (训练框架)
    ↓
WandB (实验跟踪，可选)
```

### 项目结构

```
missing-data-battery-nn/
├── configs/              # Hydra配置
│   ├── config.yaml      # 主配置
│   ├── data/            # 4数据集
│   ├── model/           # 4模型
│   └── missing/         # 2缺失机制
├── src/
│   ├── main.py          # 唯一入口
│   ├── models/          # 神经网络定义
│   ├── data/            # 数据加载+划分
│   ├── missing_data/    # 缺失模拟+插补
│   └── trainers/        # 训练器
├── results/             # 实验结果(CSV)
└── docs/                # 文档
```

---

## 🔬 核心概念

### MIM (Missing Indicator Method)

**核心思想**: 不掩盖缺失，而是明确告诉网络"哪些数据缺失"

```python
# 传统方法 (16维)
输入 = [voltage_mean, voltage_std, ..., current_entropy]

# MIM方法 (32维)
输入 = [特征值(16维)] + [缺失指示器(16维)]
      # [3.7, 0.1, ..., 0] + [0, 0, ..., 1]
```

### 混合缺失率训练

```python
# 训练时: 10个缺失率版本
for mr in [0.0, 0.1, ..., 0.9]:
    X_missing = apply_missing(X_train, mr)
    train(X_missing, y_train)

# 测试时: 指定缺失率
X_test_missing = apply_missing(X_test, mr=0.5)
predict = model(X_test_missing)
```

### 按电池划分

```
❌ 错误: 随机划分样本
✅ 正确: 整个电池只属于train/val/test之一

Train: [电池3,4,5,7]
Val:   [电池1,8]
Test:  [电池2,6]
```

---

## 📊 实验设计

### 控制变量

| 维度 | 取值 |
|------|------|
| 数据集 | XJTU, TJU, HUST, MIT |
| 缺失机制 | MCAR, MAR |
| 方法 | MIM, Mean, Median, KNN, Zero |
| 模型 | MLP, LSTM, GRU, CNN1D |
| 缺失率 | 0.1, 0.2, ..., 0.9 |
| 随机种子 | 42, 43, ..., 141 |

### 训练策略

**Baseline模型**：
- 仅在完整数据(MR=0.0)上训练
- 测试时应用缺失+插补

**MIM模型**：
- 训练时混合10个缺失率(0.0-0.9)
- 数据量扩大10倍
- 测试时直接应用MIM

---

## 🚀 快速开始

```bash
# 1. 环境配置
conda env create -f environment.yml
conda activate battery-nn

# 2. 运行测试
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42] training.epochs=10

# 3. 查看结果
cat results/battery_soh_experiment_mlp_mim.csv
```

详细步骤见 [QUICKSTART.md](QUICKSTART.md)

---

## 📚 其他文档

- [RESEARCH_LESSONS.md](RESEARCH_LESSONS.md) - 科研经验教训
- [MIM_CORRECTION_SUMMARY.md](MIM_CORRECTION_SUMMARY.md) - MIM策略修正说明
- [WANDB_GUIDE.md](WANDB_GUIDE.md) - WandB使用指南

---

*最后更新: 2026-03-07*
