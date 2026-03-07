# 新实验架构说明

本文档说明新实验架构与旧架构的主要区别。

## 核心变化

### 1. 分层实验结构

**旧架构:**
```
大实验（100次重复）
└── 循环训练（MLP×100 → MLP-MIM×100 → LSTM×100 → ...）
    └── 最后统一保存所有结果
```

**新架构:**
```
大实验（100次重复）
├── 小实验(seed=42): 训练所有10种模型 → 保存结果 → 绘图
├── 小实验(seed=43): 训练所有10种模型 → 保存结果 → 绘图
├── ...
└── 小实验(seed=141): 训练所有10种模型 → 保存结果 → 绘图
    └── 最后生成统计性图表
```

**优势:**
- 每个小实验独立保存，中断不丢失已完成结果
- 可单独运行/调试某个种子
- 失败隔离，一个种子失败不影响其他

### 2. 独立绘图脚本

**旧架构:** 绘图内嵌在实验代码中，无法单独使用

**新架构:** 
```bash
# 独立绘图脚本，可手动调用
python plot_single.py --input seed_42/results.csv
python plot_batch.py --input experiments/3C/20240203_120000/
```

**优势:**
- 中断后可手动生成图表
- 可修改绘图代码重新生成
- 解耦绘图与实验执行

### 3. 训练策略区分

**Baseline模型（不带缺失指示器）:**
```python
# 只用完整数据训练1次
train_data = X_train  # 原始数据
```

**MIM模型（带缺失指示器）:**
```python
# 复制10份，分别施加不同缺失率
for mr in [0.0, 0.1, ..., 0.9]:
    X_copy = apply_missing(X_train, missing_rate=mr)
    X_copy_with_indicator = concat([X_copy, indicator])
    train_data.append(X_copy_with_indicator)

# 拼接成10×大小的大训练集
train_data = concat(all_copies)  # [10*n_samples, 32]
```

### 4. 容错机制

**检查点系统:**
```json
{
  "completed_seeds": [42, 43, 44, 45],
  "failed_seeds": [],
  "status": "running"
}
```

**恢复流程:**
```bash
# 检查状态
python check_status.py --experiment_dir experiments/3C/20240203_120000

# 继续运行（自动跳过已完成）
python run_batch.py --batch 3C --n_repeats 100 --resume
```

## 文件对应关系

| 旧文件 | 新文件 | 说明 |
|--------|--------|------|
| `run_full_experiment.py` | `run_batch.py` | 运行大实验 |
| - | `run_single.py` | 运行单个小实验（新增） |
| `src/experiments/experiment_runner.py` | `src/experiments/batch_experiment.py` | 大实验运行器 |
| - | `src/experiments/single_experiment.py` | 小实验运行器（新增） |
| - | `src/experiments/checkpoint.py` | 检查点管理（新增） |
| 内嵌绘图 | `plot_single.py` | 单实验绘图脚本（新增） |
| 内嵌绘图 | `plot_batch.py` | 批量绘图脚本（新增） |
| - | `check_status.py` | 状态检查（新增） |

## 使用方式对比

### 运行完整实验

**旧方式:**
```bash
python run_full_experiment.py --batch 3C --n_repeats 100
```

**新方式:**
```bash
python run_batch.py --batch 3C --n_repeats 100
```

### 中断恢复

**旧方式:** 不支持，需要重新开始

**新方式:**
```bash
python run_batch.py --batch 3C --n_repeats 100 --resume
```

### 手动绘图

**旧方式:** 不支持

**新方式:**
```bash
python plot_batch.py --input experiments/3C/20240203_120000/
```

### 单独运行某种子

**旧方式:** 不支持

**新方式:**
```bash
python run_single.py --seed 100 --epochs 50
```

## 输出结构对比

### 旧结构
```
experiments/{timestamp}_{batch}/
├── models/              # 所有模型混在一起
├── results/             # 最后统一保存
├── figures/             # 最后统一绘图
└── logs/
```

### 新结构
```
experiments/{batch}/{timestamp}/
├── checkpoint.json      # 检查点
├── logs/
│   └── batch.log
├── seed_42/             # 每个种子独立目录
│   ├── results.csv      # 该种子结果
│   ├── figures/         # 该种子图表
│   └── models/          # 该种子模型
├── seed_43/
├── ...
└── aggregate/           # 统计结果
    ├── results_all.csv
    └── figures/         # 统计图表
```

## 迁移指南

如果之前已经运行了旧架构的实验，想要使用新架构：

1. 新架构代码与旧架构代码共存，不会冲突
2. 建议用新架构重新运行实验以获得完整功能
3. 旧实验结果仍可通过旧代码分析

## 新架构优势总结

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 中断恢复 | ❌ | ✅ |
| 独立绘图 | ❌ | ✅ |
| 单种子调试 | ❌ | ✅ |
| 即时保存 | ❌ | ✅ |
| 统计图表 | ❌ | ✅ |
| 进度检查 | ❌ | ✅ |
| 容错隔离 | ❌ | ✅ |
