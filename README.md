# 电池SOH预测：缺失数据下的MIM方法研究

基于 **9层实验架构** 的电池健康状态(SOH)预测框架，系统对比 **MIM (Missing Indicator Method)** 与传统插补方法在缺失数据场景下的性能。

> 📋 **权威架构定义**: 参见 [meta.md](meta.md) - 包含完整的9层实验设计原则与概念定义

---

## 核心概念

### MIM vs 插补的本质

**关键认知**: MIM与插补不是互斥关系，而是**正交组合**。

| 维度 | 无MIM (16维) | 有MIM (32维) |
|------|-------------|-------------|
| **mean** | 均值插补 | 均值插补 + 缺失指示器 |
| **knn** | KNN插补 | KNN插补 + 缺失指示器 |
| **iterative** | 迭代插补 | 迭代插补 + 缺失指示器 |
| **zero** | 零值填充 | 零值填充 + 缺失指示器 |

**研究问题**: 在相同插补策略下，添加MIM指示器是否有帮助？

---

## 9层实验架构

实验设计遵循 **meta.md** 定义的9层层次结构：

### 分界线以上（影响模型训练）

| 层级 | 变量 | 取值 | 说明 |
|------|------|------|------|
| L1 | Seed | 0-99 | 随机种子，控制可复现性 |
| L2 | Dataset | XJTU | 固定使用XJTU数据集 |
| L3 | Batch | 2C, 3C, R2.5, R3, RW, Sim_satellite | 6个电池批次 |
| L4 | Model | mlp, lstm, cnn | 3种神经网络架构 |
| L5 | use_mim | false, true | 是否使用MIM指示器 |
| L6 | Train MR | 0.0 (false) / 0.0-0.9 (true) | 训练缺失率 |

**分界线以上组合数**: 100 × 6 × 3 × 2 = **3,600个独立模型**

### 分界线以下（仅影响测试）

| 层级 | 变量 | 取值 | 说明 |
|------|------|------|------|
| L7 | Mode | MCAR, MAR, MNAR | 缺失模式 |
| L8 | Test MR | 0.0, 0.1, ..., 0.9 | 测试缺失率 |
| L9 | Imputation | mean, knn, iterative, zero | 插补方法 |

**每个模型的测试组合**: 3 × 10 × 4 = **120个测试结果**

### 总实验规模

- **训练**: 3,600个模型
- **测试**: 3,600 × 120 = **432,000个测试结果**
- **完整矩阵**: 100种子 × 6批次 × 3模型 × 2 MIM × 3模式 × 10 MR × 4插补 = **432,000行结果**

---

## 项目结构

```
.
├── meta.md                     # ⭐ 权威架构定义（只读）
├── README.md                   # 本文件
├── experiments/                # 实验入口
│   ├── run_batch_experiments.py   # 批量实验主入口
│   ├── run_experiment.py          # 单次实验执行
│   ├── run_single.py              # 快速单次测试
│   └── evaluate.py                # 结果评估
├── src/                        # 源代码
│   ├── main.py                 # 旧Hydra入口（已弃用）
│   ├── models/                 # 神经网络定义
│   ├── data/                   # 数据加载与预处理
│   ├── missing_data/           # 缺失模拟与插补
│   └── utils/                  # 工具函数
├── configs/                    # 配置文件
├── data/                       # 数据集（XJTU等）
├── models/                     # 保存的训练模型
├── results/                    # 实验结果（JSON/CSV）
├── docs/                       # 文档
│   ├── ARCHITECTURE.md         # 代码架构
│   ├── EXPERIMENTS.md          # 实验指南
│   └── archived/               # 归档文档
└── tests/                      # 测试
```

---

## 快速开始

### 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate battery-nn

# 或直接使用pip
pip install -r requirements.txt
```

### 运行实验

#### 1. 单次实验（快速测试）

```bash
# 训练单个模型
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 50

# 测试该模型
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

#### 2. 批量实验（完整矩阵）

```bash
# 运行完整实验（训练+测试）
python experiments/run_batch_experiments.py \
    --phase full \
    --epochs 50

# 仅训练阶段
python experiments/run_batch_experiments.py \
    --phase train \
    --epochs 50

# 仅测试阶段（需已有训练好的模型）
python experiments/run_batch_experiments.py \
    --phase test

# 指定子集（如仅2个种子、2个批次）
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 42 43 \
    --batches 2C 3C \
    --epochs 50
```

#### 3. 大规模实验（100种子）

```bash
# 后台运行（耗时约10-15小时）
nohup python experiments/run_batch_experiments.py \
    --phase full \
    --seeds $(seq 0 99) \
    --epochs 50 \
    --model-dir models/100seeds \
    --results-dir results/100seeds \
    > experiment_100seeds.log 2>&1 &

# 监控进度
tail -f experiment_100seeds.log
```

---

## 监控实验进度

批量实验内置tqdm进度条，显示：
- 实时进度百分比
- 预计剩余时间(ETA)
- 当前配置
- 成功/失败状态

```
Training:  25%|████▌| 900/3600 [2:30:15<7:30:45, 10.2s/it, ✓ 2C-mlp-MIM (4.5s)]
```

---

## 结果分析

### 结果文件结构

```
results/
├── seed42_batch2C_modelmlp_mimtrue.json      # 单次实验结果
├── seed42_batch2C_modellstm_mimfalse.json
├── ...
├── train_summary.json                        # 训练摘要
└── aggregated_results.csv                    # 聚合结果（自动生成的CSV）
```

### 结果格式

```json
{
  "seed": 42,
  "batch": "2C",
  "model": "mlp",
  "use_mim": "true",
  "mode": "MCAR",
  "test_mr": 0.3,
  "imputation": "mean",
  "test_mae": 0.0234,
  "test_rmse": 0.0312,
  "status": "success",
  "elapsed": 4.5
}
```

### 统计分析示例

```python
import pandas as pd

# 加载聚合结果
df = pd.read_csv('results/aggregated_results.csv')

# MIM vs 非MIM对比
mim_comparison = df.groupby(['use_mim', 'imputation'])['test_mae'].mean()
print(mim_comparison)

# 不同缺失率下的性能
mr_performance = df.groupby('test_mr')['test_mae'].mean()
print(mr_performance)
```

---

## 关键特性

### 1. 按电池划分（Battery-wise Split）

同一电池的所有循环只属于train/val/test中的一个，避免数据泄漏。

### 2. 分界线原则

- **分界线以上**（L1-L6）: 每个组合需独立训练模型
- **分界线以下**（L7-L9）: 同一模型可复用测试所有组合
- **关键洞察**: 训练缺失率与测试缺失率独立，可评估泛化能力

### 3. MIM训练策略

- **use_mim=false**: 使用完整数据训练（MR=0.0）
- **use_mim=true**: 使用MCAR多MR混合训练（0.0-0.9）
- **测试时**: 可面对任意缺失模式（MCAR/MAR/MNAR）

---

## 文档导航

| 文档 | 内容 | 读者 |
|------|------|------|
| [meta.md](meta.md) | ⭐ 9层架构定义、实验原则（**权威参考**） | 所有人 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 代码架构、模块设计 | 开发者 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 详细实验指南 | 研究者 |
| [docs/DATASETS.md](docs/DATASETS.md) | 数据集说明 | 数据使用者 |

---

## 故障排查

### 问题: 显存不足
**解决**: 减小batch size或模型隐藏层维度

### 问题: 实验中断
**解决**: 支持断点续传，重新运行相同命令会自动跳过已完成的实验

### 问题: 缺少依赖
**解决**: `pip install -r requirements.txt`

---

## 引用

```bibtex
@article{battery_mim_2026,
  title={Battery SOH Prediction with Missing Data using Missing Indicator Method},
  author={...},
  journal={Journal of Energy Storage},
  year={2026}
}
```

## License

MIT License

---

*Last updated: 2026-03-18*
