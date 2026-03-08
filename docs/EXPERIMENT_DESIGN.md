# 实验设计规范文档

> 本文档记录MIM与Baseline对比实验的完整设计规范
> 
> 版本: v2.0 | 日期: 2026-03-08 | 分支: exp-execution

---

## 1. 核心设计理念

### 1.1 循环层级结构（重要性从内到外递增）

```
for dataset in datasets:              # 最外层 - 跨数据集泛化
    for batch in batches:             # 外层 - 同数据集不同批次
        for seed in seeds:            # 中外层 - 重复性
            for model in models:      # 中层 - 架构泛化
                for method in methods: # 最内层 - 核心对比（必须完整）
                    run_experiment()
```

**完整循环层级**: `dataset → batch → seed → model → method`

**设计原则**：
- **方法（method）是最核心的变量**，位于最内层循环
- **数据集（dataset）是最外层的泛化维度**，验证跨数据集鲁棒性
- 一个最小完整实验 = `(dataset, batch, seed, model)` + **both methods**
- 只有方法对比完整，才能称之为一次有效实验
- 外层变量（dataset/batch/seed/model）即使只有一个值，也能构成相对完整的实验

### 1.2 新增维度说明

| 维度 | 层级 | 说明 | 示例值 |
|------|------|------|--------|
| **dataset** | 最外层 | 不同来源的电池数据集 | xjtu, nasa, calce |
| **batch** | 外层 | 同一数据集内的不同批次 | 0.5C, 1C, 2C |
| **seed** | 中外层 | 随机种子重复性 | 42-141 |
| **model** | 中层 | 深度学习架构 | mlp, lstm, gru, cnn1d |
| **method** | 最内层 | 缺失数据处理方法 | baseline, mim |

### 1.3 实验单元定义（更新）

| 术语 | 定义 | 计算方式 |
|------|------|----------|
| **Run** | 单次执行 | `(dataset, batch, seed, model, method)` |
| **Comparison Unit** | 最小完整对比单元 | `(dataset, batch, seed, model)` 包含两种方法 |
| **Batch Experiment** | 批次实验 | `(dataset, batch)` 的所有seeds × models × methods |
| **Full Experiment** | 完整实验 | 所有datasets × batches × seeds × models × methods |

---

## 2. 缺失率设置

### 2.1 测试集缺失率（10档）

```python
eval_missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**设计理由**：
- **MR=0.0（无缺失）**：现实中最普遍的场景，必须作为基准
- **MR=0.1-0.8**：覆盖低、中、高各类缺失水平
- **MR=0.9（极高缺失）**：压力测试，评估极端情况下的鲁棒性
- 测试时复制为10份，各赋予一个缺失率，进行10次独立测试

### 2.2 MIM训练集缺失率

```python
mim_train_missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**关键约束**：
- **严格与测试集缺失率相同**（10档完全一致）
- 复制10份训练集，各份赋予一个缺失率
- **不是进行10次训练**，而是将10份合并为一份大训练集进行一次训练
- 数据量扩展为原始训练集的10倍

### 2.3 Baseline训练集缺失率

```python
baseline_train_missing_rates = [0.0]  # 仅在完整数据上训练
```

**设计理由**：
- Baseline方法（插补范式）的标准做法
- 训练阶段假设数据完整
- 测试阶段通过插补处理缺失

---

## 3. 方法对比设计

### 3.1 两种范式的核心差异

| 维度 | MIM (端到端学习) | Baseline (插补范式) |
|------|------------------|---------------------|
| **训练数据** | 10份合并（10种MR混合） | 仅完整数据（MR=0.0） |
| **输入维度** | 32维（16特征 + 16掩码） | 16维（仅特征） |
| **缺失处理** | 模型学习 P(Y\|X_observed, Mask) | 插补后学习 P(Y\|X_imputed) |
| **测试方式** | 直接输入（无需插补） | 先插补，后预测 |

### 3.2 控制变量

- **数据集**: XJTU 2C电池数据集
- **模型架构**: MLP/LSTM/GRU/CNN1D（约27K参数）
- **缺失机制**: MCAR（完全随机缺失）
- **评价指标**: MAE（平均绝对误差）

### 3.3 独立变量

- **处理范式**: MIM vs Baseline
- **缺失率**: 0.0 - 0.9（10档）
- **模型架构**: 4种深度学习模型
- **随机种子**: 100个独立重复

---

## 4. 实验规模与统计

### 4.1 完整实验规模（含dataset和batch）

```python
# 最外层循环
datasets = ["xjtu", "nasa", "calce"]  # 3个数据集（示例）
batches = {
    "xjtu": ["0.5C", "1C", "2C"],
    "nasa": ["batch1", "batch2"],
    "calce": ["CS2", "CX2"]
}

# 中外层循环
seeds = list(range(42, 142))  # 100个种子

# 中间循环
models = ["mlp", "lstm", "gru", "cnn1d"]  # 4种架构

# 最内层循环（核心）
methods = ["baseline", "mim"]  # 2种方法（必须成对）

# 总执行次数（以XJTU为例：3 batches × 100 seeds × 4 models × 2 methods = 2,400 runs）
# 如果3个数据集各2-3 batches：约 7,200 runs

# 对比单元数（最小完整实验单位）
total_comparison_units = total_batches × 100 × 4

# 测试结果行数
total_result_rows = total_runs × 10 MRs
```

### 4.2 时间戳管理

**每次实验运行创建独立的时间戳命名空间**：

```
experiments/
├── runs/                          # 按时间戳组织的实验运行
│   ├── 20260308_143052/          # 2026-03-08 14:30:52 的实验
│   │   ├── experiment_db.csv     # 该次实验的数据库
│   │   ├── logs/                 # 训练日志
│   │   │   ├── xjtu_2C_seed42_mlp_baseline.log
│   │   │   └── ...
│   │   └── results/              # 结果CSV
│   │       ├── xjtu_2C_mlp_baseline.csv
│   │       └── ...
│   └── 20260309_091215/          # 另一次实验
└── aggregated/                    # 汇总结果（跨多次实验）
    └── all_results.csv
```

**时间戳命名规则**: `YYYYMMDD_HHMMSS`

### 4.3 简化测试规模（含dataset/batch）

| 模式 | Datasets | Batches | Seeds | Models | Methods | Runs | Units |
|------|----------|---------|-------|--------|---------|------|-------|
| `--tiny` | 1 | 1 | 1 | 1 | 2 | 2 | 1 |
| `--mini` | 1 | 1 | 2 | 1 | 2 | 4 | 2 |
| `--test` | 1 | 2 | 2 | 2 | 2 | 16 | 8 |
| `--all` | 3 | 2-3 | 100 | 4 | 2 | ~7,200 | ~3,600 |

**层级递进关系**：
- `--tiny`: 验证单次运行的正确性
- `--mini`: 验证一个完整comparison unit
- `--test`: 验证跨batch泛化
- `--all`: 完整跨数据集实验

---

## 5. 数据记录格式

### 5.1 实验数据库字段（含dataset/batch/时间戳）

```csv
exp_id,dataset,batch,seed,model,method,status,metrics,eval_missing_rates,mim_train_missing_rates,timestamp,run_dir,...
```

**关键字段说明**：
- `exp_id`: `{dataset}_{batch}_seed{seed}_{model}_{method}`（完整层级标识）
- `dataset`: 最外层循环 - 数据集名称（xjtu/nasa/calce）
- `batch`: 外层循环 - 批次标识（2C/1C/0.5C等）
- `seed`: 中外层循环 - 随机种子
- `model`: 中层循环 - 模型架构
- `method`: 最内层循环（核心）- 方法名称
- `timestamp`: 实验启动时间 `YYYYMMDD_HHMMSS`
- `run_dir`: 该次实验的运行目录 `experiments/runs/{timestamp}/`
- `eval_missing_rates`: 测试MR列表（10档）
- `mim_train_missing_rates`: MIM训练MR列表（10档）
- `metrics`: JSON格式，包含所有10个MR的测试结果

### 5.2 时间戳管理规范

**目的**：区分不同时间、不同次的实验，避免结果覆盖

#### 5.2.1 时间戳生成

```python
from datetime import datetime

def generate_timestamp():
    """生成实验时间戳: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

timestamp = generate_timestamp()  # 例如: "20260308_143052"
```

#### 5.2.2 目录结构

```
experiments/
├── runs/                           # 所有实验运行的根目录
│   ├── {timestamp}_xjtu_2C/       # 某次针对XJTU 2C的实验
│   │   ├── experiment_db.csv      # 该次实验的数据库
│   │   ├── config.yaml            # 实验配置备份
│   │   ├── logs/                  # 训练日志
│   │   │   └── seed42_mlp_baseline.log
│   │   └── results/               # 结果文件
│   │       └── seed42_mlp_baseline.csv
│   └── {timestamp}_multi/         # 跨数据集实验
└── latest -> runs/{timestamp}/    # 软链接指向最新实验
```

#### 5.2.3 实验标识命名

```python
# 完整实验ID: {dataset}_{batch}_seed{seed}_{model}_{method}
exp_id = f"{dataset}_{batch}_seed{seed}_{model}_{method}"
# 例如: "xjtu_2C_seed42_mlp_baseline"

# 带时间戳的全局唯一ID
global_id = f"{timestamp}_{exp_id}"
# 例如: "20260308_143052_xjtu_2C_seed42_mlp_baseline"
```

#### 5.2.4 结果汇总

```python
# 跨时间戳汇总所有实验结果
def aggregate_results(timestamps: List[str]):
    """汇总多次实验的结果"""
    all_results = []
    for ts in timestamps:
        db_path = f"experiments/runs/{ts}/experiment_db.csv"
        results = load_results(db_path)
        all_results.extend(results)
    
    # 保存汇总结果
    save_to = f"experiments/aggregated/aggregate_{timestamps[0]}_to_{timestamps[-1]}.csv"
    save_results(all_results, save_to)
```

### 5.3 Metrics JSON结构

```json
{
  "mr_results": [
    {"missing_rate": 0.0, "test_mae": 0.0139},
    {"missing_rate": 0.1, "test_mae": 0.0151},
    ...
    {"missing_rate": 0.9, "test_mae": 0.0364}
  ],
  "test_mae_mean": 0.02377,  // 10档MR的平均MAE
  "test_mae_max": 0.0364,     // 最差情况
  "test_mae_min": 0.0139,     // 最好情况（通常MR=0.0）
  "test_mae": 0.0364,         // 最后一个MR的结果（兼容）
  "missing_rate": 0.9         // 最后一个MR（兼容）
}
```

---

## 6. 结果分析方法

### 6.1 鲁棒性曲线（Robustness Curve）

对于每个 `(seed, model)` 组合，绘制两条曲线：
- **X轴**: 缺失率（0.0 - 0.9）
- **Y轴**: MAE
- **曲线**: Baseline vs MIM

### 6.2 统计指标

**单点指标**（每个MR单独计算）：
- 均值: `mean(MAE)` across 100 seeds
- 标准差: `std(MAE)` across 100 seeds
- 95%置信区间

**综合指标**（跨所有MR）：
- 平均鲁棒性: `mean(MAE across MRs)`
- 最差情况性能: `max(MAE at MR=0.9)`
- 无缺失性能: `MAE at MR=0.0`
- 性能下降率: `(MAE@0.9 - MAE@0.0) / MAE@0.0`

### 6.3 假设检验

对每个MR，检验：
- H0: MIM_MAE = Baseline_MAE
- H1: MIM_MAE < Baseline_MAE（单侧检验）

使用配对t检验（paired t-test），因为两种方法使用相同的seeds和models。

---

## 7. 关键实现细节

### 7.1 MIM训练数据生成

```python
def create_mim_training_data(X, y, missing_rates, seed):
    """
    Args:
        X: 原始特征 [N, D]
        y: 目标值 [N]
        missing_rates: [0.0, 0.1, ..., 0.9]
        seed: 基础随机种子
    
    Returns:
        X_merged: [N×10, D×2]（特征+掩码）
        y_merged: [N×10]
    """
    all_inputs = []
    all_targets = []
    
    for i, mr in enumerate(missing_rates):
        mr_seed = seed + i * 100  # 每个MR使用不同种子
        X_with_missing = simulate_mcar(X, mr, seed=mr_seed)
        all_inputs.append(X_with_missing)
        all_targets.append(y)
    
    return torch.cat(all_inputs), torch.cat(all_targets)
```

### 7.2 Baseline训练

```python
# Baseline仅在MR=0.0数据上训练
X_train_complete = X_train  # 无缺失
train_model(X_train_complete, y_train)

# 测试时施加缺失并插补
for mr in [0.0, 0.1, ..., 0.9]:
    X_test_missing = simulate_mcar(X_test, mr)
    X_test_imputed = impute(X_test_missing, method='mean')
    evaluate(model, X_test_imputed, y_test)
```

### 7.3 命令行构建

```bash
python src/main.py \
    model=paper_mlp \
    method=mim \                          # 或 baseline
    experiment.seeds=[42] \               # 外层：种子
    +experiment.run_name=seed42_mlp_mim \
    +missing.missing_rates_eval=[0.0,0.1,...,0.9] \    # 10档测试
    +missing.missing_rates_train=[0.0,0.1,...,0.9]     # MIM训练用
```

---

## 8. 常见问题与注意事项

### 8.1 为什么MR=0.0必须包含？

- **现实意义**: 很多实际场景数据是完整的
- **公平对比**: Baseline在MR=0.0时理论上最优
- **性能锚点**: 提供性能比较的基准点

### 8.2 为什么MIM训练要和测试MR严格相同？

- **分布匹配**: 训练分布与测试分布一致
- **公平性**: 避免训练时未见过的缺失率
- **科学严谨**: 控制变量，只比较"是否使用掩码"

### 8.3 为什么不把MR作为最外层循环？

如果 `for mr in MRs: for seed in seeds: ...`，会导致：
- 同一seed在不同MR下训练多个模型（冗余）
- 违背"一个模型测试多MR"的设计
- 实验量爆炸（800 runs × 10 MRs = 8000次训练）

正确做法：一个模型训练一次，测试所有MR。

### 8.4 为什么方法必须在最内层？

保证最小完整实验包含两种方法的对比：
- ✅ `seed42_mlp_baseline` + `seed42_mlp_mim` = 完整对比
- ❌ 如果method在外层，可能只运行了baseline没运行mim

---

## 9. 论文写作建议

### 9.1 图表建议

1. **鲁棒性曲线图**: 每个model一个子图，展示10档MR下的MAE对比
2. **热力图**: 展示不同(model, MR)组合下MIM相对提升百分比
3. **箱线图**: 展示100个seeds下的MAE分布稳定性
4. **表格**: 汇总4种模型在关键MR点（0.0, 0.5, 0.9）的均值±标准差

### 9.2 关键结论表述

- MIM在**所有缺失率水平**下均优于Baseline
- 随着缺失率增加，Baseline性能**线性恶化**，MIM性能**缓慢下降**
- 在MR=0.9极端情况下，MIM相对提升**X%**
- MIM在MR=0.0（无缺失）时性能**不劣于**Baseline

---

## 10. 附录

### 10.1 实验运行命令（含dataset/batch）

```bash
# 初始化时指定数据集和批次
python scripts/run_experiments.py init \
    --dataset xjtu \
    --batch 2C \
    --seeds 42 43

# 或者使用预设配置
python scripts/run_experiments.py init --tiny    # 1 dataset, 1 batch, 1 seed
python scripts/run_experiments.py init --mini    # 1 dataset, 1 batch, 2 seeds
python scripts/run_experiments.py init --test    # 1 dataset, 2 batches, 2 seeds
python scripts/run_experiments.py init --all     # 所有datasets和batches

# 运行（自动使用时间戳目录）
python scripts/run_experiments.py run --gpu-workers 1 --cpu-workers 3

# 查看特定时间戳的实验
python scripts/run_experiments.py status --timestamp 20260308_143052

# 汇总多次实验结果
python scripts/run_experiments.py aggregate \
    --timestamps 20260308_143052 20260309_091215 \
    --output aggregated_results.csv
```

### 10.2 状态监控

```bash
# 查看进度
python scripts/run_experiments.py status

# 查看结果
python -c "
import json, csv
with open('experiments/experiment_db.csv') as f:
    for r in csv.DictReader(f):
        if r['status'] == 'completed':
            m = json.loads(r['metrics'])
            print(f\"{r['exp_id']}: {len(m['mr_results'])} MRs\")
"
```

### 10.3 故障恢复

```bash
# 重置失败实验
python scripts/run_experiments.py retry

# 恢复卡住的实验（running → pending）
python scripts/run_experiments.py recover
```

---

**文档维护**: 如实验设计有重大变更，请更新本文档并注明版本和日期。
