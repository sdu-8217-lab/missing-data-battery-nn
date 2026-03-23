# 实验设计规范文档

<!-- 
  confidence: A
  reviewed: 2026-03-19
  reviewer: chen
  ai-assisted: true
  commit: c87f6b7
-->

> **文档说明**
>
> 本文档基于 **meta.md** 定义的9层实验架构（L1-L9）。
> 旧版5层循环架构已废弃，相关内容参见[HISTORY.md](HISTORY.md)。

---

## 1. 9层实验架构

### 1.1 架构概览

```
╔══════════════════════════════════════════════════════════════════════╗
║ 分界线以上：影响模型训练                                              ║
╠══════════════════════════════════════════════════════════════════════╣
L1: Seed          - 随机种子（0-99），控制可复现性
L2: Dataset       - 数据集（当前固定为XJTU）
L3: Batch         - 电池批次（2C, 3C, R2.5, R3, RW, Sim_satellite）
L4: Model         - 模型架构（mlp, lstm, cnn）
L5: use_mim       - 是否使用MIM（true/false）
L6: Train MR      - 训练缺失率（由L5决定：0.0或0.0-0.9混合）
╠══════════════════════════════════════════════════════════════════════╣
║ 分界线：模型训练完成，参数固定                                        ║
╠══════════════════════════════════════════════════════════════════════╣
L7: Mode          - 测试缺失模式（MCAR, MAR, MNAR）
L8: Test MR       - 测试缺失率（0.0, 0.1, ..., 0.9）
L9: Imputation    - 插补方法（mean, knn, iterative, zero）
╚══════════════════════════════════════════════════════════════════════╝
```

### 1.2 分界线原则

**核心洞察**：分界线以上（L1-L6）每个组合需要**独立训练**一个模型；分界线以下（L7-L9）同一模型可以**复用测试**所有组合。

| 阶段 | 层级 | 影响 | 组合数（单种子） |
|------|------|------|-----------------|
| 训练 | L1-L6 | 改变模型参数 | 6×3×2 = **36个模型** |
| 测试 | L7-L9 | 不改变模型参数 | 3×10×4 = **120个测试/模型** |

**总结果数**：36个模型 × 120个测试 = **4,320行结果/种子**

**100种子总计**：4,320 × 100 = **432,000行结果**

---

## 2. 层级详解

### 2.1 L1: 随机种子（Seed）

**取值范围**：0-99（100个独立重复）

**作用**：
- 控制数据划分（电池随机分配）
- 控制模型初始化
- 控制缺失模式生成

**执行方式**：
```bash
python experiments/run_batch_experiments.py --seeds 0 1 2 ... 99
```

### 2.2 L2: 数据集（Dataset）

**当前状态**：固定为XJTU数据集

**未来扩展**：可添加TJU、HUST、MIT等数据集

### 2.3 L3: 电池批次（Batch）

**取值**：
- `2C` - 2C充放电
- `3C` - 3C充放电
- `R2.5` - 2.5C随机 walk
- `R3` - 3C随机 walk
- `RW` - Random Walk工况
- `Sim_satellite` - 卫星仿真工况

**注意**：每个批次视为**独立分布**，必须分别训练模型

### 2.4 L4: 模型架构（Model）

**取值**：
- `mlp` - 多层感知机
- `lstm` - 长短期记忆网络
- `cnn` - 卷积神经网络

### 2.5 L5: MIM使用（use_mim）

**取值**：
- `false` - 不使用MIM（16维输入）
- `true` - 使用MIM（32维输入）

**关键影响**：
| 特性 | use_mim=false | use_mim=true |
|------|--------------|--------------|
| 输入维度 | 16 | 32（16特征+16掩码）|
| 训练数据 | 完整数据（MR=0.0） | 混合MR（0.0-0.9） |
| 验证策略 | 完整验证 | 多MR平均验证 |

### 2.6 L6: 训练缺失率（Train MR）

**由L5自动决定**：
- `use_mim=false` → Train MR = 0.0（完整数据）
- `use_mim=true` → Train MR = 0.0-0.9（10种MR混合）

**MIM训练策略**：
```python
# 复制10份训练集
for mr in [0.0, 0.1, ..., 0.9]:
    X_mr = apply_mcar(X_train, mr)  # 应用MCAR缺失
    X_imputed = impute(X_mr)         # 插补
    X_mim = concat([X_imputed, mask]) # 拼接掩码
    
# 合并为10倍大数据集训练
X_train_mim = concat([X_mim_0.0, X_mim_0.1, ..., X_mim_0.9])
```

### 2.7 L7: 测试缺失模式（Mode）

**取值**：
- `MCAR` - 完全随机缺失（Missing Completely At Random）
- `MAR` - 随机缺失，依赖观测值（Missing At Random）
- `MNAR` - 非随机缺失，依赖缺失值本身（Missing Not At Random）

**实现差异**：
| 模式 | 缺失概率依赖 | 代码实现 |
|------|-------------|---------|
| MCAR | 均匀随机 | `src/missing_data/mcar.py` |
| MAR | 电压特征 | `src/missing_data/mar.py` |
| MNAR | SOH目标值 | `src/missing_data/mnar.py` |

### 2.8 L8: 测试缺失率（Test MR）

**取值**：0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95（10档）

**设计理由**：
- **MR=0.0**：无缺失基准
- **MR=0.1-0.8**：覆盖低、中、高缺失水平
- **MR=0.9**：极端压力测试

### 2.9 L9: 插补方法（Imputation）

**取值**：
- `mean` - 均值插补
- `knn` - K近邻插补
- `iterative` - 迭代插补（MICE）
- `zero` - 零值填充

**关键认知**：
- MIM方法也**需要插补**（填充缺失位置的具体数值）
- 与纯插补的区别：MIM**额外输入缺失掩码**
- 研究问题：在**相同插补策略**下，MIM指示器是否有帮助？

---

## 3. 实验执行

### 3.1 批量实验（完整矩阵）

**入口**：`experiments/run_batch_experiments.py`

**完整实验（100种子）**：
```bash
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds $(seq 0 99) \
    --epochs 50 \
    --model-dir models/100seeds \
    --results-dir results/100seeds
```

**小规模测试（2种子 × 2批次）**：
```bash
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 42 43 \
    --batches 2C 3C \
    --epochs 50
```

### 3.2 仅训练阶段

```bash
python experiments/run_batch_experiments.py \
    --phase train \
    --epochs 50
```

**训练组合数**：100 seeds × 6 batches × 3 models × 2 use_mim = **3,600个模型**

### 3.3 仅测试阶段（需已有训练好的模型）

```bash
python experiments/run_batch_experiments.py \
    --phase test \
    --model-dir models/100seeds \
    --results-dir results/100seeds
```

**测试组合数**：3,600个模型 × 3 modes × 10 test MRs × 4 imputations = **432,000个测试**

### 3.4 单次实验（调试用）

**训练**：
```bash
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 50 \
    --save-model
```

**测试**：
```bash
python experiments/run_experiment.py \
    --phase test \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --mode MCAR \
    --test-mr 0.3 \
    --imputation mean \
    --model-dir models/100seeds
```

---

## 4. 实验规模统计

### 4.1 完整规模（100种子）

| 层级 | 取值 | 说明 |
|------|------|------|
| L1 (Seed) | 100 | 0-99 |
| L3 (Batch) | 6 | 2C, 3C, R2.5, R3, RW, Sim_satellite |
| L4 (Model) | 3 | mlp, lstm, cnn |
| L5 (use_mim) | 2 | true, false |
| **训练组合** | **3,600** | L1×L3×L4×L5 |
| L7 (Mode) | 3 | MCAR, MAR, MNAR |
| L8 (Test MR) | 10 | 0.0-0.9 |
| L9 (Imputation) | 4 | mean, knn, iterative, zero |
| **测试组合/模型** | **120** | L7×L8×L9 |
| **总结果数** | **432,000** | 3,600 × 120 |

### 4.2 小规模测试模式

| 模式 | Seeds | Batches | 训练模型数 | 总结果数 | 用途 |
|------|-------|---------|-----------|---------|------|
| `--tiny` | 1 | 1 | 6 | 720 | 验证单次运行 |
| `--mini` | 1 | 2 | 12 | 1,440 | 验证跨批次 |
| `--test` | 2 | 3 | 36 | 4,320 | 验证完整流程 |
| `--full` | 100 | 6 | 3,600 | 432,000 | 完整实验 |

---

## 5. 结果格式

### 5.1 模型文件命名

```
models/{experiment_dir}/
└── seed{seed}_batch{batch}_model{model}_{mim_suffix}.pt

示例:
├── seed0_batch2C_modelmlp_mim.pt        # use_mim=true
├── seed0_batch2C_modelmlp_no_mim.pt     # use_mim=false
├── seed0_batch2C_modellstm_mim.pt
├── seed0_batch3C_modelcnn_no_mim.pt
└── ...
```

### 5.2 结果文件（JSON）

**单条测试结果**：
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

**聚合结果**（CSV）：
```bash
results/{experiment_dir}/
├── seed42_batch2C_modelmlp_mim_modeMCAR_mr0.3_impmean.json
├── ...
└── aggregated_results.csv  # 自动聚合
```

CSV列：
- `seed`, `batch`, `model`, `use_mim` - L1, L3, L4, L5
- `mode`, `test_mr`, `imputation` - L7, L8, L9
- `test_mae`, `test_rmse` - 评估指标
- `status` - success/failed/timeout
- `elapsed` - 耗时（秒）

---

## 6. 关键设计对比

### 6.1 与旧5层循环架构的区别

| 特性 | 旧5层循环 | 新9层架构 |
|------|----------|----------|
| 核心对比 | MIM vs Baseline方法 | use_mim（有/无指示器） |
| 架构维度 | dataset→batch→seed→model→method | L1-L9，明确分界线 |
| 训练阶段 | 不明确 | L1-L6，每个组合独立训练 |
| 测试阶段 | 仅MCAR | L7-L9，支持三种缺失模式 |
| 缺失处理 | 测试时单一MR | 训练多MR混合，测试多MR评估 |

### 6.2 MIM与插补的关系

**正确理解**：
```
MIM ≠ 不插补
MIM = 插补 + 缺失指示器（Mask）

对比维度：
- use_mim=false: [插补值] → 16维
- use_mim=true:  [插补值] + [掩码] → 32维
```

**2×4实验矩阵**：

| use_mim ↓ / Imputation → | mean | knn | iterative | zero |
|--------------------------|------|-----|-----------|------|
| **false** | 16维 | 16维 | 16维 | 16维 |
| **true** | 32维 | 32维 | 32维 | 32维 |

---

## 7. 监控与故障排查

### 7.1 监控实验进度

```bash
# 实时日志
tail -f logs/experiments/100seeds_*.log

# 进度统计
grep "Training:" logs/experiments/100seeds_*.log | tail -5

# 成功计数
grep -c "✓" logs/experiments/100seeds_*.log

# 失败计数
grep -c "✗" logs/experiments/100seeds_*.log
```

### 7.2 检查模型保存

```bash
# 已保存模型数
ls models/100seeds/*.pt | wc -l
# 预期: 3,600

# 检查特定种子
ls models/100seeds/seed42*.pt
# 预期: 6×3×2 = 36个模型
```

### 7.3 常见问题

**问题1：实验中断后如何恢复？**
```bash
# 重新运行相同命令，自动跳过已完成的模型
python experiments/run_batch_experiments.py --phase full ...
```

**问题2：如何只运行特定种子/批次？**
```bash
python experiments/run_batch_experiments.py \
    --seeds 42 43 \
    --batches 2C 3C \
    ...
```

**问题3：如何测试特定配置？**
```bash
# 单次实验
python experiments/run_experiment.py \
    --phase test \
    --seed 42 --batch 2C --model mlp --use-mim true \
    --mode MAR --test-mr 0.5 --imputation knn \
    --model-dir models/100seeds
```

---

## 8. 相关文档

| 文档 | 内容 | 置信度 |
|------|------|--------|
| [../meta.md](../meta.md) | 9层架构定义（权威来源） | 🟢 S |
| [CONFIDENCE.md](CONFIDENCE.md) | 文档置信度体系 | 🟢 S |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 代码架构实现 | 🟡 A |
| [QUICKSTART.md](QUICKSTART.md) | 快速入门 | 🟡 A |
| [DATASETS.md](DATASETS.md) | 数据集说明 | 🟠 B |
| [HISTORY.md](HISTORY.md) | 历史文档索引 | 🟠 B |

---

*文档版本: v3.0*
*架构版本: 9层实验架构*
*最后更新: 2026-03-19*
*对应代码: c87f6b7*
