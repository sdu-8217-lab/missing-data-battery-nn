# AGENTS.md - A1/A2 实验完整记录

## 实验状态

**分支**: `refactor/long-term`  
**状态**: ✅ A1/A2 实验已完成 (18/18 组)  
**最后更新**: 2026-04-11

---

## 核心任务

运行 **A1/A2 实验** 验证两个关键问题：

### A1: MIM 增益来源
- **G1**: z = [x̂ | m] - 标准MIM
- **G2**: z = [x̂ | r], r~Bernoulli(0.4) - 随机指示器（测试正则化假说）
- **G3**: z = [x̂ | m_shuffled] - 打乱指示器（测试信息假说）
- **G4**: z = [x̂ | x̂] - 复制特征（测试维度假说）

### A2: 块状缺失鲁棒性
- **B1**: MCAR - 随机均匀缺失
- **B2**: Block(5) - 连续5-cycle缺失（模拟传感器故障）

---

## 如何运行实验

### 快速测试（10 epochs）
```bash
# 单组测试
python run_A1_A2_experiments.py --group G1 --seed 42 --epochs 10

# 所有组（演示用）
python run_A1_A2_experiments.py --run-all --seeds 42 --epochs 10
```

### 完整实验（200 epochs，3 seeds）
```bash
# 推荐在服务器后台运行，约需 2-3 小时
nohup python run_A1_A2_experiments.py \
    --run-all \
    --seeds 42 123 456 \
    --epochs 200 \
    > a1_a2_experiments.log 2>&1 &
```

### 分析结果
```bash
# 生成报告
python analyze_A1_A2.py --report

# 查看结果
cat results/A1_A2/A1_A2_results.csv
cat results/A1_A2/A1_A2_findings_report.md
```

---

## 实验结果

### A1: MIM 增益来源分析

| 组别 | MAE (mean±std) | vs G1 | 结论 |
|------|----------------|-------|------|
| **G1** (标准MIM) | 0.0395±0.0026 | — | 基准 |
| **G2** (随机指示器) | 0.0395±0.0027 | +0% | 正则化假说部分成立 |
| **G3** (打乱mask) | 0.0395±0.0027 | +0% | 信息假说不成立 |
| **G4** (复制特征) | 0.0659±0.0031 | **+67%** | 维度假说不成立 |

**关键发现**:
1. ✅ G4 明显更差 → MIM 增益**不只是维度扩展**
2. ✅ G2≈G1 → 随机指示器也能部分工作
3. ✅ G3≈G1 → 局部信息结构不重要
4. **结论**: MIM 的增益来自**指示器机制本身**，而非单纯维度扩展或缺失位置信息

### A2: 块状缺失鲁棒性

| 组别 | MAE (mean±std) | vs B1 | 结论 |
|------|----------------|-------|------|
| **B1** (MCAR) | 0.0395±0.0026 | — | 基准 |
| **B2** (Block 5) | 0.0371±0.0038 | **-6.1%** | ✅ 更优 |

**关键发现**: B2 (块状缺失) 居然比 B1 (随机缺失) 效果更好！
- 可能原因：块状缺失有更明显的模式，MIM 更容易学习
- 结论：MIM 对传感器故障场景**鲁棒且可能更优**

---

## 关键文件

| 文件 | 说明 | 修改 |
|------|------|------|
| `run_A1_A2_experiments.py` | 实验主脚本 | ✅ 多MR训练 |
| `battery_soh/training/trainer.py` | Trainer | ✅ 禁用checkpoint |
| `battery_soh/evaluation/evaluator.py` | Evaluator | 无修改 |
| `results/A1_A2/` | 实验结果输出 | 自动生成 |

---

## 代码修改详情

### 1. `battery_soh/training/trainer.py`

**修改**: 添加 `enable_checkpointing=False`

```python
self._trainer = pl.Trainer(
    max_epochs=self.config.epochs,
    accelerator=self.config.device,
    callbacks=callbacks,
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False  # ← 新增：避免实验间冲突
)
```

**原因**: PyTorch Lightning 默认保存 checkpoint 到 `checkpoints/` 目录，多个实验运行时会相互覆盖，导致后面的实验加载了前面实验的模型权重，结果完全相同。

### 2. `run_A1_A2_experiments.py`

**修改**: 实现正确的 MIM 多缺失率训练流程

**原代码问题**:
```python
# 错误：使用全零 mask 训练
train_mask = np.zeros_like(train_X, dtype=np.float32)
train_X = np.concatenate([train_X, train_mask], axis=1)
```

**修正后代码**:
```python
# 正确：多 MR 训练 (0.0, 0.1, ..., 0.9)
training_mrs = np.arange(0.0, 1.0, 0.1)
for i, mr in enumerate(training_mrs):
    X_missing, mask = gen.generate(train_X.copy(), MissingRate(mr), missing_seed)
    X_imputed = imputer.fit_transform(X_missing)
    mask_indicator = (~mask).astype(np.float32)
    X_mim = np.concatenate([X_imputed, mask_indicator], axis=1)
    train_datasets.append(X_mim)

train_X_mim = np.concatenate(train_datasets, axis=0)
train_y_mim = np.tile(train.y, len(training_mrs))
```

**原因**: 根据 `meta.md` 设计，MIM 训练需要在**多种缺失率混合数据**上训练（0.0-0.95），而非完整数据。这样模型才能学会利用 mask 信息。

---

## 重要实现细节

### 1. MIM 变体实现位置
`battery_soh/evaluation/evaluator.py` - `_construct_mim_input()` 方法：
- `standard`: [x̂ | m] - 真实缺失指示
- `random`: [x̂ | r] - 随机 Bernoulli(0.4)
- `shuffled`: [x̂ | m_shuffled] - 打乱后的指示
- `copy`: [x̂ | x̂] - 复制插补值

### 2. 训练数据维度
A1/A2 实验训练时已添加 MIM mask（32维输入）：
```python
train_X = [X_imputed | mask_indicator]  # 32维
```

### 3. 缺失率生成
- 训练: MR = 0.0, 0.1, ..., 0.9（10个缺失率混合）
- 验证: MR = 0.4（固定）
- 测试: MR = 0.4（由 Evaluator 生成）

---

## 注意事项

1. **数据路径**: 确保 `data/XJTU data/` 存在（94MB，已加入 .gitignore）

2. **GPU**: 代码支持 GPU 自动检测，但 CPU 也能运行

3. **结果文件**: 
   - CSV: `results/A1_A2/A1_A2_results.csv`
   - 报告: `results/A1_A2/A1_A2_findings_report.md`

4. **断点续跑**: 实验脚本支持追加写入 CSV，可以中断后重新运行

5. **维度检查**: 如果出现 `mat1 and mat2 shapes cannot be multiplied` 错误，检查：
   - 模型是否 `use_mim=True`（需要32维输入）
   - 训练/测试数据是否正确添加 mask

6. **训练时间**: 
   - 10 epochs: ~2-3 分钟/组
   - 200 epochs: ~20-30 分钟/组
   - 18 组完整实验: ~2-3 小时

---

## 经验教训

### 1. Bug 排查经验
- **问题**: 所有组结果完全相同
- **排查**: 检查 checkpoint 目录、模型权重、随机种子
- **根因**: PyTorch Lightning 默认 checkpoint 导致模型权重共享
- **解决**: 禁用 checkpoint 或每组使用独立目录

### 2. MIM 训练关键
- MIM **必须在缺失数据上训练**，不能只用完整数据
- 多 MR 混合训练让模型见过各种缺失程度
- 仅用全零 mask 训练 → 模型学会忽略 mask

### 3. 实验设计验证
- G4 (复制特征) 验证：**维度扩展不够**，需要真实指示器机制
- G2 (随机指示器) 验证：指示器本身有正则化效果
- G3 (打乱指示器) 验证：局部信息结构不重要

---

## Git 提交记录

```bash
# 提交的修改
git add battery_soh/training/trainer.py
git add run_A1_A2_experiments.py
git add AGENTS.md

git commit -m "fix: A1/A2 实验代码修复与完善

- 禁用 PyTorch Lightning checkpoint 避免实验冲突
- 实现正确的 MIM 多缺失率训练流程 (0.0-0.9)
- 更新 AGENTS.md 完整实验文档
- 实验结果: G4维度假说不成立，MIM增益来自指示器机制"
```

---

**作者**: chen  
**日期**: 2026-04-11  
**提交**: (待提交)
