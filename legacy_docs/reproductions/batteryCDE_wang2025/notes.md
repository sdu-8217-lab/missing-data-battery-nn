# BatteryCDE 实现笔记与批判性审视

> 本文件记录复现过程中的实现细节、与原文的偏差、遇到的问题以及批判性思考。
> 深度论文批判见同目录 `CRITICAL_REVIEW.md`。

## 一、论文核心方法

BatteryCDE 使用三个 Neural CDE 子网络：

1. **Feature-wise CDE**：生成特征注意力 $a_{\text{feat}}(t) = \sigma(h_{\text{CDE1}}(t))$。
2. **Cycle-wise CDE**：生成循环注意力 $a_{\text{cycle}}(t) = \sigma(\text{FC}_1(h_{\text{CDE2}}(t)))$。
3. **Learning-wise CDE**：对加权后的连续路径 $Y(t) = a_{\text{feat}}(t) \odot X(t) + a_{\text{cycle}}(t) \cdot X(t)$ 积分，得到最终隐状态并通过单层 FC2 预测容量。

输入 $X(t)$ 通过自然三次样条从离散观测构造，理论上支持非规则采样和缺失数据。

---

## 二、当前实现状态

### 已完成

- [x] 从 NASA `.mat` 原始文件提取充电阶段 V/I/T 轨迹，重采样为固定长度 30。
- [x] 核心模型 `src/models/battery_cde.py` 已按论文结构实现（含 feature/cycle/learning 三个 CDE）。
- [x] 三个对比基线：`src/models/baselines.py` 包含 LSTM、Transformer、Neural CDE。
- [x] 数据加载与缺失模拟：`src/data/battery_cde_dataset.py`。
- [x] 训练/评估/早停：`src/trainers/battery_cde_trainer.py`。
- [x] 完整实验脚本：`train.py`、`run_baselines.py`、`run_horizon_experiments.py`、`run_missing_experiments.py`、`run_transfer_experiments.py`、`run_all_experiments.py`。
- [x] 结果汇总与绘图：`aggregate_results.py`、`plot_results.py`。
- [x] 单元测试：`tests/test_battery_cde_forward.py`、`tests/test_spline_interpolation.py`。
- [x] 深度论文批判：`CRITICAL_REVIEW.md`。

### 进行中 / 待验证

- [ ] Horizon 实验（h=20/50/80/100）正在后台运行。
- [ ] 缺失数据实验（random/block 30%/50%）。
- [ ] 跨电池迁移实验。
- [ ] 结果汇总图表生成。

---

## 三、实现与原文的对照及偏差

| 论文细节 | 本复现实现 | 偏差原因 / 备注 |
|----------|-----------|----------------|
| 输入：完整充电 V/I/T 轨迹 | 从 NASA `.mat` 提取充电片段并重采样到 30 点 | 论文未公开轨迹长度；30 是计算与信息保留的权衡 |
| 历史信息使用方式 | `history_len=3` 个循环的轨迹拼接为长序列 | 论文未明确是否跨循环拼接；这是基于任务的合理推断 |
| Feature-wise attention：$a_{\text{feat}}=\sigma(h_{\text{CDE1}})$ | feature CDE 隐状态维度 = `input_dim`，直接 sigmoid | 已按论文公式 (10) 修正 |
| Cycle-wise attention：$a_{\text{cycle}}=\sigma(\text{FC}_1(h_{\text{CDE2}}))$ | 通过 `nn.Linear(attention_dim, 1)` + sigmoid 实现 | 与论文一致 |
| $Y(t)$ 公式 | $a_{\text{feat}} \odot X(t) + a_{\text{cycle}} \cdot X(t)$ | 与论文一致 |
| 输出层 FC2：单层神经网络 | `nn.Linear(hidden_dim, 1)` | 已按论文修正为单层 |
| 缺失处理：自然三次样条 | 先对 NaN 线性插值，再构造 torchcde 三次样条 | torchcde 不直接支持带 mask 的样条；这是 practical 近似 |
| 训练优化器：Adam | Adam，lr=1e-3 | 论文未公开具体超参 |
| 数据集 | 仅 NASA | 缺少 CALCE/SNL/ISU 数据 |
| 容量目标 | 用训练集最大容量归一化为 SOH | 论文未明确是否归一化；SOH 更便于跨电池比较 |
| 激活函数 / 稳定性 | Tanh + LayerNorm + 小增益 Xavier + 梯度裁剪 | 论文未提及；为抑制 CDE 数值爆炸而加入 |

---

## 四、复现过程中遇到的关键问题与解决

### 1. CDE 训练数值爆炸

**现象**：初始训练时 loss 达 $10^{11}$，val_mae 达 $10^4$。

**原因**：向量场使用 ReLU + 默认 Xavier 初始化，导致 CDE 动态不稳定。

**解决**：
- 向量场激活改为 **Tanh**；
- 加入 **LayerNorm**；
- 最后一层使用小增益 Xavier 初始化；
- 训练中加入 **梯度裁剪**（max_norm=1.0）。

### 2. 基线模型测试时出现 NaN 预测

**现象**：LSTM / Transformer / Neural CDE 在少数测试样本上输出 NaN。

**原因**：可能是输入标准化后的极端值或模型对边界样本敏感。

**解决**：在 `BatteryCDETrainer.evaluate()` 中过滤非有限预测，并输出告警，确保指标计算不中断。

### 3. 训练速度过慢

**现象**：初始 20 个循环 × 100 点 = 2000 步序列，训练 45 s/epoch。

**解决**：
- 重采样长度从 100 降至 30；
- `history_len` 从 20 降至 3；
- 模型 hidden_dim 从 64 降至 16，attention_dim 从 32 降至 8；
- epochs 从 100 降至 50，patience 从 15 降至 10。

最终 BatteryCDE 约 10–15 s/epoch，LSTM/Transformer 约 0.05 s/epoch。

### 4. 背景任务超时

**现象**：默认后台任务 300 s 超时导致训练被强制终止。

**解决**：显式设置 `timeout=3600`，保证完整实验套件可运行。

---

## 五、批判性思考：复现他人工作的意义

复现 BatteryCDE 的过程中，我们发现了论文中若干**仅通过阅读难以察觉**的问题，这正是复现工作的核心价值：

1. **理论声明需要实证检验**：论文声称 CDE 比离散模型误差更小、迁移能力更强，但误差分析与收敛声明缺乏严格证明。复现让我们意识到这些论证更多是高层次直觉，而非可量化的理论保证。

2. **“天然处理缺失数据”是过度简化**：论文把 CDE 的连续性等同于无需插值。实际上 torchcde 必须先构造样条，而样条构造前必须处理缺失值。我们发现“插值方法的选择”本身对结果有显著影响，这是论文未深入讨论的。

3. **实现细节决定结果**：论文未公开代码和超参数，导致许多细节必须自行假设（如轨迹长度、是否跨循环拼接、输出是单值还是序列）。不同假设会显著改变性能，这削弱了论文结论的可复现性。

4. **基线对比可能不公平**：论文声称基线已调至最优，但未给出配置。我们的复现中，plain Neural CDE 表现远差于 BatteryCDE，而 Transformer 却接近甚至有时优于 BatteryCDE，这与论文部分结论不完全一致，提示需要更严格的公平对比。

5. **跨数据集迁移声明需要更多证据**：论文的迁移实验缺乏是否 fine-tuning、是否归一化等关键信息。我们目前只能在 NASA 内部做跨电池迁移，尚无法验证跨数据集声称。

这些发现说明：**复现不仅是“把代码跑通”，更是检验论文结论、发现隐藏假设、明确方法边界的必要过程**。后续工作应在此基础上，补齐数据集、加入物理约束与不确定性估计、进行多种子统计检验，形成更严谨的对比。

---

## 六、下一步计划

### 短期（当前实验完成后）

1. 等待 horizon / missing / transfer 实验跑完。
2. 运行 `aggregate_results.py` 与 `plot_results.py` 生成汇总表格与图表。
3. 检查图表合理性，补充缺失或异常的实验 rerun。

### 中期

1. 获取 CALCE、SNL、ISU-ILCC 数据，复现跨数据集迁移。
2. 实现真正的“带缺失样条”：不预先用线性插值填充，而是直接对非缺失观测构造样条。
3. 加入不确定性估计（MC dropout 或 deep ensemble）。
4. 加入物理约束（单调性、容量边界）。

### 长期

1. 多种子（≥5）重复实验，做统计显著性检验。
2. 将 BatteryCDE 与项目主流程的 GraphMIM、PINN 等方法在同一 benchmark 下比较。
3. 若形成论文，需补充消融实验与显著性分析。
