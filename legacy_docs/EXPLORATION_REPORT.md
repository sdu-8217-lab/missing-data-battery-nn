# 预实验探索报告

> 生成时间：2026-06-17 12:40
> 目的：评估 GraphMIM 方向可行性、方法对比、以及下一步核心实验设计。

---

## 1. 已完成的实验

### 1.1 代码与模型
- ✅ 实现 **GraphMIM**（`src/models/graphmim.py`）：GNN + 语义组级注意力池化
- ✅ 接入模型工厂与实验配置（`MLP-GraphMIM-Uniform`, `LSTM-GraphMIM-Uniform`）
- ✅ 修复 `src/main.py` 缺失 `road_course` 选项
- ✅ 实现 **两阶段基线**（`scripts/run_two_stage_baseline.py`）：Mean / KNN / Iterative imputation + MLP
- ✅ 实现 **跨数据集线性基线**（`scripts/cross_dataset_linear.py`）
- ✅ 实现 **综合对比脚本**（`scripts/compare_all_methods.py`）

### 1.2 数据收集
| 实验 | 状态 | 规模 | 备注 |
|---|---|---|---|
| battle_royale_*（端到端 8 方法） | ✅ 完成 | 6 模式 × 8 方法 × 5 seeds | 在 `results/battle_royale_*/` |
| battle_royale bernoulli/channel/road_course | 🔄 运行中 | 3 模式 × 8 方法 × 30 seeds | GPU，预计 6/17 15:36 完成 |
| linear baseline（6 模式） | ✅ 完成 | 6 模式 × 2 方法 × 30 seeds | 在 `results/linear_baseline_n30/` |
| two-stage baseline | ✅ 完成 | channel/road_course × 3 imputers × 5 seeds | 在 `results/two_stage_n5/` |
| cross-dataset linear | ✅ 完成 | 5 数据集 × 5 数据集 × 5 MR | 在 `results/cross_dataset_linear/` |
| GraphMIM n=5 | ⏸ 已调度 | CPU 太慢，已设置 GPU 自动调度器 | 代码已验证可跑通 |
| linear imputer baseline | 🔄 运行中 | 6 模式 × 2 imputers × 30 seeds | 预计 6/17 13:00 完成 |
| time-split MLP-MIM | 🔄 运行中 | 3 模式 × 1 模型 | 评估真实时间泛化 |

---

## 2. 关键发现

### 2.1 端到端方法对比（n=5，XJTU 3C）

**MAE 中位数汇总：**

| 方法 | bernoulli | block | channel | group | mixed | road_course |
|---|---|---|---|---|---|---|
| MLP-GNN-Uniform | 0.0155 | 0.0151 | **0.0146** | 0.0160 | **0.0150** | 0.0146 |
| LSTM-MIM-Uniform | **0.0154** | 0.0164 | 0.0187 | **0.0149** | 0.0165 | **0.0149** |
| LSTM-GroupMIM-Uniform | 0.0170 | 0.0175 | 0.0186 | 0.0175 | 0.0168 | 0.0151 |
| TwoStage-Mean | — | — | 0.0172 | — | — | 0.0168 |
| Linear | 0.0181 | 0.0184 | 0.0217 | 0.0233 | 0.0183 | 0.0195 |

**洞察：**
1. **GNN 在 channel / mixed 上最强**，说明特征级图消息传递对通道结构化缺失非常有效。
2. **LSTM-MIM 在 group / road_course 上最强**，说明序列模型 + MIM 对组级/强结构化缺失有优势。
3. **GroupMIM 稳定但 rarely 最佳**，可能需要与 GNN 融合才能发挥最大价值。
4. **TwoStage-Mean 是强基线**，在 channel/road_course 上接近部分端到端方法。
5. **线性模型明显落后**，但该任务本身对神经网络友好。

### 2.2 两阶段 vs 端到端
- TwoStage-Mean 在 channel 上 MAE=0.0172，优于 LSTM-MIM (0.0187)、接近 MLP-GroupMIM (0.0173)。
- 这说明**简单插补 + 强神经网络 backbone 已经很有竞争力**。
- GraphMIM 要证明价值，必须在 channel/road_course 上显著超过 TwoStage-Mean 和 GNN-Uniform。

### 2.3 跨数据集线性基线
- 同数据集平均 MAE：0.0744
- 跨数据集平均 MAE：0.1204
- **跨数据集泛化非常困难**，尤其是 NASA 作为目标时。
- 这意味着主实验必须**在每个数据集上独立训练/测试**，不能依赖零样本迁移。

### 2.4 时间泛化实验
- 按每个电池内部时间顺序划分：前 70% 训练，后 30% 测试。
- MLP-MIM-Uniform 在时间划分下的 MAE：**0.048–0.054**。
- 相比之下，跨电池随机划分下 MLP-MIM 的 MAE 仅 **0.015–0.020**。
- **结论**：当前主实验本质上是跨电池分布内插值；真实未来退化预测（时间外推）是更难的问题，需作为独立实验讨论。

---

## 3. GraphMIM 可行性评估

### 3.1 设计合理性
GraphMIM 的设计逻辑成立：
- GNN-Uniform 已证明对 channel 最有效；
- GroupMIM 已证明对 group/road_course 有效；
- 融合两者理应同时获得 channel 和 group/road_course 的优势。

### 3.2 尚未验证的风险
- **组级池化是否真的带来提升？** 需要消融实验（GraphMIM vs GNN-Uniform with group_ids）。
- **参数量增加是否导致过拟合？** GraphMIM 35K params vs GNN-Uniform 33K params，差距小，风险低。
- **CPU 上训练极慢**，说明 GraphMIM 对 GPU 有依赖，但这不是方法问题。

### 3.3 建议
- **必须跑 GraphMIM n=5 对比**，尤其是 channel 和 road_course。
- 若 GraphMIM n=5 优于 GNN-Uniform，则纳入 n=30 核心实验。
- 若 GraphMIM 不优于 GNN，则论文核心方法可退化为 **GNN-Uniform + 统一 benchmark**，GraphMIM 作为消融/扩展。

---

## 4. 下一步核心实验建议

### 优先级 1（今天下午必须完成）
1. **等待 battle royale n=30 完成**（预计 15:36），获得统计稳健的主结果。
2. **GPU 空闲后立即跑 GraphMIM n=5**（channel/road_course，约 20-30 分钟）。
3. **根据 GraphMIM n=5 结果决定**：
   - 若显著优于 GNN → 启动 GraphMIM n=30（3 模式，约 1.5 小时，可能到 17:00）。
   - 若不优于 GNN → 不跑 n=30，把 GNN-Uniform 作为核心方法。

### 优先级 2（今天或明天）
4. **跨数据集端到端验证**：XJTU/HUST/MIT/TJU/NASA 上跑 MLP-GNN-Uniform 和 MLP-MIM-Uniform（n=30）。
5. **两阶段基线扩展**：实现 self-attention imputation + LSTM（文献主流）。

### 优先级 3（后续）
6. **单调性物理约束**：加入 SOH 单调不增约束。
7. **自适应方法选择**：根据缺失结构自动选择最佳策略。

---

## 5. 当前运行状态

```
GPU: battle_royale n=30 (bernoulli → channel → road_course)
     进度: ~75/240 for bernoulli, 预计 6/17 15:36 全部完成
CPU: 空闲（linear、two-stage、cross-dataset 均已完成）
```

---

## 6. 决策点

1. **是否现在停止 battle royale n=30 去跑 GraphMIM n=5？**
   - 建议：否。n=30 主实验更重要，且已跑 30+ 分钟。

2. **GraphMIM n=30 是否值得跑？**
   - 取决于 15:36 后的 GraphMIM n=5 结果。

3. **论文核心方法定位：**
   - 当前证据支持："结构感知端到端缺失处理（GNN/GraphMIM）+ 公平 benchmark"
   - 若 GraphMIM 胜，升级为 "GraphMIM"
   - 若 GraphMIM 平/负，降级为 "GNN-Uniform + benchmark"
