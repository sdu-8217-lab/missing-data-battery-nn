# 论文修订计划 v2：结构感知缺失处理 + 公平 Benchmark

> 基于文献分析（134 篇 WoS）与当前 battle royale 结果，论文方向从“MIM 自证”升级为 **GraphMIM：面向电池 SOH 结构化特征缺失的图注意力方法 + 统一 Benchmark**。

---

## 1. 新论文核心信息

### 1.1 题目
**GraphMIM: Graph-Aware Missing Indicator Learning for Robust Battery State-of-Health Estimation under Structured Sensor Failures**

### 1.2 核心贡献
1. **方法**：GraphMIM —— 融合特征图神经网络（GNN）与语义分组池化的端到端缺失处理模块。
2. **基准**：覆盖 6 种缺失模式、10+ 方法、多数据集的电池 SOH 缺失数据公平 benchmark。
3. **洞察**：
   - 结构化缺失（channel/group/road_course）下，结构感知端到端方法优于传统两阶段插补→估计；
   - 简单的 mean imputation + MLP 已经是强基线，但 GNN/GraphMIM 在高结构化缺失下更稳健；
   - 跨数据集结论需控制特征语义一致性。

---

## 2. 方法设计（GraphMIM）

### 2.1 输入与问题设定
- 输入：每个循环 16 维 handcrafted 特征 + 缺失指示器（MIM 形式）。
- 目标：SOH = capacity / initial capacity。
- 训练：uniform_multi_rate（MR 0–90% 混合）。
- 测试：在 MR 0.1–0.9 上逐点评估。

### 2.2 GraphMIM 架构
1. **特征图构建**：16 个特征为节点；同组内全连接，跨组按物理相关性连接。
2. **GAT 消息传递**：观测节点向缺失节点传播信息。
3. **组级注意力池化**：按电压/电流/充电量/斜率等语义组池化，得到组级上下文。
4. **节点增强**：组级上下文广播回节点，增强缺失位置表示。
5. **SOH 估计**：插补后的完整特征 + 缺失指示器 → MLP/LSTM。

### 2.3 对比方法
- **端到端**：Baseline-Uniform, MultiMR-Uniform, MIM-Uniform, GroupMIM-Uniform, GNN-Uniform, FMG-Uniform, GraphMIM-Uniform。
- **两阶段**：Mean/KNN/Iterative imputation + MLP。
- **线性**：Linear, Linear-MIM。

---

## 3. 实验计划

### 3.1 主实验
- 数据集：XJTU 3C（主），HUST 1 / MIT 2017-05-12 / TJU NCA / NASA（跨数据集验证）。
- 缺失模式：bernoulli, block, channel, group, mixed, road_course。
- 重复：n=30 seeds（主实验），n=5 smoke test（探索）。
- 评估：MAE, RMSE, R²；Wilcoxon + BH 校正。

### 3.2 关键消融
- GNN vs GraphMIM（验证组级池化贡献）
- MIM vs MultiMR（验证缺失指示器贡献）
- 两阶段 vs 端到端（验证联合优化价值）

### 3.3 当前已收集/正在收集的数据
| 实验 | 状态 | 备注 |
|---|---|---|
| battle_royale_* n=5（6 模式 × 8 方法） | ✅ 完成 | 在 `results/battle_royale_*/` |
| battle_royale bernoulli/channel/road_course n=30 | 🔄 运行中 | GPU，预计 2 小时完成 |
| linear baseline n=30（6 模式） | ✅/🔄 部分完成 | channel/road_course 完成，其余运行中 |
| two-stage n=5（mean/knn/iterative × channel/road_course） | ✅ 完成 | 在 `results/two_stage_n5/` |
| GraphMIM n=5（6 模式） | ⏸ 暂停 | CPU 太慢，待 GPU 空闲后重跑 |

---

## 4. 论文结构

1. **Introduction**
   - 电池 SOH 估计的重要性与 BMS 数据缺失的普遍性。
   - 现有工作多聚焦独立随机缺失或曲线重建，忽略特征级结构化缺失。
   - 本文提出 GraphMIM 与统一 benchmark。

2. **Related Work**
   - 电池 SOH 估计（LSTM/GRU/Transformer/CNN）。
   - 缺失数据处理：插补（KNN/GPR/CGAN/self-attention）vs 端到端 MIM。
   - GNN 在电池领域的应用（稀少，多为曲线/图像图）。
   - 物理约束（PINN/单调性）。

3. **Method**
   - 问题形式化。
   - 缺失模式定义。
   - GraphMIM 架构细节。
   - 训练策略与损失函数。

4. **Experimental Setup**
   - 数据集与特征工程。
   - 缺失模式实现细节。
   - 对比方法与超参数。
   - 评估指标与统计检验。

5. **Results**
   - XJTU 3C 主结果：6 模式 × 方法对比。
   - 消融实验。
   - 跨数据集验证。
   - 统计显著性分析。

6. **Discussion**
   - 为什么 GraphMIM 在结构化缺失下更有效？
   - 为什么 mean imputation + MLP 已经是强基线？
   - 局限：单一数据集（当前）、未加入物理约束、未覆盖真实 BMS 数据。

7. **Conclusion**

---

## 5. 待决策事项

1. 是否将 GraphMIM 纳入正在跑的 n=30 battle royale？（需停止当前 GPU 任务并重启）
2. 是否增加 self-attention imputation + LSTM 两阶段基线？
3. 是否加入单调性物理约束？
4. NASA 是作为主数据集还是挑战性案例？

---

*最后更新：2026-06-17*
