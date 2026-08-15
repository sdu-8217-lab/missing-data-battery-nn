# 技术逻辑与技术哲学导向的文献复现路线图

> 本文档从我们所研究的核心问题——**数据缺失条件下的锂离子电池 SOH 估计**——出发，结合我们已经提出的 MIM 方法，从技术逻辑和技术哲学两个维度，规划接下来应该复现哪些文献方法。

---

## 一、核心问题不变

**研究问题**：在数据缺失条件下，如何准确、稳健、可解释地估计锂离子电池健康状态（SOH）？

缺失可以发生在多个维度：
- **时间维度**：某个循环内的时间步缺失；
- **特征维度**：某个传感器通道缺失；
- **循环维度**：连续多个循环缺失；
- **域维度**：跨电池、跨数据集、跨化学体系的数据缺失或分布偏移。

我们的目标不是“修复”缺失，而是让模型在缺失存在的情况下仍能做出可靠估计，并理解缺失带来的影响。

---

## 二、MIM 的方法论定位

### 2.1 技术逻辑定位

MIM（Missingness Indicator Method）的核心是：

$$
\hat{y} = f(\tilde{x}, m)
$$

其中 $x$ 是原始特征，$\tilde{x}$ 是填充后的特征，$m$ 是缺失指示器。

MIM 解决的是一个**信息论问题**：
- 它把“哪些数据缺失”这一信息显式输入模型；
- 让模型区分“真实观测值”与“填充值”；
- 针对不同缺失模式学习不同的条件映射。

### 2.2 技术哲学定位

MIM 完成了一次认识论转向：

> 从“缺失是需要修复的缺陷” → “缺失是携带系统信息的现象”。

但它仍停留在**现象学层面**：它承认缺失存在，却没有追问缺失为何发生、缺失与物理系统有何关系、缺失的不确定性如何量化。

### 2.3 MIM 的边界

MIM 没有解决：
- 时间维度上的连续动态缺失；
- 特征之间的物理/语义结构关系（GroupMIM/GNN 部分弥补）；
- 缺失值的分布建模；
- 缺失条件下的预测不确定性；
- 缺失的生成机制。

---

## 三、缺失处理层次模型

基于 MIM 的边界，我们建立一个从浅到深的缺失处理能力层次模型：

```
Level 0: 无知（直接填充，不告诉模型缺失）
Level 1: 承认缺失 —— MIM
Level 2: 结构化理解缺失 —— GroupMIM / GNN
Level 3: 连续动态理解缺失 —— Neural CDE / ODE
Level 4: 物理约束下的缺失 —— PINN / Physics-Augmented CDE
Level 5: 生成式缺失建模 —— GAN / VAE / Diffusion
Level 6: 概率化缺失建模 —— GP / Neural SDE / Bayesian NN
Level 7: 因果/机制化缺失建模 —— Missing Mechanism Identification
```

### 各层次的技术逻辑含义

| 层次 | 核心问题 | 所需数学工具 |
|-----|---------|------------|
| Level 3 | 时间维度缺失如何用连续动态填补？ | Neural CDE / ODE-RNN / Latent ODE |
| Level 4 | 如何利用电池物理约束减少缺失带来的不确定性？ | PINN / Physics-Augmented CDE / 单调性约束 |
| Level 5 | 如何学习缺失值的分布，而不是只做点预测？ | VAE / GAN / Diffusion / Normalizing Flow |
| Level 6 | 如何量化缺失条件下的预测不确定性？ | GP / Neural SDE / Bayesian Deep Learning |
| Level 7 | 如何识别缺失机制并针对性处理？ | Causal Inference / Missing Data Theory |

### 各层次的技术哲学含义

| 层次 | 哲学立场 | 对“缺失”的理解 |
|-----|---------|--------------|
| Level 1 | 现象学 | 缺失是一种需要被标记的现象 |
| Level 2 | 结构主义 | 缺失在特征/语义结构中有位置关系 |
| Level 3 | 过程哲学 | 缺失是连续时间路径上的“空隙”，可用动态过程弥合 |
| Level 4 | 物理实在论 | 缺失必须服从物理定律，物理约束是“免费信息” |
| Level 5 | 生成论 | 缺失数据本身是从完整数据分布中采样得到的 |
| Level 6 | 认识论谦逊 | 缺失导致不确定性，预测应以概率形式呈现 |
| Level 7 | 因果论 | 缺失有其生成机制，理解机制才能真正鲁棒 |

---

## 四、推荐复现的文献清单

基于上述层次模型，从 `literature/savedrecs (44).xls` 筛选出的 80 篇核心相关文献中，挑选出最能填补每一层缺口的论文。

### Level 3：连续动态缺失处理

**已复现**：
- **BatteryCDE**: A Transferable Future Capacity Estimation Method for Battery Degradation With Irregular Sampling and Missing Data (Wang et al., IEEE TTE 2025)

**推荐复现**：
- **Physics-augmented neural controlled differential equations for lithium-ion battery state-of-health prediction under missing cycling data** (Toom et al., Journal of Power Sources 2026)
  - DOI: 10.1016/j.jpowsour.2026.239987
  - 理由：与 BatteryCDE 同范式，但加入物理约束；可直接复用现有 CDE 代码；验证物理约束是否让 CDE 更稳健。

### Level 4：物理约束下的缺失处理

**推荐复现 A**：
- **Physics-Informed neural SOH Estimation method for Lithium-ion battery under partial observability and sparse sensor data** (Ma et al., Ionics 2026)
  - DOI: 10.1007/s11581-025-06805-0
  - 理由：标题直接命中核心问题；对应项目计划中的物理约束模块；可检验物理约束在传感器稀疏时的作用。

**推荐复现 B**：
- **State-of-charge estimation of batteries using parameterized physics-informed neural networks** (Jang et al., Journal of Energy Storage 2026)
  - DOI: 10.1016/j.est.2026.122235
  - 理由：把物理参数嵌入神经网络，是物理约束的另一种形式；SOC 与 SOH 方法论可迁移。

### Level 5：生成式缺失建模

**推荐复现**：
- **A Novel Method Based on Hybridization of Generative Adversarial Imputation Nets and SDAE-Kriging for RUL Prediction of Lithium-Ion Battery in Scenarios of Missing and Incomplete Data** (Li et al., IEEE Transactions on Industry Applications 2025)
  - DOI: 10.1109/TIA.2025.3549408
  - 理由：代表领域主流“插补 + 估计”两阶段范式；与我们的端到端方法形成哲学对比；可直接对比“修复现实”与“适应现实”。

**备选**：
- **When Do We Need Complex Generative Models for Time Series Imputation? A Case Study on Battery and Electrical Degradation Data** (Electronics 2026)
  - 理由：直接讨论复杂生成模型在电池插补中的必要性。

### Level 6：概率化缺失建模

**推荐复现**：
- **Transfer learning with composite kernel sparse Gaussian process-aided model for probabilistic state of health estimation of lithium-ion batteries against multi-source coupled harsh scenarios** (Xiong et al., Applied Energy 2025)
  - 理由：同时覆盖迁移学习和概率估计；GP 天然提供不确定性量化；对应真实多源、恶劣、缺失场景。

### Level 2 的补充：注意力机制在结构化缺失下

**推荐复现**：
- **SOH estimation of lithium-ion batteries subject to partly missing data: A Kolmogorov-Arnold-Linformer model** (Shao et al., Neurocomputing 2025)
  - 理由：系统测试 Transformer/Linformer/KAN 在结构化缺失下的表现；与 MIM/GroupMIM 形成全局注意力 vs 语义分组的对比。

---

## 五、推荐的复现顺序

```
第 1 站：BatteryCDE ✅（已完成，Level 3 入门）
    ↓
第 2 站：Physics-augmented Neural CDE（Level 3 → Level 4 的桥梁）
    ↓
第 3 站：Physics-Informed SOH under sparse data（Level 4 深入）
    ↓
第 4 站：GAN + SDAE-Kriging（Level 5，与端到端方法哲学对比）
    ↓
第 5 站：Transformer / Linformer for missing SOH（Level 2 的注意力补充）
    ↓
第 6 站：Transfer learning + GP（Level 6，概率 + 迁移）
    ↓
未来：Level 7 因果缺失建模（需要进一步文献搜索）
```

### 下一轮优先复现的两篇

如果只能选两篇作为接下来的重点：

1. **Physics-augmented Neural CDE (Toom et al., 2026)**
   - 技术逻辑：在现有代码上扩展物理约束，工作量可控；
   - 技术哲学：从“连续时间动态”走向“物理约束下的连续动态”。

2. **GAN + SDAE-Kriging (Li et al., 2025)**
   - 技术逻辑：领域主流两阶段范式，与端到端方法直接对比；
   - 技术哲学：代表“修复现实”的哲学，与 MIM 的“适应现实”形成张力。

---

## 六、这个路线图对项目论文的意义

按此路线复现，最终可以构建一个**层次化的方法对比框架**：

> **在统一数据集和统一缺失模式下，从 Level 1（MIM）到 Level 6（概率迁移）的方法表现如何？**

这将成为项目论文的核心贡献：

1. 提出结构化缺失的分类（bernoulli / block / channel / group / mixed / road_course）；
2. 建立从 MIM 到物理约束 CDE 的端到端方法谱系；
3. 系统复现文献主流方法，在统一协议下公平对比；
4. 证明：在真实缺失场景下，**结构感知 + 物理约束 + 概率量化**的融合方向最有前景。

---

## 七、相关文档

- `docs/11_MIM_method_reflection.md`：MIM 方法的技术逻辑与技术哲学反思；
- `reproductions/batteryCDE_wang2025/TECHNICAL_REFLECTION.md`：BatteryCDE 复现的技术反思；
- `literature/core_battery_missing_papers.csv`：从文献库中筛选出的 80 篇核心相关论文；
- `reproductions/batteryCDE_wang2025/`：第一站复现代码与结果。
