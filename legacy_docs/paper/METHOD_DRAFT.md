# 方法草稿：GraphMIM

## 3.1 符号与问题设定

设电池数据集包含 $N$ 个循环样本，每个样本由 $F$ 维 handcrafted 特征 $\mathbf{x}_i \in \mathbb{R}^F$ 和对应的健康状态标签 $y_i \in \mathbb{R}$ 组成。本文中 $F=16$，$y_i$ 为相对容量（当前容量 / 初始容量）。

缺失由二元掩码 $\mathbf{m}_i \in \{0,1\}^F$ 表示，其中 $m_{ij}=1$ 表示第 $i$ 个样本的第 $j$ 个特征缺失。观测到的零填充特征为：

$$\tilde{x}_{ij} = x_{ij} \cdot (1 - m_{ij})$$

传统 Missing Indicator Mechanism (MIM) 将模型输入构造为：

$$\mathbf{z}_i = [\tilde{\mathbf{x}}_i; \mathbf{m}_i] \in \mathbb{R}^{2F}$$

即把缺失指示器与零填充特征拼接。

## 3.2 缺失模式

为模拟真实 BMS 中的不同故障场景，定义六种缺失模式：

1. **Bernoulli**：每个特征-样本独立以概率 $p$ 缺失，作为随机缺失基线。
2. **Block**：对每个特征通道独立生成交替的观测段与缺失段，模拟传感器通信中断。
3. **Channel**：随机选择若干特征通道整体置为缺失，模拟某传感器完全失效。
4. **Group**：按物理语义组（电压、电流、充电量、斜率/熵）整体缺失，模拟某类传感器组失效。
5. **Mixed**：以一定权重混合上述多种模式。
6. **Road_course**：强结构化缺失，高缺失率且集中在特定通道/组，模拟实车恶劣工况。

所有模式均通过校准使实际缺失率与目标缺失率偏差小于 0.01。

## 3.3 GraphMIM 架构

GraphMIM 的核心思想是：把 $F$ 个特征视为图上的 $F$ 个节点，利用图注意力网络在节点间传播观测信息，从而估计缺失值；同时引入基于物理语义组的注意力池化，增强对组级结构化缺失的鲁棒性。

### 3.3.1 节点构建

对于输入 $\mathbf{z}_i = [\tilde{\mathbf{x}}_i; \mathbf{m}_i]$，第 $j$ 个特征节点的初始表示为：

$$\mathbf{h}_j^{(0)} = \text{MLP}_{\text{node}}\left([\tilde{x}_{ij}; m_{ij}; \mathbf{e}_{g_j}]\right)$$

其中：
- $\tilde{x}_{ij}$ 为零填充特征值；
- $m_{ij}$ 为缺失指示；
- $\mathbf{e}_{g_j} \in \mathbb{R}^{E}$ 为特征 $j$ 所属语义组 $g_j$ 的可学习组嵌入；
- $[\cdot; \cdot]$ 表示拼接。

### 3.3.2 图注意力消息传递

节点间构建全连接图，边权重由多头图注意力机制学习。对每一层 $l$：

$$\alpha_{jk}^{(l)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_j^{(l)}; \mathbf{W}\mathbf{h}_k^{(l)}] + b_{g_j g_k}\right)\right)}{\sum_{k' \in \mathcal{N}_j} \exp\left(\cdots\right)}$$

其中 $b_{g_j g_k}$ 为组间注意力偏置，使同组节点更容易互相注意。缺失节点不作为信息源（key masking），避免向其他节点传播噪声。

节点更新：

$$\mathbf{h}_j^{(l+1)} = \text{ReLU}\left(\sum_{k \in \mathcal{N}_j} \alpha_{jk}^{(l)} \mathbf{W}\mathbf{h}_k^{(l)}\right)$$

### 3.3.3 组级注意力池化

为显式建模组级上下文，对每个语义组 $g$ 计算组内注意力池化：

$$\mathbf{c}_g = \sum_{j: g_j=g} \beta_{gj} \mathbf{h}_j^{(L)}$$

其中注意力权重：

$$\beta_{gj} = \frac{\exp\left(\mathbf{q}_g^T \mathbf{h}_j^{(L)}\right)}{\sum_{j': g_{j'}=g} \exp\left(\mathbf{q}_g^T \mathbf{h}_{j'}^{(L)}\right)}$$

$\mathbf{q}_g$ 为可学习的组查询向量。

将各组上下文拼接并投影：

$$\mathbf{c} = \text{MLP}_{\text{group}}\left([\mathbf{c}_1; \cdots; \mathbf{c}_G]\right)$$

### 3.3.4 节点增强与插补

把组级上下文广播回每个节点：

$$\hat{\mathbf{h}}_j = \mathbf{h}_j^{(L)} + \mathbf{c}$$

插补值为：

$$\hat{x}_{ij} = \text{MLP}_{\text{imp}}(\hat{\mathbf{h}}_j)$$

最终完整特征：

$$\bar{x}_{ij} = \tilde{x}_{ij} \cdot (1 - m_{ij}) + \hat{x}_{ij} \cdot m_{ij}$$

### 3.3.5 SOH 估计

将插补后的完整特征与缺失指示器拼接：

$$\hat{y}_i = f_{\text{soh}}\left([\bar{\mathbf{x}}_i; \mathbf{m}_i]\right)$$

其中 $f_{\text{soh}}$ 为标准 MLP 或 LSTM backbone。

## 3.4 训练策略

### 3.4.1 Uniform Multi-Rate 训练

为模拟 BMS 中不同缺失率混合出现的场景，训练时每个 epoch 从缺失率集合 $\{0.0, 0.1, \dots, 0.9\}$ 中均匀采样并混合。所有对比方法均采用此策略，以保证公平。

### 3.4.2 损失函数

主损失为 SOH 估计的均方误差：

$$\mathcal{L}_{\text{SOH}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

可选的重建辅助损失（用于消融）：

$$\mathcal{L}_{\text{rec}} = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^F m_{ij} (x_{ij} - \hat{x}_{ij})^2$$

总损失：

$$\mathcal{L} = \mathcal{L}_{\text{SOH}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}$$

本文默认 $\lambda_{\text{rec}}=0$（不直接监督重建，让插补由 SOH 目标隐式驱动）。

## 3.5 对比方法

### 3.5.1 端到端方法
- **Baseline-Uniform**：仅在完整数据（MR=0）上训练，测试时直接面对缺失数据。
- **MultiMR-Uniform**：uniform multi-rate 训练，不拼接缺失指示器。
- **MIM-Uniform**：uniform multi-rate 训练，拼接缺失指示器。
- **GroupMIM-Uniform**：MIM + 特征组嵌入。
- **GNN-Uniform**：MIM + 特征图注意力插补（无组级池化）。
- **FMG-Uniform**：Feature-wise Missing Gate。
- **GraphMIM-Uniform**：本文方法。

### 3.5.2 两阶段方法
- **Mean/KNN/Iterative Imputation + MLP**：先在训练集上 fit 插补器，再训练 MLP 估计 SOH。

### 3.5.3 线性方法
- **Linear / Linear-MIM**：使用均值插补或 MIM 输入的线性回归。

## 3.6 实现细节

- GNN 隐藏维度：32
- GNN 层数：2
- 组嵌入维度：4
- 注意力头数：1
- 优化器：Adam，学习率 0.001
- 早停耐心值：15
- 最大 epoch：100
- 批量大小：32
