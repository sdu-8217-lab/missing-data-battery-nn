# BatteryCDE 复现

> Wang, T., Ren, C., Xiong, H., Song, Q., & Dong, Z. (2025). BatteryCDE: A Transferable Future Capacity Estimation Method for Battery Degradation With Irregular Sampling and Missing Data. *IEEE Transactions on Transportation Electrification*, 11(4), 10427–10440. DOI: [10.1109/TTE.2025.3547740](https://doi.org/10.1109/TTE.2025.3547740)

## 状态

🚧 实验进行中：核心模型、数据预处理、基线对比已完成；horizon / 缺失数据 / 迁移实验正在后台顺序运行。

> **批判性审视**：本复现不仅是代码实现，更包含对论文方法、理论声明与实验设计的深度批判。详见 `CRITICAL_REVIEW.md` 与 `notes.md`。

## 核心贡献（论文原文）

1. 用神经控制微分方程（Neural CDE）连续更新隐状态，支持长期未来容量预测（20–1000 个循环）。
2. 通过自然三次样条（natural cubic spline）处理非规则采样与缺失数据。
3. 引入 feature-wise 与 cycle-wise 双重注意力，对关键特征和关键时间区间加权。
4. 展示跨电池、跨温度、跨数据集的迁移能力。

## 当前已获得的参考结果

在 NASA 数据上（train: B5/B6, val: B7, test: B18, horizon=50, history_len=3）：

| 模型 | Test MAE (SOH) | Test RMSE | R² |
|------|---------------|-----------|-----|
| BatteryCDE | 0.037 | 0.049 | -0.41 |
| Transformer | 0.033 | 0.045 | -0.18 |
| LSTM | 0.038 | 0.054 | -0.70 |
| Neural CDE | 0.106 | 0.111 | -6.20 |

> 注：R² 为负说明模型尚未优于简单均值基线，可能与测试集容量变化范围小、单种子有关；需多种子进一步验证。

## 复现范围

### 已完成实验

- [x] BatteryCDE 主实验（horizon=50）
- [x] LSTM / Transformer / Neural CDE 基线对比
- [x] Horizon h=20（h=50/80/100 进行中）

### 计划复现的实验

- [ ] **长期预测**：NASA 上 20–100 步未来容量预测（论文扩展到 1000，受数据循环数限制）。
- [ ] **缺失数据实验**：
  - Scenario 1：原始充电轨迹随机缺失（30%、50%）。
  - Scenario 2：原始充电轨迹连续缺失（30%、50%）。
  - Scenario 3–6：提取特征上的随机/连续/时间/特征维度缺失（受数据限制，部分待实现）。
- [ ] **迁移实验**：
  - 同数据集不同电池（已实现）。
  - 跨数据集（NASA→CALCE、NASA→ISU 等，受数据可及性限制）。

### 与原文的已知差异

| 方面 | 原文 | 本复现 | 原因 |
|------|------|--------|------|
| 输入 | 完整充电 V/I/T 轨迹 | 从 `.mat` 提取充电片段并重采样到 30 点 | 论文未公开长度；计算权衡 |
| 历史信息 | 未明确 | `history_len=3` 个循环拼接成长序列 | 合理推断 |
| 数据集 | NASA/CALCE/SNL/ISU | 目前仅 NASA | 数据可及性 |
| 评估指标 | MAE | MAE / RMSE / R² | 与主项目指标对齐 |
| 输出层 | 单层 FC2 | 单层 FC2 | 已按论文修正 |
| Feature attention | $a_{\text{feat}}=\sigma(h_{\text{CDE1}})$ | 直接对 feature CDE 隐状态 sigmoid | 已按论文修正 |
| 缺失处理 | 自然三次样条 | 先线性插值 NaN，再构造 torchcde 样条 | torchcde 不直接支持带 mask 样条 |
| 稳定性 | 未明确 | Tanh + LayerNorm + 梯度裁剪 | CDE 数值稳定所需 |

## 环境要求

除项目根 `requirements.txt` 外，本复现依赖 `torchcde`：

```bash
pip install torchcde
```

## 快速开始

```bash
# 1. 激活环境
source ~/research/battery-research/.venv/bin/activate

# 2. 安装额外依赖
pip install torchcde

# 3. 提取 NASA 充电轨迹
python scripts/prepare_nasa_trajectories.py \
  --data_dir "../../data/NASA data" \
  --output_dir data/nasa_trajectories \
  --batteries 5,6,7,18,29,30 \
  --target_len 30

# 4. 训练 BatteryCDE
python scripts/train.py --config configs/nasa.yaml --model_type batteryCDE

# 5. 训练所有基线并运行全部对比实验
python scripts/run_all_experiments.py --config configs/nasa.yaml --seed 42

# 6. 汇总结果与绘图
python scripts/aggregate_results.py --results_dir ../results --output ../results/summary.csv
python scripts/plot_results.py --results_dir ../results --output_dir ../results/figures

# 7. 运行测试
python tests/test_battery_cde_forward.py
python tests/test_spline_interpolation.py
```

## 文件说明

```
batteryCDE_wang2025/
├── src/
│   ├── models/battery_cde.py          # BatteryCDE 模型（已贴近论文）
│   ├── models/baselines.py            # LSTM / Transformer / Neural CDE 基线
│   ├── data/battery_cde_dataset.py    # 轨迹数据集与缺失模拟
│   └── trainers/battery_cde_trainer.py # 训练/评估循环
├── configs/
│   ├── default.yaml                   # 默认超参
│   ├── nasa.yaml                      # NASA 长期预测
│   ├── calce.yaml                     # CALCE 长期预测
│   └── transfer.yaml                  # 迁移学习设置
├── scripts/
│   ├── prepare_nasa_trajectories.py   # NASA 充电轨迹提取
│   ├── train.py                       # 单模型训练（支持 4 种模型）
│   ├── evaluate.py                    # 单模型评估
│   ├── run_baselines.py               # 基线对比
│   ├── run_horizon_experiments.py     # 不同 horizon 实验
│   ├── run_missing_experiments.py     # 缺失数据实验
│   ├── run_transfer_experiments.py    # 迁移实验
│   ├── run_all_experiments.py         # 一键运行全部实验
│   ├── aggregate_results.py           # 结果汇总到 CSV
│   └── plot_results.py                # 结果绘图
├── tests/
│   ├── test_battery_cde_forward.py    # 前向传播与 NaN 输入测试
│   └── test_spline_interpolation.py   # 样条插值测试
├── CRITICAL_REVIEW.md                 # 论文批判性审视
├── notes.md                           # 实现笔记与问题记录
└── AGENTS.md                          # 本地约定
```

## 批判性审视要点

通过复现，我们发现论文存在以下值得注意的问题：

1. **理论分析松散**：误差累积与跨域收敛声明缺乏严格证明。
2. **“天然处理缺失数据”过度简化**：CDE 仍需先构造样条，本质上仍是插值。
3. **输出定义模糊**：未明确是单步预测还是序列预测。
4. **基线对比不透明**：未公开超参数与代码。
5. **迁移实验设定不清**：是否 fine-tuning、是否归一化等未说明。
6. **缺少不确定性估计与物理约束**：对 BMS 应用不够完整。

完整批判见 `CRITICAL_REVIEW.md`。

## 参考

- 原始论文 PDF：`literature/01_battery_soh_degradation/2024_BatteryCDE_Transferable_Capacity_Estimation_Irregular_Missing_Liu.pdf`
- 项目内阅读笔记：`literature/01_battery_soh_degradation/2024_BatteryCDE_Transferable_Capacity_Estimation_Irregular_Missing_Liu_notes.md`
- 项目内基础 Neural CDE 实现：`src/models/neural_cde_model.py`
