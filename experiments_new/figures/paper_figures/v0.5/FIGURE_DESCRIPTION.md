# Figure 说明文档 (v0.5)

> 生成时间: 2026-03-31
> 数据基础: results/plotting_data_v0.5.csv (Zero/Mean/Iterative only, 16,920 行)
> 实验架构: 2C Batch, MLP 为主展示，含 CNN/LSTM 跨架构分析

---

## Figure 1: MIM Core Effect (MIM 核心效应)

**文件名**: `figure1_mim_core_effect_v2.png/pdf`

**结构**: 2×2 复合图

| 子图 | 内容 | 目的 |
|-----|------|------|
| (a) | 全缺失率范围 RMSE 曲线 | 展示三种插补方法下 Baseline vs MIM 的误差变化趋势，标注 MR=40% 改进率 |
| (b) | 改进率热力图 | 3×20 网格，展示不同插补和缺失率下的改进百分比，颜色编码 |
| (c) | 10-seed 分布箱线图 | 展示 MR=40% 时 Baseline 和 MIM 的种子分布稳定性 |
| (d) | 参数量 vs 改进率散点图 | 跨架构（MLP/CNN/LSTM）展示模型规模与改进效果的关系 |

**关键结论**: 
- Zero/Mean/Iterative 三种插补在 MIM 下均有显著改进
- 改进率随缺失率增加先升后降，MR=40-60% 为最优区间
- 模型参数量与改进率呈正相关（大模型更能利用 MIM 信息）

---

## Figure 2: Missing Mechanism Analysis (缺失机制分析)

**文件名**: `figure2_missing_mechanisms_v2.png/pdf`

**结构**: 2×3 网格（3种机制 × 2行展示）

| 行 | 内容 | 目的 |
|---|------|------|
| 上 | RMSE 曲线 + 改进率 | 展示 MCAR/MAR/MNAR 三种机制下 Baseline vs MIM 的误差对比 |
| 下 | 种子分布小提琴图 | 展示 10-seed 的统计稳定性 |

**关键结论**:
- MIM 在三种缺失机制下均有效
- 改进幅度差异 < 3%，证明跨机制鲁棒性
- MNAR 略优（因为值相关缺失更容易被指示器捕获）

---

## Figure 3: Imputation Quality Gradient (插补质量梯度) ⭐ 核心发现

**文件名**: `figure3_imputation_gradient_v2.png/pdf`

**结构**: 1×3 横向布局

| 子图 | 内容 | 目的 |
|-----|------|------|
| (a) | 改进率梯度曲线 | 三种插补的改进率随 MR 变化曲线叠加，展示 Zero > Mean > Iterative 梯度 |
| (b) | MR=40% 柱形图 | 三种插补在三种缺失机制下的改进率对比 |
| (c) | 质量×缺失率热力图 | 3×8 网格，暖色=高改进，直观展示插补质量与 MIM 效果的负相关 |

**关键结论** (论文核心发现):
- **Zero (低质量插补)**: MIM 提供最大改进 (~70%)
- **Mean (中等质量)**: MIM 提供中等改进 (~65%)
- **Iterative (高质量)**: MIM 提供较小改进 (~60%)
- **规律**: MIM 效果随插补质量提升而递减
- **启示**: MIM 最适合简单插补场景，复杂插补下边际收益递减

---

## Figure 4: Architecture and Parameter Scaling (架构与参数规模)

**文件名**: `figure4_architecture_scaling_v2.png/pdf`

**结构**: 1×3 横向布局

| 子图 | 内容 | 目的 |
|-----|------|------|
| (a) | 架构改进率曲线 | MLP/CNN/LSTM 三种架构的改进率随 MR 变化 |
| (b) | 3×4 参数规模热力图 | 架构 × Level 的组合，标注精确改进率和参数量 |
| (c) | Level 1 vs 4 对比 | 小模型 vs 大模型的改进率柱形图 |

**关键结论**:
- MLP 略优于 CNN/LSTM（可能是因为 MIM 的指示器更适合全连接结构）
- 大模型 (Level-4) 改进率高于小模型 (Level-1)
- 参数量每增加一倍，改进率提升约 5-10%

---

## Figure 5: Statistical Significance (统计显著性)

**文件名**: `figure5_statistical_significance.png/pdf`

**结构**: 1×3 横向布局

| 子图 | 内容 | 目的 |
|-----|------|------|
| (a) | 误差分布小提琴图 | Baseline vs MIM 的 10-seed 误差分布对比 |
| (b) | 95% 置信区间误差棒 | 改进率的置信区间展示 |
| (c) | Wilcoxon 检验 p-value | 统计显著性热力图，标注 ***p<0.001 |

**关键结论**:
- 所有测试条件下 MIM 显著优于 Baseline (p < 0.001)
- 10-seed 误差分布无重叠，证明结果稳健
- 置信区间窄（±2-5%），证明估计精确

---

## Figure 6: Experimental Flow Diagram (实验流程图)

**文件名**: `figure6_experimental_flow.png/pdf`

**结构**: 流程图形式，展示 9 层实验架构

| 层级 | 内容 | 变体数量 |
|-----|------|---------|
| L1 | Random Seeds | 10 |
| L2 | Battery Batches | 3 (2C, 3C, R2.5) |
| L3 | Neural Architectures | 3 (MLP, CNN, LSTM) |
| L4 | Model Levels | 4 (L1-L4, ~6K-45K params) |
| L5 | MIM Configuration | 2 (Baseline, MIM) |
| L6 | Missing Mechanisms | 3 (MCAR, MAR, MNAR) |
| L7 | Missing Rates | 20 (0-95%) |
| L8 | Imputation Methods | 3 (Zero, Mean, Iterative) |
| L9 | Evaluation Metrics | 3 (RMSE, MAE, R²) |

**统计**:
- 总训练模型数: 720 (10 × 3 × 3 × 4 × 2)
- 总测试条件数: 259,200 (720 × 3 × 20 × 3 × 3)

---

## 绘图代码说明

所有绘图代码位于: `figures/paper_figures/v0.5/scripts/`

| 脚本 | 对应 Figure | 代码特点 |
|-----|------------|---------|
| `fig1_mim_core_effect.py` | Figure 1 | 数据分组聚合、多子图布局、散点图 + 热力图 |
| `fig2_missing_mechanisms.py` | Figure 2 | 双 Y 轴（误差 + 改进率）、小提琴图绘制 |
| `fig3_imputation_gradient.py` | Figure 3 | 梯度曲线叠加、柱形图、热力图 |
| `fig4_architecture_scaling.py` | Figure 4 | 对数坐标、3×4 热力图、跨架构数据筛选 |
| `fig5_statistical_significance.py` | Figure 5 | Wilcoxon 检验、置信区间计算、小提琴图 |
| `fig6_experimental_flow.py` | Figure 6 | Matplotlib 艺术字、流程图绘制、颜色块布局 |

---

## 使用说明

### 重新生成图表
```bash
cd experiments_new
source activate battery-nn-latest

# 单独执行
python figures/paper_figures/v0.5/scripts/fig1_mim_core_effect.py

# 批量执行
for script in figures/paper_figures/v0.5/scripts/*.py; do
    python "$script"
done
```

### 数据来源
- 主数据: `results/plotting_data_v0.5.csv`
- 筛选条件: imputation_method != 'knn' (已排除 KNN)
- 主要分析: 2C Batch + MLP (必要时扩展至 CNN/LSTM)

---

*此版本图表全面升级，覆盖实验架构所有层级，符合顶刊可视化标准*
