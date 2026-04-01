# 项目绘图资源完整清单

> 生成时间: 2026-04-01  
> 项目: missing-data-battery-nn  
> 版本: v0.5  
> 命名规范: 描述性名称，无编号，小写下划线分隔

---

## 一、论文主图 (paper/figures/)

**仅保留 PNG 格式，按内容描述命名**

### 1.1 核心实验图
| 文件名 | 说明 |
|--------|------|
| `problem_scenario.png` | 问题场景示意 (BMS监控链 + 缺失模式) |
| `baseline_vs_mim_comparison.png` | Baseline vs MIM 方法对比 |
| `joint_training_strategy.png` | 联合训练策略说明 |

### 1.2 分析图表
| 文件名 | 说明 |
|--------|------|
| `architecture_scaling.png` | 架构缩放分析 |
| `greedy_ablation.png` | 贪婪消融实验结果 |
| `improvement_vs_missing_rate.png` | 改进率 vs 缺失率 |
| `model_distribution.png` | 模型分布 |
| `model_selection.png` | 模型选择结果 |
| `performance_parameters.png` | 性能参数分析 |

### 1.3 指标曲线与分布
| 文件名 | 说明 |
|--------|------|
| `mae_curves_with_ci.png` | MAE曲线(含置信区间) |
| `mae_distribution.png` | MAE分布 |
| `mae_heatmap.png` | MAE热力图 |
| `mae_improvement_rate.png` | MAE改进率 |
| `r2_curves_with_ci.png` | R²曲线(含置信区间) |
| `rmse_curves_with_ci.png` | RMSE曲线(含置信区间) |
| `rmse_distribution.png` | RMSE分布 |
| `rmse_improvement_rate.png` | RMSE改进率 |
| `seed_stability.png` | 种子稳定性分析 |

### 1.4 已删除
- ~~`model_architectures_comparison.png`~~ - Figure 3 已删除
- ~~`individual_models/`~~ 子目录已删除

---

## 二、v0.5 论文图表 (experiments_new/figures/paper_figures/v0.5/)

**原始文件 + 输出文件，按内容描述命名**

### 2.1 图表文件 (PNG + PDF)
| 文件名 | 原始脚本 | 说明 |
|--------|----------|------|
| `mim_core_effect.png/pdf` | `mim_core_effect.py` | MIM核心效应 |
| `missing_mechanisms.png/pdf` | `missing_mechanisms.py` | 缺失机制分析 |
| `imputation_gradient.png/pdf` | `imputation_gradient.py` | 插补质量梯度 |
| `architecture_scaling.png/pdf` | `architecture_scaling.py` | 架构与参数规模 |
| `statistical_significance.png/pdf` | `statistical_significance.py` | 统计显著性检验 |
| `experimental_flow.png` | `experimental_flow.html` | 实验流程图 |

### 2.2 绘图脚本 (scripts/)
| 原始文件 | 说明 | 输出 |
|----------|------|------|
| `mim_core_effect.py` | MIM核心效应绘图 | PNG/PDF |
| `missing_mechanisms.py` | 缺失机制分析绘图 | PNG/PDF |
| `imputation_gradient.py` | 插补梯度绘图 | PNG/PDF |
| `architecture_scaling.py` | 架构缩放绘图 | PNG/PDF |
| `statistical_significance.py` | 统计显著性绘图 | PNG/PDF |
| `experimental_flow.py` | Matplotlib流程图 | PNG/PDF |
| `experimental_flow.html` | **HTML/CSS流程图** | 手动截图 |
| `html_to_png.py` | HTML转PNG工具 | - |

### 2.3 数据与文档
| 文件 | 说明 |
|------|------|
| `plotting_data_v0.5.csv` | 绘图数据 (16,920行) |
| `FIGURE_DESCRIPTION.md` | 图表说明文档 |
| `DELIVERY_SUMMARY.md` | 交付摘要 |
| `DELIVERY_TO_CLAW.md` | 给Claw的交付说明 |

---

## 三、命名规范总结

### 规则
1. **无编号**: 不使用 fig1, figure1, fig1_xxx 等编号
2. **描述性**: 使用能准确描述图像内容的名称
3. **小写**: 全部小写字母
4. **分隔符**: 单词间用下划线 `_` 分隔
5. **格式**: 
   - `paper/figures/` 只保留 PNG
   - 原始文件可以是 Python 脚本或 HTML

### 命名对照表 (旧 → 新)
| 旧名称 | 新名称 |
|--------|--------|
| fig1_problem_scenario.png | problem_scenario.png |
| fig2_baseline_vs_mim.png | baseline_vs_mim_comparison.png |
| fig3_model_architectures.png | ~~(已删除)~~ |
| fig4_joint_training.png | joint_training_strategy.png |
| figure1_mim_core_effect_v2.png | mim_core_effect.png |
| figure2_missing_mechanisms_v2.png | missing_mechanisms.png |
| greedy_ablation_4panel.png | greedy_ablation.png |
| improvement_vs_mr.png | improvement_vs_missing_rate.png |

---

## 四、快速生成命令

### 重新生成所有 v0.5 图表
```bash
cd experiments_new/figures/paper_figures/v0.5/scripts

# Python 绘图
python mim_core_effect.py
python missing_mechanisms.py
python imputation_gradient.py
python architecture_scaling.py
python statistical_significance.py

# HTML 图表需手动截图
open experimental_flow.html
```

---

## 五、目录结构

```
paper/
└── figures/
    └── [仅PNG，描述性命名]
        ├── problem_scenario.png
        ├── baseline_vs_mim_comparison.png
        └── ...

experiments_new/
└── figures/
    └── paper_figures/
        └── v0.5/
            ├── [PNG/PDF 输出文件]
            │   ├── mim_core_effect.png
            │   └── ...
            └── scripts/
                ├── [Python/HTML 原始文件]
                │   ├── mim_core_effect.py
                │   ├── experimental_flow.html
                │   └── ...
```

---

*最后更新: 2026-04-01*
