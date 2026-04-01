# 项目绘图资源完整清单

> 生成时间: 2026-04-01  
> 项目: missing-data-battery-nn  
> 版本: v0.5

---

## 一、论文主图 (paper/figures/)

### 1.1 实验流程图
| 文件 | 格式 | 说明 |
|------|------|------|
| `fig1_problem_scenario.png` | PNG | 问题场景示意 (BMS监控链 + 缺失模式) |
| `fig1_problem_scenario.html` | HTML | 上述图片的HTML源码 |
| `fig2_baseline_vs_mim.png` | PNG | Baseline vs MIM对比 |
| `fig2_baseline_vs_mim.html` | HTML | 上述图片的HTML源码 |
| `fig4_joint_training.png` | PNG | 联合训练策略说明 |
| `fig4_joint_training.html` | HTML | 上述图片的HTML源码 |

### 1.2 旧版分析图表 (experiments/scripts/生成)
| 文件 | 格式 | 说明 |
|------|------|------|
| `fig11_distribution_v2.png` | PNG | 模型分布图 (v2) |
| `fig12_model_selection_v2.png` | PNG | 模型选择图 (v2) |
| `fig13_performance_params_v2.png` | PNG | 性能参数图 (v2) |
| `fig3_model_architectures.png` | PNG | 模型架构对比 (MLP/LSTM/CNN) |
| `fig3_model_architectures.html` | HTML | 上述图片的HTML源码 |
| `fig5_mae_heatmap.png` | PNG | MAE热力图 |
| `fig6_model_selection.png` | PNG | 模型选择结果 |
| `fig6_model_selection.html` | HTML | 上述图片的HTML源码 |
| `fig7_performance_params.png` | PNG | 性能参数分析 |
| `fig7_performance_params.html` | HTML | 上述图片的HTML源码 |
| `fig9_seed_stability_v2.png` | PNG | 种子稳定性分析 (v2) |

### 1.3 v0.5 版本补充图表
| 文件 | 格式 | 说明 |
|------|------|------|
| `figure4_architecture_scaling.png/pdf` | PNG/PDF | 架构缩放分析 |
| `greedy_ablation_4panel.png/pdf` | PNG/PDF | 贪婪消融四宫格 |
| `improvement_vs_mr.png/pdf` | PNG/PDF | 改进率vs缺失率 |
| `mae_curves_ci.png` | PNG | MAE曲线(置信区间) |
| `mae_distribution.png` | PNG | MAE分布 |
| `mae_heatmap.png` | PNG | MAE热力图 |
| `mae_improvement_rate.png` | PNG | MAE改进率 |
| `r2_curves_ci.png` | PNG | R²曲线(置信区间) |
| `rmse_curves_ci.png` | PNG | RMSE曲线(置信区间) |
| `rmse_distribution.png` | PNG | RMSE分布 |
| `rmse_improvement_rate.png` | PNG | RMSE改进率 |
| `seed_stability.png` | PNG | 种子稳定性 |

### 1.4 单模型详细图表 (paper/figures/individual_models/)
| 文件 | 说明 |
|------|------|
| `all_models_individual_improvement.png` | 所有模型改进率 |
| `all_models_individual_mae.png` | 所有模型MAE |
| `all_models_individual_r2.png` | 所有模型R² |
| `high_missing_rate_detail.png` | 高缺失率细节 |
| `README.md` | 说明文档 |

---

## 二、v0.5 论文图表 (experiments_new/figures/paper_figures/v0.5/)

### 2.1 Figure 1: MIM Core Effect
| 文件 | 说明 |
|------|------|
| `figure1_mim_core_effect.png/pdf` | MIM核心效应 (v1) |
| `figure1_mim_core_effect_v2.png/pdf` | MIM核心效应 (v2, 当前使用) |

### 2.2 Figure 2: Missing Mechanisms
| 文件 | 说明 |
|------|------|
| `figure2_missing_mechanisms.png/pdf` | 缺失机制分析 (v1) |
| `figure2_missing_mechanisms_v2.png/pdf` | 缺失机制分析 (v2, 当前使用) |

### 2.3 Figure 3: Imputation Gradient
| 文件 | 说明 |
|------|------|
| `figure3_imputation_gradient.png/pdf` | 插补梯度 (v1) |
| `figure3_imputation_gradient_v2.png/pdf` | 插补梯度 (v2, 当前使用) |

### 2.4 Figure 4: Architecture Scaling
| 文件 | 说明 |
|------|------|
| `figure4_architecture_scaling_v2.png/pdf` | 架构缩放 (v2, 当前使用) |
| `figure4_model_levels.png/pdf` | 模型级别对比 |

### 2.5 Figure 5: Statistical Significance
| 文件 | 说明 |
|------|------|
| `figure5_statistical_significance.png/pdf` | 统计显著性 |

### 2.6 Figure 6: Experimental Flow
| 文件 | 说明 |
|------|------|
| `figure6_experimental_flow.png/pdf` | 实验流程 (Matplotlib生成) |
| `figure6_experimental_flow_v2.png` | 实验流程 (HTML手动截图) |

### 2.7 数据文件
| 文件 | 说明 |
|------|------|
| `plotting_data_v0.5.csv` | 绘图数据 (16,920行) |

### 2.8 文档
| 文件 | 说明 |
|------|------|
| `FIGURE_DESCRIPTION.md` | 图表说明文档 |
| `DELIVERY_SUMMARY.md` | 交付摘要 |
| `DELIVERY_TO_CLAW.md` | 给Claw的交付说明 |

---

## 三、绘图脚本清单

### 3.1 v0.5 Python 绘图脚本 (experiments_new/.../scripts/)
| 脚本 | 对应图表 | 功能说明 |
|------|----------|----------|
| `fig1_mim_core_effect.py` | Figure 1 | MIM核心效应 (2×2复合图) |
| `fig2_missing_mechanisms.py` | Figure 2 | 缺失机制分析 (2×3网格) |
| `fig3_imputation_gradient.py` | Figure 3 | 插补质量梯度 (1×3横向) |
| `fig4_architecture_scaling.py` | Figure 4 | 架构与参数规模 |
| `fig5_statistical_significance.py` | Figure 5 | 统计显著性检验 |
| `fig6_experimental_flow.py` | Figure 6 | Matplotlib流程图 |
| `figure6_experimental_flow.html` | Figure 6 v2 | **HTML/CSS版本 (手动截图)** |
| `html_to_png.py` | - | HTML转PNG截图工具 |

### 3.2 旧版实验脚本 (experiments/scripts/)
| 脚本 | 说明 |
|------|------|
| `generate_fig11_distribution.py` | 生成fig11分布图 |
| `generate_fig12_model_selection.py` | 生成fig12模型选择图 |
| `generate_fig13_performance_params.py` | 生成fig13性能参数图 |
| `generate_fig9_seed_stability.py` | 生成fig9种子稳定性图 |
| `generate_individual_model_plots.py` | 生成单模型图 |
| `generate_model_comparison_plots.py` | 生成模型对比图 |
| `plot_batch.py` | 批次结果绘图 |
| `plot_batch_results.py` | 批次结果绘图(v2) |
| `plot_batch_results_no_xgb.py` | 批次结果绘图(无XGB) |
| `plot_single.py` | 单结果绘图 |

### 3.3 核心可视化模块 (src/visualization/)
| 模块 | 说明 |
|------|------|
| `batch_plots.py` | 批次绘图功能 |
| `single_plots.py` | 单实验绘图功能 |
| `heatmaps.py` | 热力图生成 |
| `missing_rate_curves.py` | 缺失率曲线 |
| `architecture_analysis.py` | 架构分析可视化 |

### 3.4 HTML图表源码 (paper/figures/)
| 文件 | 说明 |
|------|------|
| `fig1_problem_scenario.html` | 问题场景 (HTML版) |
| `fig2_baseline_vs_mim.html` | Baseline对比 (HTML版) |
| `fig3_model_architectures.html` | 模型架构 (HTML版) |
| `fig4_joint_training.html` | 联合训练 (HTML版) |
| `fig6_model_selection.html` | 模型选择 (HTML版) |
| `fig7_performance_params.html` | 性能参数 (HTML版) |
| `html_to_png.py` | HTML转PNG工具 |

### 3.5 其他分析脚本
| 脚本 | 位置 | 说明 |
|------|------|------|
| `plot_youth_baseline.py` | src/analysis/ | 基线绘图 |

---

## 四、图片使用状态

### 4.1 当前论文使用 (main_en.tex)
| Figure | 使用文件 | 状态 |
|--------|----------|------|
| Fig 1 | `fig1_problem_scenario.png` | ✅ 保留 |
| Fig 2 | `fig2_baseline_vs_mim.png` | ✅ 保留 |
| Fig 3 | `fig3_model_architectures.png` | ❌ **已删除** |
| Fig 4 | `fig4_joint_training.png` | ✅ 保留 |
| Fig 6 | `figure6_experimental_flow.png` | ✅ 保留 (v2 HTML版) |
| 其他 | v0.5版本图表 | ✅ 跨栏大图 |

### 4.2 图表层级关系
```
论文主图 (paper/figures/)
├── 旧版图表 (fig1-fig13) - 部分仍在使用
└── v0.5图表 (引用自experiments_new/)
    ├── Figure 1-5: Matplotlib生成
    └── Figure 6: HTML/CSS手动截图
```

---

## 五、快速生成命令

### 重新生成所有 v0.5 图表
```bash
cd experiments_new
source activate battery-nn-latest

# 执行所有绘图脚本
for script in figures/paper_figures/v0.5/scripts/fig*.py; do
    python "$script"
done

# 单独生成Figure 6 (HTML版 - 需手动截图)
open figures/paper_figures/v0.5/scripts/figure6_experimental_flow.html
```

---

## 六、注意事项

1. **Figure 6** 有两个版本：
   - `fig6_experimental_flow.py` - Matplotlib自动生成
   - `figure6_experimental_flow.html` - HTML/CSS手动截图 (**当前使用**)

2. **HTML图表**：
   - `paper/figures/*.html` - 旧版图表，部分仍在使用
   - `figure6_experimental_flow.html` - 新版，样式更好

3. **数据依赖**：
   - v0.5图表依赖 `plotting_data_v0.5.csv`
   - 旧版图表依赖 `results/` 下的各种CSV

---

*最后更新: 2026-04-01*
