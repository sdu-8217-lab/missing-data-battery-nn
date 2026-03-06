# 论文写作指南

## 文件结构

```
paper/
├── main_zh.tex           # 主文件（中文版本）
├── references.bib        # 参考文献数据库
├── README.md             # 本文件
├── FIGURE_GUIDE.md       # 插图使用指南
├── figures/              # 图片文件夹
│   ├── mae_curves_ci.png         # 主结果图（已插入）
│   ├── rmse_curves_ci.png        # RMSE曲线（已插入）
│   ├── r2_curves_ci.png          # R2曲线（已插入）
│   ├── mae_improvement_rate.png  # 改进率（已插入）
│   ├── mae_distribution.png      # 分布图（已插入）
│   ├── seed_stability.png        # 稳定性（已插入）
│   ├── single_mae_curves.png     # 单种子示例（已插入）
│   └── ...
└── tables/               # 数据表格（可选）
```

## 编译方法

### 使用 XeLaTeX + BibTeX

```bash
# 编译主文件
xelatex main_zh.tex
bibtex main_zh
xelatex main_zh.tex
xelatex main_zh.tex

# 或使用 latexmk 自动编译
latexmk -xelatex -synctex=1 -interaction=nonstopmode main_zh.tex
```

### 推荐编辑器

- **TeXstudio**: 免费，支持中文，有自动补全
- **VS Code + LaTeX Workshop**: 轻量级，配合SumatraPDF可实现正向/反向搜索
- **Overleaf**: 在线编辑，无需本地配置

## 图片引用

将实验生成的图片复制到 `figures/` 文件夹：

```bash
# 从实验结果复制图片
cp ../experiments/2C/20260203_045645/figures_manual/*.png figures/

# 或使用 PowerShell
Copy-Item ../experiments/2C/20260203_045645/figures_manual/*.png figures/
```

然后在正文中引用：

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{mae_curves_ci.png}
    \caption{不同缺失率下各模型的MAE表现}
    \label{fig:mae_curves}
\end{figure}
```

## 常见问题

### 1. 中文显示问题

确保使用 `ctexart` 文档类，并使用 XeLaTeX 编译：

```latex
\documentclass[12pt,a4paper]{ctexart}
```

### 2. 参考文献不显示

需要运行 BibTeX：

```bash
xelatex main_zh.tex
bibtex main_zh
xelatex main_zh.tex
xelatex main_zh.tex
```

### 3. 图片路径问题

确保 `graphicspath` 设置正确：

```latex
\graphicspath{{figures/}}
```

图片文件名不要包含中文或特殊字符。

## 论文结构说明

根据新版论文大纲，本文结构如下：

1. **引言**: 研究背景、现状不足、研究目标与贡献
2. **相关工作**: 缺失数据处理、SOH预测模型、研究定位
3. **方法**: 数据集、特征提取、缺失模拟、MIM构造、模型架构、训练设置
4. **实验结果**: 整体表现、改进率分析、种子稳定性
5. **讨论**: 主要发现、工程建议、局限性
6. **结论**: 总结与未来工作

## 待完成事项

- [x] 扩展文本内容，采用学术化自然段
- [x] 生成并插入多种类型图像（MAE/RMSE/R2曲线、分布图、稳定性图、单种子示例等）
- [ ] 补充真实实验数据到表格
- [ ] 添加更多参考文献（当前约12篇，建议15-20篇）
- [ ] 完善作者信息和单位
- [ ] 校对公式和符号一致性
- [ ] 检查图表引用顺序
- [ ] 润色语言表达，确保学术规范

## 已插入图像清单

共7张图像已插入论文：

| 图像名 | 说明 | 章节位置 |
|--------|------|----------|
| mae_curves_ci.png | MAE曲线+95%置信区间 | 图1（4.1节） |
| rmse_curves_ci.png | RMSE曲线+置信区间 | 图4b（4.2节后） |
| r2_curves_ci.png | R²曲线+置信区间 | 图4c（4.2节后） |
| mae_improvement_rate.png | MIM改进率分析 | 图2（4.2节） |
| seed_stability.png | 100种子稳定性 | 图3（4.3节） |
| mae_distribution.png | MAE分布小提琴图 | 图5（4.3节后） |
| single_mae_curves.png | 单种子示例 | 图6（3.6节） |

所有图像均为英文标注，符合期刊要求。

## 字数统计

```bash
# 使用 texcount
texcount -inc -total main_zh.tex

# 或使用 VS Code 插件
# LaTeX Word Counter
```

当前版本：
- 文件大小：44,556 bytes
- 行数：~540行（含公式、表格）
- 预计中文字数：约15,000-18,000字
- 符合期刊论文要求（通常8000-15000字）

## 写作特点

本文采用学术化的自然段写作风格，避免了简单的要点罗列，具体体现在：

1. **引言部分**：从研究背景逐步深入，通过逻辑递进阐述问题，而非直接列要点
2. **相关工作**：系统综述各类方法的发展脉络，强调方法间的联系与区别
3. **方法描述**：详细阐述每个技术环节的设计动机和理论依据
4. **结果分析**：不仅呈现数据，更深入分析现象背后的机制
5. **讨论部分**：将发现归纳为理论洞察，并给出具体工程建议

## 待完善事项

- [x] 扩展文本内容，采用学术化自然段
- [ ] 补充真实实验数据到表格
- [ ] 添加更多参考文献（当前约12篇，建议15-20篇）
- [ ] 完善作者信息和单位
- [ ] 校对公式和符号一致性
- [ ] 检查图表引用顺序
- [ ] 润色语言表达，确保学术规范
