#!/usr/bin/env python3
"""
L1-L4 贪心分层选择消融图
4 个子图展示各维度的 MIM 改进率
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置中文字体和 Times New Roman
matplotlib.rcParams['font.family'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 8

# 数据
# L1: 架构选择
l1_archs = ['MLP', 'CNN', 'LSTM']
l1_improvements = [27.32, 3.26, 1.33]
l1_params = [11801, 12249, 10957]
l1_selected = 0  # MLP

# L2: 层级选择
l2_levels = ['L1', 'L2', 'L3', 'L4']
l2_improvements = [-2.19, 27.32, 19.06, 4.79]
l2_params = [5953, 11801, 20537, 44529]
l2_selected = 1  # Level 2

# L3: 插补选择
l3_imputations = ['Zero', 'Mean', 'KNN', 'Iterative']
l3_improvements = [15.05, 27.32, 35.83, 3.56]
l3_selected = 1  # Mean (论文选用)
l3_archive = 2   # KNN (存档)

# L4: 缺失模式
l4_patterns = ['MCAR', 'MAR', 'MNAR']
l4_improvements = [27.32, 24.61, 20.38]
l4_selected = 0  # MCAR

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), dpi=150)
fig.patch.set_facecolor('white')

# 颜色方案 - 蓝灰色系
color_selected = '#1f4e79'  # 深蓝色 - 选中
color_normal = '#5b9bd5'    # 中蓝色 - 普通
color_archive = '#a6a6a6'   # 灰色 - 存档

# ========== 子图 A: L1 架构选择 ==========
ax = axes[0, 0]
bars = ax.bar(l1_archs, l1_improvements, color=[color_selected if i == l1_selected else color_normal for i in range(len(l1_archs))], 
              edgecolor='black', linewidth=0.5, width=0.6)

# 添加数值标签
for i, (bar, imp, param) in enumerate(zip(bars, l1_improvements, l1_params)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{imp:+.1f}%\n({param:,})',
            ha='center', va='bottom', fontsize=7)

ax.set_ylabel('MIM Improvement (%)', fontsize=9)
ax.set_title('(A) L1: Architecture Selection', fontweight='bold', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(-5, 35)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 标注选中
ax.annotate('Selected', xy=(l1_selected, l1_improvements[l1_selected]), 
            xytext=(l1_selected + 0.3, 33),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=8, color='white', fontweight='bold', ha='center')

# ========== 子图 B: L2 层级选择 ==========
ax = axes[0, 1]
# 使用折线图展示倒U型
x_pos = np.arange(len(l2_levels))
bars = ax.bar(x_pos, l2_improvements, color=[color_selected if i == l2_selected else color_normal for i in range(len(l2_levels))],
              edgecolor='black', linewidth=0.5, width=0.6)

# 添加数值标签
for i, (bar, imp, param) in enumerate(zip(bars, l2_improvements, l2_params)):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 0.5 if height >= 0 else -0.5
    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
            f'{imp:+.1f}%\n({param:,})',
            ha='center', va=va, fontsize=7)

ax.set_xticks(x_pos)
ax.set_xticklabels(l2_levels)
ax.set_xlabel('Model Level', fontsize=9)
ax.set_ylabel('MIM Improvement (%)', fontsize=9)
ax.set_title('(B) L2: Model Scale Selection', fontweight='bold', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(-10, 35)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加倒U型曲线示意
ax.annotate('', xy=(0.5, 30), xytext=(3, 30),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1, ls='--'))
ax.text(1.75, 31, 'Inverted-U Shape', fontsize=7, color='gray', ha='center')

# 标注选中
ax.annotate('Selected\n(Optimal)', xy=(l2_selected, l2_improvements[l2_selected]), 
            xytext=(l2_selected + 0.4, 34),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=8, color='white', fontweight='bold', ha='center')

# ========== 子图 C: L3 插补选择 ==========
ax = axes[1, 0]
colors_l3 = [color_normal, color_selected, color_archive, color_normal]
bars = ax.bar(l3_imputations, l3_improvements, color=colors_l3,
              edgecolor='black', linewidth=0.5, width=0.6)

# 添加数值标签
for bar, imp in zip(bars, l3_improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{imp:+.1f}%',
            ha='center', va='bottom', fontsize=8)

ax.set_ylabel('MIM Improvement (%)', fontsize=9)
ax.set_title('(C) L3: Imputation Strategy', fontweight='bold', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(0, 40)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 标注选中和存档
ax.annotate('Selected\n(Paper)', xy=(l3_selected, l3_improvements[l3_selected]), 
            xytext=(l3_selected + 0.3, 38),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=8, color='white', fontweight='bold', ha='center')

ax.annotate('Archived', xy=(l3_archive, l3_improvements[l3_archive]), 
            xytext=(l3_archive + 0.6, 33),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=8, color='white', ha='center', style='italic')

# ========== 子图 D: L4 缺失模式 ==========
ax = axes[1, 1]
bars = ax.bar(l4_patterns, l4_improvements, color=[color_selected if i == l4_selected else color_normal for i in range(len(l4_patterns))],
              edgecolor='black', linewidth=0.5, width=0.6)

# 添加数值标签
for bar, imp in zip(bars, l4_improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{imp:+.1f}%',
            ha='center', va='bottom', fontsize=8)

ax.set_ylabel('MIM Improvement (%)', fontsize=9)
ax.set_xlabel('Missingness Mechanism', fontsize=9)
ax.set_title('(D) L4: Missingness Mechanism', fontweight='bold', fontsize=10)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(0, 35)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 标注选中
ax.annotate('Selected', xy=(l4_selected, l4_improvements[l4_selected]), 
            xytext=(l4_selected + 0.2, 33),
            arrowprops=dict(arrowstyle='->', color='white', lw=2),
            fontsize=8, color='white', fontweight='bold', ha='center')

# 添加说明文字
ax.text(1, 5, 'MIM effective across\nall mechanisms', 
        fontsize=7, color='gray', ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(pad=1.5)

# 保存
output_dir = Path(__file__).parent.parent / "figures" / "paper_figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "greedy_ablation_4panel.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_dir / "greedy_ablation_4panel.pdf", bbox_inches='tight', facecolor='white')
print(f"✅ 4-panel ablation plot saved to:")
print(f"   {output_dir / 'greedy_ablation_4panel.png'}")
print(f"   {output_dir / 'greedy_ablation_4panel.pdf'}")

plt.close()

# 验证输出
from PIL import Image
img = Image.open(output_dir / "greedy_ablation_4panel.png")
print(f"\n📐 Image size: {img.size} pixels")
print(f"📊 Aspect ratio: {img.size[0]/img.size[1]:.2f} (target ~1.3 for 8.5x6.5cm)")
