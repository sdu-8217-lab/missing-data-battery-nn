"""
MIM效果深度分析可视化
揭示MIM在什么条件下有效/无效
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = Path('results/100seeds/figures/mim_analysis')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 加载数据
df = pd.read_csv('results/100seeds/test_results.csv')

# 移除异常值
df = df[df['test_mae'] < 100]

def savefig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / f'{name}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {name}")
    plt.close()

# ==================== 图1: 按模型+插补方法分面 ====================
print("生成图1: 按模型和插补方法分面...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

models = ['mlp', 'lstm', 'cnn']
imputations = ['mean', 'knn', 'iterative', 'zero']

for i, model in enumerate(models):
    for j, imp in enumerate(imputations):
        ax = axes[i, j]
        
        subset = df[(df['model'] == model) & (df['imputation'] == imp)]
        
        # 按MR分组计算
        for use_mim in [False, True]:
            mim_data = subset[subset['use_mim'] == use_mim]
            grouped = mim_data.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
            
            label = 'With MIM' if use_mim else 'Without MIM'
            color = '#E64B35' if use_mim else '#4DBBD5'
            linestyle = '-' if use_mim else '--'
            
            ax.plot(grouped.index, grouped['mean'], 
                   marker='o', markersize=4, linewidth=2,
                   label=label, color=color, linestyle=linestyle)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.2, color=color)
        
        # 计算整体MIM效果
        w = subset[subset['use_mim'] == False]['test_mae'].mean()
        m = subset[subset['use_mim'] == True]['test_mae'].mean()
        diff = w - m
        improvement = diff / w * 100
        status = '✓' if diff > 0 else '✗'
        
        ax.set_title(f'{model.upper()} + {imp.capitalize()}\n{status} MIM改进 {improvement:+.1f}%', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Missing Rate', fontsize=9)
        if j == 0:
            ax.set_ylabel('MAE', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

plt.suptitle('Figure 1: MIM Effectiveness by Model and Imputation Method', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
savefig('mim_01_model_imputation_grid')

# ==================== 图2: 按插补方法+缺失机制分面 ====================
print("生成图2: 按插补方法和缺失机制分面...")
fig, axes = plt.subplots(4, 3, figsize=(15, 16))

imputations = ['mean', 'knn', 'iterative', 'zero']
modes = ['MCAR', 'MAR', 'MNAR']

for i, imp in enumerate(imputations):
    for j, mode in enumerate(modes):
        ax = axes[i, j]
        
        subset = df[(df['imputation'] == imp) & (df['mode'] == mode)]
        
        for use_mim in [False, True]:
            mim_data = subset[subset['use_mim'] == use_mim]
            grouped = mim_data.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
            
            label = 'With MIM' if use_mim else 'Without MIM'
            color = '#E64B35' if use_mim else '#4DBBD5'
            linestyle = '-' if use_mim else '--'
            
            ax.plot(grouped.index, grouped['mean'], 
                   marker='o', markersize=4, linewidth=2,
                   label=label, color=color, linestyle=linestyle)
            ax.fill_between(grouped.index,
                           grouped['mean'] - grouped['sem'],
                           grouped['mean'] + grouped['sem'],
                           alpha=0.2, color=color)
        
        w = subset[subset['use_mim'] == False]['test_mae'].mean()
        m = subset[subset['use_mim'] == True]['test_mae'].mean()
        diff = w - m
        improvement = diff / w * 100
        status = '✓' if diff > 0 else '✗'
        
        ax.set_title(f'{imp.capitalize()} + {mode}\n{status} MIM改进 {improvement:+.1f}%', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Missing Rate', fontsize=9)
        if j == 0:
            ax.set_ylabel('MAE', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

plt.suptitle('Figure 2: MIM Effectiveness by Imputation Method and Missingness Mechanism', 
            fontsize=14, fontweight='bold', y=0.997)
plt.tight_layout()
savefig('mim_02_imputation_mode_grid')

# ==================== 图3: 关键发现 - 高MR情况 ====================
print("生成图3: 高缺失率下的MIM效果...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图: 按MR分组的MIM改进率
ax1 = axes[0]
mrs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
improvements = []

for mr in mrs:
    subset = df[df['test_mr'] == mr]
    w = subset[subset['use_mim'] == False]['test_mae'].mean()
    m = subset[subset['use_mim'] == True]['test_mae'].mean()
    imp = (w - m) / w * 100
    improvements.append(imp)

colors = ['#00A087' if imp > 0 else '#E64B35' for imp in improvements]
bars = ax1.bar(range(len(mrs)), improvements, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_xticks(range(len(mrs)))
ax1.set_xticklabels([f'{int(mr*100)}%' for mr in mrs])
ax1.set_xlabel('Missing Rate', fontweight='bold')
ax1.set_ylabel('MIM Improvement (%)', fontweight='bold')
ax1.set_title('(a) MIM Improvement by Missing Rate\n(Positive = MIM Better)', 
             fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:.0f}%',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=8, fontweight='bold')

# 右图: 按插补方法在高MR下的表现
ax2 = axes[1]
high_mr = df[df['test_mr'] >= 0.6]
imputations = ['mean', 'knn', 'iterative', 'zero']
x = np.arange(len(imputations))
width = 0.35

means_without = []
means_with = []
for imp in imputations:
    subset = high_mr[high_mr['imputation'] == imp]
    w = subset[subset['use_mim'] == False]['test_mae'].mean()
    m = subset[subset['use_mim'] == True]['test_mae'].mean()
    means_without.append(w)
    means_with.append(m)

bars1 = ax2.bar(x - width/2, means_without, width, label='Without MIM', 
               color='#4DBBD5', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, means_with, width, label='With MIM',
               color='#E64B35', alpha=0.8, edgecolor='black')

ax2.set_xlabel('Imputation Method', fontweight='bold')
ax2.set_ylabel('MAE (MR >= 0.6)', fontweight='bold')
ax2.set_title('(b) MIM Effect at High Missing Rates (≥60%)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([imp.capitalize() for imp in imputations])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 3: Key Finding - MIM Effectiveness at High Missing Rates', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('mim_03_high_mr_analysis')

# ==================== 图4: 最差组合分析 ====================
print("生成图4: MIM最差的组合...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 找出最差的组合
worst_combos = [
    ('lstm', 'knn', 'LSTM + KNN'),
    ('cnn', 'iterative', 'CNN + Iterative'),
    ('lstm', 'mean', 'LSTM + Mean'),
    ('cnn', 'knn', 'CNN + KNN'),
]

for idx, (model, imp, title) in enumerate(worst_combos):
    ax = axes[idx // 2, idx % 2]
    
    subset = df[(df['model'] == model) & (df['imputation'] == imp)]
    
    for use_mim in [False, True]:
        mim_data = subset[subset['use_mim'] == use_mim]
        grouped = mim_data.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
        
        label = 'With MIM' if use_mim else 'Without MIM'
        color = '#E64B35' if use_mim else '#4DBBD5'
        
        ax.plot(grouped.index, grouped['mean'], 
               marker='o', markersize=5, linewidth=2.5,
               label=label, color=color)
        ax.fill_between(grouped.index,
                       grouped['mean'] - grouped['sem'],
                       grouped['mean'] + grouped['sem'],
                       alpha=0.2, color=color)
    
    w = subset[subset['use_mim'] == False]['test_mae'].mean()
    m = subset[subset['use_mim'] == True]['test_mae'].mean()
    diff = w - m
    improvement = diff / w * 100
    
    ax.set_title(f'{title}\nMIM改进: {improvement:.1f}% (最差组合)', 
                fontsize=11, fontweight='bold', color='#C44E52')
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('MAE')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Figure 4: Worst Combinations for MIM (Where MIM Fails Most)', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
savefig('mim_04_worst_combinations')

# ==================== 图5: 最佳组合分析 ====================
print("生成图5: MIM最佳的组合...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

best_combos = [
    ('mlp', 'iterative', 'MCAR', 'MLP + Iterative + MCAR'),
    ('mlp', 'zero', None, 'MLP + Zero (All Modes)'),
    (None, 'zero', None, 'Zero Imputation (All Models)'),
]

for idx, (model, imp, mode, title) in enumerate(best_combos):
    ax = axes[idx]
    
    subset = df.copy()
    if model:
        subset = subset[subset['model'] == model]
    if imp:
        subset = subset[subset['imputation'] == imp]
    if mode:
        subset = subset[subset['mode'] == mode]
    
    for use_mim in [False, True]:
        mim_data = subset[subset['use_mim'] == use_mim]
        grouped = mim_data.groupby('test_mr')['test_mae'].agg(['mean', 'sem'])
        
        label = 'With MIM' if use_mim else 'Without MIM'
        color = '#E64B35' if use_mim else '#4DBBD5'
        
        ax.plot(grouped.index, grouped['mean'], 
               marker='o', markersize=5, linewidth=2.5,
               label=label, color=color)
        ax.fill_between(grouped.index,
                       grouped['mean'] - grouped['sem'],
                       grouped['mean'] + grouped['sem'],
                       alpha=0.2, color=color)
    
    w = subset[subset['use_mim'] == False]['test_mae'].mean()
    m = subset[subset['use_mim'] == True]['test_mae'].mean()
    diff = w - m
    improvement = diff / w * 100
    
    ax.set_title(f'{title}\nMIM改进: {improvement:.1f}%', 
                fontsize=10, fontweight='bold', 
                color='#00A087' if improvement > 0 else '#C44E52')
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('MAE')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle('Figure 5: Best Combinations for MIM (Where MIM Works)', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('mim_05_best_combinations')

# ==================== 图6: 综合决策图 ====================
print("生成图6: MIM使用决策图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 创建热力图：MR × Imputation，颜色表示MIM效果
pivot_data = []
for mr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    row = []
    for imp in ['mean', 'knn', 'iterative', 'zero']:
        subset = df[(df['test_mr'] == mr) & (df['imputation'] == imp)]
        w = subset[subset['use_mim'] == False]['test_mae'].mean()
        m = subset[subset['use_mim'] == True]['test_mae'].mean()
        improvement = (w - m) / w * 100
        row.append(improvement)
    pivot_data.append(row)

pivot_df = pd.DataFrame(pivot_data, 
                       index=[f'{int(mr*100)}%' for mr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]],
                       columns=['Mean', 'KNN', 'Iterative', 'Zero'])

# 使用diverging colormap
im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', vmin=-150, vmax=50)

# 添加数值标签
for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        value = pivot_df.values[i, j]
        color = 'white' if abs(value) > 75 else 'black'
        text = ax.text(j, i, f'{value:.0f}%',
                      ha="center", va="center", color=color, fontweight='bold')

ax.set_xticks(range(len(pivot_df.columns)))
ax.set_xticklabels(pivot_df.columns)
ax.set_yticks(range(len(pivot_df.index)))
ax.set_yticklabels(pivot_df.index)
ax.set_xlabel('Imputation Method', fontweight='bold', fontsize=12)
ax.set_ylabel('Missing Rate', fontweight='bold', fontsize=12)
ax.set_title('MIM Improvement by Missing Rate and Imputation Method\n(Green = MIM Better, Red = MIM Worse)', 
            fontweight='bold', fontsize=13)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('MIM Improvement (%)', fontweight='bold')

plt.tight_layout()
savefig('mim_06_decision_heatmap')

print()
print('='*70)
print(f'所有MIM分析图像已保存至: {OUTPUT_DIR}')
print('='*70)
