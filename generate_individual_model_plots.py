#!/usr/bin/env python3
"""生成每个模型单独的对比图表"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# 查找最新的实验结果
result_files = glob.glob('experiments_v2/3C/*/results_all.csv')
if not result_files:
    print("No experiment results found!")
    exit(1)

latest_file = sorted(result_files)[-1]
print(f"Loading: {latest_file}")

df = pd.read_csv(latest_file)
output_dir = Path('paper/figures/individual_models')
output_dir.mkdir(parents=True, exist_ok=True)

# 模型配置
models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
colors = {'Baseline': '#ff7f0e', 'MIM': '#2ca02c'}

# 1. 每个模型单独的MAE对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    
    # 获取该模型的数据 (model_name格式: 'MLP' 或 'MLP-MIM')
    baseline_data = df[df['model_name'] == model]
    mim_data = df[df['model_name'] == f'{model}-MIM']
    
    # 按缺失率分组计算均值和标准差
    baseline_stats = baseline_data.groupby('missing_rate')['mae'].agg(['mean', 'std']).reset_index()
    mim_stats = mim_data.groupby('missing_rate')['mae'].agg(['mean', 'std']).reset_index()
    
    # 绘制
    ax.plot(baseline_stats['missing_rate'], baseline_stats['mean'], 
            'o-', color=colors['Baseline'], label='Baseline', linewidth=2, markersize=6)
    ax.fill_between(baseline_stats['missing_rate'], 
                    baseline_stats['mean'] - baseline_stats['std'],
                    baseline_stats['mean'] + baseline_stats['std'],
                    color=colors['Baseline'], alpha=0.2)
    
    ax.plot(mim_stats['missing_rate'], mim_stats['mean'], 
            's-', color=colors['MIM'], label='MIM', linewidth=2, markersize=6)
    ax.fill_between(mim_stats['missing_rate'], 
                    mim_stats['mean'] - mim_stats['std'],
                    mim_stats['mean'] + mim_stats['std'],
                    color=colors['MIM'], alpha=0.2)
    
    ax.set_xlabel('Missing Rate', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title(f'{model}: Baseline vs MIM', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)

plt.tight_layout()
plt.savefig(output_dir / 'all_models_individual_mae.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'all_models_individual_mae.pdf', bbox_inches='tight')
print(f"Saved: all_models_individual_mae.png/pdf")
plt.close()

# 2. 每个模型单独的改进率图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    
    # 计算改进率 (model_name格式: 'MLP' 或 'MLP-MIM')
    baseline_data = df[df['model_name'] == model]
    mim_data = df[df['model_name'] == f'{model}-MIM']
    
    improvement_data = []
    for mr in sorted(df['missing_rate'].unique()):
        base_mae = baseline_data[baseline_data['missing_rate'] == mr]['mae'].mean()
        mim_mae = mim_data[mim_data['missing_rate'] == mr]['mae'].mean()
        improvement = (base_mae - mim_mae) / base_mae * 100
        improvement_data.append({'missing_rate': mr, 'improvement': improvement})
    
    imp_df = pd.DataFrame(improvement_data)
    
    # 绘制
    colors_imp = ['green' if x > 0 else 'red' for x in imp_df['improvement']]
    ax.bar(imp_df['missing_rate'], imp_df['improvement'], 
           width=0.08, color=colors_imp, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Missing Rate', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title(f'{model}: MIM Improvement Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, row in imp_df.iterrows():
        ax.text(row['missing_rate'], row['improvement'] + 2, 
                f"{row['improvement']:.1f}%", 
                ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig(output_dir / 'all_models_individual_improvement.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'all_models_individual_improvement.pdf', bbox_inches='tight')
print(f"Saved: all_models_individual_improvement.png/pdf")
plt.close()

# 3. 高缺失率区域放大图 (0.6-0.9)
fig, ax = plt.subplots(figsize=(12, 7))

high_mr_df = df[df['missing_rate'] >= 0.6]
model_colors = {'MLP': '#1f77b4', 'LSTM': '#ff7f0e', 'GRU': '#2ca02c', 'CNN1D': '#d62728'}

x_pos = np.arange(4)  # 4个缺失率: 0.6, 0.7, 0.8, 0.9
width = 0.1

for i, model in enumerate(models):
    baseline_vals = []
    mim_vals = []
    
    for mr in [0.6, 0.7, 0.8, 0.9]:
        base_mae = df[(df['model_name'] == model) & 
                      (df['missing_rate'] == mr)]['mae'].mean()
        mim_mae = df[(df['model_name'] == f'{model}-MIM') & 
                     (df['missing_rate'] == mr)]['mae'].mean()
        baseline_vals.append(base_mae)
        mim_vals.append(mim_mae)
    
    offset = (i - 1.5) * width * 2
    ax.bar(x_pos + offset, baseline_vals, width, 
           label=f'{model}-Baseline', color=model_colors[model], alpha=0.4)
    ax.bar(x_pos + offset + width, mim_vals, width,
           label=f'{model}-MIM', color=model_colors[model], alpha=0.9)

ax.set_xlabel('Missing Rate', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('High Missing Rate Region (0.6-0.9): Detailed Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos + width/2)
ax.set_xticklabels(['0.6', '0.7', '0.8', '0.9'])
ax.legend(loc='upper left', ncol=2, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'high_missing_rate_detail.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'high_missing_rate_detail.pdf', bbox_inches='tight')
print(f"Saved: high_missing_rate_detail.png/pdf")
plt.close()

# 4. R²对比图（展示模型失效情况）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    
    # R²对比 (model_name格式: 'MLP' 或 'MLP-MIM')
    baseline_data = df[df['model_name'] == model]
    mim_data = df[df['model_name'] == f'{model}-MIM']
    
    baseline_stats = baseline_data.groupby('missing_rate')['r2'].agg(['mean', 'std']).reset_index()
    mim_stats = mim_data.groupby('missing_rate')['r2'].agg(['mean', 'std']).reset_index()
    
    ax.plot(baseline_stats['missing_rate'], baseline_stats['mean'], 
            'o-', color=colors['Baseline'], label='Baseline', linewidth=2, markersize=6)
    ax.fill_between(baseline_stats['missing_rate'], 
                    baseline_stats['mean'] - baseline_stats['std'],
                    baseline_stats['mean'] + baseline_stats['std'],
                    color=colors['Baseline'], alpha=0.2)
    
    ax.plot(mim_stats['missing_rate'], mim_stats['mean'], 
            's-', color=colors['MIM'], label='MIM', linewidth=2, markersize=6)
    ax.fill_between(mim_stats['missing_rate'], 
                    mim_stats['mean'] - mim_stats['std'],
                    mim_stats['mean'] + mim_stats['std'],
                    color=colors['MIM'], alpha=0.2)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='R²=0 (random guess)')
    ax.set_xlabel('Missing Rate', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title(f'{model}: R² Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-1.5, 1.1)

plt.tight_layout()
plt.savefig(output_dir / 'all_models_individual_r2.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'all_models_individual_r2.pdf', bbox_inches='tight')
print(f"Saved: all_models_individual_r2.png/pdf")
plt.close()

print(f"\nAll individual model plots saved to: {output_dir}")
print("Files generated:")
for f in sorted(output_dir.iterdir()):
    print(f"  {f.name}")
