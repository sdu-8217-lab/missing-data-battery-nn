#!/usr/bin/env python3
"""
100种子实验完整可视化分析
复用 visualization_framework.py 和 visualize_100seeds_results.py 的代码

生成8组科研级图表：
1. 整体性能对比
2. 热力图 (MR × Imputation)  
3. MIM改进百分比热力图
4. MAE随缺失率变化线图
5. Zero+MIM对比
6. R²分数对比
7. 统计显著性分析
8. 批次性能对比
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 样式设置 (来自 visualization_framework.py) ====================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# 色盲友好的颜色方案
COLORS = {
    'imputation': {'mean': '#E64B35', 'knn': '#4DBBD5', 'iterative': '#00A087', 'zero': '#3C5488'},
    'mode': {'MCAR': '#F39B7F', 'MAR': '#8491B4', 'MNAR': '#91D1C2'},
    'model': {'mlp': '#E64B35', 'lstm': '#4DBBD5', 'cnn': '#00A087'},
    'mim': {False: '#7E6148', True: '#B09C85'}
}

# 输出目录
OUTPUT_DIR = Path('results/analysis_v2/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = 'results/100seeds_v2_final'


def load_all_data():
    """加载所有CSV结果文件"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    files = glob.glob(f'{DATA_DIR}/*.csv')
    print(f"找到 {len(files)} 个结果文件")
    
    dfs = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  已加载 {i}/{len(files)}...")
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  错误: {f} - {e}")
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # 数据清洗：移除异常值
    df_clean = df_all[df_all['test_mae'] < 100].copy()
    
    print(f"\n总计: {len(df_all):,} 条记录")
    print(f"有效记录: {len(df_clean):,}")
    print(f"异常值: {len(df_all) - len(df_clean):,}")
    
    return df_clean


def savefig(name, dpi=300):
    """保存图像"""
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"  ✓ Saved: {name}.png")
    plt.close()


# ==================== Figure 1: 整体性能对比 ====================
def plot_overall_performance(df):
    """1. 整体性能对比 (基于 visualize_100seeds_results.py)"""
    print("\n生成 Figure 1: Overall Performance...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Overall Performance (100 Seeds)', fontsize=16, fontweight='bold')
    
    # 1.1 各模型平均MAE
    ax = axes[0, 0]
    models = ['mlp', 'lstm', 'cnn']
    x = np.arange(len(models))
    width = 0.35
    
    non_mim_mae = [df[(df['model']==m) & (df['use_mim']==False)]['test_mae'].mean() for m in models]
    mim_mae = [df[(df['model']==m) & (df['use_mim']==True)]['test_mae'].mean() for m in models]
    
    ax.bar(x - width/2, non_mim_mae, width, label='non-MIM', color=COLORS['mim'][False])
    ax.bar(x + width/2, mim_mae, width, label='MIM', color=COLORS['mim'][True])
    ax.set_ylabel('Average MAE', fontweight='bold')
    ax.set_title('Average MAE by Model', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.2 各插补方法平均MAE
    ax = axes[0, 1]
    imputations = ['zero', 'mean', 'knn', 'iterative']
    x2 = np.arange(len(imputations))
    
    non_mim_mae = [df[(df['imputation']==imp) & (df['use_mim']==False)]['test_mae'].mean() for imp in imputations]
    mim_mae = [df[(df['imputation']==imp) & (df['use_mim']==True)]['test_mae'].mean() for imp in imputations]
    
    ax.bar(x2 - width/2, non_mim_mae, width, label='non-MIM', color=COLORS['mim'][False])
    ax.bar(x2 + width/2, mim_mae, width, label='MIM', color=COLORS['mim'][True])
    ax.set_ylabel('Average MAE', fontweight='bold')
    ax.set_title('Average MAE by Imputation Method', fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels([i.upper() for i in imputations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.3 High MR场景
    ax = axes[1, 0]
    high_mr = df[df['test_mr'] >= 0.6]
    x3 = np.arange(len(models))
    
    non_mim_mae = [high_mr[(high_mr['model']==m) & (high_mr['use_mim']==False)]['test_mae'].mean() for m in models]
    mim_mae = [high_mr[(high_mr['model']==m) & (high_mr['use_mim']==True)]['test_mae'].mean() for m in models]
    
    ax.bar(x3 - width/2, non_mim_mae, width, label='non-MIM', color=COLORS['mim'][False])
    ax.bar(x3 + width/2, mim_mae, width, label='MIM', color=COLORS['mim'][True])
    ax.set_ylabel('Average MAE', fontweight='bold')
    ax.set_title('High MR (>=0.6) - Average MAE', fontweight='bold')
    ax.set_xticks(x3)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.4 High MR + Zero Imputation
    ax = axes[1, 1]
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    x4 = np.arange(len(models))
    
    non_mim_mae = []
    mim_mae = []
    improvements = []
    for m in models:
        nm = high_mr_zero[(high_mr_zero['model']==m) & (high_mr_zero['use_mim']==False)]['test_mae'].mean()
        mm = high_mr_zero[(high_mr_zero['model']==m) & (high_mr_zero['use_mim']==True)]['test_mae'].mean()
        non_mim_mae.append(nm)
        mim_mae.append(mm)
        improvements.append((nm - mm) / nm * 100)
    
    ax.bar(x4 - width/2, non_mim_mae, width, label='non-MIM', color=COLORS['mim'][False])
    ax.bar(x4 + width/2, mim_mae, width, label='MIM', color=COLORS['mim'][True])
    
    for i, imp in enumerate(improvements):
        color = 'green' if imp > 0 else 'red'
        ax.text(i, max(non_mim_mae[i], mim_mae[i]) + 0.01, f'{imp:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', color=color, fontsize=9)
    
    ax.set_ylabel('Average MAE', fontweight='bold')
    ax.set_title('High MR + Zero - MAE with Improvement', fontweight='bold')
    ax.set_xticks(x4)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig('01_overall_performance')


# ==================== Figure 2: 热力图 ====================
def plot_heatmaps(df):
    """2. 热力图 (MR × Imputation)"""
    print("\n生成 Figure 2: Heatmaps...")
    
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for model in ['mlp', 'lstm', 'cnn']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{model.upper()} - MAE Heatmap (100 Seeds Average)', fontsize=14, fontweight='bold')
        
        for idx, (use_mim, mode) in enumerate([
            (False, 'MCAR'), (True, 'MCAR'),
            (False, 'MAR'), (True, 'MAR'),
            (False, 'MNAR'), (True, 'MNAR')
        ]):
            ax = axes[idx // 3, idx % 3]
            
            heatmap_data = []
            for imp in imputations:
                row = []
                for mr in missing_rates:
                    subset = df[(df['model']==model) & (df['use_mim']==use_mim) & 
                               (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                    if len(subset) > 0:
                        row.append(subset['test_mae'].mean())
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data)
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            ax.set_xticks(range(len(missing_rates)))
            ax.set_xticklabels([f'{mr:.1f}' for mr in missing_rates])
            ax.set_yticks(range(len(imputations)))
            ax.set_yticklabels(imputations)
            ax.set_xlabel('Missing Rate', fontweight='bold')
            ax.set_ylabel('Imputation Method', fontweight='bold')
            ax.set_title(f'Mode={mode}, MIM={use_mim}', fontweight='bold')
            
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(heatmap_data[i, j]):
                        ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=7)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        savefig(f'02_heatmap_{model}')


# ==================== Figure 3: MIM改进百分比热力图 ====================
def plot_mim_improvement_heatmap(df):
    """3. MIM改进百分比热力图"""
    print("\n生成 Figure 3: MIM Improvement Heatmaps...")
    
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    for model in ['mlp', 'lstm', 'cnn']:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{model.upper()} - MIM Improvement % (100 Seeds)', fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            improvement_data = []
            for imp in imputations:
                row = []
                for mr in missing_rates:
                    non_mim = df[(df['model']==model) & (df['use_mim']==False) & 
                                (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                    mim = df[(df['model']==model) & (df['use_mim']==True) & 
                            (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                    
                    if len(non_mim) > 0 and len(mim) > 0:
                        nm_mae = non_mim['test_mae'].mean()
                        m_mae = mim['test_mae'].mean()
                        improvement = (nm_mae - m_mae) / nm_mae * 100
                        row.append(improvement)
                    else:
                        row.append(np.nan)
                improvement_data.append(row)
            
            improvement_data = np.array(improvement_data)
            vmax = max(abs(np.nanmin(improvement_data)), abs(np.nanmax(improvement_data)))
            vmin = -vmax
            im = ax.imshow(improvement_data, cmap='RdYlGn', aspect='auto', 
                          interpolation='nearest', vmin=vmin, vmax=vmax)
            
            ax.set_xticks(range(len(missing_rates)))
            ax.set_xticklabels([f'{mr:.1f}' for mr in missing_rates])
            ax.set_yticks(range(len(imputations)))
            ax.set_yticklabels(imputations)
            ax.set_xlabel('Missing Rate', fontweight='bold')
            ax.set_ylabel('Imputation Method', fontweight='bold')
            ax.set_title(f'Mode={mode}', fontweight='bold')
            
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(improvement_data[i, j]):
                        color = 'white' if abs(improvement_data[i, j]) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{improvement_data[i, j]:.0f}%',
                               ha="center", va="center", color=color, fontsize=8, fontweight='bold')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Improvement %')
        
        plt.tight_layout()
        savefig(f'03_mim_improvement_{model}')


# ==================== Figure 4: MAE随缺失率变化 ====================
def plot_mae_by_missing_rate(df):
    """4. MAE随缺失率变化线图 (科研级样式)"""
    print("\n生成 Figure 4: MAE by Missing Rate...")
    
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    for imp in imputations:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Imputation={imp.upper()} - MAE vs Missing Rate (100 Seeds)', 
                    fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            for model in ['mlp', 'lstm', 'cnn']:
                for use_mim in [False, True]:
                    mae_values = []
                    sem_values = []
                    for mr in missing_rates:
                        subset = df[(df['model']==model) & (df['use_mim']==use_mim) & 
                                   (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                        if len(subset) > 0:
                            mae_values.append(subset['test_mae'].mean())
                            sem_values.append(subset['test_mae'].std() / np.sqrt(len(subset)))
                        else:
                            mae_values.append(np.nan)
                            sem_values.append(0)
                    
                    linestyle = '-' if use_mim == False else '--'
                    marker = 'o' if model == 'mlp' else ('s' if model == 'lstm' else '^')
                    label = f'{model.upper()}{"+MIM" if use_mim else ""}'
                    color = COLORS['model'][model]
                    
                    ax.plot(missing_rates, mae_values, linestyle=linestyle, marker=marker,
                           label=label, linewidth=2, markersize=6, color=color,
                           alpha=0.8 if use_mim else 1.0)
                    ax.fill_between(missing_rates, 
                                   np.array(mae_values) - np.array(sem_values),
                                   np.array(mae_values) + np.array(sem_values),
                                   alpha=0.15, color=color)
            
            ax.set_xlabel('Missing Rate', fontweight='bold')
            ax.set_ylabel('MAE', fontweight='bold')
            ax.set_title(f'Mode={mode}', fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        savefig(f'04_line_{imp}')


# ==================== Figure 5: Zero+MIM对比 ====================
def plot_zero_mim_comparison(df):
    """5. Zero+MIM vs 所有non-MIM方法对比"""
    print("\n生成 Figure 5: Zero+MIM Comparison...")
    
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Zero+MIM vs All non-MIM Methods (100 Seeds Average)', 
                fontsize=14, fontweight='bold')
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # non-MIM方法
        for imp in ['zero', 'mean', 'knn', 'iterative']:
            mae_values = []
            sem_values = []
            for mr in missing_rates:
                subset = df[(df['use_mim']==False) & (df['imputation']==imp) & 
                           (df['mode']==mode) & (df['test_mr']==mr)]
                if len(subset) > 0:
                    mae_values.append(subset['test_mae'].mean())
                    sem_values.append(subset['test_mae'].std() / np.sqrt(len(subset)))
                else:
                    mae_values.append(np.nan)
                    sem_values.append(0)
            
            ax.plot(missing_rates, mae_values, linestyle='-', marker='o', 
                   color=COLORS['imputation'][imp],
                   label=f'non-MIM+{imp.upper()}', linewidth=2, markersize=5, alpha=0.8)
            ax.fill_between(missing_rates, 
                           np.array(mae_values) - np.array(sem_values),
                           np.array(mae_values) + np.array(sem_values),
                           alpha=0.1, color=COLORS['imputation'][imp])
        
        # MIM+Zero
        mae_values_mim = []
        sem_values_mim = []
        for mr in missing_rates:
            subset = df[(df['use_mim']==True) & (df['imputation']=='zero') & 
                       (df['mode']==mode) & (df['test_mr']==mr)]
            if len(subset) > 0:
                mae_values_mim.append(subset['test_mae'].mean())
                sem_values_mim.append(subset['test_mae'].std() / np.sqrt(len(subset)))
            else:
                mae_values_mim.append(np.nan)
                sem_values_mim.append(0)
        
        ax.plot(missing_rates, mae_values_mim, linestyle='--', marker='D', 
               color='#9467bd', label='MIM+ZERO', linewidth=3, markersize=7, 
               markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(missing_rates,
                       np.array(mae_values_mim) - np.array(sem_values_mim),
                       np.array(mae_values_mim) + np.array(sem_values_mim),
                       alpha=0.2, color='#9467bd')
        
        ax.set_xlabel('Missing Rate', fontweight='bold')
        ax.set_ylabel('MAE', fontweight='bold')
        ax.set_title(f'Mode={mode}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    savefig('05_zero_mim_comparison')


# ==================== Figure 6: R²对比 ====================
def plot_r2_comparison(df):
    """6. R²分数对比"""
    print("\n生成 Figure 6: R² Comparison...")
    
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('R² Score vs Missing Rate (100 Seeds)', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(['mlp', 'lstm', 'cnn']):
        ax = axes[idx]
        
        for use_mim in [False, True]:
            r2_values = []
            sem_values = []
            for mr in missing_rates:
                subset = df[(df['model']==model) & (df['use_mim']==use_mim) & (df['test_mr']==mr)]
                if len(subset) > 0:
                    r2_values.append(subset['test_r2'].mean())
                    sem_values.append(subset['test_r2'].std() / np.sqrt(len(subset)))
                else:
                    r2_values.append(np.nan)
                    sem_values.append(0)
            
            label = 'non-MIM' if use_mim == False else 'MIM'
            linestyle = '-' if use_mim == False else '--'
            color = COLORS['mim'][use_mim]
            
            ax.plot(missing_rates, r2_values, linestyle=linestyle, marker='o', 
                   label=label, linewidth=2, markersize=6, color=color)
            ax.fill_between(missing_rates,
                           np.array(r2_values) - np.array(sem_values),
                           np.array(r2_values) + np.array(sem_values),
                           alpha=0.15, color=color)
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Missing Rate', fontweight='bold')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title(f'{model.upper()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig('06_r2_comparison')


# ==================== Figure 7: 统计显著性分析 ====================
def plot_statistical_significance(df):
    """7. 统计显著性分析"""
    print("\n生成 Figure 7: Statistical Significance...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Statistical Significance Analysis (100 Seeds)', fontsize=16, fontweight='bold')
    
    # 7.1 各模型MIM效果箱线图
    ax = axes[0, 0]
    models = ['mlp', 'lstm', 'cnn']
    box_data = []
    labels = []
    for model in models:
        for use_mim in [False, True]:
            data = df[(df['model']==model) & (df['use_mim']==use_mim)]['test_mae'].values
            box_data.append(data)
            labels.append(f'{model.upper()}\n{"MIM" if use_mim else "non-MIM"}')
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = [COLORS['mim'][False], COLORS['mim'][True]] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('MAE Distribution by Model and MIM', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 7.2 High MR + Zero场景下的t检验结果
    ax = axes[0, 1]
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    
    models = ['mlp', 'lstm', 'cnn']
    p_values = []
    effect_sizes = []
    
    for model in models:
        non_mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']==False)]['test_mae']
        mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']==True)]['test_mae']
        
        if len(non_mim) > 0 and len(mim) > 0:
            t_stat, p_val = stats.ttest_ind(non_mim, mim)
            p_values.append(p_val)
            # Cohen's d
            pooled_std = np.sqrt((np.std(non_mim)**2 + np.std(mim)**2) / 2)
            cohens_d = (np.mean(non_mim) - np.mean(mim)) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(cohens_d)
        else:
            p_values.append(1.0)
            effect_sizes.append(0)
    
    x = np.arange(len(models))
    bars = ax.bar(x, -np.log10(p_values), 
                 color=['green' if p < 0.05 else 'red' for p in p_values],
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel('-log10(p-value)', fontweight='bold')
    ax.set_title('Statistical Significance (High MR + Zero)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (p, d) in enumerate(zip(p_values, effect_sizes)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(i, -np.log10(p) + 0.2, f'{sig}\nd={d:.2f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 7.3 各批次性能稳定性
    ax = axes[1, 0]
    batch_means = []
    batch_stds = []
    batch_sems = []
    for batch in sorted(df['batch'].unique()):
        batch_data = df[df['batch']==batch]['test_mae']
        batch_means.append(batch_data.mean())
        batch_stds.append(batch_data.std())
        batch_sems.append(batch_data.std() / np.sqrt(len(batch_data)))
    
    x = np.arange(len(df['batch'].unique()))
    ax.bar(x, batch_means, yerr=batch_sems, capsize=5, color='steelblue', alpha=0.7,
          edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(df['batch'].unique()), rotation=45, ha='right')
    ax.set_ylabel('MAE (mean ± SEM)', fontweight='bold')
    ax.set_title('Performance Stability Across Batches', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 7.4 MIM改进效果分布
    ax = axes[1, 1]
    improvements = []
    for model in ['mlp', 'lstm', 'cnn']:
        for batch in df['batch'].unique():
            non_mim = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']==False)]['test_mae'].mean()
            mim = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']==True)]['test_mae'].mean()
            if not (pd.isna(non_mim) or pd.isna(mim)):
                improvement = (non_mim - mim) / non_mim * 100
                improvements.append(improvement)
    
    ax.hist(improvements, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.axvline(x=np.mean(improvements), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(improvements):.1f}%')
    ax.set_xlabel('MIM Improvement (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of MIM Improvement\n(across models and batches)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig('07_statistical_analysis')


# ==================== Figure 8: 批次对比 ====================
def plot_batch_comparison(df):
    """8. 各批次性能对比"""
    print("\n生成 Figure 8: Batch Comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Performance Comparison Across Batches (100 Seeds)', 
                fontsize=14, fontweight='bold')
    
    batches = sorted(df['batch'].unique())
    models = ['mlp', 'lstm', 'cnn']
    
    # 8.1 各批次平均MAE
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        
        batch_non_mim = []
        batch_mim = []
        batch_sems_nm = []
        batch_sems_m = []
        
        for batch in batches:
            nm_data = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']==False)]['test_mae']
            m_data = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']==True)]['test_mae']
            batch_non_mim.append(nm_data.mean())
            batch_mim.append(m_data.mean())
            batch_sems_nm.append(nm_data.std() / np.sqrt(len(nm_data)))
            batch_sems_m.append(m_data.std() / np.sqrt(len(m_data)))
        
        x = np.arange(len(batches))
        width = 0.35
        ax.bar(x - width/2, batch_non_mim, width, yerr=batch_sems_nm, 
              label='non-MIM', color=COLORS['mim'][False], capsize=3)
        ax.bar(x + width/2, batch_mim, width, yerr=batch_sems_m,
              label='MIM', color=COLORS['mim'][True], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(batches, rotation=45, ha='right')
        ax.set_ylabel('Average MAE', fontweight='bold')
        ax.set_title(f'{model.upper()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 8.2 High MR + Zero场景
    for idx, model in enumerate(models):
        ax = axes[1, idx]
        
        high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
        
        batch_non_mim = []
        batch_mim = []
        improvements = []
        
        for batch in batches:
            nm_data = high_mr_zero[(high_mr_zero['model']==model) & 
                                  (high_mr_zero['batch']==batch) & 
                                  (high_mr_zero['use_mim']==False)]['test_mae']
            m_data = high_mr_zero[(high_mr_zero['model']==model) & 
                                 (high_mr_zero['batch']==batch) & 
                                 (high_mr_zero['use_mim']==True)]['test_mae']
            batch_non_mim.append(nm_data.mean())
            batch_mim.append(m_data.mean())
            if not (pd.isna(nm_data.mean()) or pd.isna(m_data.mean())):
                improvements.append((nm_data.mean() - m_data.mean()) / nm_data.mean() * 100)
            else:
                improvements.append(0)
        
        x = np.arange(len(batches))
        width = 0.35
        ax.bar(x - width/2, batch_non_mim, width, label='non-MIM', color=COLORS['mim'][False])
        ax.bar(x + width/2, batch_mim, width, label='MIM', color=COLORS['mim'][True])
        
        for i, imp in enumerate(improvements):
            color = 'green' if imp > 0 else 'red'
            ax.text(i, max(batch_non_mim[i], batch_mim[i]) + 0.01, f'{imp:+.1f}%',
                   ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(batches, rotation=45, ha='right')
        ax.set_ylabel('Average MAE', fontweight='bold')
        ax.set_title(f'{model.upper()} - High MR + Zero', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    savefig('08_batch_comparison')


# ==================== 生成摘要报告 ====================
def generate_summary_report(df):
    """生成摘要报告"""
    report = []
    report.append("="*70)
    report.append("100随机种子实验结果摘要 (v2)")
    report.append("="*70)
    report.append(f"\n总记录数: {len(df):,}")
    report.append(f"成功记录: {len(df[df['status']=='success']):,}")
    report.append(f"异常值(MAE>100): {len(df[df['test_mae']>=100]):,}")
    
    # 整体MIM效果
    report.append("\n" + "-"*70)
    report.append("整体MIM效果")
    report.append("-"*70)
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = df[(df['model']==model) & (df['use_mim']==False)]['test_mae'].mean()
        mim = df[(df['model']==model) & (df['use_mim']==True)]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    # High MR + Zero场景
    report.append("\n" + "-"*70)
    report.append("High MR (>=0.6) + Zero Imputation")
    report.append("-"*70)
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']==False)]['test_mae'].mean()
        mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']==True)]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    # MR=0.9场景
    report.append("\n" + "-"*70)
    report.append("极高缺失率 (MR=0.9) + Zero Imputation")
    report.append("-"*70)
    mr09_zero = df[(df['test_mr'] == 0.9) & (df['imputation'] == 'zero')]
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = mr09_zero[(mr09_zero['model']==model) & (mr09_zero['use_mim']==False)]['test_mae'].mean()
        mim = mr09_zero[(mr09_zero['model']==model) & (mr09_zero['use_mim']==True)]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    # 按缺失模式
    report.append("\n" + "-"*70)
    report.append("按缺失模式的MIM改进 (Zero Imputation, MR>=0.6)")
    report.append("-"*70)
    for mode in ['MCAR', 'MAR', 'MNAR']:
        mode_data = high_mr_zero[high_mr_zero['mode']==mode]
        non_mim = mode_data[mode_data['use_mim']==False]['test_mae'].mean()
        mim = mode_data[mode_data['use_mim']==True]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{mode}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(OUTPUT_DIR / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n✓ Saved: summary_report.txt")


def main():
    print("="*70)
    print("100种子实验完整可视化分析")
    print("="*70)
    
    # 加载数据
    df = load_all_data()
    
    # 生成所有图表
    print("\n" + "="*70)
    print("开始生成图表...")
    print("="*70)
    
    plot_overall_performance(df)
    plot_heatmaps(df)
    plot_mim_improvement_heatmap(df)
    plot_mae_by_missing_rate(df)
    plot_zero_mim_comparison(df)
    plot_r2_comparison(df)
    plot_statistical_significance(df)
    plot_batch_comparison(df)
    generate_summary_report(df)
    
    print("\n" + "="*70)
    print("所有可视化完成！")
    print("="*70)
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n生成文件列表:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")
    print(f"  - summary_report.txt")


if __name__ == '__main__':
    main()
