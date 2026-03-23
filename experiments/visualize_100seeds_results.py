#!/usr/bin/env python3
"""
100随机种子实验结果全套可视化

生成图表:
1. 整体性能对比图
2. 热力图 (MR × Imputation)
3. MIM改进百分比热力图
4. MAE随缺失率变化线图
5. Zero+MIM vs 其他方法对比
6. R²分数对比
7. 统计显著性分析
8. 各批次性能分布
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

from pathlib import Path
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(jsonl_path):
    """加载JSONL数据"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def plot_overall_performance(df, output_dir):
    """1. 整体性能对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Overall Performance (100 Seeds)', fontsize=16, fontweight='bold')
    
    # 1.1 各模型平均MAE
    ax = axes[0, 0]
    models = ['mlp', 'lstm', 'cnn']
    x = np.arange(len(models))
    width = 0.35
    
    non_mim_mae = [df[(df['model']==m) & (df['use_mim']=='false')]['test_mae'].mean() for m in models]
    mim_mae = [df[(df['model']==m) & (df['use_mim']=='true')]['test_mae'].mean() for m in models]
    
    ax.bar(x - width/2, non_mim_mae, width, label='non-MIM', color='#1f77b4')
    ax.bar(x + width/2, mim_mae, width, label='MIM', color='#ff7f0e')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average MAE by Model')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.2 各插补方法平均MAE
    ax = axes[0, 1]
    imputations = ['zero', 'mean', 'knn', 'iterative']
    x2 = np.arange(len(imputations))
    
    non_mim_mae = [df[(df['imputation']==imp) & (df['use_mim']=='false')]['test_mae'].mean() for imp in imputations]
    mim_mae = [df[(df['imputation']==imp) & (df['use_mim']=='true')]['test_mae'].mean() for imp in imputations]
    
    ax.bar(x2 - width/2, non_mim_mae, width, label='non-MIM', color='#1f77b4')
    ax.bar(x2 + width/2, mim_mae, width, label='MIM', color='#ff7f0e')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average MAE by Imputation Method')
    ax.set_xticks(x2)
    ax.set_xticklabels([i.upper() for i in imputations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.3 High MR场景
    ax = axes[1, 0]
    high_mr = df[df['test_mr'] >= 0.6]
    x3 = np.arange(len(models))
    
    non_mim_mae = [high_mr[(high_mr['model']==m) & (high_mr['use_mim']=='false')]['test_mae'].mean() for m in models]
    mim_mae = [high_mr[(high_mr['model']==m) & (high_mr['use_mim']=='true')]['test_mae'].mean() for m in models]
    
    ax.bar(x3 - width/2, non_mim_mae, width, label='non-MIM', color='#1f77b4')
    ax.bar(x3 + width/2, mim_mae, width, label='MIM', color='#ff7f0e')
    ax.set_ylabel('Average MAE')
    ax.set_title('High MR (>=0.6) - Average MAE')
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
        nm = high_mr_zero[(high_mr_zero['model']==m) & (high_mr_zero['use_mim']=='false')]['test_mae'].mean()
        mm = high_mr_zero[(high_mr_zero['model']==m) & (high_mr_zero['use_mim']=='true')]['test_mae'].mean()
        non_mim_mae.append(nm)
        mim_mae.append(mm)
        improvements.append((nm - mm) / nm * 100)
    
    ax.bar(x4 - width/2, non_mim_mae, width, label='non-MIM', color='#1f77b4')
    ax.bar(x4 + width/2, mim_mae, width, label='MIM', color='#ff7f0e')
    
    for i, imp in enumerate(improvements):
        color = 'green' if imp > 0 else 'red'
        ax.text(i, max(non_mim_mae[i], mim_mae[i]) + 0.005, f'{imp:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', color=color, fontsize=9)
    
    ax.set_ylabel('Average MAE')
    ax.set_title('High MR + Zero - MAE with Improvement')
    ax.set_xticks(x4)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_overall_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_overall_performance.png")


def plot_heatmaps(df, output_dir):
    """2. 热力图 (MR × Imputation)"""
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for model in ['mlp', 'lstm', 'cnn']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{model.upper()} - MAE Heatmap (100 Seeds Average)', fontsize=14, fontweight='bold')
        
        for idx, (use_mim, mode) in enumerate([
            ('false', 'MCAR'), ('true', 'MCAR'),
            ('false', 'MAR'), ('true', 'MAR'),
            ('false', 'MNAR'), ('true', 'MNAR')
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
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Imputation Method')
            ax.set_title(f'Mode={mode}, MIM={use_mim}')
            
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(heatmap_data[i, j]):
                        ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=7)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_heatmap_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 02_heatmap_{model}.png")


def plot_mim_improvement_heatmap(df, output_dir):
    """3. MIM改进百分比热力图"""
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
                    non_mim = df[(df['model']==model) & (df['use_mim']=='false') & 
                                (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                    mim = df[(df['model']==model) & (df['use_mim']=='true') & 
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
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Imputation Method')
            ax.set_title(f'Mode={mode}')
            
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(improvement_data[i, j]):
                        color = 'white' if abs(improvement_data[i, j]) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{improvement_data[i, j]:.0f}%',
                               ha="center", va="center", color=color, fontsize=8, fontweight='bold')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Improvement %')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_mim_improvement_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 03_mim_improvement_{model}.png")


def plot_mae_by_missing_rate(df, output_dir):
    """4. MAE随缺失率变化线图"""
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    for imp in imputations:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Imputation={imp.upper()} - MAE vs Missing Rate (100 Seeds)', fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            for model in ['mlp', 'lstm', 'cnn']:
                for use_mim in ['false', 'true']:
                    mae_values = []
                    for mr in missing_rates:
                        subset = df[(df['model']==model) & (df['use_mim']==use_mim) & 
                                   (df['mode']==mode) & (df['imputation']==imp) & (df['test_mr']==mr)]
                        if len(subset) > 0:
                            mae_values.append(subset['test_mae'].mean())
                        else:
                            mae_values.append(np.nan)
                    
                    linestyle = '-' if use_mim == 'false' else '--'
                    marker = 'o' if model == 'mlp' else ('s' if model == 'lstm' else '^')
                    label = f'{model.upper()}{"+MIM" if use_mim=="true" else ""}'
                    ax.plot(missing_rates, mae_values, linestyle=linestyle, marker=marker,
                           label=label, linewidth=2, markersize=6)
            
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('MAE')
            ax.set_title(f'Mode={mode}')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_line_{imp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 04_line_{imp}.png")


def plot_zero_mim_comparison(df, output_dir):
    """5. Zero+MIM vs 所有non-MIM方法对比"""
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    colors = {'zero': '#1f77b4', 'mean': '#ff7f0e', 'knn': '#2ca02c', 
              'iterative': '#d62728', 'zero_mim': '#9467bd'}
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Zero+MIM vs All non-MIM Methods (100 Seeds Average)', fontsize=14, fontweight='bold')
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # non-MIM方法
        for imp in ['zero', 'mean', 'knn', 'iterative']:
            mae_values = []
            for mr in missing_rates:
                subset = df[(df['use_mim']=='false') & (df['imputation']==imp) & 
                           (df['mode']==mode) & (df['test_mr']==mr)]
                if len(subset) > 0:
                    mae_values.append(subset['test_mae'].mean())
                else:
                    mae_values.append(np.nan)
            ax.plot(missing_rates, mae_values, linestyle='-', marker='o', color=colors[imp],
                   label=f'non-MIM+{imp.upper()}', linewidth=2, markersize=5, alpha=0.8)
        
        # MIM+Zero
        mae_values_mim = []
        for mr in missing_rates:
            subset = df[(df['use_mim']=='true') & (df['imputation']=='zero') & 
                       (df['mode']==mode) & (df['test_mr']==mr)]
            if len(subset) > 0:
                mae_values_mim.append(subset['test_mae'].mean())
            else:
                mae_values_mim.append(np.nan)
        ax.plot(missing_rates, mae_values_mim, linestyle='--', marker='D', color=colors['zero_mim'],
               label='MIM+ZERO', linewidth=3, markersize=7, markerfacecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('MAE')
        ax.set_title(f'Mode={mode}')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_zero_mim_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_zero_mim_comparison.png")


def plot_r2_comparison(df, output_dir):
    """6. R²分数对比"""
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('R² Score vs Missing Rate (100 Seeds)', fontsize=14, fontweight='bold')
    
    for idx, model in enumerate(['mlp', 'lstm', 'cnn']):
        ax = axes[idx]
        
        for use_mim in ['false', 'true']:
            r2_values = []
            for mr in missing_rates:
                subset = df[(df['model']==model) & (df['use_mim']==use_mim) & (df['test_mr']==mr)]
                if len(subset) > 0:
                    r2_values.append(subset['test_r2'].mean())
                else:
                    r2_values.append(np.nan)
            
            label = 'non-MIM' if use_mim == 'false' else 'MIM'
            linestyle = '-' if use_mim == 'false' else '--'
            ax.plot(missing_rates, r2_values, linestyle=linestyle, marker='o', 
                   label=label, linewidth=2, markersize=6)
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{model.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_r2_comparison.png")


def plot_statistical_significance(df, output_dir):
    """7. 统计显著性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Statistical Significance Analysis (100 Seeds)', fontsize=16, fontweight='bold')
    
    # 7.1 各模型MIM效果箱线图
    ax = axes[0, 0]
    models = ['mlp', 'lstm', 'cnn']
    box_data = []
    labels = []
    for model in models:
        for use_mim in ['false', 'true']:
            data = df[(df['model']==model) & (df['use_mim']==use_mim)]['test_mae'].values
            box_data.append(data)
            labels.append(f'{model.upper()}\n{"MIM" if use_mim=="true" else "non-MIM"}')
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('MAE')
    ax.set_title('MAE Distribution by Model and MIM')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 7.2 High MR + Zero场景下的t检验结果
    ax = axes[0, 1]
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    
    models = ['mlp', 'lstm', 'cnn']
    p_values = []
    effect_sizes = []
    
    for model in models:
        non_mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']=='false')]['test_mae']
        mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']=='true')]['test_mae']
        
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
    ax.bar(x, -np.log10(p_values), color=['green' if p < 0.05 else 'red' for p in p_values])
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance (High MR + Zero)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (p, d) in enumerate(zip(p_values, effect_sizes)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(i, -np.log10(p) + 0.2, f'{sig}\nd={d:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 7.3 各批次性能稳定性
    ax = axes[1, 0]
    batch_means = []
    batch_stds = []
    for batch in df['batch'].unique():
        batch_data = df[df['batch']==batch]['test_mae']
        batch_means.append(batch_data.mean())
        batch_stds.append(batch_data.std())
    
    x = np.arange(len(df['batch'].unique()))
    ax.bar(x, batch_means, yerr=batch_stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df['batch'].unique(), rotation=45, ha='right')
    ax.set_ylabel('MAE (mean ± std)')
    ax.set_title('Performance Stability Across Batches')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 7.4 MIM改进效果分布
    ax = axes[1, 1]
    improvements = []
    for model in ['mlp', 'lstm', 'cnn']:
        for batch in df['batch'].unique():
            non_mim = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']=='false')]['test_mae'].mean()
            mim = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']=='true')]['test_mae'].mean()
            if not (pd.isna(non_mim) or pd.isna(mim)):
                improvement = (non_mim - mim) / non_mim * 100
                improvements.append(improvement)
    
    ax.hist(improvements, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='No improvement')
    ax.axvline(x=np.mean(improvements), color='green', linestyle='--', 
               label=f'Mean: {np.mean(improvements):.1f}%')
    ax.set_xlabel('MIM Improvement (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of MIM Improvement\n(across models and batches)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_statistical_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_statistical_analysis.png")


def plot_batch_comparison(df, output_dir):
    """8. 各批次性能对比"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Performance Comparison Across Batches (100 Seeds)', fontsize=14, fontweight='bold')
    
    batches = sorted(df['batch'].unique())
    models = ['mlp', 'lstm', 'cnn']
    
    # 8.1 各批次平均MAE
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        
        batch_non_mim = []
        batch_mim = []
        for batch in batches:
            nm = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']=='false')]['test_mae'].mean()
            m = df[(df['model']==model) & (df['batch']==batch) & (df['use_mim']=='true')]['test_mae'].mean()
            batch_non_mim.append(nm)
            batch_mim.append(m)
        
        x = np.arange(len(batches))
        width = 0.35
        ax.bar(x - width/2, batch_non_mim, width, label='non-MIM', color='#1f77b4')
        ax.bar(x + width/2, batch_mim, width, label='MIM', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(batches, rotation=45, ha='right')
        ax.set_ylabel('Average MAE')
        ax.set_title(f'{model.upper()}')
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
            nm = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['batch']==batch) & 
                             (high_mr_zero['use_mim']=='false')]['test_mae'].mean()
            m = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['batch']==batch) & 
                            (high_mr_zero['use_mim']=='true')]['test_mae'].mean()
            batch_non_mim.append(nm)
            batch_mim.append(m)
            if not (pd.isna(nm) or pd.isna(m)):
                improvements.append((nm - m) / nm * 100)
            else:
                improvements.append(0)
        
        x = np.arange(len(batches))
        width = 0.35
        ax.bar(x - width/2, batch_non_mim, width, label='non-MIM', color='#1f77b4')
        ax.bar(x + width/2, batch_mim, width, label='MIM', color='#ff7f0e')
        
        for i, imp in enumerate(improvements):
            color = 'green' if imp > 0 else 'red'
            ax.text(i, max(batch_non_mim[i], batch_mim[i]) + 0.01, f'{imp:+.1f}%',
                   ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(batches, rotation=45, ha='right')
        ax.set_ylabel('Average MAE')
        ax.set_title(f'{model.upper()} - High MR + Zero')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_batch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_batch_comparison.png")


def generate_summary_report(df, output_dir):
    """生成摘要报告"""
    report = []
    report.append("="*70)
    report.append("100随机种子实验结果摘要")
    report.append("="*70)
    report.append(f"\n总记录数: {len(df):,}")
    report.append(f"成功记录: {len(df[df['status']=='success']):,}")
    
    # 整体MIM效果
    report.append("\n" + "-"*70)
    report.append("整体MIM效果")
    report.append("-"*70)
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = df[(df['model']==model) & (df['use_mim']=='false')]['test_mae'].mean()
        mim = df[(df['model']==model) & (df['use_mim']=='true')]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    # High MR + Zero场景
    report.append("\n" + "-"*70)
    report.append("High MR (>=0.6) + Zero Imputation")
    report.append("-"*70)
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']=='false')]['test_mae'].mean()
        mim = high_mr_zero[(high_mr_zero['model']==model) & (high_mr_zero['use_mim']=='true')]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    # KNN场景
    report.append("\n" + "-"*70)
    report.append("KNN Imputation (所有MR)")
    report.append("-"*70)
    knn_data = df[df['imputation'] == 'knn']
    for model in ['mlp', 'lstm', 'cnn']:
        non_mim = knn_data[(knn_data['model']==model) & (knn_data['use_mim']=='false')]['test_mae'].mean()
        mim = knn_data[(knn_data['model']==model) & (knn_data['use_mim']=='true')]['test_mae'].mean()
        improvement = (non_mim - mim) / non_mim * 100
        report.append(f"{model.upper()}: non-MIM={non_mim:.4f}, MIM={mim:.4f}, 改进={improvement:+.2f}%")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nSaved: summary_report.txt")


def main():
    jsonl_path = 'results/100seeds_new_config/test_results.jsonl'
    output_dir = 'results/100seeds_new_config/figures'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {jsonl_path}...")
    df = load_data(jsonl_path)
    print(f"Loaded {len(df):,} records\n")
    
    print("Generating visualizations...")
    
    print("\n1. Overall performance...")
    plot_overall_performance(df, output_dir)
    
    print("\n2. Heatmaps...")
    plot_heatmaps(df, output_dir)
    
    print("\n3. MIM improvement heatmaps...")
    plot_mim_improvement_heatmap(df, output_dir)
    
    print("\n4. MAE by missing rate...")
    plot_mae_by_missing_rate(df, output_dir)
    
    print("\n5. Zero+MIM comparison...")
    plot_zero_mim_comparison(df, output_dir)
    
    print("\n6. R2 comparison...")
    plot_r2_comparison(df, output_dir)
    
    print("\n7. Statistical analysis...")
    plot_statistical_significance(df, output_dir)
    
    print("\n8. Batch comparison...")
    plot_batch_comparison(df, output_dir)
    
    print("\n9. Summary report...")
    generate_summary_report(df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to {output_dir}/")
    print(f"Generated files:")
    for f in sorted(Path(output_dir).glob('*.png')):
        print(f"  - {f.name}")
    print(f"  - summary_report.txt")


if __name__ == '__main__':
    main()
