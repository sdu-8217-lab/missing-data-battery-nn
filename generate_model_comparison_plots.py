#!/usr/bin/env python3
"""
生成论文用的模型对比图
使用 errorbar 显示95%置信区间（与旧代码一致）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_path):
    """加载实验结果数据"""
    df = pd.read_csv(csv_path)
    return df

def process_model_data(df, model_name):
    """
    处理单个模型的数据，计算统计量
    """
    # Baseline: model_name (e.g., 'MLP')
    # MIM: model_name-MIM (e.g., 'MLP-MIM')
    baseline_df = df[df['model_name'] == model_name].copy()
    mim_df = df[df['model_name'] == f'{model_name}-MIM'].copy()
    
    # 获取所有缺失率
    missing_rates = sorted(baseline_df['missing_rate'].unique())
    
    stats = {
        'missing_rate': [],
        'baseline_mae_mean': [],
        'baseline_mae_std': [],
        'baseline_rmse_mean': [],
        'baseline_rmse_std': [],
        'baseline_r2_mean': [],
        'baseline_r2_std': [],
        'mim_mae_mean': [],
        'mim_mae_std': [],
        'mim_rmse_mean': [],
        'mim_rmse_std': [],
        'mim_r2_mean': [],
        'mim_r2_std': [],
        'improvement_mean': [],
        'improvement_std': []
    }
    
    for mr in missing_rates:
        baseline_mr = baseline_df[baseline_df['missing_rate'] == mr]
        mim_mr = mim_df[mim_df['missing_rate'] == mr]
        
        # Baseline 统计
        baseline_mae = baseline_mr['mae'].values
        baseline_rmse = baseline_mr['rmse'].values
        baseline_r2 = baseline_mr['r2'].values
        
        # MIM 统计
        mim_mae = mim_mr['mae'].values
        mim_rmse = mim_mr['rmse'].values
        mim_r2 = mim_mr['r2'].values
        
        # 计算改进率 (%)
        improvement = ((baseline_mae - mim_mae) / baseline_mae) * 100
        
        stats['missing_rate'].append(mr)
        
        stats['baseline_mae_mean'].append(np.mean(baseline_mae))
        stats['baseline_mae_std'].append(np.std(baseline_mae))
        stats['baseline_rmse_mean'].append(np.mean(baseline_rmse))
        stats['baseline_rmse_std'].append(np.std(baseline_rmse))
        stats['baseline_r2_mean'].append(np.mean(baseline_r2))
        stats['baseline_r2_std'].append(np.std(baseline_r2))
        
        stats['mim_mae_mean'].append(np.mean(mim_mae))
        stats['mim_mae_std'].append(np.std(mim_mae))
        stats['mim_rmse_mean'].append(np.mean(mim_rmse))
        stats['mim_rmse_std'].append(np.std(mim_rmse))
        stats['mim_r2_mean'].append(np.mean(mim_r2))
        stats['mim_r2_std'].append(np.std(mim_r2))
        
        stats['improvement_mean'].append(np.mean(improvement))
        stats['improvement_std'].append(np.std(improvement))
    
    return pd.DataFrame(stats)

def create_comparison_plot(df_stats, model_name, output_path):
    """
    创建对比图，使用 errorbar 显示95%置信区间
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Model: Baseline vs MIM Comparison\n(100 Random Seeds, 95% CI)', 
                 fontsize=16, fontweight='bold')
    
    missing_rates = df_stats['missing_rate']
    n_experiments = 100  # 样本数，用于计算置信区间
    
    # 计算95%置信区间 = mean ± 1.96 * std / sqrt(n)
    ci_factor = 1.96 / np.sqrt(n_experiments)
    
    # (a) MAE对比
    ax = axes[0, 0]
    baseline_mae_ci = df_stats['baseline_mae_std'] * ci_factor
    mim_mae_ci = df_stats['mim_mae_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_mae_mean'], 
                yerr=baseline_mae_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_mae_mean'], 
                yerr=mim_mae_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('(a) MAE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (b) RMSE对比
    ax = axes[0, 1]
    baseline_rmse_ci = df_stats['baseline_rmse_std'] * ci_factor
    mim_rmse_ci = df_stats['mim_rmse_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_rmse_mean'], 
                yerr=baseline_rmse_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_rmse_mean'], 
                yerr=mim_rmse_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('(b) RMSE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (c) R²对比
    ax = axes[1, 0]
    baseline_r2_ci = df_stats['baseline_r2_std'] * ci_factor
    mim_r2_ci = df_stats['mim_r2_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_r2_mean'], 
                yerr=baseline_r2_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_r2_mean'], 
                yerr=mim_r2_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('(c) R² Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (d) 改进率曲线
    ax = axes[1, 1]
    improvement_ci = df_stats['improvement_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['improvement_mean'], 
                yerr=improvement_ci,
                fmt='D-', color='#2ECC71', label='MAE Improvement', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('(d) MAE Improvement Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # 添加平均改进率文本
    avg_improvement = df_stats['improvement_mean'].mean()
    ax.text(0.5, 0.95, f'Average Improvement: {avg_improvement:.1f}%', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 数据路径
    csv_path = 'experiments_v2/3C/20260204_084913/results_all.csv'
    
    # 输出目录
    output_dir = 'my_figures/model_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    
    # 模型列表
    models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
    
    # 为每个模型生成对比图
    for model in models:
        print(f"\nProcessing {model} model...")
        df_stats = process_model_data(df, model)
        output_path = os.path.join(output_dir, f'{model}_baseline_vs_mim_comparison.png')
        create_comparison_plot(df_stats, model, output_path)
    
    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == '__main__':
    main()
