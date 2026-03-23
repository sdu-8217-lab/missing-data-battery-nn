#!/usr/bin/env python3
"""
按Batch细粒度可视化 - MCAR模式
每张图包含：3模型 × 2 MIM × 4插补方法
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9

# 输出目录
OUTPUT_DIR = Path('results/analysis_v2/by_batch_detail')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = 'results/100seeds_v2_final'


def load_data():
    """加载所有数据"""
    print("加载数据...")
    files = glob.glob(f'{DATA_DIR}/*.csv')
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df[df['test_mae'] < 100].copy()


def plot_batch_detail(df, batch_name):
    """
    为指定batch绘制详细对比图
    布局: 2×2 subplot，每个subplot对应一种插补方法
    每个subplot中: 3模型 × 2 MIM = 6条线
    """
    print(f"  生成 {batch_name} 的图表...")
    
    # 筛选数据：指定batch + MCAR模式
    batch_df = df[(df['batch'] == batch_name) & (df['mode'] == 'MCAR')]
    
    if len(batch_df) == 0:
        print(f"    警告: {batch_name} 没有MCAR数据")
        return
    
    # 创建大图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Batch: {batch_name} | Mode: MCAR | Detailed Comparison\n'
                f'(3 Models × 2 MIM × 10 Missing Rates)',
                fontsize=14, fontweight='bold')
    
    # 插补方法和对应位置
    imputations = [
        ('zero', 0, 0), ('mean', 0, 1),
        ('knn', 1, 0), ('iterative', 1, 1)
    ]
    
    # 模型配置
    models = ['mlp', 'lstm', 'cnn']
    model_markers = {'mlp': 'o', 'lstm': 's', 'cnn': '^'}
    model_colors = {'mlp': '#E64B35', 'lstm': '#4DBBD5', 'cnn': '#00A087'}
    
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for imp_name, row, col in imputations:
        ax = axes[row, col]
        
        # 绘制每种模型和MIM组合
        for model in models:
            for use_mim in [False, True]:
                # 收集该配置的MAE值
                mae_values = []
                sem_values = []
                
                for mr in missing_rates:
                    subset = batch_df[
                        (batch_df['model'] == model) &
                        (batch_df['use_mim'] == use_mim) &
                        (batch_df['imputation'] == imp_name) &
                        (batch_df['test_mr'] == mr)
                    ]
                    
                    if len(subset) > 0:
                        mae_values.append(subset['test_mae'].mean())
                        sem_values.append(subset['test_mae'].std() / np.sqrt(len(subset)))
                    else:
                        mae_values.append(np.nan)
                        sem_values.append(0)
                
                # 线条样式
                linestyle = '-' if use_mim == False else '--'
                marker = model_markers[model]
                color = model_colors[model]
                alpha = 1.0 if use_mim == False else 0.7
                linewidth = 2.5 if use_mim == False else 2.0
                
                label = f'{model.upper()}{"+MIM" if use_mim else ""}'
                
                ax.plot(missing_rates, mae_values, 
                       linestyle=linestyle, marker=marker, 
                       color=color, linewidth=linewidth, markersize=6,
                       label=label, alpha=alpha)
                
                # 添加SEM阴影
                ax.fill_between(missing_rates,
                               np.array(mae_values) - np.array(sem_values),
                               np.array(mae_values) + np.array(sem_values),
                               alpha=0.1, color=color)
        
        # 子图标题和标签
        ax.set_title(f'{imp_name.upper()} Imputation', fontweight='bold', fontsize=11)
        ax.set_xlabel('Missing Rate', fontweight='bold')
        if col == 0:
            ax.set_ylabel('MAE', fontweight='bold')
        
        ax.set_xticks(missing_rates)
        ax.set_xticklabels([f'{mr:.1f}' for mr in missing_rates], fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 图例 - 放在子图内右上角
        ax.legend(loc='upper left', fontsize=7, ncol=2,
                 frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    output_file = OUTPUT_DIR / f'batch_{batch_name}_MCAR_detail.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_file.name}")


def plot_batch_summary(df, batch_name):
    """
    为指定batch生成一张汇总图
    展示Zero插补方法下各模型的MIM改进效果
    """
    print(f"  生成 {batch_name} 的汇总图...")
    
    batch_df = df[(df['batch'] == batch_name) & (df['mode'] == 'MCAR')]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'Batch: {batch_name} | Mode: MCAR | MIM Improvement by Model',
                fontsize=14, fontweight='bold')
    
    models = ['mlp', 'lstm', 'cnn']
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    imp_colors = {'zero': '#3C5488', 'mean': '#E64B35', 
                  'knn': '#4DBBD5', 'iterative': '#00A087'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        for imp in imputations:
            improvements = []
            
            for mr in missing_rates:
                non_mim = batch_df[
                    (batch_df['model'] == model) &
                    (batch_df['use_mim'] == False) &
                    (batch_df['imputation'] == imp) &
                    (batch_df['test_mr'] == mr)
                ]
                mim = batch_df[
                    (batch_df['model'] == model) &
                    (batch_df['use_mim'] == True) &
                    (batch_df['imputation'] == imp) &
                    (batch_df['test_mr'] == mr)
                ]
                
                if len(non_mim) > 0 and len(mim) > 0:
                    nm_mae = non_mim['test_mae'].mean()
                    m_mae = mim['test_mae'].mean()
                    improvement = (nm_mae - m_mae) / nm_mae * 100
                    improvements.append(improvement)
                else:
                    improvements.append(np.nan)
            
            ax.plot(missing_rates, improvements,
                   marker='o', linewidth=2, markersize=5,
                   label=imp.upper(), color=imp_colors[imp])
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Missing Rate', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('MIM Improvement (%)', fontweight='bold')
        ax.set_title(f'{model.upper()}', fontweight='bold')
        ax.set_xticks(missing_rates)
        ax.set_xticklabels([f'{mr:.1f}' for mr in missing_rates], fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_file = OUTPUT_DIR / f'batch_{batch_name}_MCAR_improvement.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {output_file.name}")


def generate_batch_comparison_table(df):
    """生成各batch的对比表格"""
    print("\n生成对比表格...")
    
    batches = sorted(df['batch'].unique())
    models = ['mlp', 'lstm', 'cnn']
    
    results = []
    
    for batch in batches:
        batch_df = df[(df['batch'] == batch) & (df['mode'] == 'MCAR')]
        
        for model in models:
            # Zero插补 + MR=0.9
            subset = batch_df[
                (batch_df['model'] == model) &
                (batch_df['imputation'] == 'zero') &
                (batch_df['test_mr'] == 0.9)
            ]
            
            if len(subset) > 0:
                non_mim = subset[subset['use_mim'] == False]['test_mae'].mean()
                mim = subset[subset['use_mim'] == True]['test_mae'].mean()
                improvement = (non_mim - mim) / non_mim * 100
                
                results.append({
                    'batch': batch,
                    'model': model,
                    'non_mim_mae': non_mim,
                    'mim_mae': mim,
                    'improvement': improvement
                })
    
    results_df = pd.DataFrame(results)
    
    # 保存CSV
    output_file = OUTPUT_DIR / 'batch_comparison_MR09_Zero.csv'
    results_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file.name}")
    
    # 打印汇总
    print("\n" + "="*80)
    print("各Batch在MR=0.9 + Zero插补下的MIM改进对比")
    print("="*80)
    
    pivot = results_df.pivot(index='batch', columns='model', values='improvement')
    print(pivot.to_string())
    
    print("\n" + "="*80)
    print("各Batch平均MIM改进:")
    print("="*80)
    avg_by_batch = results_df.groupby('batch')['improvement'].mean().sort_values(ascending=False)
    for batch, avg_imp in avg_by_batch.items():
        print(f"  {batch}: {avg_imp:+.2f}%")


def main():
    print("="*80)
    print("按Batch细粒度可视化 - MCAR模式")
    print("="*80)
    
    # 加载数据
    df = load_data()
    print(f"总记录数: {len(df):,}")
    
    # 获取所有batch
    batches = sorted(df['batch'].unique())
    print(f"发现 {len(batches)} 个batch: {batches}")
    print()
    
    # 为每个batch生成详细图
    print("生成各Batch详细对比图...")
    for batch in batches:
        plot_batch_detail(df, batch)
    
    print()
    print("生成各Batch MIM改进汇总图...")
    for batch in batches:
        plot_batch_summary(df, batch)
    
    # 生成对比表格
    generate_batch_comparison_table(df)
    
    print()
    print("="*80)
    print(f"分析完成！所有图表保存至: {OUTPUT_DIR}")
    print("="*80)
    print("\n生成文件列表:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")
    print(f"  - batch_comparison_MR09_Zero.csv")


if __name__ == '__main__':
    main()
