"""
单随机种子实验结果可视化
生成完整的图表集
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

import os
from pathlib import Path


def load_data(json_path):
    """加载实验结果数据"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return pd.DataFrame(results)


def plot_heatmap_by_model(df, output_dir):
    """为每个模型绘制热力图"""
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for model in ['mlp', 'lstm', 'cnn']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{model.upper()} - MAE Heatmap by Mode and MIM', fontsize=14, fontweight='bold')
        
        for idx, (use_mim, mode) in enumerate([
            ('false', 'MCAR'), ('true', 'MCAR'),
            ('false', 'MAR'), ('true', 'MAR'),
            ('false', 'MNAR'), ('true', 'MNAR')
        ]):
            ax = axes[idx // 3, idx % 3]
            
            # 构建热力图数据
            heatmap_data = []
            for imp in imputations:
                row = []
                for mr in missing_rates:
                    subset = df[(df['model'] == model) & 
                               (df['use_mim'] == use_mim) & 
                               (df['mode'] == mode) & 
                               (df['imputation'] == imp) & 
                               (df['test_mr'] == mr)]
                    if len(subset) > 0:
                        row.append(subset['test_mae'].values[0])
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            heatmap_data = np.array(heatmap_data)
            
            # 绘制热力图
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # 设置标签
            ax.set_xticks(range(len(missing_rates)))
            ax.set_xticklabels([f'{mr:.1f}' for mr in missing_rates])
            ax.set_yticks(range(len(imputations)))
            ax.set_yticklabels(imputations)
            
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Imputation Method')
            ax.set_title(f'Mode={mode}, MIM={use_mim}')
            
            # 添加数值标注
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(heatmap_data[i, j]):
                        text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=7)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/heatmap_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: heatmap_{model}.png")


def plot_line_by_imputation(df, output_dir):
    """按插补方法绘制MAE随MR变化的线图"""
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    for imp in imputations:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Imputation={imp.upper()} - MAE vs Missing Rate', fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            for model in ['mlp', 'lstm', 'cnn']:
                for use_mim in ['false', 'true']:
                    mae_values = []
                    for mr in missing_rates:
                        subset = df[(df['model'] == model) & 
                                   (df['use_mim'] == use_mim) & 
                                   (df['mode'] == mode) & 
                                   (df['imputation'] == imp) & 
                                   (df['test_mr'] == mr)]
                        if len(subset) > 0:
                            mae_values.append(subset['test_mae'].values[0])
                        else:
                            mae_values.append(np.nan)
                    
                    linestyle = '-' if use_mim == 'false' else '--'
                    marker = 'o' if model == 'mlp' else ('s' if model == 'lstm' else '^')
                    label = f'{model.upper()}{"+MIM" if use_mim == "true" else ""}'
                    
                    ax.plot(missing_rates, mae_values, linestyle=linestyle, 
                           marker=marker, label=label, linewidth=2, markersize=6)
            
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('MAE')
            ax.set_title(f'Mode={mode}')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/line_{imp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: line_{imp}.png")


def plot_mim_improvement(df, output_dir):
    """绘制MIM改进百分比热力图"""
    imputations = ['zero', 'mean', 'knn', 'iterative']
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    for model in ['mlp', 'lstm', 'cnn']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model.upper()} - MIM Improvement (%)', fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            # 计算改进百分比
            improvement_data = []
            for imp in imputations:
                row = []
                for mr in missing_rates:
                    non_mim = df[(df['model'] == model) & 
                                (df['use_mim'] == 'false') & 
                                (df['mode'] == mode) & 
                                (df['imputation'] == imp) & 
                                (df['test_mr'] == mr)]
                    mim = df[(df['model'] == model) & 
                            (df['use_mim'] == 'true') & 
                            (df['mode'] == mode) & 
                            (df['imputation'] == imp) & 
                            (df['test_mr'] == mr)]
                    
                    if len(non_mim) > 0 and len(mim) > 0:
                        non_mim_mae = non_mim['test_mae'].values[0]
                        mim_mae = mim['test_mae'].values[0]
                        improvement = (non_mim_mae - mim_mae) / non_mim_mae * 100
                        row.append(improvement)
                    else:
                        row.append(np.nan)
                improvement_data.append(row)
            
            improvement_data = np.array(improvement_data)
            
            # 绘制热力图（改进为正表示MIM更好）
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
            
            # 添加数值标注
            for i in range(len(imputations)):
                for j in range(len(missing_rates)):
                    if not np.isnan(improvement_data[i, j]):
                        color = 'white' if abs(improvement_data[i, j]) > vmax * 0.5 else 'black'
                        text = ax.text(j, i, f'{improvement_data[i, j]:.0f}%',
                                     ha="center", va="center", color=color, fontsize=8, fontweight='bold')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Improvement %')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mim_improvement_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: mim_improvement_{model}.png")


def plot_overall_comparison(df, output_dir):
    """绘制整体对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Overall Performance Comparison (Single Seed)', fontsize=16, fontweight='bold')
    
    # 1. 各模型整体MAE对比
    ax = axes[0, 0]
    models = ['mlp', 'lstm', 'cnn']
    x = np.arange(len(models))
    width = 0.35
    
    non_mim_mae = []
    mim_mae = []
    for model in models:
        non_mim_mae.append(df[(df['model'] == model) & (df['use_mim'] == 'false')]['test_mae'].mean())
        mim_mae.append(df[(df['model'] == model) & (df['use_mim'] == 'true')]['test_mae'].mean())
    
    ax.bar(x - width/2, non_mim_mae, width, label='non-MIM')
    ax.bar(x + width/2, mim_mae, width, label='MIM')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average MAE by Model')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 各插补方法MAE对比
    ax = axes[0, 1]
    imputations = ['zero', 'mean', 'knn', 'iterative']
    x = np.arange(len(imputations))
    
    non_mim_mae = []
    mim_mae = []
    for imp in imputations:
        non_mim_mae.append(df[(df['imputation'] == imp) & (df['use_mim'] == 'false')]['test_mae'].mean())
        mim_mae.append(df[(df['imputation'] == imp) & (df['use_mim'] == 'true')]['test_mae'].mean())
    
    ax.bar(x - width/2, non_mim_mae, width, label='non-MIM')
    ax.bar(x + width/2, mim_mae, width, label='MIM')
    ax.set_ylabel('Average MAE')
    ax.set_title('Average MAE by Imputation Method')
    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in imputations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. High MR场景对比
    ax = axes[1, 0]
    high_mr = df[df['test_mr'] >= 0.6]
    
    x3 = np.arange(len(models))  # 使用单独的x
    non_mim_mae = []
    mim_mae = []
    for model in models:
        non_mim_mae.append(high_mr[(high_mr['model'] == model) & (high_mr['use_mim'] == 'false')]['test_mae'].mean())
        mim_mae.append(high_mr[(high_mr['model'] == model) & (high_mr['use_mim'] == 'true')]['test_mae'].mean())
    
    ax.bar(x3 - width/2, non_mim_mae, width, label='non-MIM')
    ax.bar(x3 + width/2, mim_mae, width, label='MIM')
    ax.set_ylabel('Average MAE')
    ax.set_title('High MR (>=0.6) - Average MAE by Model')
    ax.set_xticks(x3)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Zero Imputation High MR详细对比
    ax = axes[1, 1]
    high_mr_zero = df[(df['test_mr'] >= 0.6) & (df['imputation'] == 'zero')]
    
    x4 = np.arange(len(models))  # 使用单独的x
    non_mim_mae = []
    mim_mae = []
    improvements = []
    for model in models:
        nm = high_mr_zero[(high_mr_zero['model'] == model) & (high_mr_zero['use_mim'] == 'false')]['test_mae'].mean()
        m = high_mr_zero[(high_mr_zero['model'] == model) & (high_mr_zero['use_mim'] == 'true')]['test_mae'].mean()
        non_mim_mae.append(nm)
        mim_mae.append(m)
        improvements.append((nm - m) / nm * 100)
    
    ax.bar(x4 - width/2, non_mim_mae, width, label='non-MIM')
    ax.bar(x4 + width/2, mim_mae, width, label='MIM')
    
    # 在柱子上标注改进百分比
    for i, (nm, m, imp) in enumerate(zip(non_mim_mae, mim_mae, improvements)):
        ax.text(i, max(nm, m) + 0.01, f'{imp:+.1f}%', ha='center', va='bottom', 
                fontweight='bold', color='green' if imp > 0 else 'red')
    
    ax.set_ylabel('Average MAE')
    ax.set_title('High MR + Zero Imputation - MAE by Model')
    ax.set_xticks(x4)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: overall_comparison.png")


def plot_r2_comparison(df, output_dir):
    """绘制R2对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('R² Score Comparison by Missing Rate', fontsize=14, fontweight='bold')
    
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    for idx, model in enumerate(['mlp', 'lstm', 'cnn']):
        ax = axes[idx]
        
        for use_mim in ['false', 'true']:
            r2_values = []
            for mr in missing_rates:
                subset = df[(df['model'] == model) & 
                           (df['use_mim'] == use_mim) & 
                           (df['test_mr'] == mr)]
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
    plt.savefig(f'{output_dir}/r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: r2_comparison.png")


def main():
    # 设置路径
    json_path = 'results/new_config_test.json'
    output_dir = 'results/single_seed_figures'
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {json_path}...")
    df = load_data(json_path)
    print(f"Loaded {len(df)} records")
    
    # 生成图表
    print("\nGenerating figures...")
    
    print("\n1. Heatmaps by model...")
    plot_heatmap_by_model(df, output_dir)
    
    print("\n2. Line plots by imputation...")
    plot_line_by_imputation(df, output_dir)
    
    print("\n3. MIM improvement heatmaps...")
    plot_mim_improvement(df, output_dir)
    
    print("\n4. Overall comparison...")
    plot_overall_comparison(df, output_dir)
    
    print("\n5. R2 comparison...")
    plot_r2_comparison(df, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")
    print("Generated files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
