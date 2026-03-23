"""
Zero填充MIM vs 所有non-MIM插补方法对比图
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

from pathlib import Path


def load_data(json_path):
    """加载实验结果数据"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return pd.DataFrame(results)


def plot_zero_mim_vs_all_non_mim(df, output_dir):
    """
    对比图：
    - Zero填充 + MIM (1条线)
    - 所有non-MIM插补方法 (4条线: zero, mean, knn, iterative)
    """
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['MCAR', 'MAR', 'MNAR']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Zero+MIM vs All non-MIM Imputation Methods (MAE vs Missing Rate)', 
                 fontsize=14, fontweight='bold')
    
    # 颜色定义
    colors = {
        'zero': '#1f77b4',      # 蓝色
        'mean': '#ff7f0e',      # 橙色
        'knn': '#2ca02c',       # 绿色
        'iterative': '#d62728', # 红色
        'zero_mim': '#9467bd'   # 紫色 (Zero+MIM)
    }
    
    linestyles = {
        'zero': '-',
        'mean': '-',
        'knn': '-',
        'iterative': '-',
        'zero_mim': '--'  # MIM用虚线
    }
    
    markers = {
        'zero': 'o',
        'mean': 's',
        'knn': '^',
        'iterative': 'v',
        'zero_mim': 'D'  # 菱形
    }
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # 1. 绘制所有non-MIM插补方法
        for imp in ['zero', 'mean', 'knn', 'iterative']:
            mae_values = []
            for mr in missing_rates:
                # non-MIM方法
                subset = df[(df['use_mim'] == 'false') & 
                           (df['imputation'] == imp) & 
                           (df['mode'] == mode) & 
                           (df['test_mr'] == mr)]
                if len(subset) > 0:
                    mae_values.append(subset['test_mae'].mean())
                else:
                    mae_values.append(np.nan)
            
            label = f'non-MIM+{imp.upper()}'
            ax.plot(missing_rates, mae_values, 
                   linestyle=linestyles[imp], 
                   marker=markers[imp], 
                   color=colors[imp],
                   label=label, 
                   linewidth=2, 
                   markersize=6,
                   alpha=0.8)
        
        # 2. 绘制Zero+MIM方法
        mae_values_mim = []
        for mr in missing_rates:
            subset = df[(df['use_mim'] == 'true') & 
                       (df['imputation'] == 'zero') & 
                       (df['mode'] == mode) & 
                       (df['test_mr'] == mr)]
            if len(subset) > 0:
                mae_values_mim.append(subset['test_mae'].mean())
            else:
                mae_values_mim.append(np.nan)
        
        ax.plot(missing_rates, mae_values_mim, 
               linestyle=linestyles['zero_mim'], 
               marker=markers['zero_mim'], 
               color=colors['zero_mim'],
               label='MIM+ZERO', 
               linewidth=3,  # 更粗以突出显示
               markersize=8,
               markerfacecolor='white',  # 空心标记
               markeredgewidth=2)
        
        ax.set_xlabel('Missing Rate', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'Mode={mode}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/zero_mim_vs_all_non_mim.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: zero_mim_vs_all_non_mim.png")


def plot_zero_mim_vs_all_non_mim_per_model(df, output_dir):
    """
    每个模型单独一张图：Zero+MIM vs 所有non-MIM插补
    """
    missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['MCAR', 'MAR', 'MNAR']
    models = ['mlp', 'lstm', 'cnn']
    
    colors = {
        'zero': '#1f77b4',
        'mean': '#ff7f0e',
        'knn': '#2ca02c',
        'iterative': '#d62728',
        'zero_mim': '#9467bd'
    }
    
    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{model.upper()}: Zero+MIM vs All non-MIM Imputation Methods', 
                     fontsize=14, fontweight='bold')
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            
            # 绘制所有non-MIM插补方法
            for imp in ['zero', 'mean', 'knn', 'iterative']:
                mae_values = []
                for mr in missing_rates:
                    subset = df[(df['model'] == model) &
                               (df['use_mim'] == 'false') & 
                               (df['imputation'] == imp) & 
                               (df['mode'] == mode) & 
                               (df['test_mr'] == mr)]
                    if len(subset) > 0:
                        mae_values.append(subset['test_mae'].mean())
                    else:
                        mae_values.append(np.nan)
                
                label = f'non-MIM+{imp.upper()}'
                ax.plot(missing_rates, mae_values, 
                       linestyle='-', 
                       marker='o', 
                       color=colors[imp],
                       label=label, 
                       linewidth=2, 
                       markersize=5,
                       alpha=0.7)
            
            # 绘制Zero+MIM
            mae_values_mim = []
            for mr in missing_rates:
                subset = df[(df['model'] == model) &
                           (df['use_mim'] == 'true') & 
                           (df['imputation'] == 'zero') & 
                           (df['mode'] == mode) & 
                           (df['test_mr'] == mr)]
                if len(subset) > 0:
                    mae_values_mim.append(subset['test_mae'].mean())
                else:
                    mae_values_mim.append(np.nan)
            
            ax.plot(missing_rates, mae_values_mim, 
                   linestyle='--', 
                   marker='D', 
                   color=colors['zero_mim'],
                   label='MIM+ZERO', 
                   linewidth=3, 
                   markersize=7,
                   markerfacecolor='white',
                   markeredgewidth=2)
            
            ax.set_xlabel('Missing Rate', fontsize=12)
            ax.set_ylabel('MAE', fontsize=12)
            ax.set_title(f'Mode={mode}', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/zero_mim_vs_non_mim_{model}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: zero_mim_vs_non_mim_{model}.png")


def main():
    json_path = 'results/new_config_test.json'
    output_dir = 'results/single_seed_figures'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {json_path}...")
    df = load_data(json_path)
    print(f"Loaded {len(df)} records")
    
    print("\nGenerating comparison plots...")
    
    print("\n1. Overall comparison (all models averaged)...")
    plot_zero_mim_vs_all_non_mim(df, output_dir)
    
    print("\n2. Per-model comparison...")
    plot_zero_mim_vs_all_non_mim_per_model(df, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
