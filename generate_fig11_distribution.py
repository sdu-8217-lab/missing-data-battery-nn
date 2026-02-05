#!/usr/bin/env python3
"""
生成图11：MAE分布小提琴图（重绘版）
每MR一行，每行展示各模型Base vs MIM两组小提琴
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_data(csv_path):
    """加载实验结果数据"""
    df = pd.read_csv(csv_path)
    return df

def create_distribution_plot(df, output_path):
    """
    创建MAE分布小提琴图
    每行一个MR，展示各模型Base vs MIM的分布
    """
    models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
    model_names = ['MLP', 'LSTM', 'GRU', '1D-CNN']
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(len(missing_rates), 1, figsize=(14, 16))
    fig.suptitle('MAE Distribution Across Missing Rates\n(100 Random Seeds)', 
                 fontsize=16, fontweight='bold')
    
    for idx, mr in enumerate(missing_rates):
        ax = axes[idx]
        
        # 收集数据
        data_list = []
        position_labels = []
        colors = []
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            # Baseline
            baseline_df = df[(df['model_name'] == model) & (df['missing_rate'] == mr)]
            baseline_mae = baseline_df['mae'].values
            
            # MIM
            mim_df = df[(df['model_name'] == f'{model}-MIM') & (df['missing_rate'] == mr)]
            mim_mae = mim_df['mae'].values
            
            # 为每个模型添加两组数据
            for val in baseline_mae:
                data_list.append({'MAE': val, 'Model': model_name, 'Strategy': 'Baseline'})
            for val in mim_mae:
                data_list.append({'MAE': val, 'Model': model_name, 'Strategy': 'MIM'})
        
        df_plot = pd.DataFrame(data_list)
        
        # 创建小提琴图
        sns.violinplot(data=df_plot, x='Model', y='MAE', hue='Strategy', 
                       split=True, ax=ax, palette={'Baseline': '#E74C3C', 'MIM': '#3498DB'})
        
        ax.set_title(f'MR = {mr:.1f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('MAE', fontsize=11)
        if idx < len(missing_rates) - 1:
            ax.legend_.remove()
        else:
            ax.legend(title='Strategy', loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 数据路径
    csv_path = 'experiments_v2/3C/20260204_084913/results_all.csv'
    
    # 输出目录
    output_dir = 'my_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    
    # 生成图11
    output_path = os.path.join(output_dir, 'fig11_distribution_v2.png')
    create_distribution_plot(df, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
