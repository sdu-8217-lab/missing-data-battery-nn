"""缺失率-性能曲线"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_missing_rate_curves(results_df: pd.DataFrame, save_path: str = None):
    """
    绘制缺失率-性能曲线
    
    Args:
        results_df: 结果DataFrame
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['mae', 'rmse', 'r2']
    titles = ['MAE vs Missing Rate', 'RMSE vs Missing Rate', 'R² vs Missing Rate']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # 按模型分组
        for model_name in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model_name]
            
            # 按缺失率聚合
            grouped = model_data.groupby('missing_rate')[metric].agg(['mean', 'std'])
            
            ax.plot(grouped.index, grouped['mean'], marker='o', label=model_name)
            ax.fill_between(
                grouped.index,
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                alpha=0.2
            )
        
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
