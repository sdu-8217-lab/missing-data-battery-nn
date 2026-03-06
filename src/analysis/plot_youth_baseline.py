"""
Youth Experiment - Baseline Results Visualization
绘制 Baseline 实验结果图表
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def load_baseline_results(csv_path: Path) -> pd.DataFrame:
    """加载 Baseline 实验结果"""
    df = pd.read_csv(csv_path)
    print(f"[OK] 加载数据: {len(df)} 行")
    return df

def plot_mae_by_missing_rate(df: pd.DataFrame, output_dir: Path):
    """绘制 MAE 随缺失率变化图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        summary = model_data.groupby('missing_rate')['test_mae'].agg(['mean', 'std']).reset_index()
        
        ax.plot(summary['missing_rate'], summary['mean'], marker='o', label=model.upper(), linewidth=2)
        ax.fill_between(summary['missing_rate'], 
                        summary['mean'] - summary['std'],
                        summary['mean'] + summary['std'], 
                        alpha=0.2)
    
    ax.set_xlabel('Missing Rate (MR)', fontsize=12)
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    ax.set_title('Baseline: MAE vs Missing Rate', fontsize=14, fontweight='bold')
    ax.legend(title='Model', loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_mae_by_mr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存: {output_dir / 'baseline_mae_by_mr.png'}")

def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """绘制模型对比图（箱线图）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按模型分组绘制箱线图
    models = ['mlp', 'lstm', 'cnn']
    data_to_plot = [df[df['model'] == m]['test_mae'].values for m in models]
    
    bp = ax.boxplot(data_to_plot, tick_labels=[m.upper() for m in models], 
                    patch_artist=True, showmeans=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    ax.set_title('Baseline: Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计值标签
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]['test_mae']
        mean_val = model_data.mean()
        ax.text(i+1, mean_val, f'{mean_val:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存: {output_dir / 'baseline_model_comparison.png'}")

def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """绘制热力图：模型 × 缺失率"""
    pivot_table = df.pivot_table(
        values='test_mae', 
        index='model', 
        columns='missing_rate', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'MAE'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Baseline: MAE Heatmap (Model × Missing Rate)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存: {output_dir / 'baseline_heatmap.png'}")

def plot_batch_comparison(df: pd.DataFrame, output_dir: Path):
    """绘制不同 batch 的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, batch in enumerate(df['batch_id'].unique()):
        ax = axes[idx]
        batch_data = df[df['batch_id'] == batch]
        
        for model in batch_data['model'].unique():
            model_data = batch_data[batch_data['model'] == model]
            summary = model_data.groupby('missing_rate')['test_mae'].mean().reset_index()
            ax.plot(summary['missing_rate'], summary['test_mae'], 
                   marker='o', label=model.upper(), linewidth=2)
        
        ax.set_xlabel('Missing Rate (MR)', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'Batch {batch.upper()}', fontsize=12, fontweight='bold')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Baseline: Performance by Batch', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_batch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存: {output_dir / 'baseline_batch_comparison.png'}")

def plot_r2_by_missing_rate(df: pd.DataFrame, output_dir: Path):
    """绘制 R² 随缺失率变化图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        summary = model_data.groupby('missing_rate')['test_r2'].agg(['mean', 'std']).reset_index()
        
        ax.plot(summary['missing_rate'], summary['mean'], marker='s', label=model.upper(), linewidth=2)
        ax.fill_between(summary['missing_rate'], 
                        summary['mean'] - summary['std'],
                        summary['mean'] + summary['std'], 
                        alpha=0.2)
    
    ax.set_xlabel('Missing Rate (MR)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Baseline: R² vs Missing Rate', fontsize=14, fontweight='bold')
    ax.legend(title='Model', loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_r2_by_mr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 保存: {output_dir / 'baseline_r2_by_mr.png'}")

def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """生成汇总统计表并保存为图片"""
    summary = df.groupby(['model', 'missing_rate'])['test_mae'].agg(['mean', 'std', 'count']).round(4)
    
    # 保存为 CSV
    summary.to_csv(output_dir / 'baseline_summary_stats.csv')
    print(f"[OK] 保存统计表: {output_dir / 'baseline_summary_stats.csv'}")
    
    # 打印到控制台
    print("\n" + "="*70)
    print("Baseline 实验汇总统计")
    print("="*70)
    print(summary.to_string())
    
    # 按模型汇总
    print("\n" + "-"*70)
    print("按模型汇总 (MAE)")
    print("-"*70)
    model_summary = df.groupby('model')['test_mae'].agg(['mean', 'std']).round(4)
    print(model_summary.to_string())
    
    # 按缺失率汇总
    print("\n" + "-"*70)
    print("按缺失率汇总 (MAE)")
    print("-"*70)
    mr_summary = df.groupby('missing_rate')['test_mae'].agg(['mean', 'std']).round(4)
    print(mr_summary.to_string())

def main():
    """主函数"""
    print("="*70)
    print("Youth Experiment - Baseline 结果可视化")
    print("="*70)
    
    # 路径设置
    csv_path = Path("results/csv/youth_mar_baseline.csv")
    output_dir = Path("results/figures/youth_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"\n加载数据: {csv_path}")
    df = load_baseline_results(csv_path)
    
    # 生成汇总统计
    print("\n生成汇总统计...")
    generate_summary_table(df, output_dir)
    
    # 绘制图表
    print("\n绘制图表...")
    plot_mae_by_missing_rate(df, output_dir)
    plot_model_comparison(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_batch_comparison(df, output_dir)
    plot_r2_by_missing_rate(df, output_dir)
    
    print("\n" + "="*70)
    print(f"[完成] 所有图表已保存到: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
