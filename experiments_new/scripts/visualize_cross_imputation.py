#!/usr/bin/env python3
"""
可视化交叉插补结果
生成热力图和对角线分析图
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def plot_heatmap(results_df: pd.DataFrame, output_path: Path):
    """绘制MAE热力图"""
    pivot = results_df.pivot_table(
        values='mae', 
        index='train_imputation', 
        columns='test_imputation',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'MAE'}, ax=ax,
                annot_kws={'size': 12})
    
    ax.set_xlabel('测试插补方法', fontsize=12)
    ax.set_ylabel('训练插补方法', fontsize=12)
    ax.set_title('交叉插补评估: 训练 × 测试 MAE', fontsize=14)
    
    # 高亮对角线（同种方法）
    for i in range(len(pivot.index)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                   edgecolor='blue', lw=3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"热力图已保存: {output_path}")


def plot_diagonal_analysis(results_df: pd.DataFrame, output_path: Path):
    """绘制对角线分析图（同种插补方法训练/测试 vs 不同方法）"""
    
    # 计算每种训练方法的平均性能
    train_avg = results_df.groupby('train_imputation')['mae'].mean()
    
    # 计算对角线（同种方法）的性能
    diagonal = results_df[results_df['train_imputation'] == results_df['test_imputation']]
    diagonal_avg = diagonal.groupby('train_imputation')['mae'].mean()
    
    # 计算每种训练方法在不同测试方法下的平均性能
    train_methods = results_df['train_imputation'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：每种训练方法的整体表现
    ax1 = axes[0]
    x_pos = np.arange(len(train_methods))
    ax1.bar(x_pos, train_avg[train_methods], color='skyblue', edgecolor='navy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(train_methods)
    ax1.set_ylabel('平均 MAE', fontsize=12)
    ax1.set_xlabel('训练插补方法', fontsize=12)
    ax1.set_title('训练方法平均性能', fontsize=14)
    ax1.axhline(y=train_avg.mean(), color='r', linestyle='--', 
                label=f'总体平均: {train_avg.mean():.4f}')
    ax1.legend()
    
    # 右图：同种方法 vs 不同方法
    ax2 = axes[1]
    width = 0.35
    x = np.arange(len(train_methods))
    
    # 同种方法
    same_method = [diagonal_avg[m] for m in train_methods]
    # 不同方法平均
    diff_method = []
    for m in train_methods:
        diff = results_df[(results_df['train_imputation'] == m) & 
                          (results_df['test_imputation'] != m)]['mae'].mean()
        diff_method.append(diff)
    
    ax2.bar(x - width/2, same_method, width, label='同种方法', color='green', alpha=0.7)
    ax2.bar(x + width/2, diff_method, width, label='不同方法平均', color='orange', alpha=0.7)
    
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_xlabel('训练插补方法', fontsize=12)
    ax2.set_title('同种方法 vs 不同方法性能对比', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(train_methods)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对角线分析图已保存: {output_path}")


def plot_test_method_comparison(results_df: pd.DataFrame, output_path: Path):
    """绘制不同测试插补方法的性能对比"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_methods = results_df['train_imputation'].unique()
    test_methods = results_df['test_imputation'].unique()
    
    x = np.arange(len(train_methods))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, test_method in enumerate(test_methods):
        values = [results_df[(results_df['train_imputation'] == tm) & 
                            (results_df['test_imputation'] == test_method)]['mae'].mean()
                  for tm in train_methods]
        ax.bar(x + i * width, values, width, label=f'测试: {test_method}', 
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_xlabel('训练插补方法', fontsize=12)
    ax.set_title('交叉插补性能对比', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(train_methods)
    ax.legend(title='测试插补方法', loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"测试方法对比图已保存: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化交叉插补结果")
    parser.add_argument("--results", required=True, help="结果CSV文件路径")
    parser.add_argument("--output-dir", default="results/multi_imp/figures", help="输出目录")
    
    args = parser.parse_args()
    
    results_df = pd.read_csv(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成图表
    plot_heatmap(results_df, output_dir / 'cross_imputation_heatmap.png')
    plot_diagonal_analysis(results_df, output_dir / 'cross_imputation_diagonal.png')
    plot_test_method_comparison(results_df, output_dir / 'cross_imputation_comparison.png')
    
    # 打印汇总
    print("\n" + "="*80)
    print("交叉插补实验结果汇总")
    print("="*80)
    
    pivot = results_df.pivot_table(
        values='mae', 
        index='train_imputation', 
        columns='test_imputation',
        aggfunc='mean'
    )
    print("\nMAE 矩阵:")
    print(pivot.round(4))
    
    # 找出最佳和最差组合
    best_idx = results_df['mae'].idxmin()
    worst_idx = results_df['mae'].idxmax()
    best = results_df.loc[best_idx]
    worst = results_df.loc[worst_idx]
    
    print(f"\n最佳组合: 训练={best['train_imputation']}, 测试={best['test_imputation']}, MAE={best['mae']:.4f}")
    print(f"最差组合: 训练={worst['train_imputation']}, 测试={worst['test_imputation']}, MAE={worst['mae']:.4f}")
    
    # 对角线分析
    diagonal = results_df[results_df['train_imputation'] == results_df['test_imputation']]
    print(f"\n同种方法训练/测试平均MAE: {diagonal['mae'].mean():.4f}")
    
    off_diagonal = results_df[results_df['train_imputation'] != results_df['test_imputation']]
    print(f"不同方法训练/测试平均MAE: {off_diagonal['mae'].mean():.4f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
