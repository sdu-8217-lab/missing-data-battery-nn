#!/usr/bin/env python3
"""
对比粗粒度和细粒度搜索结果

使用示例:
    python compare_search_results.py --coarse results_coarse.csv --fine results_fine.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compare_results(coarse_path: str, fine_path: str, model_type: str):
    """对比粗粒度和细粒度搜索结果"""
    
    # 读取结果
    df_coarse = pd.read_csv(coarse_path)
    df_fine = pd.read_csv(fine_path)
    
    # 筛选指定模型
    df_coarse = df_coarse[df_coarse['model_type'] == model_type]
    df_fine = df_fine[df_fine['model_type'] == model_type]
    
    print("=" * 70)
    print(f"{model_type.upper()} 搜索结果对比")
    print("=" * 70)
    
    # 粗粒度最佳
    if len(df_coarse) > 0:
        best_coarse = df_coarse.loc[df_coarse['mae_mean'].idxmin()]
        print(f"\n【粗粒度最佳】")
        print(f"  配置: {best_coarse['config_name']}")
        print(f"  MAE: {best_coarse['mae_mean']:.4f}")
        print(f"  参数量: {best_coarse['param_count']:,}")
        print(f"  配置详情: {best_coarse['config']}")
    
    # 细粒度最佳
    if len(df_fine) > 0:
        best_fine = df_fine.loc[df_fine['mae_mean'].idxmin()]
        print(f"\n【细粒度最佳】")
        print(f"  配置: {best_fine['config_name']}")
        print(f"  MAE: {best_fine['mae_mean']:.4f}")
        print(f"  参数量: {best_fine['param_count']:,}")
        print(f"  配置详情: {best_fine['config']}")
        
        # 改进幅度
        if len(df_coarse) > 0:
            improvement = (best_coarse['mae_mean'] - best_fine['mae_mean']) / best_coarse['mae_mean'] * 100
            print(f"\n【改进幅度】")
            print(f"  MAE降低: {improvement:.2f}%")
            print(f"  绝对改进: {best_coarse['mae_mean'] - best_fine['mae_mean']:.4f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 散点图对比
    ax = axes[0]
    ax.scatter(df_coarse['param_count'], df_coarse['mae_mean'], 
              alpha=0.6, s=100, label='Coarse Search', color='blue')
    ax.scatter(df_fine['param_count'], df_fine['mae_mean'], 
              alpha=0.6, s=50, label='Fine Search', color='red')
    
    if len(df_coarse) > 0:
        ax.scatter(best_coarse['param_count'], best_coarse['mae_mean'], 
                  s=200, marker='*', color='blue', edgecolors='black', 
                  label='Coarse Best', zorder=5)
    if len(df_fine) > 0:
        ax.scatter(best_fine['param_count'], best_fine['mae_mean'], 
                  s=200, marker='*', color='red', edgecolors='black', 
                  label='Fine Best', zorder=5)
    
    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('MAE')
    ax.set_title(f'{model_type.upper()}: Coarse vs Fine Search')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE分布直方图
    ax = axes[1]
    if len(df_coarse) > 0 and len(df_fine) > 0:
        ax.hist(df_coarse['mae_mean'], bins=15, alpha=0.5, label='Coarse', color='blue')
        ax.hist(df_fine['mae_mean'], bins=20, alpha=0.5, label='Fine', color='red')
        ax.axvline(best_coarse['mae_mean'], color='blue', linestyle='--', 
                  label=f'Coarse Best: {best_coarse["mae_mean"]:.4f}')
        ax.axvline(best_fine['mae_mean'], color='red', linestyle='--',
                  label=f'Fine Best: {best_fine["mae_mean"]:.4f}')
        ax.set_xlabel('MAE')
        ax.set_ylabel('Count')
        ax.set_title('MAE Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(fine_path).parent / f'comparison_{model_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {output_path}")
    
    # 显示细粒度Top 10
    if len(df_fine) > 0:
        print("\n" + "=" * 70)
        print(f"细粒度搜索 Top 10 ({model_type.upper()})")
        print("=" * 70)
        top10 = df_fine.nsmallest(10, 'mae_mean')[['config_name', 'mae_mean', 'param_count']]
        print(top10.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='对比粗粒度和细粒度搜索结果')
    parser.add_argument('--coarse', required=True, help='粗粒度结果CSV路径')
    parser.add_argument('--fine', required=True, help='细粒度结果CSV路径')
    parser.add_argument('--model', default='all',
                       choices=['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost', 'all'],
                       help='要对比的模型')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        for model in ['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost']:
            compare_results(args.coarse, args.fine, model)
            print("\n" + "=" * 70 + "\n")
    else:
        compare_results(args.coarse, args.fine, args.model)


if __name__ == '__main__':
    main()
