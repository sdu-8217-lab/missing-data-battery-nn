#!/usr/bin/env python3
"""
生成九宫格对比图

图1: MAE曲线 - 同一batch、同一level下，三种缺失模式×三种架构
     每个子图展示四种插补方法的MIM vs Baseline对比

图2: 改进率 - MIM相对于Baseline的改进百分比

布局:
    行: 缺失模式 (MCAR, MAR, MNAR)
    列: 架构 (MLP, LSTM, CNN)
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.plot_batch_results import load_results_for_batch


def calculate_improvement(mim_data: pd.DataFrame, baseline_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算MIM相对于Baseline的改进率
    
    Returns:
        DataFrame with improvement percentage
    """
    # 按条件分组计算平均MAE
    mim_grouped = mim_data.groupby(['missing_rate', 'missing_mode', 'imputation_method'])['mae'].mean().reset_index()
    baseline_grouped = baseline_data.groupby(['missing_rate', 'missing_mode', 'imputation_method'])['mae'].mean().reset_index()
    
    # 合并数据
    merged = pd.merge(
        mim_grouped, 
        baseline_grouped,
        on=['missing_rate', 'missing_mode', 'imputation_method'],
        suffixes=('_mim', '_baseline')
    )
    
    # 计算改进率: (baseline - mim) / baseline * 100
    # 正值表示MIM更好（MAE更低）
    merged['improvement'] = (merged['mae_baseline'] - merged['mae_mim']) / merged['mae_baseline'] * 100
    
    return merged


def plot_mae_nine_grid(
    results: Dict[str, pd.DataFrame],
    batch: str,
    level: str,
    output_dir: Path
):
    """
    绘制MAE九宫格图
    
    行: MCAR, MAR, MNAR
    列: MLP, LSTM, CNN
    """
    missing_modes = ['MCAR', 'MAR', 'MNAR']
    architectures = ['mlp', 'lstm', 'cnn']
    imputation_methods = ['zero', 'mean', 'knn', 'iterative']
    
    colors = {
        'zero': '#1f77b4',
        'mean': '#ff7f0e', 
        'knn': '#2ca02c',
        'iterative': '#d62728'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'{batch} - {level.upper()}: MAE Comparison (MIM vs Baseline)\n', 
                 fontsize=16, fontweight='bold')
    
    for row_idx, missing_mode in enumerate(missing_modes):
        for col_idx, arch in enumerate(architectures):
            ax = axes[row_idx, col_idx]
            
            # 获取MIM和Baseline数据
            key_mim = (arch, level, 'mim')
            key_baseline = (arch, level, 'baseline')
            
            if key_mim not in results or key_baseline not in results:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            df_mim = results[key_mim]
            df_baseline = results[key_baseline]
            
            # 过滤当前缺失模式
            mim_mode_data = df_mim[df_mim['missing_mode'] == missing_mode]
            baseline_mode_data = df_baseline[df_baseline['missing_mode'] == missing_mode]
            
            # 绘制每种插补方法
            for imp_method in imputation_methods:
                mim_imp = mim_mode_data[mim_mode_data['imputation_method'] == imp_method]
                baseline_imp = baseline_mode_data[baseline_mode_data['imputation_method'] == imp_method]
                
                if not mim_imp.empty:
                    grouped_mim = mim_imp.groupby('missing_rate')['mae'].mean()
                    ax.plot(grouped_mim.index, grouped_mim.values, 
                           label=f'{imp_method} (MIM)', 
                           color=colors[imp_method], 
                           linestyle='-', marker='o', markersize=3, linewidth=2)
                
                if not baseline_imp.empty:
                    grouped_baseline = baseline_imp.groupby('missing_rate')['mae'].mean()
                    ax.plot(grouped_baseline.index, grouped_baseline.values, 
                           label=f'{imp_method} (Baseline)', 
                           color=colors[imp_method], 
                           linestyle='--', marker='s', markersize=3, linewidth=1.5, alpha=0.7)
            
            # 设置子图标题和标签
            if row_idx == 0:
                ax.set_title(f'{arch.upper()}', fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{missing_mode}\nMAE', fontsize=12)
            if row_idx == 2:
                ax.set_xlabel('Missing Rate', fontsize=12)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            # 只在第一行第一列显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    output_path = output_dir / batch / f'{level}_mae_nine_grid.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path}")


def plot_improvement_nine_grid(
    results: Dict[str, pd.DataFrame],
    batch: str,
    level: str,
    output_dir: Path
):
    """
    绘制改进率九宫格图
    
    行: MCAR, MAR, MNAR
    列: MLP, LSTM, CNN
    
    改进率 = (Baseline_MAE - MIM_MAE) / Baseline_MAE * 100
    正值表示MIM效果更好
    """
    missing_modes = ['MCAR', 'MAR', 'MNAR']
    architectures = ['mlp', 'lstm', 'cnn']
    imputation_methods = ['zero', 'mean', 'knn', 'iterative']
    
    colors = {
        'zero': '#1f77b4',
        'mean': '#ff7f0e',
        'knn': '#2ca02c',
        'iterative': '#d62728'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'{batch} - {level.upper()}: MIM Improvement Rate over Baseline (%)\n'
                 f'Positive = MIM Better', 
                 fontsize=16, fontweight='bold')
    
    # 添加零线参考
    for row_idx, missing_mode in enumerate(missing_modes):
        for col_idx, arch in enumerate(architectures):
            ax = axes[row_idx, col_idx]
            
            key_mim = (arch, level, 'mim')
            key_baseline = (arch, level, 'baseline')
            
            if key_mim not in results or key_baseline not in results:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 计算改进率
            improvement_df = calculate_improvement(
                results[key_mim], 
                results[key_baseline]
            )
            
            # 过滤当前缺失模式
            mode_data = improvement_df[improvement_df['missing_mode'] == missing_mode]
            
            # 绘制每种插补方法的改进率
            for imp_method in imputation_methods:
                imp_data = mode_data[mode_data['imputation_method'] == imp_method]
                
                if not imp_data.empty:
                    grouped = imp_data.groupby('missing_rate')['improvement'].mean()
                    ax.plot(grouped.index, grouped.values, 
                           label=imp_method, 
                           color=colors[imp_method], 
                           linestyle='-', marker='o', markersize=4, linewidth=2)
            
            # 添加零线
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # 添加阴影区域表示改进/恶化
            ax.fill_between([0, 1], 0, 100, alpha=0.1, color='green', label='MIM Better')
            ax.fill_between([0, 1], -100, 0, alpha=0.1, color='red', label='Baseline Better')
            
            # 设置子图标题和标签
            if row_idx == 0:
                ax.set_title(f'{arch.upper()}', fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{missing_mode}\nImprovement (%)', fontsize=12)
            if row_idx == 2:
                ax.set_xlabel('Missing Rate', fontsize=12)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            # 设置y轴范围，留一些边距
            ax.set_ylim(-50, 50)
            
            # 只在第一行第一列显示图例
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / batch / f'{level}_improvement_nine_grid.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成九宫格对比图")
    parser.add_argument("--results-dir", default="./results/full_scale",
                       help="结果目录")
    parser.add_argument("--output-dir", default="./outputs/nine_grid",
                       help="输出目录")
    parser.add_argument("--batches", nargs="+", 
                       default=["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"],
                       help="要绘制的batch列表")
    parser.add_argument("--levels", nargs="+", 
                       default=["level_1", "level_2", "level_3", "level_4"],
                       help="要绘制的级别列表")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("生成九宫格对比图")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Batches: {args.batches}")
    print(f"Levels: {args.levels}")
    print("=" * 60)
    
    for batch in args.batches:
        print(f"\n[{batch}] 处理中...")
        
        # 加载结果
        results = load_results_for_batch(results_dir, batch)
        
        if not results:
            print(f"  警告: 未找到结果")
            continue
        
        print(f"  加载了 {len(results)} 个结果文件")
        
        # 为每个level生成九宫格图
        for level in args.levels:
            print(f"  生成 {level} 九宫格图...")
            plot_mae_nine_grid(results, batch, level, output_dir)
            plot_improvement_nine_grid(results, batch, level, output_dir)
    
    print("\n" + "=" * 60)
    print("九宫格对比图生成完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
