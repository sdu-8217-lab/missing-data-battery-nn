#!/usr/bin/env python3
"""
分层绘图脚本
生成每个batch的可视化结果，按参数量级别分组

结构:
    outputs/batch_plots/
        {batch}/
            level_comparison/           # 所有级别对比
                mlp_mim_vs_baseline.png
                lstm_mim_vs_baseline.png
                cnn_mim_vs_baseline.png
            by_level/                   # 每个级别独立图
                level_1_comparison.png  # MLP/LSTM/CNN × MIM/Baseline
                level_2_comparison.png
                ...
            architecture_comparison/    # 架构间对比
                mim_comparison.png      # 所有架构MIM模式对比
                baseline_comparison.png
"""
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: No external plotter dependency, using matplotlib directly


# 实验矩阵定义
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
ARCHITECTURES = ["mlp", "lstm", "cnn"]
LEVELS = ["level_1", "level_2", "level_3", "level_4"]


def load_results_for_batch(
    results_dir: Path,
    batch: str,
    architectures: List[str] = None,
    levels: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    加载某个batch的所有结果
    实际路径结构: results/full_scale/{batch}_{arch}_{level}_{mim_mode}/batch_{batch}_{arch}_{mim_mode}/{batch}/{timestamp}/test_results.csv
    
    Returns:
        {(arch, level, mim_mode): DataFrame}
    """
    if architectures is None:
        architectures = ARCHITECTURES
    if levels is None:
        levels = LEVELS
    
    results = {}
    
    print(f"  调试: results_dir = {results_dir.absolute()}")
    print(f"  调试: batch = {batch}")
    
    # 搜索所有子目录中的test_results.csv
    files = list(results_dir.rglob("test_results.csv"))
    print(f"  调试: 找到 {len(files)} 个test_results.csv文件")
    
    for idx, results_file in enumerate(files):
        try:
            # 从路径中提取信息 - 使用相对于results_dir的路径
            try:
                rel_path = results_file.relative_to(results_dir)
                parts = rel_path.parts
            except ValueError:
                # 如果无法相对化，使用绝对路径
                parts = results_file.parts
            
            # 查找包含 batch 名称的顶级目录 (例如: "3C_lstm_level_1_baseline")
            # 使用相对于results_dir的路径后，parts[0]就是实验目录
            if len(parts) < 1:
                continue
                
            exp_dir = parts[0]  # e.g., "3C_lstm_level_2_mim" or "R2.5_mlp_level_1_baseline"
            
            if exp_dir.startswith(f"{batch}_"):
                # 解析目录名: {batch}_{arch}_{level}_{num}_{mim_mode}
                name_parts = exp_dir.split('_')
                
                # 从后向前解析
                if len(name_parts) >= 5:
                    mim_mode = name_parts[-1]
                    level_num = name_parts[-2]
                    level = f"level_{level_num}"
                    arch = name_parts[-4]
                    potential_batch = '_'.join(name_parts[:-4])
                    
                    # 调试: 打印第一个匹配项
                    if len(results) == 0:
                        print(f"    调试: exp_dir={exp_dir}")
                        print(f"    调试: batch={potential_batch}, arch={arch}, level={level}, mim={mim_mode}")
                        print(f"    调试: arch in architectures: {arch in architectures}")
                        print(f"    调试: level in levels: {level in levels}")
                    
                    # 验证
                    if (potential_batch == batch and 
                        arch in architectures and 
                        level in levels and 
                        mim_mode in ['mim', 'baseline']):
                        
                        df = pd.read_csv(results_file)
                        key = (arch, level, mim_mode)
                        results[key] = df
        except Exception as e:
            print(f"  警告: 无法加载 {results_file}: {e}")
    
    
    return results


def plot_level_comparison(
    results: Dict[str, pd.DataFrame],
    batch: str,
    architecture: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    绘制同一架构所有级别的MIM vs Baseline对比图
    2x2子图，每个子图一个级别
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = {'mim': '#1f77b4', 'baseline': '#ff7f0e'}
    linestyles = {'MCAR': '-', 'MAR': '--', 'MNAR': ':'}
    
    for idx, level in enumerate(LEVELS):
        ax = axes[idx]
        
        # 获取该级别的MIM和Baseline数据
        key_mim = (architecture, level, 'mim')
        key_base = (architecture, level, 'baseline')
        
        has_data = False
        
        for mode in ['mim', 'baseline']:
            key = (architecture, level, mode)
            if key not in results:
                continue
            
            df = results[key]
            
            # 按缺失模式分组绘图
            for missing_mode in ['MCAR', 'MAR', 'MNAR']:
                mode_data = df[df['missing_mode'] == missing_mode]
                if mode_data.empty:
                    continue
                
                has_data = True
                grouped = mode_data.groupby('missing_rate')['mae'].mean()
                
                label = f"{mode.upper()}-{missing_mode}"
                ax.plot(
                    grouped.index, grouped.values,
                    label=label,
                    color=colors[mode],
                    linestyle=linestyles[missing_mode],
                    marker='o', markersize=3
                )
        
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('MAE')
        ax.set_title(f'{level.upper().replace("_", " ")}')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle(f'{batch} - {architecture.upper()}: MIM vs Baseline (All Levels)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / batch / 'level_comparison' / f'{architecture}_mim_vs_baseline.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path}")


def plot_by_level(
    results: Dict[str, pd.DataFrame],
    batch: str,
    level: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    绘制同一级别的所有架构对比
    3x2子图: 行=架构(MLP/LSTM/CNN), 列=MIM/Baseline
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    linestyles = {'MCAR': '-', 'MAR': '--', 'MNAR': ':'}
    colors_by_mode = {'MCAR': '#1f77b4', 'MAR': '#ff7f0e', 'MNAR': '#2ca02c'}
    
    for arch_idx, architecture in enumerate(ARCHITECTURES):
        for mode_idx, mim_mode in enumerate(['mim', 'baseline']):
            ax = axes[arch_idx, mode_idx]
            
            key = (architecture, level, mim_mode)
            
            if key not in results:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{architecture.upper()} - {mim_mode.upper()}')
                continue
            
            df = results[key]
            
            # 绘制所有缺失模式
            for missing_mode in ['MCAR', 'MAR', 'MNAR']:
                mode_data = df[df['missing_mode'] == missing_mode]
                if mode_data.empty:
                    continue
                
                grouped = mode_data.groupby('missing_rate')['mae'].mean()
                
                ax.plot(
                    grouped.index, grouped.values,
                    label=missing_mode,
                    color=colors_by_mode[missing_mode],
                    linestyle=linestyles[missing_mode],
                    marker='o', markersize=4
                )
            
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('MAE')
            ax.set_title(f'{architecture.upper()} - {mim_mode.upper()}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
    
    fig.suptitle(f'{batch} - {level.upper().replace("_", " ")}: All Architectures Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / batch / 'by_level' / f'{level}_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path}")


def plot_architecture_comparison(
    results: Dict[str, pd.DataFrame],
    batch: str,
    mim_mode: str,
    level: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    绘制所有架构的对比图（固定MIM模式和级别）
    3x1子图: 每个缺失模式一个子图
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    colors = {'mlp': '#1f77b4', 'lstm': '#ff7f0e', 'cnn': '#2ca02c'}
    
    for idx, missing_mode in enumerate(['MCAR', 'MAR', 'MNAR']):
        ax = axes[idx]
        
        for architecture in ARCHITECTURES:
            key = (architecture, level, mim_mode)
            
            if key not in results:
                continue
            
            df = results[key]
            mode_data = df[df['missing_mode'] == missing_mode]
            
            if mode_data.empty:
                continue
            
            grouped = mode_data.groupby('missing_rate')['mae'].mean()
            
            ax.plot(
                grouped.index, grouped.values,
                label=architecture.upper(),
                color=colors[architecture],
                marker='o', markersize=4
            )
        
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('MAE')
        ax.set_title(f'{missing_mode}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    fig.suptitle(f'{batch} - {level.upper().replace("_", " ")} - {mim_mode.upper()}: Architecture Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / batch / 'architecture_comparison' / f'{level}_{mim_mode}_arch_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path}")


def create_batch_summary(
    results: Dict[str, pd.DataFrame],
    batch: str,
    output_dir: Path
):
    """创建batch级别的汇总统计"""
    summary = []
    
    for (arch, level, mim_mode), df in results.items():
        for missing_mode in ['MCAR', 'MAR', 'MNAR']:
            mode_data = df[df['missing_mode'] == missing_mode]
            if mode_data.empty:
                continue
            
            stats = {
                'batch': batch,
                'architecture': arch,
                'level': level,
                'mode': mim_mode,
                'missing_mode': missing_mode,
                'mean_mae': mode_data['mae'].mean(),
                'std_mae': mode_data['mae'].std(),
                'max_mae': mode_data['mae'].max(),
                'min_mae': mode_data['mae'].min(),
            }
            
            # 计算高缺失率下的表现 (MR >= 0.5)
            high_mr_data = mode_data[mode_data['missing_rate'] >= 0.5]
            if not high_mr_data.empty:
                stats['high_mr_mean_mae'] = high_mr_data['mae'].mean()
            
            summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    
    output_path = output_dir / batch / f'{batch}_summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"  保存汇总: {output_path}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="批量绘图")
    parser.add_argument("--results-dir", default="./results/batch_runs",
                       help="结果目录")
    parser.add_argument("--output-dir", default="./outputs/batch_plots",
                       help="输出目录")
    parser.add_argument("--batches", nargs="+", default=BATCHES,
                       help="要绘制的batch列表")
    parser.add_argument("--architectures", nargs="+", default=ARCHITECTURES,
                       help="要绘制的架构列表")
    parser.add_argument("--levels", nargs="+", default=LEVELS,
                       help="要绘制的级别列表")
    parser.add_argument("--plots", nargs="+", 
                       choices=["level_comparison", "by_level", "arch_comparison", "all"],
                       default=["all"],
                       help="要生成的图表类型")
    
    args = parser.parse_args()
    
    results_dir = Path(__file__).parent.parent / args.results_dir
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_types = args.plots
    if "all" in plot_types:
        plot_types = ["level_comparison", "by_level", "arch_comparison"]
    
    print("=" * 60)
    print("分层绘图")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Batches: {args.batches}")
    print(f"图表类型: {plot_types}")
    print("=" * 60)
    
    for batch in args.batches:
        print(f"\n[{batch}] 处理中...")
        
        # 加载结果
        print(f"  调试: architectures={args.architectures}")
        print(f"  调试: levels={args.levels}")
        results = load_results_for_batch(
            results_dir, batch, args.architectures, args.levels
        )
        
        if not results:
            print(f"  警告: 未找到结果")
            continue
        
        print(f"  加载了 {len(results)} 个结果文件")
        
        # 创建汇总
        create_batch_summary(results, batch, output_dir)
        
        # 绘制图表
        if "level_comparison" in plot_types:
            print(f"  生成级别对比图...")
            for arch in args.architectures:
                plot_level_comparison(results, batch, arch, output_dir)
        
        if "by_level" in plot_types:
            print(f"  生成按级别分组图...")
            for level in args.levels:
                plot_by_level(results, batch, level, output_dir)
        
        if "arch_comparison" in plot_types:
            print(f"  生成架构对比图...")
            for level in args.levels:
                for mim_mode in ['mim', 'baseline']:
                    plot_architecture_comparison(results, batch, mim_mode, level, output_dir)
    
    print("\n" + "=" * 60)
    print("绘图完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
