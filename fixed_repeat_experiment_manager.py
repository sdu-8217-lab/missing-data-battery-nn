import os
import subprocess
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
import glob
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def run_single_experiment(config):
    """
    运行单次实验
    """
    cmd = [
        "python", config['script_path'],
        "--epochs", str(config['epochs']),
        "--data_dir", config['data_dir'],
        "--batch", config['batch'],
        "--seed", str(config['seed']),
        "--batch_size", str(config['batch_size']),
        "--results_dir", config['results_dir']
    ]
    
    if config.get('pretrained_model'):
        cmd.extend(["--pretrained_model", config['pretrained_model']])
    
    if config.get('training_missing_rates'):
        rates_str = " ".join(map(str, config['training_missing_rates']))
        cmd.extend(["--training_missing_rates"] + [str(rate) for rate in config['training_missing_rates']])
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"实验失败: {result.stderr}")
        return False
    
    print(f"实验完成: {config['results_dir']}")
    return True

def extract_results_from_csv(results_dir):
    """
    从结果目录提取CSV结果文件
    """
    csv_files = glob.glob(os.path.join(results_dir, "all_missing_combinations_*.csv"))
    if not csv_files:
        return None
    
    # 取最新的CSV文件
    latest_csv = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_csv)
    return df

def aggregate_results(results_dirs):
    """
    汇总多个实验结果
    """
    all_results = []
    
    for i, results_dir in enumerate(results_dirs):
        df = extract_results_from_csv(results_dir)
        if df is not None:
            df['experiment_id'] = i
            all_results.append(df)
        else:
            print(f"警告: 无法从 {results_dir} 提取结果")
    
    if not all_results:
        return None
    
    return pd.concat(all_results, ignore_index=True)

def plot_distribution_analysis(aggregated_df, output_dir):
    """
    绘制结果分布分析图
    """
    # 创建输出目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. MAE分布箱线图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Indicator MAE分布
    sns.boxplot(data=aggregated_df, y='Indicator_MAE', ax=axes[0,0])
    axes[0,0].set_title('Indicator Model MAE Distribution')
    axes[0,0].set_ylabel('MAE')
    
    # Reduced MAE分布
    sns.boxplot(data=aggregated_df, y='Reduced_MAE', ax=axes[0,1])
    axes[0,1].set_title('Reduced Model MAE Distribution')
    axes[0,1].set_ylabel('MAE')
    
    # RMSE分布
    sns.boxplot(data=aggregated_df[['Indicator_RMSE', 'Reduced_RMSE']], ax=axes[1,0])
    axes[1,0].set_title('RMSE Distribution Comparison')
    axes[1,0].set_ylabel('RMSE')
    
    # R²分布
    sns.boxplot(data=aggregated_df[['Indicator_R2', 'Reduced_R2']], ax=axes[1,1])
    axes[1,1].set_title('R² Distribution Comparison')
    axes[1,1].set_ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_rmse_r2_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 每个缺失组合的性能对比
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 按缺失组合分组
    grouped = aggregated_df.groupby('Missing_Combination').agg({
        'Indicator_MAE': ['mean', 'std'],
        'Reduced_MAE': ['mean', 'std']
    }).round(4)
    
    grouped.columns = ['Indicator_MAE_mean', 'Indicator_MAE_std', 'Reduced_MAE_mean', 'Reduced_MAE_std']
    grouped = grouped.head(20)  # 只显示前20个组合以便观察
    
    x = np.arange(len(grouped))
    width = 0.35
    
    ax.bar(x - width/2, grouped['Indicator_MAE_mean'], width, 
           label='Indicator Model', yerr=grouped['Indicator_MAE_std'], capsize=5, alpha=0.8)
    ax.bar(x + width/2, grouped['Reduced_MAE_mean'], width, 
           label='Reduced Model', yerr=grouped['Reduced_MAE_std'], capsize=5, alpha=0.8)
    
    ax.set_xlabel('Missing Combination')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Comparison by Missing Combination (with std error bars)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Comb{i}" for i in range(len(grouped))], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_comparison_by_combination.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 改进率分布
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(aggregated_df['Improvement_Percent'], bins=30, kde=True, ax=ax)
    ax.axvline(aggregated_df['Improvement_Percent'].mean(), color='red', linestyle='--', 
               label=f'Mean: {aggregated_df["Improvement_Percent"].mean():.2f}%')
    ax.set_xlabel('Improvement Percentage (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Improvement Percentages')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'improvement_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 性能散点图矩阵
    performance_cols = ['Indicator_MAE', 'Reduced_MAE', 'Indicator_RMSE', 'Reduced_RMSE', 'Indicator_R2', 'Reduced_R2']
    performance_df = aggregated_df[performance_cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(aggregated_df['Indicator_MAE'], aggregated_df['Reduced_MAE'], 
                        c=aggregated_df['Improvement_Percent'], cmap='viridis', alpha=0.6)
    ax.plot([aggregated_df['Indicator_MAE'].min(), aggregated_df['Indicator_MAE'].max()], 
            [aggregated_df['Indicator_MAE'].min(), aggregated_df['Indicator_MAE'].max()], 
            'r--', alpha=0.5, label='Perfect Match')
    ax.set_xlabel('Indicator Model MAE')
    ax.set_ylabel('Reduced Model MAE')
    ax.set_title('MAE Comparison: Indicator vs Reduced Models')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Improvement Percentage (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_scatter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_statistics(aggregated_df):
    """
    计算统计信息
    """
    stats = {}
    
    # 整体统计
    stats['overall'] = {
        'indicator_mae_mean': aggregated_df['Indicator_MAE'].mean(),
        'indicator_mae_std': aggregated_df['Indicator_MAE'].std(),
        'reduced_mae_mean': aggregated_df['Reduced_MAE'].mean(),
        'reduced_mae_std': aggregated_df['Reduced_MAE'].std(),
        'improvement_mean': aggregated_df['Improvement_Percent'].mean(),
        'improvement_std': aggregated_df['Improvement_Percent'].std()
    }
    
    # 按缺失组合分组统计
    grouped_stats = aggregated_df.groupby('Missing_Combination').agg({
        'Indicator_MAE': ['mean', 'std', 'min', 'max'],
        'Reduced_MAE': ['mean', 'std', 'min', 'max'],
        'Improvement_Percent': ['mean', 'std', 'min', 'max']
    })
    
    stats['by_combination'] = grouped_stats.round(4)
    
    return stats

def save_statistics(stats, output_dir):
    """
    保存统计信息到JSON和CSV
    """
    # 保存整体统计到JSON
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json_stats = {}
        for key, value in stats['overall'].items():
            if isinstance(value, (int, float)):
                json_stats[key] = round(value, 4)
            else:
                json_stats[key] = value
        json.dump(json_stats, f, indent=2)
    
    # 保存按组合统计到CSV
    if 'by_combination' in stats:
        stats['by_combination'].to_csv(os.path.join(output_dir, 'stats_by_combination.csv'))
    
    print(f"统计信息已保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='重复实验管理器')
    parser.add_argument('--script_path', type=str, required=True, 
                       help='下位实验脚本路径')
    parser.add_argument('--num_experiments', type=int, default=5,
                       help='重复实验次数')
    parser.add_argument('--base_results_dir', type=str, default='repeat_experiment_results',
                       help='基础结果目录')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--batch', type=str, default='3C',
                       choices=['2C','3C','R2.5','R3','RW','satellite'],
                       help='电池批次')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--seeds', type=int, nargs='+', 
                       help='随机种子列表 (如果不提供则自动生成)')
    
    args = parser.parse_args()
    
    # 创建主结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_results_dir = os.path.join(args.base_results_dir, f"repeat_experiment_{timestamp}")
    os.makedirs(main_results_dir, exist_ok=True)
    
    # 设置随机种子
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [42 + i for i in range(args.num_experiments)]
    
    if len(seeds) < args.num_experiments:
        seeds = seeds + [seeds[-1] + i + 1 for i in range(args.num_experiments - len(seeds))]
    
    print(f"将运行 {args.num_experiments} 次重复实验")
    print(f"使用种子: {seeds[:args.num_experiments]}")
    
    # 运行实验
    experiment_dirs = []
    for i in range(args.num_experiments):
        exp_results_dir = os.path.join(main_results_dir, f"experiment_{i+1}")
        os.makedirs(exp_results_dir, exist_ok=True)
        
        config = {
            'script_path': args.script_path,
            'epochs': args.epochs,
            'data_dir': args.data_dir,
            'batch': args.batch,
            'seed': seeds[i],
            'batch_size': args.batch_size,
            'results_dir': exp_results_dir
        }
        
        success = run_single_experiment(config)
        if success:
            experiment_dirs.append(exp_results_dir)
            print(f"实验 {i+1} 完成")
        else:
            print(f"实验 {i+1} 失败")
    
    # 汇总结果
    print("汇总实验结果...")
    aggregated_df = aggregate_results(experiment_dirs)
    
    if aggregated_df is not None:
        print(f"汇总了 {len(aggregated_df)} 个缺失组合的结果")
        
        # 保存汇总数据
        aggregated_df.to_csv(os.path.join(main_results_dir, 'aggregated_results.csv'), index=False)
        
        # 绘制分析图
        print("绘制分析图表...")
        plot_distribution_analysis(aggregated_df, main_results_dir)
        
        # 计算并保存统计信息
        print("计算统计信息...")
        stats = calculate_statistics(aggregated_df)
        save_statistics(stats, main_results_dir)
        
        # 打印摘要
        print("\n" + "="*60)
        print("重复实验摘要")
        print("="*60)
        overall = stats['overall']
        print(f"Indicator Model MAE: {overall['indicator_mae_mean']:.4f} ± {overall['indicator_mae_std']:.4f}")
        print(f"Reduced Model MAE: {overall['reduced_mae_mean']:.4f} ± {overall['reduced_mae_std']:.4f}")
        print(f"Improvement: {overall['improvement_mean']:.2f}% ± {overall['improvement_std']:.2f}%")
        print(f"实验总数: {args.num_experiments}")
        print(f"结果保存至: {main_results_dir}")
        print("="*60)
    else:
        print("未能汇总任何实验结果")

if __name__ == "__main__":
    main()
