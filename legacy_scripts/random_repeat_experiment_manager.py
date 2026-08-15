#!/usr/bin/env python3
"""
SOH估计重复实验管理与分析代码
用于运行多次实验并分析结果分布
"""

import os
import subprocess
import sys
import json
import glob
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import datetime
import logging
from typing import List, Dict, Any
import re

# =========================
# 1. 日志配置
# =========================
def setup_logging(timestamp):
    """设置日志配置"""
    # 配置日志处理器以支持UTF-8编码
    class UTF8FileHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
            super().__init__(filename, mode, encoding, delay)

    # 创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 文件处理器（UTF-8编码）
    log_file = f'repeat_experiment_{timestamp}.log'
    file_handler = UTF8FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# =========================
# 2. 实验参数配置
# =========================
def get_default_params():
    """获取默认参数配置"""
    return {
        'missing_rates': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'training_missing_rates': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        'epochs': 100,
        'data_dir': './data/XJTU data',
        'batch': '2C',
        'batch_size': 32,
        'results_dir': 'repeat_experiment_results'
    }

def generate_seeds(n_experiments, base_seed=42):
    """生成不同的随机种子"""
    np.random.seed(base_seed)
    seeds = []
    for i in range(n_experiments):
        seed = np.random.randint(1, 10000)
        seeds.append(seed)
    return seeds

# =========================
# 3. 运行实验函数
# =========================
def run_single_experiment(params: Dict[str, Any], experiment_id: int, timestamp: str):
    """运行单次实验"""
    # 创建实验特定的目录
    exp_dir = f"exp_{timestamp}_id{experiment_id}"
    params_copy = params.copy()
    params_copy['results_dir'] = exp_dir
    
    # 构建命令
    cmd = [
        sys.executable, 'soh_random_missing_analysis.py', 
        '--missing_rates'
    ] + [str(r) for r in params_copy['missing_rates']]
    cmd += ['--training_missing_rates'] + [str(r) for r in params_copy['training_missing_rates']]
    cmd += ['--epochs', str(params_copy['epochs'])]
    cmd += ['--data_dir', params_copy['data_dir']]
    cmd += ['--batch', params_copy['batch']]
    cmd += ['--seed', str(params_copy['seed'])]
    cmd += ['--batch_size', str(params_copy['batch_size'])]
    cmd += ['--results_dir', params_copy['results_dir']]
    
    print(f"Running experiment {experiment_id}: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode != 0:
            print(f"Experiment {experiment_id} failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
            
        # 查找生成的CSV结果文件
        csv_pattern = f"{exp_dir}/experiment_results_*.csv"
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"No CSV results found for experiment {experiment_id}")
            return None
            
        # 返回结果文件路径
        return csv_files[0]
        
    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_id} timed out")
        return None
    except Exception as e:
        print(f"Error running experiment {experiment_id}: {e}")
        return None

def run_multiple_experiments(params: Dict[str, Any], n_experiments: int, timestamp: str):
    """运行多个实验"""
    seeds = generate_seeds(n_experiments)
    results = {}
    
    print(f"Starting {n_experiments} experiments...")
    
    for i in range(n_experiments):
        print(f"\nRunning experiment {i+1}/{n_experiments}")
        
        # 更新参数中的种子
        current_params = params.copy()
        current_params['seed'] = seeds[i]
        
        # 运行单次实验
        result_file = run_single_experiment(current_params, i+1, timestamp)
        
        if result_file:
            results[f'exp_{i+1}'] = result_file
            print(f"Experiment {i+1} completed successfully: {result_file}")
        else:
            print(f"Experiment {i+1} failed")
    
    return results

# =========================
# 4. 结果分析与可视化
# =========================
def load_experiment_results(result_files: Dict[str, str]):
    """加载所有实验结果"""
    all_results = {}
    
    for exp_name, csv_file in result_files.items():
        try:
            df = pd.read_csv(csv_file)
            all_results[exp_name] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return all_results

def calculate_statistics(all_results: Dict[str, pd.DataFrame]):
    """计算统计信息"""
    # 获取所有缺失率
    missing_rates = list(all_results.values())[0]['Missing_Rate'].values
    
    # 初始化统计字典
    stats_dict = {
        'missing_rates': missing_rates,
        'indicator_mae_mean': [],
        'indicator_mae_std': [],
        'indicator_rmse_mean': [],
        'indicator_rmse_std': [],
        'indicator_r2_mean': [],
        'indicator_r2_std': [],
        'fill_mae_mean': [],
        'fill_mae_std': [],
        'fill_rmse_mean': [],
        'fill_rmse_std': [],
        'fill_r2_mean': [],
        'fill_r2_std': [],
        'improvement_mean': [],
        'improvement_std': [],
        'baseline_mae_mean': [],
        'baseline_rmse_mean': [],
        'baseline_r2_mean': []
    }
    
    # 计算每种缺失率下的统计信息
    for i, mr in enumerate(missing_rates):
        indicator_mae_vals = []
        indicator_rmse_vals = []
        indicator_r2_vals = []
        fill_mae_vals = []
        fill_rmse_vals = []
        fill_r2_vals = []
        improvement_vals = []
        
        baseline_mae_vals = []
        baseline_rmse_vals = []
        baseline_r2_vals = []
        
        for exp_name, df in all_results.items():
            row = df[df['Missing_Rate'] == mr]
            if not row.empty:
                indicator_mae_vals.append(row['Indicator_MAE'].iloc[0])
                indicator_rmse_vals.append(row['Indicator_RMSE'].iloc[0])
                indicator_r2_vals.append(row['Indicator_R2'].iloc[0])
                fill_mae_vals.append(row['Fill_MAE'].iloc[0])
                fill_rmse_vals.append(row['Fill_RMSE'].iloc[0])
                fill_r2_vals.append(row['Fill_R2'].iloc[0])
                improvement_vals.append(row['Improvement_Percent'].iloc[0])
            
            # 基线值（相同，只需取一次）
            if not baseline_mae_vals:  # 只需取第一个实验的基线值
                baseline_mae_vals.append(row['Baseline_MAE'].iloc[0])
                baseline_rmse_vals.append(row['Baseline_RMSE'].iloc[0])
                baseline_r2_vals.append(row['Baseline_R2'].iloc[0])
        
        # 计算统计值
        stats_dict['indicator_mae_mean'].append(np.mean(indicator_mae_vals))
        stats_dict['indicator_mae_std'].append(np.std(indicator_mae_vals))
        stats_dict['indicator_rmse_mean'].append(np.mean(indicator_rmse_vals))
        stats_dict['indicator_rmse_std'].append(np.std(indicator_rmse_vals))
        stats_dict['indicator_r2_mean'].append(np.mean(indicator_r2_vals))
        stats_dict['indicator_r2_std'].append(np.std(indicator_r2_vals))
        stats_dict['fill_mae_mean'].append(np.mean(fill_mae_vals))
        stats_dict['fill_mae_std'].append(np.std(fill_mae_vals))
        stats_dict['fill_rmse_mean'].append(np.mean(fill_rmse_vals))
        stats_dict['fill_rmse_std'].append(np.std(fill_rmse_vals))
        stats_dict['fill_r2_mean'].append(np.mean(fill_r2_vals))
        stats_dict['fill_r2_std'].append(np.std(fill_r2_vals))
        stats_dict['improvement_mean'].append(np.mean(improvement_vals))
        stats_dict['improvement_std'].append(np.std(improvement_vals))
    
    # 基线值只需要一次计算（因为对所有实验都一样）
    stats_dict['baseline_mae_mean'] = [np.mean(baseline_mae_vals)] * len(missing_rates)
    stats_dict['baseline_rmse_mean'] = [np.mean(baseline_rmse_vals)] * len(missing_rates)
    stats_dict['baseline_r2_mean'] = [np.mean(baseline_r2_vals)] * len(missing_rates)
    
    return stats_dict

def plot_repeated_experiment_results(stats_dict, n_experiments, save_dir):
    """绘制重复实验结果"""
    missing_rates = stats_dict['missing_rates']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE对比
    ax1 = axes[0, 0]
    ax1.errorbar(missing_rates, stats_dict['indicator_mae_mean'], 
                 yerr=stats_dict['indicator_mae_std'], 
                 fmt='o-', label='Missing Indicators', capsize=5, capthick=2)
    ax1.errorbar(missing_rates, stats_dict['fill_mae_mean'], 
                 yerr=stats_dict['fill_mae_std'], 
                 fmt='s--', label='Mean Filling (Base Model)', capsize=5, capthick=2)
    ax1.axhline(y=stats_dict['baseline_mae_mean'][0], color='k', linestyle='-', 
                label='Baseline (Complete Data)', linewidth=2)
    ax1.set_xlabel('Missing Rate')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE vs Missing Rate (with error bars)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE对比
    ax2 = axes[0, 1]
    ax2.errorbar(missing_rates, stats_dict['indicator_rmse_mean'], 
                 yerr=stats_dict['indicator_rmse_std'], 
                 fmt='o-', label='Missing Indicators', capsize=5, capthick=2)
    ax2.errorbar(missing_rates, stats_dict['fill_rmse_mean'], 
                 yerr=stats_dict['fill_rmse_std'], 
                 fmt='s--', label='Mean Filling (Base Model)', capsize=5, capthick=2)
    ax2.axhline(y=stats_dict['baseline_rmse_mean'][0], color='k', linestyle='-', 
                label='Baseline (Complete Data)', linewidth=2)
    ax2.set_xlabel('Missing Rate')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Missing Rate (with error bars)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # R²对比
    ax3 = axes[1, 0]
    ax3.errorbar(missing_rates, stats_dict['indicator_r2_mean'], 
                 yerr=stats_dict['indicator_r2_std'], 
                 fmt='o-', label='Missing Indicators', capsize=5, capthick=2)
    ax3.errorbar(missing_rates, stats_dict['fill_r2_mean'], 
                 yerr=stats_dict['fill_r2_std'], 
                 fmt='s--', label='Mean Filling (Base Model)', capsize=5, capthick=2)
    ax3.axhline(y=stats_dict['baseline_r2_mean'][0], color='k', linestyle='-', 
                label='Baseline (Complete Data)', linewidth=2)
    ax3.set_xlabel('Missing Rate')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² Score vs Missing Rate (with error bars)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Improvement对比
    ax4 = axes[1, 1]
    ax4.errorbar(missing_rates, stats_dict['improvement_mean'], 
                 yerr=stats_dict['improvement_std'], 
                 fmt='D-', label='Improvement', capsize=5, capthick=2, color='purple')
    ax4.set_xlabel('Missing Rate')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title(f'Improvement of Missing Indicators over Mean Filling\n(Average across {n_experiments} experiments)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/repeated_experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_box_plots(all_results, save_dir):
    """绘制箱线图展示结果分布"""
    # 收集数据
    metrics_data = {
        'MAE': [],
        'RMSE': [],
        'R2': [],
        'Method': [],
        'Missing_Rate': []
    }
    
    for exp_name, df in all_results.items():
        for _, row in df.iterrows():
            # Missing Indicators
            metrics_data['MAE'].append(row['Indicator_MAE'])
            metrics_data['RMSE'].append(row['Indicator_RMSE'])
            metrics_data['R2'].append(row['Indicator_R2'])
            metrics_data['Method'].append('Missing Indicators')
            metrics_data['Missing_Rate'].append(row['Missing_Rate'])
            
            # Fill Method
            metrics_data['MAE'].append(row['Fill_MAE'])
            metrics_data['RMSE'].append(row['Fill_RMSE'])
            metrics_data['R2'].append(row['Fill_R2'])
            metrics_data['Method'].append('Mean Filling')
            metrics_data['Missing_Rate'].append(row['Missing_Rate'])
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAE箱线图
    sns.boxplot(data=df_metrics, x='Missing_Rate', y='MAE', hue='Method', ax=axes[0])
    axes[0].set_title('MAE Distribution Across Missing Rates')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE箱线图
    sns.boxplot(data=df_metrics, x='Missing_Rate', y='RMSE', hue='Method', ax=axes[1])
    axes[1].set_title('RMSE Distribution Across Missing Rates')
    axes[1].tick_params(axis='x', rotation=45)
    
    # R²箱线图
    sns.boxplot(data=df_metrics, x='Missing_Rate', y='R2', hue='Method', ax=axes[2])
    axes[2].set_title('R² Distribution Across Missing Rates')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/box_plots_repeated_experiments.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_intervals(stats_dict, n_experiments, save_dir):
    """绘制置信区间图"""
    missing_rates = stats_dict['missing_rates']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAE置信区间
    ax1 = axes[0]
    indicator_lower = np.array(stats_dict['indicator_mae_mean']) - 1.96 * np.array(stats_dict['indicator_mae_std']) / np.sqrt(n_experiments)
    indicator_upper = np.array(stats_dict['indicator_mae_mean']) + 1.96 * np.array(stats_dict['indicator_mae_std']) / np.sqrt(n_experiments)
    fill_lower = np.array(stats_dict['fill_mae_mean']) - 1.96 * np.array(stats_dict['fill_mae_std']) / np.sqrt(n_experiments)
    fill_upper = np.array(stats_dict['fill_mae_mean']) + 1.96 * np.array(stats_dict['fill_mae_std']) / np.sqrt(n_experiments)
    
    ax1.plot(missing_rates, stats_dict['indicator_mae_mean'], 'o-', label='Missing Indicators', color='blue')
    ax1.fill_between(missing_rates, indicator_lower, indicator_upper, alpha=0.2, color='blue')
    ax1.plot(missing_rates, stats_dict['fill_mae_mean'], 's--', label='Mean Filling', color='orange')
    ax1.fill_between(missing_rates, fill_lower, fill_upper, alpha=0.2, color='orange')
    ax1.set_xlabel('Missing Rate')
    ax1.set_ylabel('MAE')
    ax1.set_title(f'MAE Confidence Intervals (95%, n={n_experiments})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE置信区间
    ax2 = axes[1]
    indicator_lower = np.array(stats_dict['indicator_rmse_mean']) - 1.96 * np.array(stats_dict['indicator_rmse_std']) / np.sqrt(n_experiments)
    indicator_upper = np.array(stats_dict['indicator_rmse_mean']) + 1.96 * np.array(stats_dict['indicator_rmse_std']) / np.sqrt(n_experiments)
    fill_lower = np.array(stats_dict['fill_rmse_mean']) - 1.96 * np.array(stats_dict['fill_rmse_std']) / np.sqrt(n_experiments)
    fill_upper = np.array(stats_dict['fill_rmse_mean']) + 1.96 * np.array(stats_dict['fill_rmse_std']) / np.sqrt(n_experiments)
    
    ax2.plot(missing_rates, stats_dict['indicator_rmse_mean'], 'o-', label='Missing Indicators', color='blue')
    ax2.fill_between(missing_rates, indicator_lower, indicator_upper, alpha=0.2, color='blue')
    ax2.plot(missing_rates, stats_dict['fill_rmse_mean'], 's--', label='Mean Filling', color='orange')
    ax2.fill_between(missing_rates, fill_lower, fill_upper, alpha=0.2, color='orange')
    ax2.set_xlabel('Missing Rate')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'RMSE Confidence Intervals (95%, n={n_experiments})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # R²置信区间
    ax3 = axes[2]
    indicator_lower = np.array(stats_dict['indicator_r2_mean']) - 1.96 * np.array(stats_dict['indicator_r2_std']) / np.sqrt(n_experiments)
    indicator_upper = np.array(stats_dict['indicator_r2_mean']) + 1.96 * np.array(stats_dict['indicator_r2_std']) / np.sqrt(n_experiments)
    fill_lower = np.array(stats_dict['fill_r2_mean']) - 1.96 * np.array(stats_dict['fill_r2_std']) / np.sqrt(n_experiments)
    fill_upper = np.array(stats_dict['fill_r2_mean']) + 1.96 * np.array(stats_dict['fill_r2_std']) / np.sqrt(n_experiments)
    
    ax3.plot(missing_rates, stats_dict['indicator_r2_mean'], 'o-', label='Missing Indicators', color='blue')
    ax3.fill_between(missing_rates, indicator_lower, indicator_upper, alpha=0.2, color='blue')
    ax3.plot(missing_rates, stats_dict['fill_r2_mean'], 's--', label='Mean Filling', color='orange')
    ax3.fill_between(missing_rates, fill_lower, fill_upper, alpha=0.2, color='orange')
    ax3.set_xlabel('Missing Rate')
    ax3.set_ylabel('R² Score')
    ax3.set_title(f'R² Confidence Intervals (95%, n={n_experiments})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_aggregated_results(stats_dict, save_dir, n_experiments):
    """保存聚合结果"""
    df = pd.DataFrame({
        'Missing_Rate': stats_dict['missing_rates'],
        'Baseline_MAE': stats_dict['baseline_mae_mean'],
        'Baseline_RMSE': stats_dict['baseline_rmse_mean'],
        'Baseline_R2': stats_dict['baseline_r2_mean'],
        'Indicator_MAE_Mean': stats_dict['indicator_mae_mean'],
        'Indicator_MAE_Std': stats_dict['indicator_mae_std'],
        'Indicator_RMSE_Mean': stats_dict['indicator_rmse_mean'],
        'Indicator_RMSE_Std': stats_dict['indicator_rmse_std'],
        'Indicator_R2_Mean': stats_dict['indicator_r2_mean'],
        'Indicator_R2_Std': stats_dict['indicator_r2_std'],
        'Fill_MAE_Mean': stats_dict['fill_mae_mean'],
        'Fill_MAE_Std': stats_dict['fill_mae_std'],
        'Fill_RMSE_Mean': stats_dict['fill_rmse_mean'],
        'Fill_RMSE_Std': stats_dict['fill_rmse_std'],
        'Fill_R2_Mean': stats_dict['fill_r2_mean'],
        'Fill_R2_Std': stats_dict['fill_r2_std'],
        'Improvement_Mean': stats_dict['improvement_mean'],
        'Improvement_Std': stats_dict['improvement_std']
    })
    
    csv_path = f'{save_dir}/aggregated_results_{n_experiments}_experiments.csv'
    df.to_csv(csv_path, index=False)
    print(f"Aggregated results saved to {csv_path}")
    
    return csv_path

# =========================
# 5. 主程序
# =========================
def main():
    parser = argparse.ArgumentParser(description='Run repeated SOH estimation experiments')
    parser.add_argument('--n_experiments', type=int, default=100, help='Number of repeated experiments')
    parser.add_argument('--missing_rates', type=float, nargs='+', 
                       default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help='Missing rates to test')
    parser.add_argument('--training_missing_rates', type=float, nargs='+', 
                       default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help='Training missing rates')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data/XJTU data', help='Data directory')
    parser.add_argument('--batch', type=str, default='2C', 
                       choices=['2C','3C','R2.5','R3','RW','satellite'], help='Battery batch')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--results_dir', type=str, default='repeat_experiment_results', 
                       help='Results directory')
    parser.add_argument('--script_path', type=str, default='soh_random_missing_analysis.py',
                       help='Path to the original experiment script')
    
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志
    logger = setup_logging(timestamp)
    logger.info(f"Starting repeated experiments at {timestamp}")
    logger.info(f"Parameters: {args}")
    
    # 确保结果目录存在
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    params = {
        'missing_rates': args.missing_rates,
        'training_missing_rates': args.training_missing_rates,
        'epochs': args.epochs,
        'data_dir': args.data_dir,
        'batch': args.batch,
        'batch_size': args.batch_size,
        'results_dir': args.results_dir
    }
    
    # 运行多个实验
    result_files = run_multiple_experiments(params, args.n_experiments, timestamp)
    
    if not result_files:
        logger.error("No experiments completed successfully")
        return
    
    logger.info(f"Completed {len(result_files)} experiments")
    
    # 加载所有结果
    all_results = load_experiment_results(result_files)
    
    if not all_results:
        logger.error("Failed to load any results")
        return
    
    logger.info(f"Loaded results from {len(all_results)} experiments")
    
    # 计算统计信息
    stats_dict = calculate_statistics(all_results)
    
    # 绘制结果
    plot_repeated_experiment_results(stats_dict, args.n_experiments, args.results_dir)
    plot_box_plots(all_results, args.results_dir)
    plot_confidence_intervals(stats_dict, args.n_experiments, args.results_dir)
    
    # 保存聚合结果
    csv_path = save_aggregated_results(stats_dict, args.results_dir, args.n_experiments)
    
    # 打印汇总统计
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Missing_Rate':<12} {'Method':<20} {'Mean':<10} {'Std':<10} {'CI_Lower':<10} {'CI_Upper':<10}")
    logger.info("-"*80)
    
    for i, mr in enumerate(stats_dict['missing_rates']):
        # Missing Indicators
        indicator_mae_mean = stats_dict['indicator_mae_mean'][i]
        indicator_mae_std = stats_dict['indicator_mae_std'][i]
        ci_lower = indicator_mae_mean - 1.96 * indicator_mae_std / np.sqrt(args.n_experiments)
        ci_upper = indicator_mae_mean + 1.96 * indicator_mae_std / np.sqrt(args.n_experiments)
        
        logger.info(f"{mr*100:>10.0f}%  {'Indicators_MAE':<20} {indicator_mae_mean:<10.4f} {indicator_mae_std:<10.4f} {ci_lower:<10.4f} {ci_upper:<10.4f}")
        
        # Fill Method
        fill_mae_mean = stats_dict['fill_mae_mean'][i]
        fill_mae_std = stats_dict['fill_mae_std'][i]
        ci_lower = fill_mae_mean - 1.96 * fill_mae_std / np.sqrt(args.n_experiments)
        ci_upper = fill_mae_mean + 1.96 * fill_mae_std / np.sqrt(args.n_experiments)
        
        logger.info(f"{'':<12} {'Fill_MAE':<20} {fill_mae_mean:<10.4f} {fill_mae_std:<10.4f} {ci_lower:<10.4f} {ci_upper:<10.4f}")
        
        logger.info("-"*80)
    
    logger.info("\n" + "="*80)
    logger.info("REPEATED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Results directory: {os.path.abspath(args.results_dir)}")
    logger.info(f"Aggregated results: {csv_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
