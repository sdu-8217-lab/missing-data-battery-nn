import os
import subprocess
import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(script_path, params):
    """
    运行单次实验
    
    Args:
        script_path (str): 实验脚本路径
        params (dict): 参数字典
    
    Returns:
        dict: 实验结果
    """
    # 构建命令
    cmd = ["python", script_path]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            cmd.append(f"--{key}")
            for v in value:
                cmd.append(str(v))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode != 0:
            logger.error(f"实验执行失败: {result.stderr}")
            return {"status": "error", "error": result.stderr}
        
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "params": params
        }
    except subprocess.TimeoutExpired:
        logger.error("实验超时")
        return {"status": "timeout", "error": "实验超时"}
    except Exception as e:
        logger.error(f"实验执行异常: {str(e)}")
        return {"status": "error", "error": str(e)}

def extract_metrics_from_log(log_content):
    """
    从日志内容中提取关键指标
    
    Args:
        log_content (str): 日志内容
    
    Returns:
        dict: 提取的指标
    """
    metrics = {}
    
    # 提取MAE, RMSE, R²指标
    lines = log_content.split('\n')
    
    # 查找缺失指示器模型的最终性能
    for line in reversed(lines):  # 从最后开始查找，因为最终结果在末尾
        if '缺失指示器通用模型:' in line:
            # MAE=0.0045, RMSE=0.0052, R²=0.9991
            mae_match = re.search(r'MAE=([0-9.]+)', line)
            rmse_match = re.search(r'RMSE=([0-9.]+)', line)
            r2_match = re.search(r'R²=([0-9.]+)', line)
            
            if mae_match:
                metrics['indicator_mae'] = float(mae_match.group(1))
            if rmse_match:
                metrics['indicator_rmse'] = float(rmse_match.group(1))
            if r2_match:
                metrics['indicator_r2'] = float(r2_match.group(1))
            break
    
    # 查找缩减模型的最终性能
    for line in reversed(lines):
        if '缩减模型 (' in line and 'MAE=' in line:
            mae_match = re.search(r'MAE=([0-9.]+)', line)
            rmse_match = re.search(r'RMSE=([0-9.]+)', line)
            r2_match = re.search(r'R²=([0-9.]+)', line)
            
            if mae_match:
                metrics['reduced_mae'] = float(mae_match.group(1))
            if rmse_match:
                metrics['reduced_rmse'] = float(rmse_match.group(1))
            if r2_match:
                metrics['reduced_r2'] = float(r2_match.group(1))
            break
    
    # 统计实验完成情况
    if "实验成功完成！" in log_content:
        metrics['completed_successfully'] = True
    else:
        metrics['completed_successfully'] = False
    
    return metrics

def parse_experiment_results(results):
    """
    解析实验结果，提取关键指标
    
    Args:
        results: 实验结果列表
    
    Returns:
        dict: 解析后的结果数据
    """
    parsed_results = []
    
    for i, result in enumerate(results):
        experiment_result = {
            "experiment_id": i,
            "params": result["params"],
            "status": result["status"]
        }
        
        if result["status"] == "success":
            # 从stdout中提取指标
            metrics = extract_metrics_from_log(result.get("stdout", ""))
            experiment_result.update(metrics)
            
            # 如果找不到指标，也尝试从stderr中查找
            if not metrics.get('indicator_mae'):
                stderr_metrics = extract_metrics_from_log(result.get("stderr", ""))
                experiment_result.update(stderr_metrics)
        
        parsed_results.append(experiment_result)
    
    return parsed_results

def plot_results_distribution(results, output_dir):
    """
    绘制实验结果分布图
    
    Args:
        results: 解析后的实验结果
        output_dir: 输出目录
    """
    if not results:
        logger.warning("没有有效的实验结果可用于绘图")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取指标数据
    indicator_maes = []
    indicator_rmses = []
    indicator_r2s = []
    reduced_maes = []
    reduced_rmses = []
    reduced_r2s = []
    seeds = []
    
    for result in results:
        if result.get("completed_successfully", False):
            if 'indicator_mae' in result:
                indicator_maes.append(result['indicator_mae'])
            if 'indicator_rmse' in result:
                indicator_rmses.append(result['indicator_rmse'])
            if 'indicator_r2' in result:
                indicator_r2s.append(result['indicator_r2'])
            if 'reduced_mae' in result:
                reduced_maes.append(result['reduced_mae'])
            if 'reduced_rmse' in result:
                reduced_rmses.append(result['reduced_rmse'])
            if 'reduced_r2' in result:
                reduced_r2s.append(result['reduced_r2'])
            seeds.append(result['params']['seed'])
    
    # 绘制结果分布图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 指标直方图
    if indicator_maes:
        axes[0, 0].hist(indicator_maes, bins=20, edgecolor='black', alpha=0.7, color='blue', label='Indicator Model')
        axes[0, 0].set_title('MAE Distribution - Indicator Model')
        axes[0, 0].set_xlabel('MAE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
    
    if reduced_maes:
        axes[0, 0].hist(reduced_maes, bins=20, edgecolor='black', alpha=0.7, color='orange', label='Reduced Model')
        axes[0, 0].set_title('MAE Distribution - Both Models')
        axes[0, 0].legend()
    
    if indicator_rmses:
        axes[0, 1].hist(indicator_rmses, bins=20, edgecolor='black', alpha=0.7, color='blue', label='Indicator Model')
        axes[0, 1].set_title('RMSE Distribution - Indicator Model')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    
    if reduced_rmses:
        axes[0, 1].hist(reduced_rmses, bins=20, edgecolor='black', alpha=0.7, color='orange', label='Reduced Model')
        axes[0, 1].set_title('RMSE Distribution - Both Models')
        axes[0, 1].legend()
    
    if indicator_r2s:
        axes[0, 2].hist(indicator_r2s, bins=20, edgecolor='black', alpha=0.7, color='blue', label='Indicator Model')
        axes[0, 2].set_title('R² Distribution - Indicator Model')
        axes[0, 2].set_xlabel('R²')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
    
    if reduced_r2s:
        axes[0, 2].hist(reduced_r2s, bins=20, edgecolor='black', alpha=0.7, color='orange', label='Reduced Model')
        axes[0, 2].set_title('R² Distribution - Both Models')
        axes[0, 2].legend()
    
    # 指标随随机种子的变化趋势
    if seeds and indicator_maes:
        axes[1, 0].plot(seeds[:len(indicator_maes)], indicator_maes, 'bo-', label='Indicator Model', markersize=4)
        if len(reduced_maes) > 0:
            axes[1, 0].plot(seeds[:len(reduced_maes)], reduced_maes, 'ro-', label='Reduced Model', markersize=4)
        axes[1, 0].set_title('MAE vs Random Seed')
        axes[1, 0].set_xlabel('Random Seed')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if seeds and indicator_rmses:
        axes[1, 1].plot(seeds[:len(indicator_rmses)], indicator_rmses, 'bo-', label='Indicator Model', markersize=4)
        if len(reduced_rmses) > 0:
            axes[1, 1].plot(seeds[:len(reduced_rmses)], reduced_rmses, 'ro-', label='Reduced Model', markersize=4)
        axes[1, 1].set_title('RMSE vs Random Seed')
        axes[1, 1].set_xlabel('Random Seed')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    if seeds and indicator_r2s:
        axes[1, 2].plot(seeds[:len(indicator_r2s)], indicator_r2s, 'bo-', label='Indicator Model', markersize=4)
        if len(reduced_r2s) > 0:
            axes[1, 2].plot(seeds[:len(reduced_r2s)], reduced_r2s, 'ro-', label='Reduced Model', markersize=4)
        axes[1, 2].set_title('R² vs Random Seed')
        axes[1, 2].set_xlabel('Random Seed')
        axes[1, 2].set_ylabel('R²')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_results_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制箱线图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # MAE箱线图
    if indicator_maes and reduced_maes:
        box_data = [indicator_maes, reduced_maes]
        ax[0].boxplot(box_data, labels=['Indicator Model', 'Reduced Model'])
        ax[0].set_title('MAE Box Plot Comparison')
        ax[0].set_ylabel('MAE')
        ax[0].grid(True, alpha=0.3)
    
    # R²箱线图
    if indicator_r2s and reduced_r2s:
        box_data = [indicator_r2s, reduced_r2s]
        ax[1].boxplot(box_data, labels=['Indicator Model', 'Reduced Model'])
        ax[1].set_title('R² Box Plot Comparison')
        ax[1].set_ylabel('R²')
        ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"结果分布图已保存至: {output_dir}")

def calculate_statistics(results):
    """
    计算实验结果的统计信息
    
    Args:
        results: 解析后的实验结果
    
    Returns:
        dict: 统计信息
    """
    successful_results = [r for r in results if r.get("completed_successfully", False)]
    
    stats = {
        "total_experiments": len(results),
        "successful_experiments": len(successful_results),
        "failed_experiments": len(results) - len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0
    }
    
    # 计算各项指标的统计信息
    indicator_maes = [r['indicator_mae'] for r in successful_results if 'indicator_mae' in r]
    indicator_rmses = [r['indicator_rmse'] for r in successful_results if 'indicator_rmse' in r]
    indicator_r2s = [r['indicator_r2'] for r in successful_results if 'indicator_r2' in r]
    reduced_maes = [r['reduced_mae'] for r in successful_results if 'reduced_mae' in r]
    reduced_rmses = [r['reduced_rmse'] for r in successful_results if 'reduced_rmse' in r]
    reduced_r2s = [r['reduced_r2'] for r in successful_results if 'reduced_r2' in r]
    
    if indicator_maes:
        stats['indicator_mae_mean'] = np.mean(indicator_maes)
        stats['indicator_mae_std'] = np.std(indicator_maes)
        stats['indicator_mae_min'] = np.min(indicator_maes)
        stats['indicator_mae_max'] = np.max(indicator_maes)
    
    if indicator_rmses:
        stats['indicator_rmse_mean'] = np.mean(indicator_rmses)
        stats['indicator_rmse_std'] = np.std(indicator_rmses)
        stats['indicator_rmse_min'] = np.min(indicator_rmses)
        stats['indicator_rmse_max'] = np.max(indicator_rmses)
    
    if indicator_r2s:
        stats['indicator_r2_mean'] = np.mean(indicator_r2s)
        stats['indicator_r2_std'] = np.std(indicator_r2s)
        stats['indicator_r2_min'] = np.min(indicator_r2s)
        stats['indicator_r2_max'] = np.max(indicator_r2s)
    
    if reduced_maes:
        stats['reduced_mae_mean'] = np.mean(reduced_maes)
        stats['reduced_mae_std'] = np.std(reduced_maes)
        stats['reduced_mae_min'] = np.min(reduced_maes)
        stats['reduced_mae_max'] = np.max(reduced_maes)
    
    if reduced_rmses:
        stats['reduced_rmse_mean'] = np.mean(reduced_rmses)
        stats['reduced_rmse_std'] = np.std(reduced_rmses)
        stats['reduced_rmse_min'] = np.min(reduced_rmses)
        stats['reduced_rmse_max'] = np.max(reduced_rmses)
    
    if reduced_r2s:
        stats['reduced_r2_mean'] = np.mean(reduced_r2s)
        stats['reduced_r2_std'] = np.std(reduced_r2s)
        stats['reduced_r2_min'] = np.min(reduced_r2s)
        stats['reduced_r2_max'] = np.max(reduced_r2s)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='批量运行SOH估计实验（100次重复实验）')
    
    # 实验脚本路径
    parser.add_argument('--script-path', type=str, default='soh_fixed_feature_missing.py',
                        help='实验脚本路径 (默认: soh_fixed_feature_missing.py)')
    
    # 固定的训练缺失率
    parser.add_argument('--training-missing-rates', type=str, 
                        default='0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95',
                        help='训练缺失率列表 (默认: 0.0 0.05 ... 0.95)')
    
    # 其他参数
    parser.add_argument('--data-dir', type=str, default='./data/XJTU data',
                        help='数据目录 (默认: ./data/XJTU data)')
    parser.add_argument('--batch', type=str, default='2C',
                        help='电池批次 (默认: 2C)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小 (默认: 32)')
    parser.add_argument('--start-seed', type=int, default=1,
                        help='起始随机种子 (默认: 1)')
    parser.add_argument('--num-experiments', type=int, default=100,
                        help='实验次数 (默认: 100)')
    parser.add_argument('--results-dir', type=str, default='batch_experiment_results',
                        help='结果保存目录 (默认: batch_experiment_results)')
    parser.add_argument('--output-dir', type=str, default='experiment_analysis',
                        help='分析结果输出目录 (默认: experiment_analysis)')
    
    args = parser.parse_args()
    
    # 验证实验脚本是否存在
    if not os.path.exists(args.script_path):
        logger.error(f"实验脚本不存在: {args.script_path}")
        return
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析训练缺失率
    training_missing_rates = [float(r) for r in args.training_missing_rates.split()]
    
    logger.info(f"总共 {args.num_experiments} 次实验，每次使用不同的随机种子")
    logger.info(f"训练缺失率: {training_missing_rates}")
    
    # 运行实验
    all_results = []
    for i in range(args.num_experiments):
        seed = args.start_seed + i
        logger.info(f"运行实验 {i+1}/{args.num_experiments} (随机种子: {seed})")
        
        # 每次实验都有独立的结果目录
        exp_results_dir = os.path.join(args.results_dir, f"experiment_{i+1:03d}_seed_{seed}")
        
        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': seed,
            'training_missing_rates': training_missing_rates,
            'data_dir': args.data_dir,
            'batch': args.batch,
            'results_dir': exp_results_dir,
            'pretrained_model': None  # 不使用预训练模型进行批量实验
        }
        
        result = run_experiment(args.script_path, params)
        all_results.append(result)
        
        # 保存中间结果
        with open(os.path.join(args.output_dir, f"experiment_{i+1:03d}_result.json"), 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    # 解析结果
    logger.info("解析实验结果...")
    parsed_results = parse_experiment_results(all_results)
    
    # 保存所有结果
    results_summary = {
        "total_experiments": len(all_results),
        "successful_experiments": len([r for r in all_results if r["status"] == "success"]),
        "failed_experiments": len([r for r in all_results if r["status"] != "success"]),
        "parameters_used": {
            "script_path": args.script_path,
            "training_missing_rates": training_missing_rates,
            "data_dir": args.data_dir,
            "batch": args.batch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "start_seed": args.start_seed,
            "num_experiments": args.num_experiments,
            "results_dir": args.results_dir
        },
        "raw_results": all_results,
        "parsed_results": parsed_results
    }
    
    summary_path = os.path.join(args.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"实验总结已保存至: {summary_path}")
    
    # 计算统计信息
    stats = calculate_statistics(parsed_results)
    stats_path = os.path.join(args.output_dir, 'experiment_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logger.info(f"实验统计信息已保存至: {stats_path}")
    
    # 绘制结果分布图
    logger.info("绘制结果分布图...")
    plot_results_distribution(parsed_results, args.output_dir)
    
    # 打印统计摘要
    logger.info("="*80)
    logger.info("批量实验完成总结:")
    logger.info(f"总实验数: {stats['total_experiments']}")
    logger.info(f"成功实验数: {stats['successful_experiments']}")
    logger.info(f"失败实验数: {stats['failed_experiments']}")
    logger.info(f"成功率: {stats['success_rate']*100:.2f}%")
    
    if 'indicator_mae_mean' in stats:
        logger.info(f"\n缺失指示器模型性能:")
        logger.info(f"  MAE: 均值={stats['indicator_mae_mean']:.6f}, 标准差={stats['indicator_mae_std']:.6f}")
        logger.info(f"  RMSE: 均值={stats['indicator_rmse_mean']:.6f}, 标准差={stats['indicator_rmse_std']:.6f}")
        logger.info(f"  R²: 均值={stats['indicator_r2_mean']:.6f}, 标准差={stats['indicator_r2_std']:.6f}")
    
    if 'reduced_mae_mean' in stats:
        logger.info(f"\n缩减模型性能:")
        logger.info(f"  MAE: 均值={stats['reduced_mae_mean']:.6f}, 标准差={stats['reduced_mae_std']:.6f}")
        logger.info(f"  RMSE: 均值={stats['reduced_rmse_mean']:.6f}, 标准差={stats['reduced_rmse_std']:.6f}")
        logger.info(f"  R²: 均值={stats['reduced_r2_mean']:.6f}, 标准差={stats['reduced_r2_std']:.6f}")
    
    logger.info(f"\n结果保存目录: {args.results_dir}")
    logger.info(f"分析结果保存目录: {args.output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
