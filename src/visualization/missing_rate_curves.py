"""
缺失率曲线可视化脚本

绘制 MAE/RMSE/R² 随缺失率变化的曲线图，对比 Baseline 和 MIM 方法。

Usage:
    python src/visualization/missing_rate_curves.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_mar_results(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    加载 MAR 实验结果文件
    
    Args:
        csv_dir: CSV 文件目录
        
    Returns:
        字典，键为配置名，值为 DataFrame
    """
    results = {}
    
    # 定义要加载的文件
    files = {
        'mim_0.3': 'mar_0.3_cnn1d_mim.csv',
        'mim_0.6': 'mar_0.6_cnn1d_mim.csv',
        'mim_0.9': 'mar_0.9_cnn1d_mim.csv',
        'baseline_0.3': 'mar_0.3_cnn1d_baseline.csv',
        'baseline_0.6': 'mar_0.6_cnn1d_baseline.csv',
        'baseline_0.9': 'mar_0.9_cnn1d_baseline.csv',
    }
    
    for key, filename in files.items():
        filepath = csv_dir / filename
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
            print(f"[OK] Loaded {filename}: {len(results[key])} rows")
        else:
            print(f"[MISSING] File not found: {filename}")
    
    return results


def compute_statistics(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    计算统计量（均值 ± 标准差）
    
    Args:
        df: 结果 DataFrame
        
    Returns:
        字典，键为指标名，值为 (mean, std) 元组
    """
    return {
        'mae': (df['test_mae'].mean(), df['test_mae'].std()),
        'rmse': (df['test_rmse'].mean(), df['test_rmse'].std()),
        'r2': (df['test_r2'].mean(), df['test_r2'].std()),
    }


def plot_missing_rate_curves(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    model_name: str = "CNN1D",
    missing_mode: str = "MAR"
) -> None:
    """
    绘制缺失率曲线图
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        model_name: 模型名称
        missing_mode: 缺失机制
    """
    # 准备数据
    missing_rates = [0.3, 0.6, 0.9]
    
    # 存储统计数据
    stats = {
        'mim': {'mae': [], 'rmse': [], 'r2': []},
        'baseline': {'mae': [], 'rmse': [], 'r2': []}
    }
    
    for mr in missing_rates:
        # MIM
        key = f'mim_{mr}'
        if key in results:
            s = compute_statistics(results[key])
            stats['mim']['mae'].append(s['mae'])
            stats['mim']['rmse'].append(s['rmse'])
            stats['mim']['r2'].append(s['r2'])
        else:
            stats['mim']['mae'].append((np.nan, np.nan))
            stats['mim']['rmse'].append((np.nan, np.nan))
            stats['mim']['r2'].append((np.nan, np.nan))
        
        # Baseline
        key = f'baseline_{mr}'
        if key in results:
            s = compute_statistics(results[key])
            stats['baseline']['mae'].append(s['mae'])
            stats['baseline']['rmse'].append(s['rmse'])
            stats['baseline']['r2'].append(s['r2'])
        else:
            stats['baseline']['mae'].append((np.nan, np.nan))
            stats['baseline']['rmse'].append((np.nan, np.nan))
            stats['baseline']['r2'].append((np.nan, np.nan))
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['mae', 'rmse', 'r2']
    titles = ['MAE vs Missing Rate', 'RMSE vs Missing Rate', 'R² vs Missing Rate']
    ylabels = ['MAE', 'RMSE', 'R²']
    
    colors = {'mim': '#2E86AB', 'baseline': '#A23B72'}
    labels = {'mim': 'MIM', 'baseline': 'Baseline'}
    
    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]
        
        for method in ['baseline', 'mim']:
            means = [s[0] for s in stats[method][metric]]
            stds = [s[1] for s in stats[method][metric]]
            
            ax.errorbar(
                missing_rates, means, yerr=stds,
                marker='o', markersize=8,
                linewidth=2, capsize=5,
                color=colors[method],
                label=labels[method]
            )
        
        ax.set_xlabel('Missing Rate', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(missing_rates)
    
    plt.suptitle(
        f'{model_name} Performance vs Missing Rate ({missing_mode})',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'{metric}_vs_missing_rate_cnn1d_mar.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        print(f"[SAVED] {filepath}")
    
    plt.close()


def main():
    """主函数"""
    # 路径设置
    project_root = Path(__file__).parent.parent.parent
    csv_dir = project_root / 'results' / 'csv'
    output_dir = project_root / 'results' / 'images'
    
    print("=" * 60)
    print("Missing Rate Curves Visualization")
    print("=" * 60)
    print(f"CSV directory: {csv_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 加载数据
    results = load_mar_results(csv_dir)
    
    if not results:
        print("No data files found. Please run experiments first.")
        return
    
    print()
    
    # 绘制曲线
    plot_missing_rate_curves(results, output_dir)
    
    print()
    print("=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
