"""
热力图可视化脚本

绘制 MIM 相对 Baseline 的 MAE 改善百分比热力图。

Usage:
    python src/visualization/heatmaps.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_improvement(baseline_df: pd.DataFrame, mim_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算 MIM 相对 Baseline 的改善百分比
    
    improvement = (MAE_baseline - MAE_mim) / MAE_baseline * 100%
    
    Args:
        baseline_df: Baseline 结果 DataFrame
        mim_df: MIM 结果 DataFrame
        
    Returns:
        字典，包含各指标的改善百分比
    """
    baseline_mae = baseline_df['test_mae'].mean()
    mim_mae = mim_df['test_mae'].mean()
    
    baseline_rmse = baseline_df['test_rmse'].mean()
    mim_rmse = mim_df['test_rmse'].mean()
    
    baseline_r2 = baseline_df['test_r2'].mean()
    mim_r2 = mim_df['test_r2'].mean()
    
    return {
        'mae': (baseline_mae - mim_mae) / baseline_mae * 100 if baseline_mae != 0 else 0,
        'rmse': (baseline_rmse - mim_rmse) / baseline_rmse * 100 if baseline_rmse != 0 else 0,
        'r2': (mim_r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0,
    }


def plot_improvement_heatmap(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    model_name: str = "CNN1D",
    missing_mode: str = "MAR"
) -> None:
    """
    绘制改善百分比热力图
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        model_name: 模型名称
        missing_mode: 缺失机制
    """
    # 计算改善百分比
    missing_rates = [0.3, 0.6, 0.9]
    
    improvements = {'mae': [], 'rmse': [], 'r2': []}
    
    for mr in missing_rates:
        baseline_key = f'baseline_{mr}'
        mim_key = f'mim_{mr}'
        
        if baseline_key in results and mim_key in results:
            imp = compute_improvement(results[baseline_key], results[mim_key])
            improvements['mae'].append(imp['mae'])
            improvements['rmse'].append(imp['rmse'])
            improvements['r2'].append(imp['r2'])
        else:
            improvements['mae'].append(np.nan)
            improvements['rmse'].append(np.nan)
            improvements['r2'].append(np.nan)
    
    # 创建热力图数据 (1×3)
    data = np.array([improvements['mae']])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 绘制热力图
    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        vmin=-20,
        vmax=50,
        cbar_kws={'label': 'Improvement (%)'},
        ax=ax,
        linewidths=1,
        linecolor='white'
    )
    
    # 设置标签
    ax.set_xticklabels([f'{mr}' for mr in missing_rates], fontsize=12)
    ax.set_yticklabels([model_name], fontsize=12, rotation=0)
    
    ax.set_xlabel('Missing Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    ax.set_title(
        f'MIM Improvement over Baseline ({missing_mode})\nMAE Reduction (%)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    plt.tight_layout()
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'mim_improvement_cnn1d_mar.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        print(f"[SAVED] {filepath}")
    
    plt.close()
    
    # 打印数值表格
    print("\n" + "=" * 60)
    print("MIM Improvement over Baseline (%)")
    print("=" * 60)
    print(f"{'Missing Rate':<15} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
    print("-" * 60)
    for i, mr in enumerate(missing_rates):
        mae_imp = improvements['mae'][i]
        rmse_imp = improvements['rmse'][i]
        r2_imp = improvements['r2'][i]
        print(f"{mr:<15} {mae_imp:>10.1f}% {rmse_imp:>10.1f}% {r2_imp:>10.1f}%")
    print("=" * 60)


def main():
    """主函数"""
    # 路径设置
    project_root = Path(__file__).parent.parent.parent
    csv_dir = project_root / 'results' / 'csv'
    output_dir = project_root / 'results' / 'images'
    
    print("=" * 60)
    print("Improvement Heatmap Visualization")
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
    
    # 绘制热力图
    plot_improvement_heatmap(results, output_dir)
    
    print()
    print("=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
