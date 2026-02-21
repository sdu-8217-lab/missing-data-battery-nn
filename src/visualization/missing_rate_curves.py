"""
缺失率-性能曲线绘制模块
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_missing_rate_curves(
    csv_path: str,
    output_dir: str = "results/images",
    metrics: list = None,
):
    """
    绘制性能指标随缺失率变化的曲线
    
    参数:
        csv_path: 结果 CSV 文件路径
        output_dir: 图像输出目录
        metrics: 要绘制的指标列表
    """
    if metrics is None:
        metrics = ["mae", "rmse", "r2"]
    
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # 按 missing_rate 分组计算均值和标准差
        grouped = df.groupby("missing_rate")[metric].agg(["mean", "std"])
        
        plt.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            marker='o',
            capsize=5,
        )
        
        plt.xlabel("Missing Rate")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Missing Rate")
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / f"{metric}_vs_mr.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
