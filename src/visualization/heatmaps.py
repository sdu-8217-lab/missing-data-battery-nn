"""
热力图绘制模块
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_mim_improvement_heatmap(
    baseline_csv: str,
    mim_csv: str,
    output_path: str = "results/images/mim_improvement.png",
    metric: str = "mae",
):
    """
    绘制 MIM 相对于 Baseline 的改善热力图
    
    参数:
        baseline_csv: Baseline 结果 CSV
        mim_csv: MIM 结果 CSV
        output_path: 输出图像路径
        metric: 评估指标（越小越好，如 mae/rmse）
    """
    df_base = pd.read_csv(baseline_csv)
    df_mim = pd.read_csv(mim_csv)
    
    # 按 missing_rate 计算均值
    base_mean = df_base.groupby("missing_rate")[metric].mean()
    mim_mean = df_mim.groupby("missing_rate")[metric].mean()
    
    # 计算改善率（百分比）
    improvement = (base_mean - mim_mean) / base_mean * 100
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    
    # 创建 2D 数据用于热力图（这里用条形图代替）
    plt.bar(improvement.index, improvement.values, color='steelblue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Missing Rate")
    plt.ylabel(f"{metric.upper()} Improvement (%)")
    plt.title(f"MIM Improvement over Baseline ({metric.upper()})")
    plt.grid(True, alpha=0.3, axis='y')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_model_comparison_heatmap(
    results_dict: dict,
    output_path: str = "results/images/model_comparison.png",
    metric: str = "mae",
):
    """
    多模型对比热力图
    
    参数:
        results_dict: 模型名到CSV路径的映射
        output_path: 输出路径
        metric: 评估指标
    """
    # 收集数据
    data = []
    model_names = []
    
    for model_name, csv_path in results_dict.items():
        df = pd.read_csv(csv_path)
        mean_by_mr = df.groupby("missing_rate")[metric].mean()
        data.append(mean_by_mr.values)
        model_names.append(model_name)
    
    # 创建 DataFrame
    missing_rates = mean_by_mr.index
    df_plot = pd.DataFrame(data, index=model_names, columns=missing_rates)
    
    # 绘制热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_plot, annot=True, fmt=".4f", cmap="YlOrRd", cbar_kws={"label": metric.upper()})
    plt.title(f"Model Comparison ({metric.upper()})")
    plt.xlabel("Missing Rate")
    plt.ylabel("Model")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
