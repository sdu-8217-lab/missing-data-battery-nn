#!/usr/bin/env python3
"""
预实验结果分析脚本

生成报告和可视化，展示不同预算下的最优架构

用法:
    python scripts/analyze_preexperiment.py --input-dir results/preexperiment
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_results(input_dir: Path) -> pd.DataFrame:
    """加载所有预实验结果"""
    records = []
    
    for json_file in sorted(input_dir.glob("*.json")):
        # 跳过汇总文件
        if json_file.name.startswith("_"):
            continue
        with open(json_file) as f:
            data = json.load(f)
        
        # 解析文件名获取模型类型和预算
        parts = json_file.stem.split("_")
        budget = int(parts[-1])
        model_type = "_".join(parts[:-1])
        
        record = {
            "model_type": model_type,
            "budget": budget,
            "n_params": data["n_params"],
            "test_mae_mean": data["performance"]["test_mae_mean"],
            "test_mae_std": data["performance"]["test_mae_std"],
            "test_r2_mean": data["performance"]["test_r2_mean"],
            "test_r2_std": data["performance"]["test_r2_std"],
            "config": data["best_config"],
        }
        records.append(record)
    
    return pd.DataFrame(records)


def plot_scaling_curves(df: pd.DataFrame, output_path: Path):
    """绘制参数量-性能 scaling 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 按模型类型分组
    model_types = df["model_type"].unique()
    colors = {"mlp": "#1f77b4", "lstm": "#ff7f0e", "gru": "#2ca02c", "cnn1d": "#d62728"}
    markers = {"mlp": "o", "lstm": "s", "gru": "^", "cnn1d": "D"}
    
    # 左图：MAE vs Budget
    ax1 = axes[0]
    for model_type in sorted(model_types):
        subset = df[df["model_type"] == model_type].sort_values("budget")
        ax1.errorbar(
            subset["budget"], 
            subset["test_mae_mean"],
            yerr=subset["test_mae_std"],
            label=model_type.upper(),
            color=colors.get(model_type, "gray"),
            marker=markers.get(model_type, "o"),
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    
    ax1.set_xlabel("Parameter Budget", fontsize=12)
    ax1.set_ylabel("Test MAE (lower is better)", fontsize=12)
    ax1.set_title("Model Performance vs Parameter Budget", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    
    # 右图：R2 vs Budget
    ax2 = axes[1]
    for model_type in sorted(model_types):
        subset = df[df["model_type"] == model_type].sort_values("budget")
        ax2.errorbar(
            subset["budget"],
            subset["test_r2_mean"],
            yerr=subset["test_r2_std"],
            label=model_type.upper(),
            color=colors.get(model_type, "gray"),
            marker=markers.get(model_type, "o"),
            linewidth=2,
            markersize=8,
            capsize=5,
        )
    
    ax2.set_xlabel("Parameter Budget", fontsize=12)
    ax2.set_ylabel("Test R² (higher is better)", fontsize=12)
    ax2.set_title("Model R² vs Parameter Budget", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] Scaling curves: {output_path}")


def generate_report(df: pd.DataFrame, output_path: Path):
    """生成 Markdown 报告"""
    lines = ["# 预实验结果报告\n"]
    lines.append("自动架构搜索结果：不同参数量预算下的最优模型配置\n")
    lines.append("---\n")
    
    # 汇总表
    lines.append("\n## 最优架构汇总\n")
    lines.append("| Model | Budget | Params | Test MAE | Test R² | Config |")
    lines.append("|-------|--------|--------|----------|---------|--------|")
    
    for _, row in df.sort_values(["model_type", "budget"]).iterrows():
        config_str = str(row["config"]).replace("\n", " ")
        lines.append(
            f"| {row['model_type'].upper()} | {row['budget']:,} | {row['n_params']:,} | "
            f"{row['test_mae_mean']:.4f}±{row['test_mae_std']:.4f} | "
            f"{row['test_r2_mean']:.4f}±{row['test_r2_std']:.4f} | `{config_str}` |"
        )
    
    # 效率分析
    lines.append("\n## 参数效率分析\n")
    lines.append("每 1000 参数带来的 MAE 改善：\n")
    
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type].sort_values("budget")
        if len(subset) >= 2:
            lines.append(f"\n### {model_type.upper()}\n")
            
            # 计算边际效益
            for i in range(1, len(subset)):
                prev = subset.iloc[i - 1]
                curr = subset.iloc[i]
                
                delta_params = curr["n_params"] - prev["n_params"]
                delta_mae = prev["test_mae_mean"] - curr["test_mae_mean"]
                efficiency = delta_mae / (delta_params / 1000) if delta_params > 0 else 0
                
                lines.append(
                    f"- {prev['budget']:,} → {curr['budget']:,}: "
                    f"MAE 改善 {delta_mae:.4f} ({efficiency:.4f} / 1K params)"
                )
    
    # 推荐配置
    lines.append("\n## 推荐配置\n")
    lines.append("### 小模型优先 (8K params)\n")
    best_small = df[df["budget"] == 8192].nsmallest(1, "test_mae_mean").iloc[0]
    lines.append(f"- **{best_small['model_type'].upper()}**: `{best_small['config']}`\n")
    lines.append(f"  - MAE: {best_small['test_mae_mean']:.4f}, R²: {best_small['test_r2_mean']:.4f}\n")
    
    lines.append("\n### 性能优先 (65K params)\n")
    best_large = df[df["budget"] == 65536].nsmallest(1, "test_mae_mean").iloc[0]
    lines.append(f"- **{best_large['model_type'].upper()}**: `{best_large['config']}`\n")
    lines.append(f"  - MAE: {best_large['test_mae_mean']:.4f}, R²: {best_large['test_r2_mean']:.4f}\n")
    
    lines.append("\n### 性价比最优\n")
    # 找每个模型在预算范围内的最佳性价比点
    for model_type in sorted(df["model_type"].unique()):
        subset = df[df["model_type"] == model_type]
        # 计算每 1K 参数的 MAE
        subset = subset.copy()
        subset["mae_per_1k"] = subset["test_mae_mean"] / (subset["n_params"] / 1000)
        best = subset.nsmallest(1, "mae_per_1k").iloc[0]
        lines.append(
            f"- **{model_type.upper()}**: {best['budget']:,} budget "
            f"({best['mae_per_1k']:.4f} MAE per 1K params)\n"
        )
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"[Saved] Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析预实验结果")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/preexperiment",
        help="预实验结果目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/preexperiment",
        help="分析报告输出目录"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {input_dir}")
    df = load_results(input_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} results")
    print(df[["model_type", "budget", "n_params", "test_mae_mean"]].to_string())
    
    # 生成可视化
    plot_scaling_curves(df, output_dir / "scaling_curves.png")
    
    # 生成报告
    generate_report(df, output_dir / "report.md")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
