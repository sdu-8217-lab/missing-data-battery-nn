#!/usr/bin/env python3
"""聚合多组实验结果并生成论文用汇总表与图。

用法：
    python scripts/aggregate_results.py --input ./results --output ./results/aggregated
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

METRICS = ["mae", "rmse", "r2"]


def find_result_csvs(root: Path):
    """递归查找所有 experiment_results_*.csv。"""
    return sorted(root.rglob("experiment_results_*.csv"))


def load_config(csv_path: Path):
    """读取同目录下的 config.json，失败则返回空字典。"""
    config_path = csv_path.parent.parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def parse_experiment_path(csv_path: Path, cfg: dict):
    """从 config.json 或路径推断实验元信息。"""
    dataset = cfg.get("dataset_name", "Unknown")
    batch = cfg.get("batch", "Unknown")
    pattern = cfg.get("missing_pattern", "Unknown")

    # 如果 config.json 没有 missing_pattern，尝试从路径推断
    if pattern == "Unknown":
        parts = csv_path.parts
        for part in parts:
            if "bernoulli" in part:
                pattern = "bernoulli"
            elif "block" in part:
                pattern = "block"
            elif "channel" in part:
                pattern = "channel"
            elif "state_dependent" in part:
                pattern = "state_dependent"
            elif "mixed" in part:
                pattern = "mixed"

    return {
        "csv_path": str(csv_path),
        "dataset": dataset,
        "batch": batch,
        "missing_pattern": pattern,
        "timestamp": cfg.get("timestamp", csv_path.stem.split("_")[-1]),
    }


def aggregate_group(df: pd.DataFrame, group_cols, value_cols=METRICS):
    """按 group_cols 聚合，计算 mean/std/count/se/95% CI。"""
    agg = df.groupby(group_cols)[value_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    for metric in value_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        se_col = f"{metric}_se"
        ci_col = f"{metric}_ci95"
        agg[se_col] = agg[std_col] / np.sqrt(agg[count_col])
        # 95% CI using t-distribution
        agg[ci_col] = stats.t.ppf(0.975, agg[count_col] - 1) * agg[se_col]
        agg[ci_col] = agg[ci_col].fillna(0.0)

    return agg


def _paired_samples(df: pd.DataFrame, name_a: str, name_b: str,
                    metric: str = "mae", group_cols=None):
    """获取两个模型在同 seed 同分组下的配对样本。"""
    if group_cols is None:
        group_cols = ["missing_rate", "seed"]
    a = df[df["model"] == name_a][group_cols + [metric]].rename(
        columns={metric: f"{metric}_a"}
    )
    b = df[df["model"] == name_b][group_cols + [metric]].rename(
        columns={metric: f"{metric}_b"}
    )
    return pd.merge(a, b, on=group_cols)


def paired_test(df: pd.DataFrame, baseline_name: str, comp_name: str,
                metric: str = "mae", method: str = "wilcoxon"):
    """对两个模型在同 seed 同 missing_rate 下的结果做配对检验。

    Args:
        method: 'ttest' 或 'wilcoxon'。
    Returns:
        statistic, p_value；若无法计算返回 (nan, nan)。
    """
    merged = _paired_samples(df, baseline_name, comp_name, metric)
    if len(merged) == 0:
        return np.nan, np.nan
    a = merged[f"{metric}_a"]
    b = merged[f"{metric}_b"]
    if (a - b).std() == 0 or len(a) < 2:
        return np.nan, np.nan
    if method == "wilcoxon":
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
    else:
        stat, p_value = stats.ttest_rel(a, b)
    return stat, p_value


def fdr_correction(p_values: np.ndarray):
    """Benjamini-Hochberg FDR 校正。"""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    q_sorted = sorted_p * n / np.arange(1, n + 1)
    # 保证单调性：从最大 p 值向最小 p 值方向累积最小值
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_values = np.empty_like(q_sorted)
    q_values[order] = q_sorted
    q_values = np.clip(q_values, 0.0, 1.0)
    return q_values


def make_improvement_table(df: pd.DataFrame):
    """生成 MIM 相对 Baseline 与 MultiMR 的改进率表（按 dataset/model/mr）。"""
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        ddf = df[df["dataset"] == dataset]
        for model_type in ["MLP", "LSTM", "GRU", "CNN1D"]:
            base_name = model_type
            mim_name = f"{model_type}-MIM"
            multimr_name = f"{model_type}-MultiMR"
            if base_name not in ddf["model"].values or mim_name not in ddf["model"].values:
                continue
            for mr in sorted(ddf["missing_rate"].unique()):
                b = ddf[(ddf["model"] == base_name) & (ddf["missing_rate"] == mr)]
                m = ddf[(ddf["model"] == mim_name) & (ddf["missing_rate"] == mr)]
                if len(b) == 0 or len(m) == 0:
                    continue
                base_mae = b["mae"].mean()
                mim_mae = m["mae"].mean()
                # 仅在该 missing_rate 内做配对检验
                mr_df = ddf[ddf["missing_rate"] == mr]
                _, p_base = paired_test(mr_df, base_name, mim_name, metric="mae")
                _, p_multi = np.nan, np.nan
                if multimr_name in ddf["model"].values:
                    _, p_multi = paired_test(mr_df, multimr_name, mim_name, metric="mae")
                rows.append(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "missing_rate": mr,
                        "baseline_mae_mean": base_mae,
                        "baseline_mae_std": b["mae"].std(),
                        "mim_mae_mean": mim_mae,
                        "mim_mae_std": m["mae"].std(),
                        "improvement_vs_baseline_pct": (base_mae - mim_mae) / base_mae * 100,
                        "p_mim_vs_baseline": p_base,
                        "p_mim_vs_multimr": p_multi,
                    }
                )
    imp_df = pd.DataFrame(rows)
    if not imp_df.empty:
        # 在每个数据集×模型组合内对 9 个 missing rate 做 FDR 校正
        imp_df["q_mim_vs_baseline"] = np.nan
        imp_df["q_mim_vs_multimr"] = np.nan
        for (dataset, model_type), sub in imp_df.groupby(["dataset", "model_type"]):
            pvals = sub["p_mim_vs_baseline"].values
            mask = ~np.isnan(pvals)
            if mask.any():
                qvals = fdr_correction(pvals[mask])
                imp_df.loc[sub.index[mask], "q_mim_vs_baseline"] = qvals
            pvals = sub["p_mim_vs_multimr"].values
            mask = ~np.isnan(pvals)
            if mask.any():
                qvals = fdr_correction(pvals[mask])
                imp_df.loc[sub.index[mask], "q_mim_vs_multimr"] = qvals
    return imp_df


def make_strategy_comparison_table(df: pd.DataFrame):
    """按数据集×缺失率汇总 Baseline / MultiMR / MIM 的平均 MAE。"""
    rows = []
    for (dataset, pattern, mr), sub in df.groupby(["dataset", "missing_pattern", "missing_rate"]):
        for strategy in ["baseline", "multi_missing_rate", "mim"]:
            ssub = sub[sub["strategy"] == strategy]
            if len(ssub) == 0:
                continue
            rows.append({
                "dataset": dataset,
                "missing_pattern": pattern,
                "missing_rate": mr,
                "strategy": strategy,
                "mae_mean": ssub["mae"].mean(),
                "mae_std": ssub["mae"].std(),
                "rmse_mean": ssub["rmse"].mean(),
                "r2_mean": ssub["r2"].mean(),
                "n_seeds": ssub["seed"].nunique(),
            })
    return pd.DataFrame(rows)


def plot_metric_curves(summary: pd.DataFrame, metric: str, output_dir: Path):
    """绘制各数据集上不同模型的 metric 随 missing_rate 变化曲线。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(summary["dataset"].unique())
    models = sorted(summary["model"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ddf = summary[summary["dataset"] == dataset]
        for model in models:
            mdf = ddf[ddf["model"] == model].sort_values("missing_rate")
            if len(mdf) == 0:
                continue
            ax.plot(
                mdf["missing_rate"],
                mdf[f"{metric}_mean"],
                marker="o",
                label=model,
                linewidth=1.5,
            )
            ax.fill_between(
                mdf["missing_rate"],
                mdf[f"{metric}_mean"] - mdf[f"{metric}_ci95"],
                mdf[f"{metric}_mean"] + mdf[f"{metric}_ci95"],
                alpha=0.15,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Missing Rate")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # 隐藏多余子图
    for idx in range(len(datasets), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_curves_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_improvement_heatmap(imp_df: pd.DataFrame, output_dir: Path):
    """绘制 MIM 改进率热力图（dataset × model_type，按 MR 分面）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    mrs = sorted(imp_df["missing_rate"].unique())
    n = len(mrs)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, mr in enumerate(mrs):
        ax = axes[idx]
        sub = imp_df[imp_df["missing_rate"] == mr].pivot(
            index="model_type", columns="dataset", values="improvement_vs_baseline_pct"
        )
        im = ax.imshow(sub.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=40)
        ax.set_xticks(np.arange(len(sub.columns)))
        ax.set_yticks(np.arange(len(sub.index)))
        ax.set_xticklabels(sub.columns, rotation=45, ha="right")
        ax.set_yticklabels(sub.index)
        ax.set_title(f"MR={mr}")
        for i in range(len(sub.index)):
            for j in range(len(sub.columns)):
                val = sub.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="Improvement %")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "mim_improvement_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--input", type=str, default="./results", help="结果根目录")
    parser.add_argument("--output", type=str, default="./results/aggregated", help="输出目录")
    args = parser.parse_args()

    root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_result_csvs(root)
    if len(csv_files) == 0:
        print(f"未在 {root} 下找到 experiment_results_*.csv 文件")
        return

    print(f"找到 {len(csv_files)} 个结果文件")

    all_rows = []
    for csv_path in csv_files:
        cfg = load_config(csv_path)
        meta = parse_experiment_path(csv_path, cfg)
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"model_name": "model"})
        for k, v in meta.items():
            df[k] = v
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    # 统一模型名映射
    full_df["model"] = full_df["model"].str.strip()
    full_df["strategy"] = full_df["strategy"].fillna("baseline")

    # 保存完整长表
    full_df.to_csv(output_dir / "all_results_long.csv", index=False)

    # 1. 按数据集/模型/缺失率聚合
    summary = aggregate_group(
        full_df,
        ["dataset", "batch", "missing_pattern", "model", "missing_rate"],
        METRICS,
    )
    summary.to_csv(output_dir / "summary_by_dataset_model_mr.csv", index=False)
    print(f"汇总表已保存: {output_dir / 'summary_by_dataset_model_mr.csv'}")

    # 2. 按数据集/策略聚合（所有模型平均）
    strategy_summary = aggregate_group(
        full_df,
        ["dataset", "missing_pattern", "strategy", "missing_rate"],
        METRICS,
    )
    strategy_summary.to_csv(output_dir / "summary_by_dataset_strategy_mr.csv", index=False)

    # 3. 改进率表
    imp_df = make_improvement_table(full_df)
    if not imp_df.empty:
        imp_df.to_csv(output_dir / "mim_improvement.csv", index=False)
        print(f"改进率表已保存: {output_dir / 'mim_improvement.csv'}")
    
    # 4. 策略对比表（Baseline / MultiMR / MIM）
    strategy_comp = make_strategy_comparison_table(full_df)
    if not strategy_comp.empty:
        strategy_comp.to_csv(output_dir / "strategy_comparison.csv", index=False)
        print(f"策略对比表已保存: {output_dir / 'strategy_comparison.csv'}")

    # 4. 绘图
    try:
        for metric in METRICS:
            plot_metric_curves(summary, metric, output_dir / "figures")
        if not imp_df.empty:
            plot_improvement_heatmap(imp_df, output_dir / "figures")
        print(f"图表已保存: {output_dir / 'figures'}")
    except Exception as e:
        print(f"绘图失败: {e}")

    # 5. 简单打印关键结果
    print("\n=== 各数据集 MR=0.5 时 MIM vs Baseline 平均 MAE ===")
    pivot = (
        full_df[full_df["missing_rate"] == 0.5]
        .groupby(["dataset", "model"])["mae"]
        .mean()
        .unstack("model")
    )
    print(pivot)


if __name__ == "__main__":
    main()
