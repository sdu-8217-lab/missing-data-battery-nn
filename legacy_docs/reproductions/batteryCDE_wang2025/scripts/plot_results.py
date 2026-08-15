"""BatteryCDE 复现结果绘图脚本

用法:
    python scripts/plot_results.py --results_dir results --output_dir results/figures
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot BatteryCDE reproduction results")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Root results directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output figures directory")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_history(history, output_path):
    """绘制训练 loss 与验证 MAE 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_mae"], label="val MAE", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bar_comparison(df, x_col, y_col, title, output_path, hue_col=None):
    """绘制柱状对比图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue_col is not None:
        pivot = df.pivot(index=x_col, columns=hue_col, values=y_col)
        pivot.plot(kind="bar", ax=ax)
    else:
        df.plot(x=x_col, y=y_col, kind="bar", ax=ax, legend=False)
    ax.set_ylabel(y_col.upper())
    ax.set_title(title)
    ax.grid(True, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_horizon_comparison(results_dir, output_dir):
    """绘制不同 horizon 的 MAE 对比。"""
    horizon_dir = Path(results_dir) / "horizon"
    if not horizon_dir.exists():
        return

    rows = []
    for model_dir in horizon_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for h_dir in model_dir.iterdir():
            if not h_dir.is_dir():
                continue
            metric_file = h_dir / model_dir.name / "test_metrics.json"
            if not metric_file.exists():
                metric_file = h_dir / "test_metrics.json"
            if not metric_file.exists():
                continue
            metrics = load_json(metric_file)
            horizon = int(h_dir.name.replace("h", ""))
            rows.append({
                "model": model_dir.name,
                "horizon": horizon,
                "mae": metrics.get("mae", np.nan),
                "rmse": metrics.get("rmse", np.nan),
                "r2": metrics.get("r2", np.nan),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "horizon"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if df["model"].nunique() == 1:
        df.plot(x="horizon", y="mae", kind="bar", ax=ax, legend=False, color="#2c5282")
    else:
        pivot = df.pivot(index="horizon", columns="model", values="mae")
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Test MAE (SOH)", fontsize=11)
    ax.set_xlabel("Prediction Horizon (cycles)", fontsize=11)
    ax.set_title("Long-term Capacity Prediction: MAE vs Horizon", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_comparison.png", dpi=150)
    plt.close()

    df.to_csv(output_dir / "horizon_comparison.csv", index=False)


def plot_missing_comparison(results_dir, output_dir):
    """绘制缺失数据实验对比。"""
    missing_dir = Path(results_dir) / "missing"
    if not missing_dir.exists():
        return

    rows = []
    for model_dir in missing_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            metric_file = exp_dir / model_dir.name / "test_metrics.json"
            if not metric_file.exists():
                metric_file = exp_dir / "test_metrics.json"
            if not metric_file.exists():
                continue
            metrics = load_json(metric_file)
            rows.append({
                "model": model_dir.name,
                "scenario": exp_dir.name,
                "mae": metrics.get("mae", np.nan),
                "rmse": metrics.get("rmse", np.nan),
                "r2": metrics.get("r2", np.nan),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    # 定义缺失模式的有序显示顺序
    scenario_order = ["clean", "random30", "random50", "block30", "block50"]
    df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
    df = df.sort_values(["model", "scenario"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if df["model"].nunique() == 1:
        df.plot(x="scenario", y="mae", kind="bar", ax=ax, legend=False, color="#2c5282")
    else:
        pivot = df.pivot(index="scenario", columns="model", values="mae")
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Test MAE (SOH)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Missing Data Robustness", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_comparison.png", dpi=150)
    plt.close()

    df.to_csv(output_dir / "missing_comparison.csv", index=False)


def plot_baseline_comparison(results_dir, output_dir):
    """绘制基线对比。"""
    nasa_dir = Path(results_dir) / "nasa"
    if not nasa_dir.exists():
        return

    rows = []
    for model_dir in nasa_dir.iterdir():
        if not model_dir.is_dir():
            continue
        metric_file = model_dir / "test_metrics.json"
        if not metric_file.exists():
            continue
        metrics = load_json(metric_file)
        rows.append({
            "model": model_dir.name,
            "mae": metrics.get("mae", np.nan),
            "rmse": metrics.get("rmse", np.nan),
            "r2": metrics.get("r2", np.nan),
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("mae")
    plot_bar_comparison(
        df,
        x_col="model",
        y_col="mae",
        title="NASA Test Set: Model Comparison",
        output_path=output_dir / "baseline_comparison.png",
    )
    df.to_csv(output_dir / "baseline_comparison.csv", index=False)


def plot_transfer_comparison(results_dir, output_dir):
    """绘制迁移实验对比。"""
    transfer_dir = Path(results_dir) / "transfer"
    if not transfer_dir.exists():
        return

    rows = []
    for model_dir in transfer_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for scenario_dir in model_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            metric_file = scenario_dir / model_dir.name / "test_metrics.json"
            if not metric_file.exists():
                metric_file = scenario_dir / "test_metrics.json"
            if not metric_file.exists():
                continue
            metrics = load_json(metric_file)
            rows.append({
                "model": model_dir.name,
                "scenario": scenario_dir.name,
                "mae": metrics.get("mae", np.nan),
                "rmse": metrics.get("rmse", np.nan),
                "r2": metrics.get("r2", np.nan),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "scenario"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if df["model"].nunique() == 1:
        df.plot(x="scenario", y="mae", kind="bar", ax=ax, legend=False, color="#2c5282")
    else:
        pivot = df.pivot(index="scenario", columns="model", values="mae")
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Test MAE (SOH)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Cross-Battery Transfer", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "transfer_comparison.png", dpi=150)
    plt.close()

    df.to_csv(output_dir / "transfer_comparison.csv", index=False)


def plot_all_single_histories(results_dir, output_dir):
    """为每个单模型结果绘制训练曲线。"""
    results_path = Path(results_dir)
    for history_file in results_path.rglob("history.json"):
        history = load_json(history_file)
        rel = history_file.parent.relative_to(results_path)
        out_name = str(rel).replace(os.sep, "_") + "_history.png"
        plot_training_history(history, output_dir / out_name)


def plot_history_grid(histories, labels, output_path, ncols=2):
    """绘制统一刻度的训练历史子图网格。

    Args:
        histories: list of dict，每个 dict 包含 train_loss 和 val_mae
        labels: list of str，子图标题
        output_path: 输出路径
        ncols: 每行子图数
    """
    n = len(histories)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(6 * ncols, 3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes[:nrows, :ncols * 2]

    # 统一 y 轴范围
    all_train_loss = [v for h in histories for v in h["train_loss"]]
    all_val_mae = [v for h in histories for v in h["val_mae"]]
    train_ylim = (0, max(all_train_loss) * 1.1) if all_train_loss else (0, 1)
    val_ylim = (0, max(all_val_mae) * 1.1) if all_val_mae else (0, 1)

    for idx, (history, label) in enumerate(zip(histories, labels)):
        row = idx // ncols
        col = idx % ncols
        ax_loss = axes[row, col * 2]
        ax_mae = axes[row, col * 2 + 1]
        epochs = list(range(1, len(history["train_loss"]) + 1))

        ax_loss.plot(epochs, history["train_loss"], color="#2c5282", linewidth=1.5)
        ax_loss.set_ylim(train_ylim)
        ax_loss.set_xlabel("Epoch", fontsize=8)
        ax_loss.set_ylabel("MSE Loss", fontsize=8)
        ax_loss.set_title(f"{label} — Train Loss", fontsize=9)
        ax_loss.grid(True, linestyle="--", alpha=0.5)

        ax_mae.plot(epochs, history["val_mae"], color="#dd6b20", linewidth=1.5)
        ax_mae.set_ylim(val_ylim)
        ax_mae.set_xlabel("Epoch", fontsize=8)
        ax_mae.set_ylabel("MAE", fontsize=8)
        ax_mae.set_title(f"{label} — Val MAE", fontsize=9)
        ax_mae.grid(True, linestyle="--", alpha=0.5)

    # 隐藏多余的子图
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col * 2].set_visible(False)
        axes[row, col * 2 + 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_baseline_histories_grid(results_dir, output_dir):
    """绘制统一刻度的基线训练历史网格。"""
    base = Path(results_dir) / "nasa"
    configs = [
        ("transformer", "Transformer"),
        ("batteryCDE", "BatteryCDE"),
        ("lstm", "LSTM"),
        ("neuralCDE", "Neural CDE"),
    ]
    histories = []
    labels = []
    for model, label in configs:
        hist_file = base / model / "history.json"
        if hist_file.exists():
            histories.append(load_json(hist_file))
            labels.append(label)
    if histories:
        plot_history_grid(histories, labels, output_dir / "baseline_histories_grid.png", ncols=2)


def plot_horizon_histories_grid(results_dir, output_dir):
    """绘制统一刻度的 horizon 训练历史网格。"""
    base = Path(results_dir) / "horizon" / "batteryCDE"
    configs = [
        ("h20", "h=20"),
        ("h50", "h=50"),
        ("h80", "h=80"),
        ("h100", "h=100"),
    ]
    histories = []
    labels = []
    for subdir, label in configs:
        hist_file = base / subdir / "batteryCDE" / "history.json"
        if hist_file.exists():
            histories.append(load_json(hist_file))
            labels.append(label)
    if histories:
        plot_history_grid(histories, labels, output_dir / "horizon_histories_grid.png", ncols=2)


def plot_missing_histories_grid(results_dir, output_dir):
    """绘制统一刻度的 missing 训练历史网格（分两页）。"""
    base = Path(results_dir) / "missing" / "batteryCDE"

    page1_configs = [
        ("clean", "clean"),
        ("random30", "random30"),
        ("random50", "random50"),
    ]
    page2_configs = [
        ("block30", "block30"),
        ("block50", "block50"),
    ]

    for page_idx, configs in enumerate([page1_configs, page2_configs], 1):
        histories = []
        labels = []
        for subdir, label in configs:
            hist_file = base / subdir / "batteryCDE" / "history.json"
            if hist_file.exists():
                histories.append(load_json(hist_file))
                labels.append(label)
        if histories:
            ncols = 3 if len(configs) == 3 else 2
            plot_history_grid(histories, labels,
                              output_dir / f"missing_histories_grid_{page_idx}.png",
                              ncols=ncols)


def plot_transfer_histories_grid(results_dir, output_dir):
    """绘制统一刻度的 transfer 训练历史网格。"""
    base = Path(results_dir) / "transfer" / "batteryCDE"
    configs = [
        ("B5B6_to_B7", "B5B6 → B7"),
        ("B5B6_to_B18", "B5B6 → B18"),
        ("B7B18_to_B5B6", "B7B18 → B5B6"),
    ]
    histories = []
    labels = []
    for subdir, label in configs:
        hist_file = base / subdir / "batteryCDE" / "history.json"
        if hist_file.exists():
            histories.append(load_json(hist_file))
            labels.append(label)
    if histories:
        plot_history_grid(histories, labels, output_dir / "transfer_histories_grid.png", ncols=3)


def plot_r2_summary(results_dir, output_dir):
    """绘制各实验 R² 汇总对比图（2x2 子图）。"""
    results_path = Path(results_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    experiments = [
        ("nasa", "scenario", "Model"),
        ("horizon", "scenario", "Horizon"),
        ("missing", "scenario", "Missing Pattern"),
        ("transfer", "scenario", "Transfer Scenario"),
    ]
    titles = ["Baseline", "Horizon", "Missing Data", "Transfer"]

    for idx, (exp_dir, group_col, xlabel) in enumerate(experiments):
        exp_path = results_path / exp_dir
        if not exp_path.exists():
            axes[idx].set_visible(False)
            continue

        rows = []
        for model_dir in exp_path.iterdir():
            if not model_dir.is_dir():
                continue
            for sub_dir in model_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                metric_file = sub_dir / model_dir.name / "test_metrics.json"
                if not metric_file.exists():
                    metric_file = sub_dir / "test_metrics.json"
                if not metric_file.exists():
                    continue
                metrics = load_json(metric_file)
                rows.append({
                    "model": model_dir.name,
                    "scenario": sub_dir.name,
                    "r2": metrics.get("r2", np.nan),
                })

        if not rows:
            axes[idx].set_visible(False)
            continue

        df = pd.DataFrame(rows)
        df["r2"] = pd.to_numeric(df["r2"], errors="coerce")

        # 排序：baseline 按 r2 排序，其余按场景名排序
        if exp_dir == "nasa":
            df = df.sort_values("r2", ascending=True)
        else:
            df = df.sort_values("scenario")

        df.plot(x="scenario", y="r2", kind="bar", ax=axes[idx],
                legend=False, color="#c53030")
        axes[idx].axhline(0, color="black", linewidth=1, linestyle="--")
        axes[idx].set_title(titles[idx], fontsize=12)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("R²")
        axes[idx].grid(True, axis="y", linestyle="--", alpha=0.5)
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "r2_summary.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting results from {results_dir} to {output_dir}")

    plot_baseline_comparison(results_dir, output_dir)
    plot_horizon_comparison(results_dir, output_dir)
    plot_missing_comparison(results_dir, output_dir)
    plot_transfer_comparison(results_dir, output_dir)
    plot_all_single_histories(results_dir, output_dir)
    plot_baseline_histories_grid(results_dir, output_dir)
    plot_horizon_histories_grid(results_dir, output_dir)
    plot_missing_histories_grid(results_dir, output_dir)
    plot_transfer_histories_grid(results_dir, output_dir)
    plot_r2_summary(results_dir, output_dir)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
