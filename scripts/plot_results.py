#!/usr/bin/env python
"""
Generate paper figures from experiment results.

Usage:
    # Generate all figures
    python scripts/plot_results.py --output-dir results/figures/
    
    # Generate specific figure
    python scripts/plot_results.py --figure mr_mae --output-dir results/figures/
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.database import ExperimentDatabase


def load_data(db_path: str) -> pd.DataFrame:
    """Load completed experiments."""
    import json
    
    db = ExperimentDatabase(db_path)
    experiments = db.get_all_experiments()
    
    records = []
    for exp in experiments:
        if exp.status == "completed" and exp.metrics:
            # Parse JSON metrics
            try:
                metrics = json.loads(exp.metrics) if isinstance(exp.metrics, str) else exp.metrics
            except json.JSONDecodeError:
                metrics = {}
            
            record = {
                "model": exp.model.upper(),
                "method": "MIM" if exp.method == "mim" else "Baseline",
                "missing_rate": exp.mr,
                "seed": exp.seed,
                "MAE": metrics.get("test_mae", metrics.get("mae", None)),
                "RMSE": metrics.get("test_rmse", metrics.get("rmse", None)),
            }
            records.append(record)
    
    return pd.DataFrame(records)


def plot_mr_mae_curves(df: pd.DataFrame, output_path: Path):
    """Plot MR-MAE curves (Figure style from paper)."""
    if df.empty or "MAE" not in df.columns:
        print("⚠️  No data available for MR-MAE plot")
        return
    
    # Aggregate by (model, method, missing_rate)
    grouped = df.groupby(["model", "method", "missing_rate"])["MAE"].agg(["mean", "std"]).reset_index()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    models = ["MLP", "LSTM", "GRU", "CNN1D"]
    colors = {"Baseline": "#e74c3c", "MIM": "#3498db"}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = grouped[grouped["model"] == model]
        
        for method in ["Baseline", "MIM"]:
            method_data = model_data[model_data["method"] == method].sort_values("missing_rate")
            if not method_data.empty:
                ax.plot(method_data["missing_rate"], method_data["mean"], 
                       marker="o", label=method, color=colors[method], linewidth=2)
                # Only add fill_between if we have more than one point
                if len(method_data) > 1:
                    ax.fill_between(method_data["missing_rate"], 
                                   method_data["mean"] - method_data["std"],
                                   method_data["mean"] + method_data["std"],
                                   alpha=0.2, color=colors[method])
        
        ax.set_xlabel("Missing Rate", fontsize=11)
        ax.set_ylabel("MAE", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved MR-MAE curves to: {output_path}")


def plot_method_comparison(df: pd.DataFrame, output_path: Path):
    """Plot MIM vs Baseline comparison."""
    if df.empty or "MAE" not in df.columns:
        print("⚠️  No data available for comparison plot")
        return
    
    # Calculate improvement
    baseline = df[df["method"] == "Baseline"].groupby(["model", "missing_rate"])["MAE"].mean().reset_index()
    baseline.rename(columns={"MAE": "MAE_baseline"}, inplace=True)
    
    mim = df[df["method"] == "MIM"].groupby(["model", "missing_rate"])["MAE"].mean().reset_index()
    mim.rename(columns={"MAE": "MAE_mim"}, inplace=True)
    
    merged = pd.merge(baseline, mim, on=["model", "missing_rate"])
    merged["improvement"] = (merged["MAE_baseline"] - merged["MAE_mim"]) / merged["MAE_baseline"] * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = merged["model"].unique()
    x_pos = range(len(models))
    width = 0.08
    
    mrs = sorted(merged["missing_rate"].unique())
    
    for i, mr in enumerate(mrs):
        mr_data = merged[merged["missing_rate"] == mr]
        offsets = [x + (i - len(mrs)/2) * width for x in x_pos]
        ax.bar(offsets, mr_data["improvement"], width, label=f"MR={mr:.1f}")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Improvement (%)", fontsize=12)
    ax.set_title("MIM Improvement over Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend(title="Missing Rate", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved method comparison to: {output_path}")


def plot_model_comparison(df: pd.DataFrame, output_path: Path):
    """Plot model comparison at different MR levels."""
    if df.empty or "MAE" not in df.columns:
        print("⚠️  No data available for model comparison")
        return
    
    # Aggregate
    grouped = df.groupby(["model", "method", "missing_rate"])["MAE"].mean().reset_index()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, method in enumerate(["Baseline", "MIM"]):
        ax = axes[idx]
        method_data = grouped[grouped["method"] == method]
        
        for model in ["MLP", "LSTM", "GRU", "CNN1D"]:
            model_data = method_data[method_data["model"] == model].sort_values("missing_rate")
            if not model_data.empty:
                ax.plot(model_data["missing_rate"], model_data["MAE"], 
                       marker="o", label=model, linewidth=2)
        
        ax.set_xlabel("Missing Rate", fontsize=11)
        ax.set_ylabel("MAE", fontsize=11)
        ax.set_title(f"{method} Method", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved model comparison to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--db", default="experiments/experiment_db.csv", help="Database path")
    parser.add_argument("--output-dir", "-o", default="results/figures", help="Output directory")
    parser.add_argument("--figure", choices=["mr_mae", "comparison", "models", "all"], 
                       default="all", help="Which figure to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading experiment data...")
    df = load_data(args.db)
    
    if df.empty:
        print("⚠️  No completed experiments found in database")
        print("Run some experiments first: python scripts/run_experiments.py run")
        return 1
    
    print(f"Loaded {len(df)} completed experiments")
    
    # Generate figures
    if args.figure in ["mr_mae", "all"]:
        plot_mr_mae_curves(df, output_dir / "fig_mr_mae_curves.png")
    
    if args.figure in ["comparison", "all"]:
        plot_method_comparison(df, output_dir / "fig_mim_improvement.png")
    
    if args.figure in ["models", "all"]:
        plot_model_comparison(df, output_dir / "fig_model_comparison.png")
    
    print(f"\n✅ All figures saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
