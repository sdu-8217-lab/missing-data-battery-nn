"""汇总 BatteryCDE 复现实验结果到 CSV

用法:
    python scripts/aggregate_results.py --results_dir ../results --output ../results/summary.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate BatteryCDE results")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Root results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(results_dir: Path):
    rows = []

    # 单模型结果：results/nasa/{model}/test_metrics.json
    nasa_dir = results_dir / "nasa"
    if nasa_dir.exists():
        for model_dir in nasa_dir.iterdir():
            if not model_dir.is_dir():
                continue
            metric_file = model_dir / "test_metrics.json"
            if metric_file.exists():
                metrics = load_json(metric_file)
                rows.append({
                    "experiment": "nasa_baseline",
                    "model": model_dir.name,
                    "scenario": "default",
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2": metrics.get("r2"),
                    "path": str(metric_file),
                })

    # horizon 实验：results/horizon/{model}/h{H}/{model}/test_metrics.json
    horizon_dir = results_dir / "horizon"
    if horizon_dir.exists():
        for model_dir in horizon_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for h_dir in model_dir.iterdir():
                if not h_dir.is_dir():
                    continue
                metric_file = h_dir / model_dir.name / "test_metrics.json"
                if not metric_file.exists():
                    metric_file = h_dir / "test_metrics.json"
                if metric_file.exists():
                    metrics = load_json(metric_file)
                    rows.append({
                        "experiment": "horizon",
                        "model": model_dir.name,
                        "scenario": h_dir.name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "path": str(metric_file),
                    })

    # missing 实验：results/missing/{model}/{scenario}/{model}/test_metrics.json
    missing_dir = results_dir / "missing"
    if missing_dir.exists():
        for model_dir in missing_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                metric_file = exp_dir / model_dir.name / "test_metrics.json"
                if not metric_file.exists():
                    metric_file = exp_dir / "test_metrics.json"
                if metric_file.exists():
                    metrics = load_json(metric_file)
                    rows.append({
                        "experiment": "missing",
                        "model": model_dir.name,
                        "scenario": exp_dir.name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "path": str(metric_file),
                    })

    # transfer 实验：results/transfer/{model}/{scenario}/{model}/test_metrics.json
    transfer_dir = results_dir / "transfer"
    if transfer_dir.exists():
        for model_dir in transfer_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for scenario_dir in model_dir.iterdir():
                if not scenario_dir.is_dir():
                    continue
                metric_file = scenario_dir / model_dir.name / "test_metrics.json"
                if not metric_file.exists():
                    metric_file = scenario_dir / "test_metrics.json"
                if metric_file.exists():
                    metrics = load_json(metric_file)
                    rows.append({
                        "experiment": "transfer",
                        "model": model_dir.name,
                        "scenario": scenario_dir.name,
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "path": str(metric_file),
                    })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else results_dir / "summary.csv"

    df = collect_results(results_dir)
    if df.empty:
        print(f"No results found in {results_dir}")
        return

    df = df.sort_values(["experiment", "model", "scenario"])
    df.to_csv(output_path, index=False)
    print(f"Aggregated {len(df)} result entries to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
