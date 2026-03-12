#!/usr/bin/env python
"""
Collect and analyze experiment results.

Usage:
    # Generate summary CSV
    python scripts/collect_results.py --output results/summary.csv
    
    # Generate with statistics
    python scripts/collect_results.py --stats --output results/summary_with_stats.csv
    
    # Only show statistics
    python scripts/collect_results.py --stats-only
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.database import ExperimentDatabase


def load_results(db_path: str = "experiments/experiment_db.csv") -> pd.DataFrame:
    """Load all completed experiments into a DataFrame."""
    import json
    
    db = ExperimentDatabase(db_path)
    experiments = db.get_all_experiments()
    
    records = []
    for exp in experiments:
        if exp.status == "completed" and exp.metrics:
            record = {
                "exp_id": exp.exp_id,
                "model": exp.model,
                "method": exp.method,
                "missing_rate": exp.mr,
                "seed": exp.seed,
                "duration_seconds": exp.duration_seconds,
            }
            # Parse JSON metrics
            try:
                metrics = json.loads(exp.metrics) if isinstance(exp.metrics, str) else exp.metrics
                record.update(metrics)
            except json.JSONDecodeError:
                pass
            records.append(record)
    
    if not records:
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each (model, method, missing_rate) group."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by model, method, missing_rate
    grouped = df.groupby(["model", "method", "missing_rate"])
    
    stats = []
    for (model, method, mr), group in grouped:
        stat = {
            "model": model,
            "method": method,
            "missing_rate": mr,
            "n_seeds": len(group),
        }
        
        # Compute mean and std for all metric columns
        for col in group.columns:
            if col not in ["exp_id", "model", "method", "missing_rate", "seed"]:
                stat[f"{col}_mean"] = group[col].mean()
                stat[f"{col}_std"] = group[col].std()
        
        stats.append(stat)
    
    return pd.DataFrame(stats)


def compare_with_paper(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare results with paper values."""
    # Paper values from Table (approximate, you should update with actual values)
    paper_values = {
        # Format: (model, method, mr): {"MAE": value, "RMSE": value}
        # These are placeholder values - replace with actual paper values
        ("mlp", "baseline", 0.5): {"MAE": 0.050, "RMSE": 0.070},
        ("mlp", "mim", 0.5): {"MAE": 0.035, "RMSE": 0.050},
    }
    
    comparisons = []
    
    for (model, method, mr), paper_mae in paper_values.items():
        subset = df[(df["model"] == model) & (df["method"] == method) & (df["missing_rate"] == mr)]
        if not subset.empty:
            actual_mae = subset["test_mae"].mean()
            paper_mae_val = paper_mae.get("MAE", 0)
            diff = actual_mae - paper_mae_val
            diff_pct = (diff / paper_mae_val * 100) if paper_mae_val > 0 else 0
            
            comparisons.append({
                "model": model,
                "method": method,
                "missing_rate": mr,
                "actual_mae": actual_mae,
                "paper_mae": paper_mae_val,
                "diff": diff,
                "diff_pct": diff_pct
            })
    
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze experiment results")
    parser.add_argument("--db", default="experiments/experiment_db.csv", help="Database path")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--stats", action="store_true", help="Include statistics")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics")
    
    args = parser.parse_args()
    
    # Load database
    db = ExperimentDatabase(args.db)
    db_stats = db.get_statistics()
    
    print("\n📊 Database Summary")
    print("=" * 40)
    print(f"Total:      {db_stats['total']:5d}")
    print(f"Pending:    {db_stats.get('pending', 0):5d}")
    print(f"Running:    {db_stats.get('running', 0):5d}")
    print(f"Completed:  {db_stats.get('completed', 0):5d}")
    print(f"Failed:     {db_stats.get('failed', 0):5d}")
    
    # Load results
    df = load_results(args.db)
    
    if df.empty:
        print("\n⚠️  No completed experiments found")
        return 1
    
    print(f"\n✅ Loaded {len(df)} completed experiments")
    
    # Show sample
    print("\n📋 Sample Results:")
    print(df.head(10).to_string())
    
    # Compute statistics
    if args.stats or args.stats_only:
        stats_df = compute_statistics(df)
        if not stats_df.empty:
            print("\n📈 Statistics by (model, method, missing_rate):")
            print(stats_df.to_string())
            
            if args.output and args.stats:
                stats_output = args.output.replace(".csv", "_stats.csv")
                stats_df.to_csv(stats_output, index=False)
                print(f"\n✅ Statistics saved to: {stats_output}")
    
    # Save raw results
    if args.output and not args.stats_only:
        df.to_csv(args.output, index=False)
        print(f"\n✅ Raw results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
