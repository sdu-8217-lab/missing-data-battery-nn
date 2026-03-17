#!/usr/bin/env python
"""
Aggregate results from multiple batch experiments.

Usage:
    # Aggregate all batches with same timestamp
    python scripts/aggregate_results.py --timestamp 20260318_120000
    
    # Aggregate specific batches
    python scripts/aggregate_results.py --timestamp 20260318_120000 --batches 2C 3C
    
    # Aggregate by pattern (all timestamps for a batch)
    python scripts/aggregate_results.py --batch 2C
    
    # Output to specific directory
    python scripts/aggregate_results.py --timestamp 20260318_120000 --output aggregated/
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.paths import list_all_batches


def find_results(timestamp: str = None, batch_id: str = None, base_dir: str = "results") -> list:
    """Find result directories matching criteria."""
    all_batches = list_all_batches(base_dir)
    
    matches = []
    for ts, bid, path in all_batches:
        if timestamp and not ts.startswith(timestamp):
            continue
        if batch_id and bid != batch_id.replace(".", "_"):
            continue
        matches.append((ts, bid, path))
    
    return matches


def aggregate_results(result_dirs: list) -> pd.DataFrame:
    """Aggregate CSV files from multiple directories."""
    all_dfs = []
    
    for ts, bid, path in result_dirs:
        csv_path = path / "results.csv"
        if not csv_path.exists():
            print(f"⚠️  No results.csv in {path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            df['_source'] = str(path)
            df['_timestamp'] = ts
            all_dfs.append(df)
            print(f"✅ Loaded {len(df)} rows from {path}")
        except Exception as e:
            print(f"❌ Error loading {csv_path}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics."""
    if df.empty:
        return df
    
    # Group by relevant columns
    group_cols = ['batch_id', 'model', 'method', 'missing_mode', 'missing_rate']
    group_cols = [c for c in group_cols if c in df.columns]
    
    # Identify metric columns
    metric_cols = [c for c in df.columns 
                  if c.startswith('test_') or c in ['MAE', 'RMSE', 'R2']]
    
    stats = []
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys)) if isinstance(keys, tuple) else {group_cols[0]: keys}
        row['n_seeds'] = len(group)
        
        for col in metric_cols:
            if col in group.columns:
                row[f'{col}_mean'] = group[col].mean()
                row[f'{col}_std'] = group[col].std()
        
        stats.append(row)
    
    return pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description='Aggregate batch experiment results')
    parser.add_argument('--timestamp', help='Filter by timestamp prefix')
    parser.add_argument('--batch', help='Filter by batch ID')
    parser.add_argument('--batches', nargs='+', help='Specific batch IDs')
    parser.add_argument('--base-dir', default='results', help='Base results directory')
    parser.add_argument('--output', '-o', default='results/aggregated', 
                       help='Output directory')
    parser.add_argument('--stats', action='store_true', 
                       help='Generate statistics summary')
    
    args = parser.parse_args()
    
    # Find matching results
    if args.batches:
        all_matches = []
        for batch in args.batches:
            matches = find_results(args.timestamp, batch, args.base_dir)
            all_matches.extend(matches)
        matches = all_matches
    else:
        matches = find_results(args.timestamp, args.batch, args.base_dir)
    
    if not matches:
        print("❌ No matching result directories found")
        print(f"   Base dir: {args.base_dir}")
        print(f"   Timestamp: {args.timestamp}")
        print(f"   Batch: {args.batch or args.batches}")
        return 1
    
    print(f"\nFound {len(matches)} result directories:")
    for ts, bid, path in matches:
        print(f"  - {path}")
    
    # Aggregate
    print("\nAggregating results...")
    df = aggregate_results(matches)
    
    if df.empty:
        print("❌ No data to aggregate")
        return 1
    
    print(f"\n✅ Total rows aggregated: {len(df)}")
    
    # Create output directory
    output_dir = Path(args.output)
    if args.timestamp:
        output_dir = output_dir / args.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    raw_path = output_dir / "raw_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw results saved: {raw_path}")
    
    # Save statistics
    if args.stats:
        stats_df = compute_statistics(df)
        stats_path = output_dir / "statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Statistics saved: {stats_path}")
        
        # Print summary
        print("\n📊 Summary Statistics:")
        print(stats_df.to_string() if len(stats_df) < 20 else 
              stats_df.head(20).to_string() + "\n... (showing first 20 rows)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
