# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
Independent Batch Experiment Plotting Script
Generates publication-ready figures from experimental results

Usage:
    python plot_batch_results.py --input experiments/2C/20260203_045645/
    python plot_batch_results.py --input experiments/2C/20260203_045645/ --output ./figures/
    python plot_batch_results.py --input experiments/2C/20260203_045645/aggregate/results_all.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.visualization.batch_plots import BatchExperimentPlots


def load_results(input_path: Path) -> pd.DataFrame:
    """Load results from directory or CSV file"""
    if input_path.is_file():
        # Direct CSV file
        print(f"Loading results from: {input_path}")
        return pd.read_csv(input_path)
    
    elif input_path.is_dir():
        # Directory - look for aggregate results or merge seed results
        aggregate_csv = input_path / "aggregate" / "results_all.csv"
        
        if aggregate_csv.exists():
            print(f"Loading aggregated results from: {aggregate_csv}")
            return pd.read_csv(aggregate_csv)
        
        # Try to merge individual seed results
        print("Aggregated results not found. Merging individual seed results...")
        seed_dirs = list(input_path.glob("seed_*"))
        
        if not seed_dirs:
            print(f"Error: No results found in {input_path}")
            print("Expected: aggregate/results_all.csv or seed_*/results.csv")
            sys.exit(1)
        
        all_results = []
        for seed_dir in seed_dirs:
            result_file = seed_dir / "results.csv"
            if result_file.exists():
                all_results.append(pd.read_csv(result_file))
        
        if not all_results:
            print(f"Error: No result CSV files found")
            sys.exit(1)
        
        combined = pd.concat(all_results, ignore_index=True)
        print(f"Merged {len(all_results)} seed results: {len(combined)} total records")
        return combined
    
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready plots from batch experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input experiments/2C/20260203_045645/
  %(prog)s --input experiments/2C/20260203_045645/ --output ./my_figures/
  %(prog)s --input results_all.csv --output ./figures/
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory or CSV file containing experiment results')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for figures (default: input_dir/figures/)')
    parser.add_argument('--metrics', '-m', type=str, nargs='+',
                       default=['mae', 'rmse', 'r2'],
                       help='Metrics to plot (default: mae rmse r2)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    elif input_path.is_dir():
        output_dir = input_path / "figures_manual"
    else:
        output_dir = input_path.parent / "figures_manual"
    
    print("=" * 70)
    print("Batch Experiment Plotting Tool")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()
    
    # Load data
    df = load_results(input_path)
    
    # Print summary
    print("Data Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Unique seeds: {df['seed'].nunique()}")
    print(f"  Model configs: {df['model_name'].nunique()}")
    print(f"  Missing rates: {sorted(df['missing_rate'].unique())}")
    print()
    
    # Create plotter and generate all plots
    print("Generating plots...")
    plotter = BatchExperimentPlots(output_dir)
    
    # Generate plots for each requested metric
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"  Warning: Metric '{metric}' not found in data, skipping...")
            continue
        
        print(f"  Plotting {metric.upper()}...")
        metric_label = metric.upper() if metric != 'r2' else 'R虏'
        
        # Metric curves with CI
        plotter.plot_metric_curves(df, metric, metric_label)
        
        # Improvement rate
        plotter.plot_improvement_rate(df, metric)
        
        # Distribution
        plotter.plot_metric_distribution(df, metric)
        
        # Heatmap
        plotter.plot_metric_heatmap(df, metric)
    
    # Seed stability (always plot)
    print("  Plotting seed stability analysis...")
    plotter.plot_seed_stability(df)
    
    print()
    print("=" * 70)
    print("Plotting Complete!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 70)
    print("\nGenerated files:")
    
    # List generated files
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.png")):
            print(f"  - {f.name}")


if __name__ == '__main__':
    main()

