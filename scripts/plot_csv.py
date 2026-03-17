#!/usr/bin/env python
"""
Standalone plotting script for experiment results CSV.

Decoupled from experiment execution - can re-run visualization on existing results.

Usage:
    # Plot from specific CSV
    python scripts/plot_csv.py results/20260318_120000_2C/results.csv
    
    # Plot from latest results for a batch
    python scripts/plot_csv.py --batch 2C
    
    # Specify output directory
    python scripts/plot_csv.py results/xxx/results.csv --output-dir figures/
    
    # Generate specific figure only
    python scripts/plot_csv.py results/xxx/results.csv --figure mr_mae
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.paths import find_latest_results


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Standardize column names (handle variations)
    col_mapping = {
        'test_mae': 'MAE',
        'test_rmse': 'RMSE',
        'test_r2': 'R2',
        'missing_rate': 'missing_rate',
        'mr': 'missing_rate',
        'method': 'method',
        'model': 'model',
        'seed': 'seed',
    }
    
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Normalize method names
    if 'method' in df.columns:
        df['method'] = df['method'].str.upper().str.replace('MIM', 'MIM').str.replace('BASELINE', 'Baseline')
    
    # Normalize model names
    if 'model' in df.columns:
        df['model'] = df['model'].str.upper()
    
    return df


def plot_mr_mae_curves(df: pd.DataFrame, output_path: Path):
    """Plot MR-MAE curves for each model."""
    if df.empty or 'MAE' not in df.columns:
        print("⚠️  No MAE data available")
        return False
    
    required_cols = ['model', 'method', 'missing_rate', 'MAE']
    if not all(c in df.columns for c in required_cols):
        print(f"⚠️  Missing required columns: {[c for c in required_cols if c not in df.columns]}")
        return False
    
    # Aggregate statistics
    grouped = df.groupby(['model', 'method', 'missing_rate'])['MAE'].agg(['mean', 'std', 'count']).reset_index()
    
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Determine subplot layout
    if n_models <= 2:
        nrows, ncols = 1, n_models
        figsize = (6 * ncols, 5)
    elif n_models <= 4:
        nrows, ncols = 2, 2
        figsize = (12, 10)
    else:
        nrows = (n_models + 1) // 2
        ncols = 2
        figsize = (12, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = {'BASELINE': '#e74c3c', 'MIM': '#3498db', 'Baseline': '#e74c3c'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = grouped[grouped['model'] == model]
        
        for method in ['BASELINE', 'MIM', 'Baseline']:
            method_data = model_data[model_data['method'] == method].sort_values('missing_rate')
            if method_data.empty:
                continue
            
            color = colors.get(method, '#2ecc71')
            method_label = 'MIM' if 'MIM' in method else 'Baseline'
            
            ax.plot(method_data['missing_rate'], method_data['mean'],
                   marker='o', label=method_label, color=color, linewidth=2, markersize=6)
            
            # Add shaded error region if multiple seeds
            if method_data['count'].iloc[0] > 1:
                ax.fill_between(method_data['missing_rate'],
                               method_data['mean'] - method_data['std'],
                               method_data['mean'] + method_data['std'],
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Missing Rate', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved MR-MAE curves: {output_path}")
    return True


def plot_method_comparison(df: pd.DataFrame, output_path: Path):
    """Plot MIM vs Baseline improvement."""
    if df.empty or 'MAE' not in df.columns:
        print("⚠️  No MAE data available")
        return False
    
    # Calculate improvement
    # Treat non-MIM methods as baseline (mean, knn, iterative, zero, etc.)
    is_mim = df['method'].str.upper() == 'MIM'
    baseline = df[~is_mim].groupby(['model', 'missing_rate'])['MAE'].mean().reset_index()
    baseline.rename(columns={'MAE': 'MAE_baseline'}, inplace=True)
    
    mim = df[is_mim].groupby(['model', 'missing_rate'])['MAE'].mean().reset_index()
    mim.rename(columns={'MAE': 'MAE_mim'}, inplace=True)
    
    if baseline.empty or mim.empty:
        print("⚠️  Need both Baseline and MIM data for comparison")
        return False
    
    merged = pd.merge(baseline, mim, on=['model', 'missing_rate'])
    merged['improvement'] = (merged['MAE_baseline'] - merged['MAE_mim']) / merged['MAE_baseline'] * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = merged['model'].unique()
    x_pos = range(len(models))
    width = 0.08
    
    mrs = sorted(merged['missing_rate'].unique())
    
    for i, mr in enumerate(mrs):
        mr_data = merged[merged['missing_rate'] == mr]
        offsets = [x + (i - len(mrs)/2) * width for x in x_pos]
        values = [mr_data[mr_data['model'] == m]['improvement'].values[0] 
                 if len(mr_data[mr_data['model'] == m]) > 0 else 0 
                 for m in models]
        ax.bar(offsets, values, width, label=f'MR={mr:.1f}')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('MIM Improvement over Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend(title='Missing Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved method comparison: {output_path}")
    return True


def plot_summary_table(df: pd.DataFrame, output_path: Path):
    """Generate summary statistics table as figure."""
    if df.empty:
        print("⚠️  No data available")
        return False
    
    # Compute summary statistics
    summary = df.groupby(['model', 'method', 'missing_rate']).agg({
        'MAE': ['mean', 'std', 'count']
    }).reset_index()
    
    summary.columns = ['model', 'method', 'missing_rate', 'MAE_mean', 'MAE_std', 'n_runs']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(4, len(summary) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Format table
    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row['model'],
            row['method'],
            f"{row['missing_rate']:.1f}",
            f"{row['MAE_mean']:.4f}",
            f"{row['MAE_std']:.4f}" if pd.notna(row['MAE_std']) else '-',
            f"{int(row['n_runs'])}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Method', 'MR', 'MAE Mean', 'MAE Std', 'N'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.1, 0.2, 0.2, 0.1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved summary table: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot experiment results from CSV')
    parser.add_argument('csv', nargs='?', help='Path to results CSV file')
    parser.add_argument('--batch', help='Find and use latest results for batch (e.g., 2C)')
    parser.add_argument('--output-dir', '-o', help='Output directory for figures')
    parser.add_argument('--figure', choices=['mr_mae', 'comparison', 'summary', 'all'],
                       default='all', help='Which figure to generate')
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.csv:
        csv_path = Path(args.csv)
    elif args.batch:
        results_dir = find_latest_results(args.batch)
        if results_dir is None:
            print(f"❌ No results found for batch: {args.batch}")
            return 1
        csv_path = results_dir / 'results.csv'
        print(f"Using latest results: {results_dir}")
    else:
        print("❌ Please provide either --batch or a CSV path")
        return 1
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"Loaded {len(df)} rows")
    
    if df.empty:
        print("❌ No data to plot")
        return 1
    
    # Generate figures
    success = []
    
    if args.figure in ['mr_mae', 'all']:
        success.append(plot_mr_mae_curves(df, output_dir / 'fig_mr_mae_curves.png'))
    
    if args.figure in ['comparison', 'all']:
        success.append(plot_method_comparison(df, output_dir / 'fig_mim_improvement.png'))
    
    if args.figure in ['summary', 'all']:
        success.append(plot_summary_table(df, output_dir / 'fig_summary_table.png'))
    
    print(f"\n✅ Figures saved to: {output_dir}")
    return 0 if any(success) else 1


if __name__ == '__main__':
    sys.exit(main())
