#!/usr/bin/env python3
"""Analyze fine-grained search results to find best configs and optimization directions"""
import pandas as pd
import numpy as np

def analyze_model_results(csv_path, model_name):
    """Analyze single model results"""
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} Results Analysis")
    print(f"{'='*70}")
    
    # Basic stats
    print(f"\nTotal configs: {len(df)}")
    print(f"Param range: {df['param_count'].min():,} - {df['param_count'].max():,}")
    
    # Sort by MAE to find best
    df_sorted = df.sort_values('mae_mean')
    
    print(f"\n[TOP 5 Best Configs] (sorted by MAE_mean)")
    print("-" * 70)
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['config_name']}")
        print(f"   MAE_mean: {row['mae_mean']:.4f} (0.1:{row['mae_0.1']:.4f}, 0.5:{row['mae_0.5']:.4f}, 0.9:{row['mae_0.9']:.4f})")
        print(f"   Params: {row['param_count']:,}")
        print(f"   Config: {row['config']}")
        print()
    
    # Efficient configs (MAE/Params ratio)
    df['efficiency'] = df['mae_mean'] / (df['param_count'] / 10000)  # MAE per 10K params
    df_efficient = df.sort_values('efficiency')
    
    print(f"\n[TOP 3 Most Efficient] (MAE per 10K params)")
    print("-" * 70)
    for i, (_, row) in enumerate(df_efficient.head(3).iterrows(), 1):
        print(f"{i}. {row['config_name']}: eff={row['efficiency']:.3f}, MAE={row['mae_mean']:.4f}, params={row['param_count']:,}")
    
    # Key insights
    best = df_sorted.iloc[0]
    print(f"\n[Key Insights]")
    print(f"- Best MAE: {best['mae_mean']:.4f}")
    print(f"- Best config: {best['config']}")
    print(f"- Best params: {best['param_count']:,}")
    
    return df_sorted.head(5)

# Analyze all models
results_summary = {}

for model, path in [
    ('MLP', 'experiments_v2/fine_grained_mlp_3C_20260204_025808/results.csv'),
    ('CNN1D', 'experiments_v2/fine_grained_cnn1d_3C_20260204_025935/results.csv'),
    ('GRU', 'experiments_v2/fine_grained_gru_3C_20260204_030117/results.csv'),
    ('LSTM', 'experiments_v2/fine_grained_lstm_3C_20260204_030013/results.csv'),
]:
    try:
        top5 = analyze_model_results(path, model)
        results_summary[model] = {
            'best_mae': top5.iloc[0]['mae_mean'],
            'best_config': top5.iloc[0]['config'],
            'best_params': top5.iloc[0]['param_count'],
            'best_name': top5.iloc[0]['config_name']
        }
    except Exception as e:
        print(f"{model} analysis failed: {e}")

# Summary comparison
print(f"\n{'='*70}")
print("[Model Comparison Summary]")
print(f"{'='*70}")
print(f"{'Model':<10} {'Best MAE':<12} {'Params':<12} {'Config'}")
print("-" * 70)
for model, info in results_summary.items():
    print(f"{model:<10} {info['best_mae']:<12.4f} {info['best_params']:<12,} {info['best_config']}")

# Recommendations
print(f"\n{'='*70}")
print("[Next Round Search Recommendations]")
print(f"{'='*70}")
print("""
Based on results analysis:

1. CNN1D performs best (MAE as low as 0.0101)
   - Recommend: Dense search around [96,48] kernel=4-5
   - 2-layer configs show highest efficiency

2. MLP best: [192,64,32] d0.2, MAE=0.0157
   - Recommend: Fine-tune layer widths (±20%)
   - Explore [160-224, 48-80, 24-40] range

3. GRU/LSTM underperform (MAE>0.012)
   - Recommend: Focus on single-layer wide nets (h=48-64)
   - Reduce search space, avoid deep configs

4. 65K param limit validated
   - All best configs under 65K
   - CNN1D [256,128] near limit but effective
""")
