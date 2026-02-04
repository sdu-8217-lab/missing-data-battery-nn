#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('experiments_v2/arch_search_3C_fine_20260204_024951/results.csv')

print('=' * 70)
print('Previous Fine-Grained Search Results Analysis')
print('=' * 70)
print(f'Total architectures tested: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()

# Group by model
for model_type in df['model_type'].unique():
    model_df = df[df['model_type'] == model_type]
    print(f'[{model_type.upper()}] - {len(model_df)} configs')
    if len(model_df) > 0:
        best_idx = model_df['mae_mean'].idxmin()
        best = model_df.loc[best_idx]
        print(f"  Best MAE: {best['mae_mean']:.4f}")
        print(f"  Best config: {best['config_name']}")
        print(f"  Params: {best['param_count']:,}")
    print()

# Top 10 overall
print('=' * 70)
print('Top 10 Overall (by MAE)')
print('=' * 70)
top10 = df.nsmallest(10, 'mae_mean')[['config_name', 'model_type', 'mae_mean', 'param_count', 'config']]
print(top10.to_string(index=False))

# MAE distribution
print()
print('=' * 70)
print('MAE Statistics by Model')
print('=' * 70)
print(df.groupby('model_type')['mae_mean'].describe())
