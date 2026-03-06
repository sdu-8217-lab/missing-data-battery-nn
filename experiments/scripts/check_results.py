# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
import pandas as pd
import sys

results_file = 'experiments_v2/arch_search_3C_coarse_20260204_022047/results.csv'
df = pd.read_csv(results_file)

print('=' * 70)
print('Coarse Search Results Summary')
print('=' * 70)
print(f'Total architectures: {len(df)}')
print()

for model_type in df['model_type'].unique():
    model_df = df[df['model_type'] == model_type]
    print(f'[{model_type.upper()}]')
    print(f'  Passed: {len(model_df)}')
    if len(model_df) > 0:
        best_idx = model_df['mae_mean'].idxmin()
        best = model_df.loc[best_idx]
        print(f'  Best MAE: {best["mae_mean"]:.4f}')
        print(f'  Best config: {best["config_name"]}')
        print(f'  Params: {best["param_count"]:,}')
    print()

print('=' * 70)
print('Top 10 (by MAE)')
print('=' * 70)
top10 = df.nsmallest(10, 'mae_mean')[['config_name', 'model_type', 'mae_mean', 'param_count', 'training_time']]
print(top10.to_string(index=False))

print()
print('=' * 70)
print('MAE Statistics')
print('=' * 70)
print(df['mae_mean'].describe())

