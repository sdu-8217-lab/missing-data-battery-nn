#!/usr/bin/env python3
"""展示MLP优化配置详情"""
import pandas as pd
df = pd.read_csv('configs_v2/mlp_configs.csv')

print('=' * 70)
print('MLP优化配置详情 (基于上一轮[192,64,32] MAE=0.0157结果)')
print('=' * 70)

# 3层精细搜索
print('\n【策略1】3层精细搜索 (围绕最佳点[192,64,32]):')
print('-' * 50)
for tag, desc in [('fine_t1', '变化第一层'), ('fine_t2', '变化第二层'), ('fine_t3', '变化第三层')]:
    sub = df[df['config_id'].str.contains(tag)]
    if len(sub) > 0:
        print(f'\n  {desc}: {len(sub)}个配置')
        print(f'    参数范围: {sub.estimated_params.min():,} - {sub.estimated_params.max():,}')
        for _, row in sub.head(4).iterrows():
            print(f'    - {row.hidden_config:20s} d={row.dropout:.2f}  params={row.estimated_params:6,}')

# 2层高效区
print('\n\n【策略2】2层高效区扩展:')
print('-' * 50)
sub = df[df['config_id'].str.contains('efficient')]
print(f'  共{len(sub)}个配置 (上一轮[64,32]仅用4K参数获得MAE≈0.0163)')
sub_sorted = sub.sort_values('estimated_params')
print(f'  最小参数: {sub_sorted.iloc[0].hidden_config:12s} params={sub_sorted.iloc[0].estimated_params:5,}')
print(f'  最大参数: {sub_sorted.iloc[-1].hidden_config:12s} params={sub_sorted.iloc[-1].estimated_params:5,}')

# 4层深度验证
print('\n\n【策略3】4层深度验证:')
print('-' * 50)
sub = df[df['config_id'].str.contains('deep')]
print(f'  共{len(sub)}个配置')
for _, row in sub.iterrows():
    print(f'  - {row.hidden_config:30s} d={row.dropout:.2f}  params={row.estimated_params:6,}')

# 1层大宽度
print('\n\n【策略4】1层大宽度探索:')
print('-' * 50)
sub = df[df['config_id'].str.contains('wide')]
print(f'  共{len(sub)}个配置')
sub_sorted = sub.sort_values('estimated_params')
for _, row in sub_sorted.head(5).iterrows():
    print(f'  - {row.hidden_config:12s} d={row.dropout:.2f}  params={row.estimated_params:6,}')

print('\n' + '=' * 70)
print('总计: 266个MLP配置，预计运行时间: ~2.2小时')
print('=' * 70)
