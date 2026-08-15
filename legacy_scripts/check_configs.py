#!/usr/bin/env python3
"""检查生成的配置"""
import pandas as pd

for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
    df = pd.read_csv(f'configs/{model}_configs.csv')
    df = df[df['status'] == 'pending']
    print(f'\n=== {model.upper()} ({len(df)} configs) ===')
    print(f'Param range: {df["estimated_params"].min()} - {df["estimated_params"].max()}')
    print('Param distribution by layer:')
    for lt in [1, 2, 3, 4]:
        sub = df[df['layer_type'] == lt]
        if len(sub) > 0:
            print(f'  Layer {lt}: {len(sub):4d} configs, param: {sub["estimated_params"].min():6d}-{sub["estimated_params"].max():6d}')
