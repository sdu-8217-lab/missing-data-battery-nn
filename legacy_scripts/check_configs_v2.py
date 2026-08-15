#!/usr/bin/env python3
"""检查优化后的配置"""
import pandas as pd

print("=" * 80)
print("优化后配置检查 (configs_v2)")
print("=" * 80)

for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
    try:
        df = pd.read_csv(f'configs_v2/{model}_configs.csv')
        df = df[df['status'] == 'pending']
        
        print(f"\n{'='*40}")
        print(f"{model.upper()}: {len(df)} configs")
        print(f"{'='*40}")
        
        # 参数分布
        print(f"Param range: {df['estimated_params'].min():,} - {df['estimated_params'].max():,}")
        
        # 按层分布
        for lt in sorted(df['layer_type'].unique()):
            sub = df[df['layer_type'] == lt]
            print(f"  Layer {lt}: {len(sub):3d} configs, param: {sub['estimated_params'].min():6,}-{sub['estimated_params'].max():6,}")
        
        # 显示关键配置示例
        print(f"\n  示例配置:")
        sample = df[df['layer_type'] == df['layer_type'].mode()[0]].head(3)
        for _, row in sample.iterrows():
            print(f"    {row['config_id']}: {row['hidden_config']}, d={row['dropout']}, params={row['estimated_params']:,}")
            
    except Exception as e:
        print(f"{model}: Error - {e}")

print("\n" + "=" * 80)
