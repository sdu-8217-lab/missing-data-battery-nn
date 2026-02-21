# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""妫€鏌ヤ紭鍖栧悗鐨勯厤缃?""
import pandas as pd

print("=" * 80)
print("浼樺寲鍚庨厤缃鏌?(configs_v2)")
print("=" * 80)

for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
    try:
        df = pd.read_csv(f'configs_v2/{model}_configs.csv')
        df = df[df['status'] == 'pending']
        
        print(f"\n{'='*40}")
        print(f"{model.upper()}: {len(df)} configs")
        print(f"{'='*40}")
        
        # 鍙傛暟鍒嗗竷
        print(f"Param range: {df['estimated_params'].min():,} - {df['estimated_params'].max():,}")
        
        # 鎸夊眰鍒嗗竷
        for lt in sorted(df['layer_type'].unique()):
            sub = df[df['layer_type'] == lt]
            print(f"  Layer {lt}: {len(sub):3d} configs, param: {sub['estimated_params'].min():6,}-{sub['estimated_params'].max():6,}")
        
        # 鏄剧ず鍏抽敭閰嶇疆绀轰緥
        print(f"\n  绀轰緥閰嶇疆:")
        sample = df[df['layer_type'] == df['layer_type'].mode()[0]].head(3)
        for _, row in sample.iterrows():
            print(f"    {row['config_id']}: {row['hidden_config']}, d={row['dropout']}, params={row['estimated_params']:,}")
            
    except Exception as e:
        print(f"{model}: Error - {e}")

print("\n" + "=" * 80)

