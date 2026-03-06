# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""鍒嗘瀽configs_v2鐩綍涓嬬殑妯″瀷绛涢€夌粨鏋?""
import pandas as pd
import numpy as np
import json

def analyze_csv_results(csv_path, model_name):
    """鍒嗘瀽鍗曚釜CSV鏂囦欢"""
    df = pd.read_csv(csv_path)
    
    # 鍙彇瀹屾垚鐨勫疄楠?
    completed = df[df['status'] == 'completed'].copy()
    
    print(f"\n{'='*70}")
    print(f"銆恵model_name}銆戠粨鏋滃垎鏋?)
    print(f"{'='*70}")
    
    # 鍩烘湰缁熻
    total = len(df)
    success = len(completed)
    failed = len(df[df['status'] == 'failed'])
    
    print(f"\n鎬婚厤缃? {total}")
    print(f"鎴愬姛: {success} ({success/total*100:.1f}%)")
    print(f"澶辫触: {failed} ({failed/total*100:.1f}%)")
    
    if len(completed) == 0:
        print("鏃犳垚鍔熷畬鎴愮殑瀹為獙")
        return None
    
    # 鍙傛暟鑼冨洿
    print(f"\n鍙傛暟鑼冨洿: {completed['estimated_params'].min():,} - {completed['estimated_params'].max():,}")
    
    # MAE缁熻
    print(f"MAE鑼冨洿: {completed['mae'].min():.4f} - {completed['mae'].max():.4f}")
    print(f"骞冲潎MAE: {completed['mae'].mean():.4f}")
    
    # 鎸塎AE鎺掑簭鍙朤OP 10
    top10 = completed.sort_values('mae').head(10)
    
    print(f"\n銆怲OP 10 鏈€浣抽厤缃€?鎸塎AE鎺掑簭)")
    print("-" * 70)
    print(f"{'鎺掑悕':<4} {'閰嶇疆ID':<20} {'MAE':<10} {'鍙傛暟閲?:<12} {'R2':<10} {'閰嶇疆'}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        config_str = row['hidden_config'].replace('"', '')
        print(f"{i:<4} {row['config_id']:<20} {row['mae']:<10.4f} "
              f"{int(row['estimated_params']):<12,} {row['r2']:<10.4f} {config_str}")
    
    # 鏁堢巼鍒嗘瀽 (MAE/params)
    completed['efficiency'] = completed['mae'] / (completed['estimated_params'] / 10000)
    efficient = completed.sort_values('efficiency').head(5)
    
    print(f"\n銆怲OP 5 鏈€楂樻晥閰嶇疆銆?MAE per 10K params)")
    print("-" * 70)
    for i, (_, row) in enumerate(efficient.iterrows(), 1):
        print(f"{i}. {row['config_id']}: MAE={row['mae']:.4f}, "
              f"params={int(row['estimated_params']):,}, eff={row['efficiency']:.3f}")
    
    # 灞傛暟鍒嗘瀽
    print(f"\n銆愬眰鏁板垎鏋愩€?)
    for layer_type in sorted(completed['layer_type'].unique()):
        layer_df = completed[completed['layer_type'] == layer_type]
        print(f"  {int(layer_type)}灞? 鏁伴噺={len(layer_df)}, "
              f"鏈€浣矼AE={layer_df['mae'].min():.4f}, "
              f"骞冲潎MAE={layer_df['mae'].mean():.4f}")
    
    return top10.iloc[0]  # 杩斿洖鏈€浣抽厤缃?

# 鍒嗘瀽鎵€鏈夋ā鍨?
print("="*70)
print("Configs_v2 妯″瀷绛涢€夌粨鏋滃垎鏋?)
print("="*70)

results = {}
best_configs = {}

for model, csv_file in [
    ('MLP', 'configs_v2/mlp_configs.csv'),
    ('CNN1D', 'configs_v2/cnn1d_configs.csv'),
    ('GRU', 'configs_v2/gru_configs.csv'),
    ('LSTM', 'configs_v2/lstm_configs.csv'),
]:
    try:
        best = analyze_csv_results(csv_file, model)
        if best is not None:
            best_configs[model] = best
    except Exception as e:
        print(f"\n{model} 鍒嗘瀽澶辫触: {e}")

# 鎬诲姣?
print(f"\n\n{'='*70}")
print("銆愭ā鍨嬪姣旀€荤粨銆?)
print(f"{'='*70}")
print(f"{'妯″瀷':<8} {'鏈€浣矼AE':<10} {'鍙傛暟閲?:<12} {'R2':<10} {'閰嶇疆'}")
print("-" * 70)

for model, row in best_configs.items():
    config_str = row['hidden_config'].replace('"', '')
    print(f"{model:<8} {row['mae']:<10.4f} {int(row['estimated_params']):<12,} "
          f"{row['r2']:<10.4f} {config_str}")

# 鍏抽敭鍙戠幇
print(f"\n{'='*70}")
print("銆愬叧閿彂鐜颁笌寤鸿銆?)
print(f"{'='*70}")

# 鎵惧嚭鏈€浣虫ā鍨?
best_model = min(best_configs.items(), key=lambda x: x[1]['mae'])
print(f"\n1. 鏈€浣虫ā鍨? {best_model[0]}")
print(f"   - 鏈€浣矼AE: {best_model[1]['mae']:.4f}")
print(f"   - 閰嶇疆: {best_model[1]['hidden_config']}")
print(f"   - 鍙傛暟閲? {int(best_model[1]['estimated_params']):,}")

# 鏁堢巼瀵规瘮
print(f"\n2. 鏁堢巼瀵规瘮(MAE per 10K params锛岃秺浣庤秺濂?:")
for model, csv_file in [
    ('MLP', 'configs_v2/mlp_configs.csv'),
    ('CNN1D', 'configs_v2/cnn1d_configs.csv'),
    ('GRU', 'configs_v2/gru_configs.csv'),
    ('LSTM', 'configs_v2/lstm_configs.csv'),
]:
    df = pd.read_csv(csv_file)
    completed = df[df['status'] == 'completed']
    if len(completed) > 0:
        completed['efficiency'] = completed['mae'] / (completed['estimated_params'] / 10000)
        best_eff = completed.sort_values('efficiency').iloc[0]
        print(f"   {model}: {best_eff['efficiency']:.3f} (MAE={best_eff['mae']:.4f}, "
              f"params={int(best_eff['estimated_params']):,})")

print(f"\n3. 涓嬩竴杞悳绱㈠缓璁?")
print(f"   - CNN1D: 鍥寸粫[56-72, 32] kernel=2-5, dropout=0.1-0.3鍔犲瘑鎼滅储")
print(f"   - MLP: 褰撳墠缁撴灉杈冨樊锛屽缓璁噸鏂板瑙嗘悳绱㈣寖鍥?)
print(f"   - GRU/LSTM: 鍗曞眰瀹界綉缁?h=40-64)琛ㄧ幇鏇村ソ锛屽缓璁噺灏戝眰鏁?)

