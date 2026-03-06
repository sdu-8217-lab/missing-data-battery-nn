# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""鍒嗘瀽 configs_v3 绗笁杞簿缁嗘悳绱㈢祼鏋?""
import pandas as pd
import numpy as np

def analyze_model(csv_path, model_name):
    """鍒嗘瀽鍗曚釜妯″瀷缁撴灉"""
    df = pd.read_csv(csv_path)
    completed = df[df['status'] == 'completed'].copy()
    
    print(f"\n{'='*70}")
    print(f"銆恵model_name}銆慍onfigs_v3 鍒嗘瀽缁撴灉")
    print(f"{'='*70}")
    
    total = len(df)
    success = len(completed)
    failed = len(df[df['status'] == 'failed'])
    
    print(f"\n鎬婚厤缃? {total} | 鎴愬姛: {success} | 澶辫触: {failed}")
    
    if len(completed) == 0:
        return None
    
    # MAE缁熻
    print(f"MAE鑼冨洿: {completed['mae'].min():.4f} - {completed['mae'].max():.4f}")
    print(f"骞冲潎MAE: {completed['mae'].mean():.4f}")
    print(f"涓綅鏁癕AE: {completed['mae'].median():.4f}")
    
    # TOP 10鏈€浣?
    top10 = completed.sort_values('mae').head(10)
    print(f"\n銆怲OP 10 鏈€浣抽厤缃€?)
    print("-" * 70)
    print(f"{'鎺掑悕':<4} {'閰嶇疆ID':<18} {'MAE':<10} {'R2':<10} {'鍙傛暟閲?:<12} {'閰嶇疆'}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        config_str = row['hidden_config'].replace('"', '').replace(' ', '')
        print(f"{i:<4} {row['config_id']:<18} {row['mae']:<10.4f} "
              f"{row['r2']:<10.4f} {int(row['estimated_params']):<12,} {config_str}")
    
    # 涓巚2瀵规瘮
    print(f"\n銆愪笌 configs_v2 瀵规瘮銆?)
    print("-" * 70)
    
    return top10.iloc[0] if len(top10) > 0 else None

print("="*70)
print("Configs_v3 绗笁杞簿缁嗘悳绱㈢粨鏋滃垎鏋?)
print("="*70)

# 鍒嗘瀽鎵€鏈夋ā鍨?
results = {}
for model, path in [
    ('MLP', 'configs_v3/mlp_configs.csv'),
    ('CNN1D', 'configs_v3/cnn1d_configs.csv'),
    ('GRU', 'configs_v3/gru_configs.csv'),
    ('LSTM', 'configs_v3/lstm_configs.csv'),
]:
    try:
        best = analyze_model(path, model)
        if best is not None:
            results[model] = best
    except Exception as e:
        print(f"\n{model} 鍒嗘瀽澶辫触: {e}")

# 鎬诲姣?
print(f"\n\n{'='*70}")
print("銆怌onfigs_v3 vs Configs_v2 瀵规瘮銆?)
print(f"{'='*70}")

# 璇诲彇v2鏈€浣?
v2_best = {
    'CNN1D': 0.0061,
    'LSTM': 0.0069,
    'GRU': 0.0085,
    'MLP': 0.0097,
}

print(f"{'妯″瀷':<8} {'v2鏈€浣?:<10} {'v3鏈€浣?:<10} {'鍙樺寲':<10} {'鐘舵€?}")
print("-" * 70)

for model in ['CNN1D', 'LSTM', 'GRU', 'MLP']:
    if model in results:
        v3_mae = results[model]['mae']
        v2_mae = v2_best.get(model, float('inf'))
        change = v3_mae - v2_mae
        change_pct = (change / v2_mae) * 100
        status = "馃敶 閫€姝? if change > 0.001 else ("馃煝 杩涙" if change < -0.001 else "鈿?鎸佸钩")
        print(f"{model:<8} {v2_mae:<10.4f} {v3_mae:<10.4f} {change:+.4f}({change_pct:+.1f}%) {status}")
    else:
        print(f"{model:<8} {v2_best.get(model, 0):<10.4f} {'N/A':<10} {'N/A':<10} 鈿?鏃犳暟鎹?)

print(f"\n{'='*70}")
print("銆愬叧閿彂鐜般€?)
print(f"{'='*70}")

print("""
1. CNN1D v3缁撴灉娉㈠姩澶э細
   - 鏈€浣矼AE 0.0066 (涓巚2鐨?.0061鎺ヨ繎)
   - 浣嗗嚭鐜板ぇ閲廙AE>0.05鐨勫け璐?涓嶇ǔ瀹氳缁?
   - 鍙兘鍘熷洜锛氬涔犵巼杩囧ぇ鎴栧垵濮嬪寲鏁忔劅

2. MLP绋冲畾浣嗘棤绐佺牬锛?
   - 鏈€浣矼AE 0.0128 (v2鏄?.0097)
   - 4灞傛灦鏋勬湭甯︽潵鏄捐憲鎻愬崌

3. GRU/LSTM v3琛ㄧ幇锛?
   - GRU鏈€浣?.0069锛屼笌v2鐨?.0085鏈夎繘姝?
   - LSTM鏈€浣?.0077锛屾瘮v2鐨?.0069鐣ユ湁閫€姝?
   - 鏁翠綋RNN绫绘ā鍨嬬ǔ瀹氭€ц緝濂?

4. 鏀舵暃闂锛?
   - 閮ㄥ垎CNN1D閰嶇疆鍑虹幇MAE>0.1鐨勭伨闅炬€х粨鏋?
   - R2涓鸿礋鍊硷紝璇存槑妯″瀷瀹屽叏澶辨晥
   - 鍙兘闇€瑕佽皟鏁村涔犵巼鎴栧鍔犳棭鍋滄晱鎰熷害
""")

print(f"{'='*70}")
print("銆愬缓璁€?)
print(f"{'='*70}")
print("""
1. 鍥哄畾CNN1D鏈€浣抽厤缃細
   - [144,72,36] kernel=4, dropout=0.2, MAE=0.0066
   - 鎴栧洖閫€鍒皏2鐨?[160,80,40] kernel=3, MAE=0.0061

2. 閲囩敤GRU h=64閰嶇疆锛?
   - h=64,l=2,dropout=0.2, MAE=0.0069 (鎰忓鐨勫ソ缁撴灉!)

3. 缁熶竴鏈€缁堟ā鍨嬮€夋嫨锛?
   - 鏈€浣虫€ц兘: CNN1D (MAE~0.0061)
   - 鏈€浣崇ǔ瀹? GRU h=64 (MAE~0.0069)
   - 杞婚噺绾? CNN1D 2灞?[64,24] (MAE~0.0063, 7K params)
""")

