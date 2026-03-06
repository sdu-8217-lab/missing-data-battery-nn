# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
妫€鏌ュ疄楠岃繘搴﹀拰缁撴灉
"""
import os
import pandas as pd
import glob
from datetime import datetime

def find_latest_experiment():
    """鎵惧埌鏈€鏂扮殑瀹為獙鐩綍"""
    exp_dirs = glob.glob('experiments/*_3C')
    if not exp_dirs:
        return None
    return max(exp_dirs, key=os.path.getmtime)

def check_progress(exp_dir):
    """妫€鏌ュ疄楠岃繘搴?""
    print("=" * 70)
    print("瀹為獙杩涘害妫€鏌?)
    print("=" * 70)
    print(f"瀹為獙鐩綍: {exp_dir}")
    print(f"妫€鏌ユ椂闂? {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 妫€鏌ョ粨鏋淐SV
    results_dir = os.path.join(exp_dir, 'results')
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print("[!] 灏氭湭鐢熸垚缁撴灉鏂囦欢锛屽疄楠屽彲鑳藉垰寮€濮嬫垨灏氭湭瀹屾垚浠讳綍妯″瀷")
        return
    
    # 璇诲彇骞跺垎鏋愮粨鏋?
    all_results = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
        except Exception as e:
            print(f"璇诲彇 {csv_file} 澶辫触: {e}")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"[OK] 宸插畬鎴愯瘎浼版暟: {len(combined)}")
        print()
        
        # 鎸夋ā鍨嬬粺璁?
        model_counts = combined.groupby(['model_type', 'use_mim']).size().reset_index(name='count')
        model_counts['model'] = model_counts['model_type'] + '-' + model_counts['use_mim'].map({True: 'MIM', False: 'Baseline'})
        
        print("鍚勬ā鍨嬪畬鎴愯繘搴?")
        print("-" * 50)
        for _, row in model_counts.iterrows():
            print(f"  {row['model']:20s}: {row['count']:3d} / 900 (棰勬湡)")
        print()
        
        # 鎸夌瀛愮粺璁?
        seed_counts = combined.groupby('seed').size()
        print(f"宸插畬鎴愮瀛愭暟: {len(seed_counts)} / 100")
        print(f"绉嶅瓙鑼冨洿: {seed_counts.index.min()} - {seed_counts.index.max()}")
        print()
        
        # 鏄剧ず閮ㄥ垎缁撴灉绀轰緥
        print("鏈€鏂扮粨鏋滅ず渚?(MLP, seed=42):")
        print("-" * 50)
        example = combined[(combined['model_type'] == 'MLP') & (combined['use_mim'] == False) & (combined['seed'] == 42)]
        if not example.empty:
            example_display = example[['missing_rate', 'mae', 'rmse', 'r2']].sort_values('missing_rate')
            print(example_display.to_string(index=False))
        print()
        
        # 棰勪及鍓╀綑鏃堕棿
        models_done = len(combined) / 9000 * 10  # 宸插畬鎴愮殑妯″瀷鏁帮紙浼扮畻锛?
        if models_done > 0:
            exp_time = os.path.getmtime(exp_dir)
            elapsed = datetime.now().timestamp() - exp_time
            eta_seconds = elapsed / models_done * (10 - models_done)
            eta_hours = eta_seconds / 3600
            print(f"棰勪及鍓╀綑鏃堕棿: {eta_hours:.1f} 灏忔椂")
    
    # 妫€鏌ユā鍨嬫枃浠?
    models_dir = os.path.join(exp_dir, 'models')
    model_files = glob.glob(os.path.join(models_dir, '*.pth'))
    print(f"\n宸蹭繚瀛樻ā鍨嬫暟: {len(model_files)}")
    for mf in model_files[:5]:
        print(f"  - {os.path.basename(mf)}")
    if len(model_files) > 5:
        print(f"  ... 杩樻湁 {len(model_files) - 5} 涓ā鍨?)

def main():
    exp_dir = find_latest_experiment()
    if exp_dir:
        check_progress(exp_dir)
    else:
        print("鏈壘鍒板疄楠岀洰褰?)

if __name__ == '__main__':
    main()

