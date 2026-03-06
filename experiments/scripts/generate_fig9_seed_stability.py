# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚鍥?锛氱瀛愮ǔ瀹氭€у垎鏋愶紙閲嶇粯鐗堬級
浣跨敤鐏拌壊闃村奖甯︼紙min-max鎴杕ean卤std锛? 褰╄壊鍧囧€肩嚎
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_path):
    """鍔犺浇瀹為獙缁撴灉鏁版嵁"""
    df = pd.read_csv(csv_path)
    return df

def process_seed_data(df, model_name):
    """
    澶勭悊绉嶅瓙绋冲畾鎬ф暟鎹?
    杩斿洖锛歮issing_rates, baseline_mean, mim_mean, baseline_std, mim_std
    """
    # 绛涢€夎妯″瀷鐨勬暟鎹?
    baseline_df = df[df['model_name'] == model_name].copy()
    mim_df = df[df['model_name'] == f'{model_name}-MIM'].copy()
    
    # 鑾峰彇鎵€鏈夌己澶辩巼
    missing_rates = sorted(baseline_df['missing_rate'].unique())
    
    baseline_means = []
    baseline_mins = []
    baseline_maxs = []
    mim_means = []
    mim_mins = []
    mim_maxs = []
    
    for mr in missing_rates:
        baseline_mr = baseline_df[baseline_df['missing_rate'] == mr]
        mim_mr = mim_df[mim_df['missing_rate'] == mr]
        
        baseline_mae = baseline_mr['mae'].values
        mim_mae = mim_mr['mae'].values
        
        baseline_means.append(np.mean(baseline_mae))
        baseline_mins.append(np.min(baseline_mae))
        baseline_maxs.append(np.max(baseline_mae))
        
        mim_means.append(np.mean(mim_mae))
        mim_mins.append(np.min(mim_mae))
        mim_maxs.append(np.max(mim_mae))
    
    return (np.array(missing_rates), 
            np.array(baseline_means), np.array(baseline_mins), np.array(baseline_maxs),
            np.array(mim_means), np.array(mim_mins), np.array(mim_maxs))

def create_seed_stability_plot(df, output_path):
    """
    鍒涘缓绉嶅瓙绋冲畾鎬у浘锛?x2瀛愬浘锛?
    浣跨敤鐏拌壊闃村奖甯﹁〃绀簃in-max鑼冨洿锛屽僵鑹茬嚎琛ㄧず鍧囧€?
    """
    models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
    model_names = ['MLP', 'LSTM', 'GRU', '1D-CNN']
    colors = {'baseline': '#E74C3C', 'mim': '#3498DB'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seed Stability Analysis (100 Random Seeds)\nShaded Area: Min-Max Range, Solid Line: Mean', 
                 fontsize=16, fontweight='bold')
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        ax = axes[idx // 2, idx % 2]
        
        (mr, baseline_mean, baseline_min, baseline_max,
         mim_mean, mim_min, mim_max) = process_seed_data(df, model)
        
        # 缁樺埗Baseline锛氶槾褰卞甫 + 鍧囧€肩嚎
        ax.fill_between(mr, baseline_min, baseline_max, alpha=0.2, color='gray', label='_nolegend_')
        ax.plot(mr, baseline_mean, 's--', color=colors['baseline'], 
                label='Baseline (Mean)', markersize=8, linewidth=2)
        
        # 缁樺埗MIM锛氶槾褰卞甫 + 鍧囧€肩嚎
        ax.fill_between(mr, mim_min, mim_max, alpha=0.2, color='gray', label='_nolegend_')
        ax.plot(mr, mim_mean, 'o-', color=colors['mim'], 
                label='MIM (Mean)', markersize=8, linewidth=2)
        
        ax.set_xlabel('Missing Rate', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) {model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mr)
        ax.set_xticklabels([f'{x:.1f}' for x in mr])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 鏁版嵁璺緞
    csv_path = 'experiments_v2/3C/20260204_084913/results_all.csv'
    
    # 杈撳嚭鐩綍
    output_dir = 'my_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 鍔犺浇鏁版嵁
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    
    # 鐢熸垚鍥?
    output_path = os.path.join(output_dir, 'fig9_seed_stability_v2.png')
    create_seed_stability_plot(df, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

