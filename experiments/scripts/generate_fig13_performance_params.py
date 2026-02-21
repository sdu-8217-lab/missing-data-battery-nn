# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚鍥?3锛氭€ц兘-鍙傛暟閲忔潈琛★紙MR=0.5锛?
浣跨敤琛?涓殑鐪熷疄鍙傛暟閲?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_path):
    """鍔犺浇瀹為獙缁撴灉鏁版嵁"""
    df = pd.read_csv(csv_path)
    return df

def create_performance_params_plot(df, output_path):
    """
    鍒涘缓鎬ц兘-鍙傛暟閲忔潈琛℃暎鐐瑰浘
    浣跨敤琛?涓殑鐪熷疄鍙傛暟閲?
    """
    # 琛?涓殑鐪熷疄鍙傛暟閲忥紙鏉ヨ嚜璁烘枃锛?
    params = {
        'MLP': {'Baseline': 27649, 'MIM': 36865},
        'LSTM': {'Baseline': 31537, 'MIM': 40753},
        'GRU': {'Baseline': 40769, 'MIM': 49985},
        'CNN1D': {'Baseline': 16713, 'MIM': 30537}
    }
    
    model_names = {'MLP': 'MLP', 'LSTM': 'LSTM', 'GRU': 'GRU', 'CNN1D': '1D-CNN'}
    colors = {'MLP': '#E74C3C', 'LSTM': '#3498DB', 'GRU': '#F39C12', 'CNN1D': '#2ECC71'}
    markers = {'Baseline': 'x', 'MIM': 'o'}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 鑾峰彇MR=0.5鏃剁殑MAE鏁版嵁
    mr = 0.5
    
    for model in ['MLP', 'LSTM', 'GRU', 'CNN1D']:
        for strategy in ['Baseline', 'MIM']:
            if strategy == 'Baseline':
                model_df = df[(df['model_name'] == model) & (df['missing_rate'] == mr)]
            else:
                model_df = df[(df['model_name'] == f'{model}-MIM') & (df['missing_rate'] == mr)]
            
            mae_mean = model_df['mae'].mean()
            param_count = params[model][strategy]
            
            # 缁樺埗鏁ｇ偣
            label = f"{model_names[model]}-{strategy}"
            ax.scatter(param_count, mae_mean, 
                      c=colors[model], marker=markers[strategy], 
                      s=200, alpha=0.8, edgecolors='black', linewidth=1.5,
                      label=label)
            
            # 娣诲姞鏂囨湰鏍囨敞
            offset_x = 800
            offset_y = 0.001
            ax.annotate(f"{model_names[model]}-{strategy}", 
                       (param_count, mae_mean),
                       xytext=(param_count + offset_x, mae_mean + offset_y),
                       fontsize=9, fontweight='bold')
    
    # 娣诲姞甯曠疮鎵樺墠娌跨嚎锛堢ず鎰忥級
    # 甯曠疮鎵樺墠娌跨偣锛?D-CNN-MIM, GRU-MIM锛堟牴鎹鏂囨弿杩帮級
    pareto_x = [params['CNN1D']['MIM'], params['GRU']['MIM']]
    pareto_y = [0.015, 0.016]  # 杩戜技鍊?
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Front')
    
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (MR=0.5)', fontsize=14, fontweight='bold')
    ax.set_title('Performance-Parameter Trade-off (MR=0.5)\nUsing Actual Parameter Counts from Table 2', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    # 娣诲姞娉ㄩ噴
    ax.text(0.05, 0.95, 'Lower-left is better\n(fewer params, lower MAE)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 娣诲姞鏈€浣崇偣鏍囨敞
    ax.annotate('Best Trade-off:\n1D-CNN-MIM\n(30.5K params, MAE鈮?.015)', 
               xy=(params['CNN1D']['MIM'], 0.015),
               xytext=(params['CNN1D']['MIM'] + 5000, 0.020),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
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
    
    # 鐢熸垚鍥?3
    output_path = os.path.join(output_dir, 'fig13_performance_params_v2.png')
    create_performance_params_plot(df, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

