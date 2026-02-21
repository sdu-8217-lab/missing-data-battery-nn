# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚鍥?1锛歁AE鍒嗗竷灏忔彁鐞村浘锛堥噸缁樼増锛?
姣廙R涓€琛岋紝姣忚灞曠ず鍚勬ā鍨婤ase vs MIM涓ょ粍灏忔彁鐞?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_data(csv_path):
    """鍔犺浇瀹為獙缁撴灉鏁版嵁"""
    df = pd.read_csv(csv_path)
    return df

def create_distribution_plot(df, output_path):
    """
    鍒涘缓MAE鍒嗗竷灏忔彁鐞村浘
    姣忚涓€涓狹R锛屽睍绀哄悇妯″瀷Base vs MIM鐨勫垎甯?
    """
    models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
    model_names = ['MLP', 'LSTM', 'GRU', '1D-CNN']
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(len(missing_rates), 1, figsize=(14, 16))
    fig.suptitle('MAE Distribution Across Missing Rates\n(100 Random Seeds)', 
                 fontsize=16, fontweight='bold')
    
    for idx, mr in enumerate(missing_rates):
        ax = axes[idx]
        
        # 鏀堕泦鏁版嵁
        data_list = []
        position_labels = []
        colors = []
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            # Baseline
            baseline_df = df[(df['model_name'] == model) & (df['missing_rate'] == mr)]
            baseline_mae = baseline_df['mae'].values
            
            # MIM
            mim_df = df[(df['model_name'] == f'{model}-MIM') & (df['missing_rate'] == mr)]
            mim_mae = mim_df['mae'].values
            
            # 涓烘瘡涓ā鍨嬫坊鍔犱袱缁勬暟鎹?
            for val in baseline_mae:
                data_list.append({'MAE': val, 'Model': model_name, 'Strategy': 'Baseline'})
            for val in mim_mae:
                data_list.append({'MAE': val, 'Model': model_name, 'Strategy': 'MIM'})
        
        df_plot = pd.DataFrame(data_list)
        
        # 鍒涘缓灏忔彁鐞村浘
        sns.violinplot(data=df_plot, x='Model', y='MAE', hue='Strategy', 
                       split=True, ax=ax, palette={'Baseline': '#E74C3C', 'MIM': '#3498DB'})
        
        ax.set_title(f'MR = {mr:.1f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('MAE', fontsize=11)
        if idx < len(missing_rates) - 1:
            ax.legend_.remove()
        else:
            ax.legend(title='Strategy', loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
    
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
    
    # 鐢熸垚鍥?1
    output_path = os.path.join(output_dir, 'fig11_distribution_v2.png')
    create_distribution_plot(df, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

