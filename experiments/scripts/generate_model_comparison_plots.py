# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
鐢熸垚璁烘枃鐢ㄧ殑妯″瀷瀵规瘮鍥?
浣跨敤 errorbar 鏄剧ず95%缃俊鍖洪棿锛堜笌鏃т唬鐮佷竴鑷达級
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_path):
    """鍔犺浇瀹為獙缁撴灉鏁版嵁"""
    df = pd.read_csv(csv_path)
    return df

def process_model_data(df, model_name):
    """
    澶勭悊鍗曚釜妯″瀷鐨勬暟鎹紝璁＄畻缁熻閲?
    """
    # Baseline: model_name (e.g., 'MLP')
    # MIM: model_name-MIM (e.g., 'MLP-MIM')
    baseline_df = df[df['model_name'] == model_name].copy()
    mim_df = df[df['model_name'] == f'{model_name}-MIM'].copy()
    
    # 鑾峰彇鎵€鏈夌己澶辩巼
    missing_rates = sorted(baseline_df['missing_rate'].unique())
    
    stats = {
        'missing_rate': [],
        'baseline_mae_mean': [],
        'baseline_mae_std': [],
        'baseline_rmse_mean': [],
        'baseline_rmse_std': [],
        'baseline_r2_mean': [],
        'baseline_r2_std': [],
        'mim_mae_mean': [],
        'mim_mae_std': [],
        'mim_rmse_mean': [],
        'mim_rmse_std': [],
        'mim_r2_mean': [],
        'mim_r2_std': [],
        'improvement_mean': [],
        'improvement_std': []
    }
    
    for mr in missing_rates:
        baseline_mr = baseline_df[baseline_df['missing_rate'] == mr]
        mim_mr = mim_df[mim_df['missing_rate'] == mr]
        
        # Baseline 缁熻
        baseline_mae = baseline_mr['mae'].values
        baseline_rmse = baseline_mr['rmse'].values
        baseline_r2 = baseline_mr['r2'].values
        
        # MIM 缁熻
        mim_mae = mim_mr['mae'].values
        mim_rmse = mim_mr['rmse'].values
        mim_r2 = mim_mr['r2'].values
        
        # 璁＄畻鏀硅繘鐜?(%)
        improvement = ((baseline_mae - mim_mae) / baseline_mae) * 100
        
        stats['missing_rate'].append(mr)
        
        stats['baseline_mae_mean'].append(np.mean(baseline_mae))
        stats['baseline_mae_std'].append(np.std(baseline_mae))
        stats['baseline_rmse_mean'].append(np.mean(baseline_rmse))
        stats['baseline_rmse_std'].append(np.std(baseline_rmse))
        stats['baseline_r2_mean'].append(np.mean(baseline_r2))
        stats['baseline_r2_std'].append(np.std(baseline_r2))
        
        stats['mim_mae_mean'].append(np.mean(mim_mae))
        stats['mim_mae_std'].append(np.std(mim_mae))
        stats['mim_rmse_mean'].append(np.mean(mim_rmse))
        stats['mim_rmse_std'].append(np.std(mim_rmse))
        stats['mim_r2_mean'].append(np.mean(mim_r2))
        stats['mim_r2_std'].append(np.std(mim_r2))
        
        stats['improvement_mean'].append(np.mean(improvement))
        stats['improvement_std'].append(np.std(improvement))
    
    return pd.DataFrame(stats)

def create_comparison_plot(df_stats, model_name, output_path):
    """
    鍒涘缓瀵规瘮鍥撅紝浣跨敤 errorbar 鏄剧ず95%缃俊鍖洪棿
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} Model: Baseline vs MIM Comparison\n(100 Random Seeds, 95% CI)', 
                 fontsize=16, fontweight='bold')
    
    missing_rates = df_stats['missing_rate']
    n_experiments = 100  # 鏍锋湰鏁帮紝鐢ㄤ簬璁＄畻缃俊鍖洪棿
    
    # 璁＄畻95%缃俊鍖洪棿 = mean 卤 1.96 * std / sqrt(n)
    ci_factor = 1.96 / np.sqrt(n_experiments)
    
    # (a) MAE瀵规瘮
    ax = axes[0, 0]
    baseline_mae_ci = df_stats['baseline_mae_std'] * ci_factor
    mim_mae_ci = df_stats['mim_mae_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_mae_mean'], 
                yerr=baseline_mae_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_mae_mean'], 
                yerr=mim_mae_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('(a) MAE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (b) RMSE瀵规瘮
    ax = axes[0, 1]
    baseline_rmse_ci = df_stats['baseline_rmse_std'] * ci_factor
    mim_rmse_ci = df_stats['mim_rmse_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_rmse_mean'], 
                yerr=baseline_rmse_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_rmse_mean'], 
                yerr=mim_rmse_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('(b) RMSE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (c) R虏瀵规瘮
    ax = axes[1, 0]
    baseline_r2_ci = df_stats['baseline_r2_std'] * ci_factor
    mim_r2_ci = df_stats['mim_r2_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['baseline_r2_mean'], 
                yerr=baseline_r2_ci,
                fmt='s--', color='#E74C3C', label='Baseline', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.errorbar(missing_rates, df_stats['mim_r2_mean'], 
                yerr=mim_r2_ci,
                fmt='o-', color='#3498DB', label='MIM', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('R虏 Score', fontsize=12)
    ax.set_title('(c) R虏 Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # (d) 鏀硅繘鐜囨洸绾?
    ax = axes[1, 1]
    improvement_ci = df_stats['improvement_std'] * ci_factor
    
    ax.errorbar(missing_rates, df_stats['improvement_mean'], 
                yerr=improvement_ci,
                fmt='D-', color='#2ECC71', label='MAE Improvement', 
                markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('(d) MAE Improvement Rate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(missing_rates)
    ax.set_xticklabels([f'{x:.1f}' for x in missing_rates])
    
    # 娣诲姞骞冲潎鏀硅繘鐜囨枃鏈?
    avg_improvement = df_stats['improvement_mean'].mean()
    ax.text(0.5, 0.95, f'Average Improvement: {avg_improvement:.1f}%', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    # 鏁版嵁璺緞
    csv_path = 'experiments_v2/3C/20260204_084913/results_all.csv'
    
    # 杈撳嚭鐩綍
    output_dir = 'my_figures/model_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    # 鍔犺浇鏁版嵁
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    
    # 妯″瀷鍒楄〃
    models = ['MLP', 'LSTM', 'GRU', 'CNN1D']
    
    # 涓烘瘡涓ā鍨嬬敓鎴愬姣斿浘
    for model in models:
        print(f"\nProcessing {model} model...")
        df_stats = process_model_data(df, model)
        output_path = os.path.join(output_dir, f'{model}_baseline_vs_mim_comparison.png')
        create_comparison_plot(df_stats, model, output_path)
    
    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == '__main__':
    main()

