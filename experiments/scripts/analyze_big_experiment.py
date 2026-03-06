# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
#!/usr/bin/env python3
"""
澶у疄楠岀粨鏋滃垎鏋?- 鐢熸垚璁烘枃鍥捐〃

杈撳嚭:
- fig2_mae_vs_missing_rate.png: MAE闅忕己澶辩巼鍙樺寲
- fig3_rmse_vs_missing_rate.png: RMSE闅忕己澶辩巼鍙樺寲
- fig4_r2_vs_missing_rate.png: R2闅忕己澶辩巼鍙樺寲
- fig5_improvement_heatmap.png: MIM鏀硅繘鐧惧垎姣旂儹鍔涘浘
- fig6_high_missing_comparison.png: 楂樼己澶辩巼瀵规瘮
- tab3_main_results.csv: 涓荤粨鏋滆〃
- tab4_improvement_significance.csv: 鏀硅繘鐧惧垎姣斿強鏄捐憲鎬ф楠?
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_all_results():
    """鍔犺浇鎵€鏈夋ā鍨嬬殑瀹為獙缁撴灉"""
    all_data = []
    models = ['mlp', 'lstm', 'gru', 'cnn1d']
    
    for model in models:
        csv_path = f'configs_big/{model}_big_experiment.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            completed = df[df['status'] == 'completed']
            all_data.append(completed)
            print(f"{model.upper()}: {len(completed)}/{len(df)} completed")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def aggregate_results(df):
    """鎸?妯″瀷, 绛栫暐, 缂哄け鐜?鑱氬悎缁撴灉"""
    grouped = df.groupby(['model_type', 'strategy', 'missing_rate'])
    
    results = []
    for (model, strategy, mr), group in grouped:
        results.append({
            'model': model,
            'strategy': strategy,
            'missing_rate': mr,
            'n': len(group),
            'mae_mean': group['mae'].mean(),
            'mae_std': group['mae'].std(),
            'mae_ci_lower': group['mae'].quantile(0.025),
            'mae_ci_upper': group['mae'].quantile(0.975),
            'rmse_mean': group['rmse'].mean(),
            'rmse_std': group['rmse'].std(),
            'r2_mean': group['r2'].mean(),
            'r2_std': group['r2'].std(),
        })
    
    return pd.DataFrame(results)


def calculate_improvement(agg_df):
    """璁＄畻MIM鐩稿Baseline鐨勬敼杩涚櫨鍒嗘瘮"""
    improvements = []
    
    for model in agg_df['model'].unique():
        for mr in agg_df['missing_rate'].unique():
            baseline = agg_df[(agg_df['model']==model) & 
                             (agg_df['strategy']=='baseline') & 
                             (agg_df['missing_rate']==mr)]
            mim = agg_df[(agg_df['model']==model) & 
                        (agg_df['strategy']=='mim') & 
                        (agg_df['missing_rate']==mr)]
            
            if len(baseline) > 0 and len(mim) > 0:
                mae_base = baseline['mae_mean'].values[0]
                mae_mim = mim['mae_mean'].values[0]
                improvement = (mae_base - mae_mim) / mae_base * 100
                
                improvements.append({
                    'model': model,
                    'missing_rate': mr,
                    'mae_baseline': mae_base,
                    'mae_mim': mae_mim,
                    'improvement_pct': improvement
                })
    
    return pd.DataFrame(improvements)


def create_fig2_mae_curves(agg_df):
    """鍥?: MAE闅忕己澶辩巼鍙樺寲鏇茬嚎"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['mlp', 'lstm', 'gru', 'cnn1d']
    colors = {'mlp': '#1f77b4', 'lstm': '#ff7f0e', 'gru': '#2ca02c', 'cnn1d': '#d62728'}
    markers = {'baseline': 'o', 'mim': 's'}
    linestyles = {'baseline': '--', 'mim': '-'}
    
    for model in models:
        for strategy in ['baseline', 'mim']:
            data = agg_df[(agg_df['model']==model) & (agg_df['strategy']==strategy)]
            data = data.sort_values('missing_rate')
            
            label = f"{model.upper()}-{strategy.upper()}"
            ax.plot(data['missing_rate'], data['mae_mean'], 
                   color=colors[model], linestyle=linestyles[strategy],
                   marker=markers[strategy], markersize=6, label=label, linewidth=2)
            
            # 娣诲姞璇樊甯?
            ax.fill_between(data['missing_rate'], 
                           data['mae_ci_lower'], data['mae_ci_upper'],
                           color=colors[model], alpha=0.1)
    
    ax.set_xlabel('Missing Rate', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('MAE vs Missing Rate (100 repeats, 95% CI)', fontsize=16)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    
    plt.tight_layout()
    plt.savefig('experiments_big/fig2_mae_vs_missing_rate.png', dpi=300, bbox_inches='tight')
    print("Saved: fig2_mae_vs_missing_rate.png")
    plt.close()


def create_fig5_improvement_heatmap(imp_df):
    """鍥?: MIM鏀硅繘鐧惧垎姣旂儹鍔涘浘"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 鏁版嵁閫忚
    pivot = imp_df.pivot(index='model', columns='missing_rate', values='improvement_pct')
    pivot = pivot.reindex(['mlp', 'lstm', 'gru', 'cnn1d'])
    
    # 缁樺埗鐑姏鍥?
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    # 璁剧疆鍒诲害
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([x.upper() for x in pivot.index])
    
    # 娣诲姞鏁板€兼爣娉?
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Missing Rate', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    ax.set_title('MIM Improvement over Baseline (%)', fontsize=16)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('experiments_big/fig5_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: fig5_improvement_heatmap.png")
    plt.close()


def create_fig6_high_missing(agg_df):
    """鍥?: 楂樼己澶辩巼璇︾粏瀵规瘮"""
    high_mr = agg_df[agg_df['missing_rate'] >= 0.6]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = ['mlp', 'lstm', 'gru', 'cnn1d']
    colors = {'baseline': '#ff7f0e', 'mim': '#2ca02c'}
    x_pos = np.arange(len(models))
    width = 0.12
    
    for i, mr in enumerate([0.6, 0.7, 0.8, 0.9]):
        offset = (i - 1.5) * width * 2
        
        baseline_vals = []
        mim_vals = []
        
        for model in models:
            b = high_mr[(high_mr['model']==model) & 
                       (high_mr['strategy']=='baseline') & 
                       (high_mr['missing_rate']==mr)]
            m = high_mr[(high_mr['model']==model) & 
                       (high_mr['strategy']=='mim') & 
                       (high_mr['missing_rate']==mr)]
            baseline_vals.append(b['mae_mean'].values[0] if len(b)>0 else 0)
            mim_vals.append(m['mae_mean'].values[0] if len(m)>0 else 0)
        
        ax.bar(x_pos + offset, baseline_vals, width, 
              label=f'Baseline MR={mr}', color=colors['baseline'], alpha=0.7-i*0.1)
        ax.bar(x_pos + offset + width, mim_vals, width,
              label=f'MIM MR={mr}', color=colors['mim'], alpha=0.7-i*0.1)
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('High Missing Rate (0.6-0.9) Performance Comparison', fontsize=16)
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiments_big/fig6_high_missing_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: fig6_high_missing_comparison.png")
    plt.close()


def save_tables(agg_df, imp_df):
    """淇濆瓨缁撴灉琛ㄦ牸"""
    os.makedirs('experiments_big', exist_ok=True)
    
    # 琛?: 涓荤粨鏋?
    pivot_mae = agg_df.pivot_table(
        index=['model', 'strategy'],
        columns='missing_rate',
        values=['mae_mean', 'mae_std']
    )
    pivot_mae.to_csv('experiments_big/tab3_main_results.csv')
    print("Saved: tab3_main_results.csv")
    
    # 琛?: 鏀硅繘鐧惧垎姣?
    imp_df.to_csv('experiments_big/tab4_improvement_significance.csv', index=False)
    print("Saved: tab4_improvement_significance.csv")


def main():
    print("="*70)
    print("澶у疄楠岀粨鏋滃垎鏋?)
    print("="*70)
    
    # 鍔犺浇鏁版嵁
    df = load_all_results()
    if df is None or len(df) == 0:
        print("No completed experiments found!")
        return
    
    print(f"\nTotal completed experiments: {len(df)}")
    
    # 鑱氬悎缁撴灉
    print("\nAggregating results...")
    agg_df = aggregate_results(df)
    
    # 璁＄畻鏀硅繘
    print("Calculating improvements...")
    imp_df = calculate_improvement(agg_df)
    
    # 淇濆瓨琛ㄦ牸
    print("\nSaving tables...")
    save_tables(agg_df, imp_df)
    
    # 鐢熸垚鍥捐〃
    print("\nGenerating figures...")
    create_fig2_mae_curves(agg_df)
    create_fig5_improvement_heatmap(imp_df)
    create_fig6_high_missing(agg_df)
    
    # 鎵撳嵃鎽樿
    print("\n" + "="*70)
    print("鍏抽敭鍙戠幇")
    print("="*70)
    
    for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
        model_imp = imp_df[imp_df['model']==model]
        avg_imp = model_imp['improvement_pct'].mean()
        best_mr = model_imp.loc[model_imp['improvement_pct'].idxmax(), 'missing_rate']
        best_imp = model_imp['improvement_pct'].max()
        print(f"{model.upper():8s}: 骞冲潎鏀硅繘 {avg_imp:+.1f}% | 鏈€浣冲湪MR={best_mr:.1f} ({best_imp:+.1f}%)")
    
    print("\n" + "="*70)
    print("鍒嗘瀽瀹屾垚! 杈撳嚭鐩綍: experiments_big/")
    print("="*70)


if __name__ == '__main__':
    main()

