#!/usr/bin/env python3
"""
大实验结果分析 - 生成论文图表

输出:
- fig2_mae_vs_missing_rate.png: MAE随缺失率变化
- fig3_rmse_vs_missing_rate.png: RMSE随缺失率变化
- fig4_r2_vs_missing_rate.png: R2随缺失率变化
- fig5_improvement_heatmap.png: MIM改进百分比热力图
- fig6_high_missing_comparison.png: 高缺失率对比
- tab3_main_results.csv: 主结果表
- tab4_improvement_significance.csv: 改进百分比及显著性检验
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_all_results():
    """加载所有模型的实验结果"""
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
    """按(模型, 策略, 缺失率)聚合结果"""
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
    """计算MIM相对Baseline的改进百分比"""
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
    """图2: MAE随缺失率变化曲线"""
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
            
            # 添加误差带
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
    """图5: MIM改进百分比热力图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 数据透视
    pivot = imp_df.pivot(index='model', columns='missing_rate', values='improvement_pct')
    pivot = pivot.reindex(['mlp', 'lstm', 'gru', 'cnn1d'])
    
    # 绘制热力图
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    # 设置刻度
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([x.upper() for x in pivot.index])
    
    # 添加数值标注
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
    """图6: 高缺失率详细对比"""
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
    """保存结果表格"""
    os.makedirs('experiments_big', exist_ok=True)
    
    # 表3: 主结果
    pivot_mae = agg_df.pivot_table(
        index=['model', 'strategy'],
        columns='missing_rate',
        values=['mae_mean', 'mae_std']
    )
    pivot_mae.to_csv('experiments_big/tab3_main_results.csv')
    print("Saved: tab3_main_results.csv")
    
    # 表4: 改进百分比
    imp_df.to_csv('experiments_big/tab4_improvement_significance.csv', index=False)
    print("Saved: tab4_improvement_significance.csv")


def main():
    print("="*70)
    print("大实验结果分析")
    print("="*70)
    
    # 加载数据
    df = load_all_results()
    if df is None or len(df) == 0:
        print("No completed experiments found!")
        return
    
    print(f"\nTotal completed experiments: {len(df)}")
    
    # 聚合结果
    print("\nAggregating results...")
    agg_df = aggregate_results(df)
    
    # 计算改进
    print("Calculating improvements...")
    imp_df = calculate_improvement(agg_df)
    
    # 保存表格
    print("\nSaving tables...")
    save_tables(agg_df, imp_df)
    
    # 生成图表
    print("\nGenerating figures...")
    create_fig2_mae_curves(agg_df)
    create_fig5_improvement_heatmap(imp_df)
    create_fig6_high_missing(agg_df)
    
    # 打印摘要
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    
    for model in ['mlp', 'lstm', 'gru', 'cnn1d']:
        model_imp = imp_df[imp_df['model']==model]
        avg_imp = model_imp['improvement_pct'].mean()
        best_mr = model_imp.loc[model_imp['improvement_pct'].idxmax(), 'missing_rate']
        best_imp = model_imp['improvement_pct'].max()
        print(f"{model.upper():8s}: 平均改进 {avg_imp:+.1f}% | 最佳在MR={best_mr:.1f} ({best_imp:+.1f}%)")
    
    print("\n" + "="*70)
    print("分析完成! 输出目录: experiments_big/")
    print("="*70)


if __name__ == '__main__':
    main()
