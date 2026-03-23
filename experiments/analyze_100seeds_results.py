#!/usr/bin/env python3
"""
100种子实验结果分析
生成汇总统计和可视化
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置输出目录
RESULTS_DIR = 'results/100seeds_v2_final'
OUTPUT_DIR = 'results/analysis_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_results():
    """加载所有结果文件"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    files = glob.glob(f'{RESULTS_DIR}/*.csv')
    print(f"找到 {len(files)} 个结果文件")
    
    dfs = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  已加载 {i}/{len(files)}...")
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  错误: {f} - {e}")
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n总计: {len(df_all):,} 条记录")
    print(f"唯一模型数: {df_all[['seed', 'batch', 'model', 'use_mim']].drop_duplicates().shape[0]}")
    
    return df_all

def compute_summary_stats(df):
    """计算汇总统计"""
    print("\n" + "="*60)
    print("计算汇总统计...")
    print("="*60)
    
    # 1. 总体统计
    summary = {}
    summary['total_records'] = len(df)
    summary['overall_mae_mean'] = df['test_mae'].mean()
    summary['overall_mae_std'] = df['test_mae'].std()
    summary['overall_r2_mean'] = df['test_r2'].mean()
    summary['overall_r2_std'] = df['test_r2'].std()
    
    # 2. 按MIM分组
    mim_stats = df.groupby('use_mim')['test_mae'].agg(['mean', 'std', 'count']).reset_index()
    mim_stats.columns = ['use_mim', 'mae_mean', 'mae_std', 'count']
    summary['mim_comparison'] = mim_stats.to_dict('records')
    
    # 计算MIM改进
    mae_no_mim = mim_stats[mim_stats['use_mim'] == False]['mae_mean'].values[0]
    mae_mim = mim_stats[mim_stats['use_mim'] == True]['mae_mean'].values[0]
    improvement = (mae_no_mim - mae_mim) / mae_no_mim * 100
    summary['mim_improvement_percent'] = improvement
    
    print(f"\nMIM效果:")
    print(f"  Non-MIM MAE: {mae_no_mim:.4f}")
    print(f"  MIM MAE: {mae_mim:.4f}")
    print(f"  改进: {improvement:.2f}%")
    
    # 3. 按模型类型
    model_stats = df.groupby(['model', 'use_mim'])['test_mae'].agg(['mean', 'std']).reset_index()
    model_stats.columns = ['model', 'use_mim', 'mae_mean', 'mae_std']
    summary['model_comparison'] = model_stats.to_dict('records')
    
    print(f"\n按模型类型:")
    for model in df['model'].unique():
        model_data = model_stats[model_stats['model'] == model]
        no_mim = model_data[model_data['use_mim'] == False]['mae_mean'].values[0]
        mim = model_data[model_data['use_mim'] == True]['mae_mean'].values[0]
        imp = (no_mim - mim) / no_mim * 100
        print(f"  {model.upper()}: Non-MIM={no_mim:.4f}, MIM={mim:.4f}, 改进={imp:.1f}%")
    
    # 4. 按缺失模式
    mode_stats = df.groupby(['mode', 'use_mim'])['test_mae'].agg(['mean', 'std']).reset_index()
    mode_stats.columns = ['mode', 'use_mim', 'mae_mean', 'mae_std']
    summary['mode_comparison'] = mode_stats.to_dict('records')
    
    print(f"\n按缺失模式:")
    for mode in ['MCAR', 'MAR', 'MNAR']:
        mode_data = mode_stats[mode_stats['mode'] == mode]
        no_mim = mode_data[mode_data['use_mim'] == False]['mae_mean'].values[0]
        mim = mode_data[mode_data['use_mim'] == True]['mae_mean'].values[0]
        imp = (no_mim - mim) / no_mim * 100
        print(f"  {mode}: Non-MIM={no_mim:.4f}, MIM={mim:.4f}, 改进={imp:.1f}%")
    
    # 5. 按缺失率 (按0.1间隔分组)
    df['mr_group'] = (df['test_mr'] * 10).astype(int) / 10
    mr_stats = df.groupby(['mr_group', 'use_mim'])['test_mae'].agg(['mean', 'std']).reset_index()
    mr_stats.columns = ['mr_group', 'use_mim', 'mae_mean', 'mae_std']
    summary['mr_comparison'] = mr_stats.to_dict('records')
    
    # 6. 按插补方法
    imp_stats = df.groupby(['imputation', 'use_mim'])['test_mae'].agg(['mean', 'std']).reset_index()
    imp_stats.columns = ['imputation', 'use_mim', 'mae_mean', 'mae_std']
    summary['imputation_comparison'] = imp_stats.to_dict('records')
    
    print(f"\n按插补方法:")
    for imp in ['mean', 'knn', 'iterative', 'zero']:
        imp_data = imp_stats[imp_stats['imputation'] == imp]
        no_mim = imp_data[imp_data['use_mim'] == False]['mae_mean'].values[0]
        mim = imp_data[imp_data['use_mim'] == True]['mae_mean'].values[0]
        improvement = (no_mim - mim) / no_mim * 100
        print(f"  {imp}: Non-MIM={no_mim:.4f}, MIM={mim:.4f}, 改进={improvement:.1f}%")
    
    return summary

def create_summary_tables(df):
    """创建汇总表格"""
    print("\n" + "="*60)
    print("生成汇总表格...")
    print("="*60)
    
    tables = {}
    
    # 表1: 按缺失率和MIM
    table1 = df.pivot_table(
        values='test_mae', 
        index='test_mr', 
        columns='use_mim', 
        aggfunc=['mean', 'std']
    )
    table1.columns = ['MAE_mean_no_mim', 'MAE_mean_mim', 'MAE_std_no_mim', 'MAE_std_mim']
    table1['improvement'] = (table1['MAE_mean_no_mim'] - table1['MAE_mean_mim']) / table1['MAE_mean_no_mim'] * 100
    tables['by_missing_rate'] = table1
    table1.to_csv(f'{OUTPUT_DIR}/table_by_missing_rate.csv')
    print(f"  表1: 按缺失率 - 已保存")
    
    # 表2: 按模型和缺失率
    table2 = df.pivot_table(
        values='test_mae',
        index=['model', 'test_mr'],
        columns='use_mim',
        aggfunc='mean'
    )
    table2.columns = ['MAE_no_mim', 'MAE_mim']
    table2['improvement'] = (table2['MAE_no_mim'] - table2['MAE_mim']) / table2['MAE_no_mim'] * 100
    tables['by_model_mr'] = table2
    table2.to_csv(f'{OUTPUT_DIR}/table_by_model_mr.csv')
    print(f"  表2: 按模型×缺失率 - 已保存")
    
    # 表3: 按缺失模式和缺失率
    table3 = df.pivot_table(
        values='test_mae',
        index=['mode', 'test_mr'],
        columns='use_mim',
        aggfunc='mean'
    )
    table3.columns = ['MAE_no_mim', 'MAE_mim']
    table3['improvement'] = (table3['MAE_no_mim'] - table3['MAE_mim']) / table3['MAE_no_mim'] * 100
    tables['by_mode_mr'] = table3
    table3.to_csv(f'{OUTPUT_DIR}/table_by_mode_mr.csv')
    print(f"  表3: 按缺失模式×缺失率 - 已保存")
    
    # 表4: 按插补方法和缺失率
    table4 = df.pivot_table(
        values='test_mae',
        index=['imputation', 'test_mr'],
        columns='use_mim',
        aggfunc='mean'
    )
    table4.columns = ['MAE_no_mim', 'MAE_mim']
    table4['improvement'] = (table4['MAE_no_mim'] - table4['MAE_mim']) / table4['MAE_no_mim'] * 100
    tables['by_imputation_mr'] = table4
    table4.to_csv(f'{OUTPUT_DIR}/table_by_imputation_mr.csv')
    print(f"  表4: 按插补方法×缺失率 - 已保存")
    
    # 表5: 按批次
    table5 = df.pivot_table(
        values='test_mae',
        index='batch',
        columns='use_mim',
        aggfunc='mean'
    )
    table5.columns = ['MAE_no_mim', 'MAE_mim']
    table5['improvement'] = (table5['MAE_no_mim'] - table5['MAE_mim']) / table5['MAE_no_mim'] * 100
    tables['by_batch'] = table5
    table5.to_csv(f'{OUTPUT_DIR}/table_by_batch.csv')
    print(f"  表5: 按批次 - 已保存")
    
    return tables

def create_detailed_summary(df):
    """创建详细汇总"""
    print("\n" + "="*60)
    print("生成详细汇总...")
    print("="*60)
    
    # 1. 高缺失率场景 (>0.5)
    high_mr = df[df['test_mr'] > 0.5]
    high_mr_summary = high_mr.groupby(['model', 'use_mim'])['test_mae'].mean().reset_index()
    high_mr_pivot = high_mr_summary.pivot(index='model', columns='use_mim', values='test_mae')
    high_mr_pivot['improvement'] = (high_mr_pivot[False] - high_mr_pivot[True]) / high_mr_pivot[False] * 100
    high_mr_pivot.to_csv(f'{OUTPUT_DIR}/high_mr_summary.csv')
    print(f"  高缺失率(>0.5)汇总 - 已保存")
    
    # 2. Zero imputation场景
    zero_imp = df[df['imputation'] == 'zero']
    zero_summary = zero_imp.groupby(['test_mr', 'use_mim'])['test_mae'].mean().reset_index()
    zero_pivot = zero_summary.pivot(index='test_mr', columns='use_mim', values='test_mae')
    zero_pivot['improvement'] = (zero_pivot[False] - zero_pivot[True]) / zero_pivot[False] * 100
    zero_pivot.to_csv(f'{OUTPUT_DIR}/zero_imputation_summary.csv')
    print(f"  Zero imputation汇总 - 已保存")
    
    # 3. 统计显著性检验 (简单版本)
    from scipy import stats
    
    sig_results = []
    for model in df['model'].unique():
        for mr in [0.3, 0.5, 0.7, 0.9]:
            subset = df[(df['model'] == model) & (df['test_mr'] == mr)]
            if len(subset) > 0:
                no_mim = subset[subset['use_mim'] == False]['test_mae'].values
                mim = subset[subset['use_mim'] == True]['test_mae'].values
                if len(no_mim) > 10 and len(mim) > 10:
                    t_stat, p_value = stats.ttest_ind(no_mim, mim)
                    sig_results.append({
                        'model': model,
                        'mr': mr,
                        'no_mim_mean': no_mim.mean(),
                        'mim_mean': mim.mean(),
                        'improvement': (no_mim.mean() - mim.mean()) / no_mim.mean() * 100,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
    
    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(f'{OUTPUT_DIR}/statistical_significance.csv', index=False)
    print(f"  统计显著性检验 - 已保存")
    print(f"    显著改善的比例: {sig_df['significant'].mean()*100:.1f}%")

def create_visualizations(df):
    """创建可视化图表"""
    print("\n" + "="*60)
    print("生成可视化...")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        print("  警告: matplotlib 未安装，跳过可视化")
        return
    
    # 图1: 按缺失率的MAE对比
    fig, ax = plt.subplots(figsize=(10, 6))
    mr_mim = df[df['use_mim'] == True].groupby('test_mr')['test_mae'].mean()
    mr_no_mim = df[df['use_mim'] == False].groupby('test_mr')['test_mae'].mean()
    
    ax.plot(mr_mim.index, mr_mim.values, 'o-', label='MIM', linewidth=2)
    ax.plot(mr_no_mim.index, mr_no_mim.values, 's-', label='Non-MIM', linewidth=2)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE vs Missing Rate (100 Seeds Average)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_mae_vs_mr.png', dpi=150)
    plt.close()
    print(f"  图1: MAE vs Missing Rate - 已保存")
    
    # 图2: 按模型的改进百分比
    fig, ax = plt.subplots(figsize=(10, 6))
    model_improvement = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        no_mim = model_data[model_data['use_mim'] == False].groupby('test_mr')['test_mae'].mean()
        mim = model_data[model_data['use_mim'] == True].groupby('test_mr')['test_mae'].mean()
        improvement = (no_mim - mim) / no_mim * 100
        ax.plot(improvement.index, improvement.values, 'o-', label=model.upper(), linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('MIM Improvement by Model Type', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_improvement_by_model.png', dpi=150)
    plt.close()
    print(f"  图2: 改进百分比 by Model - 已保存")
    
    # 图3: 按缺失模式的改进
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ['MCAR', 'MAR', 'MNAR']:
        mode_data = df[df['mode'] == mode]
        no_mim = mode_data[mode_data['use_mim'] == False].groupby('test_mr')['test_mae'].mean()
        mim = mode_data[mode_data['use_mim'] == True].groupby('test_mr')['test_mae'].mean()
        improvement = (no_mim - mim) / no_mim * 100
        ax.plot(improvement.index, improvement.values, 'o-', label=mode, linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('MIM Improvement by Missing Pattern', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_improvement_by_mode.png', dpi=150)
    plt.close()
    print(f"  图3: 改进百分比 by Missing Pattern - 已保存")
    
    # 图4: 热力图 - 模型×缺失率
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_data = df.pivot_table(
        values='test_mae',
        index='model',
        columns='test_mr',
        aggfunc='mean'
    )
    im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns])
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([x.upper() for x in pivot_data.index])
    ax.set_xlabel('Missing Rate', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Average MAE Heatmap (All Configs)', fontsize=14)
    plt.colorbar(im, ax=ax, label='MAE')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_heatmap.png', dpi=150)
    plt.close()
    print(f"  图4: MAE热力图 - 已保存")

def main():
    print("="*60)
    print("100种子实验结果分析")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    df = load_all_results()
    
    # 计算统计
    summary = compute_summary_stats(df)
    
    # 创建表格
    tables = create_summary_tables(df)
    
    # 详细汇总
    create_detailed_summary(df)
    
    # 可视化
    create_visualizations(df)
    
    # 保存汇总JSON
    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n汇总JSON已保存: {OUTPUT_DIR}/summary.json")
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"输出目录: {OUTPUT_DIR}/")
    print(f"\n主要发现:")
    print(f"  - MIM整体改进: {summary['mim_improvement_percent']:.2f}%")
    print(f"  - 总记录数: {summary['total_records']:,}")
    print(f"  - 平均MAE (Non-MIM): {summary['overall_mae_mean']:.4f}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
