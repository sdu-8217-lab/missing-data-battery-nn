#!/usr/bin/env python3
"""
100种子实验结果分析 - 增强版 (带统计显著性检验)

生成汇总统计、可视化，以及统计显著性检验

新增的统计检验:
- 配对t检验 (paired t-test): 检验 MIM vs non-MIM 的差异
- Wilcoxon符号秩检验: 非参数替代方法
- Cohen's d 效应量: 量化改进的实际意义
- 置信区间: 均值差异的95%置信区间
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.statistical_testing import (
    paired_t_test, wilcoxon_test, compare_mim_effect,
    summarize_significance, StatisticalTestResult
)

warnings.filterwarnings('ignore')

# 设置输出目录
RESULTS_DIR = 'results/100seeds_v2_final'
OUTPUT_DIR = 'results/analysis_enhanced_v2'
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


def perform_statistical_tests(df):
    """
    执行统计显著性检验
    """
    print("\n" + "="*60)
    print("统计显著性检验")
    print("="*60)
    
    all_results = []
    
    # 1. 总体检验 (按 seed 配对)
    print("\n1. 总体 MIM 效果检验...")
    pivot_df = df.pivot_table(
        index='seed',
        columns='use_mim',
        values='test_mae',
        aggfunc='mean'
    ).dropna()
    
    if len(pivot_df) > 0:
        result = paired_t_test(pivot_df[False].values, pivot_df[True].values)
        print(result)
        all_results.append({
            'level': 'overall',
            'group': 'all',
            **result.__dict__
        })
    
    # 2. 按模型类型检验
    print("\n2. 按模型类型检验...")
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        pivot_df = model_df.pivot_table(
            index='seed',
            columns='use_mim',
            values='test_mae',
            aggfunc='mean'
        ).dropna()
        
        if len(pivot_df) < 10:
            print(f"  {model.upper()}: 样本不足 ({len(pivot_df)} seeds)")
            continue
            
        result = paired_t_test(pivot_df[False].values, pivot_df[True].values)
        sig_marker = "✅" if result.significant else "❌"
        print(f"  {model.upper()}: p={result.p_value:.2e}, Cohen's d={result.effect_size:.2f} ({result.effect_interpretation}) {sig_marker}")
        all_results.append({
            'level': 'model',
            'group': model,
            **result.__dict__
        })
    
    # 3. 按缺失率检验
    print("\n3. 按缺失率检验...")
    for mr in [0.0, 0.3, 0.5, 0.7, 0.9]:
        mr_df = df[np.isclose(df['test_mr'], mr)]
        pivot_df = mr_df.pivot_table(
            index='seed',
            columns='use_mim',
            values='test_mae',
            aggfunc='mean'
        ).dropna()
        
        if len(pivot_df) < 10:
            print(f"  MR={mr}: 样本不足 ({len(pivot_df)} seeds)")
            continue
            
        result = paired_t_test(pivot_df[False].values, pivot_df[True].values)
        sig_marker = "✅" if result.significant else "❌"
        print(f"  MR={mr}: p={result.p_value:.2e}, 改进={result.mean_diff/result.ci_lower*100:.1f}%, Cohen's d={result.effect_size:.2f} {sig_marker}")
        all_results.append({
            'level': 'mr',
            'group': str(mr),
            **result.__dict__
        })
    
    # 4. 按模型×缺失率详细检验
    print("\n4. 按模型×缺失率详细检验...")
    detailed_results = compare_mim_effect(
        df,
        groupby_cols=['model', 'test_mr'],
        metric='test_mae',
        test_type='paired_t'
    )
    
    # 打印显著改善的情况
    sig_improvements = [r for r in detailed_results if r['significant'] and r['improvement_percent'] > 0]
    print(f"  总共 {len(detailed_results)} 个检验")
    print(f"  显著改善: {len(sig_improvements)} ({len(sig_improvements)/len(detailed_results)*100:.1f}%)")
    
    if sig_improvements:
        print("\n  显著改善的前5个情况:")
        top5 = sorted(sig_improvements, key=lambda x: x['improvement_percent'], reverse=True)[:5]
        for r in top5:
            print(f"    {r['group']}: 改进 {r['improvement_percent']:.1f}%, p={r['p_value']:.2e}, d={r['effect_size']:.2f}")
    
    # 汇总统计
    summary = summarize_significance(detailed_results)
    print(f"\n汇总:")
    print(f"  总检验数: {summary['total_tests']}")
    print(f"  显著比例: {summary['significant_rate']*100:.1f}%")
    print(f"  平均改进: {summary['avg_improvement_percent']:.1f}%")
    print(f"  效应量分布: 可忽略={summary['effect_sizes']['negligible']}, 小={summary['effect_sizes']['small']}, 中={summary['effect_sizes']['medium']}, 大={summary['effect_sizes']['large']}")
    
    # 保存详细结果
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f'{OUTPUT_DIR}/statistical_tests_detailed.csv', index=False)
    print(f"\n  详细结果已保存: {OUTPUT_DIR}/statistical_tests_detailed.csv")
    
    # 保存汇总
    with open(f'{OUTPUT_DIR}/statistical_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  汇总已保存: {OUTPUT_DIR}/statistical_summary.json")
    
    return detailed_results, summary


def create_enhanced_summary_table(df):
    """
    创建增强版汇总表格 (包含置信区间)
    """
    print("\n" + "="*60)
    print("生成增强汇总表格...")
    print("="*60)
    
    tables = {}
    
    # 按模型×缺失率的置信区间表
    rows = []
    for model in sorted(df['model'].unique()):
        for mr in sorted(df['test_mr'].unique()):
            subset = df[(df['model'] == model) & (np.isclose(df['test_mr'], mr))]
            
            no_mim = subset[subset['use_mim'] == False]
            mim = subset[subset['use_mim'] == True]
            
            if len(no_mim) < 2 or len(mim) < 2:
                continue
            
            # 配对样本
            pivot = subset.pivot_table(index='seed', columns='use_mim', values='test_mae').dropna()
            if len(pivot) < 2:
                continue
            
            no_mim_mean = pivot[False].mean()
            mim_mean = pivot[True].mean()
            improvement = (no_mim_mean - mim_mean) / no_mim_mean * 100
            
            # 计算置信区间
            try:
                result = paired_t_test(pivot[False].values, pivot[True].values)
                ci_lower = (result.ci_lower / no_mim_mean) * 100 if result.ci_lower else None
                ci_upper = (result.ci_upper / no_mim_mean) * 100 if result.ci_upper else None
            except:
                ci_lower = ci_upper = None
            
            rows.append({
                'model': model,
                'test_mr': mr,
                'no_mim_mean': no_mim_mean,
                'mim_mean': mim_mean,
                'improvement_percent': improvement,
                'ci_lower_percent': ci_lower,
                'ci_upper_percent': ci_upper,
                'n_seeds': len(pivot)
            })
    
    table = pd.DataFrame(rows)
    table.to_csv(f'{OUTPUT_DIR}/table_with_confidence_intervals.csv', index=False)
    print(f"  置信区间表已保存")
    tables['with_ci'] = table
    
    return tables


def create_key_findings_report(df, statistical_summary):
    """
    生成关键发现报告
    """
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    
    findings = []
    
    # 1. 整体效果
    overall_pivot = df.pivot_table(index='seed', columns='use_mim', values='test_mae').dropna()
    if len(overall_pivot) > 0:
        overall_improvement = (overall_pivot[False].mean() - overall_pivot[True].mean()) / overall_pivot[False].mean() * 100
        findings.append(f"1. 整体改进: MIM 相比 non-MIM 平均改进 {overall_improvement:.1f}%")
    
    # 2. 按缺失率的最佳效果
    mr_improvements = []
    for mr in sorted(df['test_mr'].unique()):
        mr_df = df[np.isclose(df['test_mr'], mr)]
        pivot = mr_df.pivot_table(index='seed', columns='use_mim', values='test_mae').dropna()
        if len(pivot) > 0:
            improvement = (pivot[False].mean() - pivot[True].mean()) / pivot[False].mean() * 100
            mr_improvements.append((mr, improvement))
    
    if mr_improvements:
        best_mr, best_improvement = max(mr_improvements, key=lambda x: x[1])
        findings.append(f"2. 最佳缺失率: MR={best_mr} 时改进最大 ({best_improvement:.1f}%)")
        
        # 高缺失率效果
        high_mr_improvements = [imp for mr, imp in mr_improvements if mr >= 0.7]
        if high_mr_improvements:
            avg_high_mr = np.mean(high_mr_improvements)
            findings.append(f"3. 高缺失率效果: MR≥0.7 时平均改进 {avg_high_mr:.1f}%")
    
    # 3. 按模型的效果
    model_improvements = []
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        pivot = model_df.pivot_table(index='seed', columns='use_mim', values='test_mae').dropna()
        if len(pivot) > 0:
            improvement = (pivot[False].mean() - pivot[True].mean()) / pivot[False].mean() * 100
            model_improvements.append((model, improvement))
    
    if model_improvements:
        best_model, best_model_imp = max(model_improvements, key=lambda x: x[1])
        findings.append(f"4. 最佳模型: {best_model.upper()} 改进最大 ({best_model_imp:.1f}%)")
    
    # 4. 统计显著性
    if statistical_summary:
        sig_rate = statistical_summary['significant_rate'] * 100
        findings.append(f"5. 统计显著性: {sig_rate:.1f}% 的检验显示显著改善")
        
        large_effect = statistical_summary['effect_sizes']['large']
        total_tests = statistical_summary['total_tests']
        if total_tests > 0:
            findings.append(f"6. 大效应量: {large_effect}/{total_tests} ({large_effect/total_tests*100:.1f}%) 检验显示大效应量 (Cohen's d > 0.8)")
    
    # 打印并保存
    for finding in findings:
        print(f"  {finding}")
    
    with open(f'{OUTPUT_DIR}/key_findings.txt', 'w') as f:
        f.write("关键发现\n")
        f.write("=" * 60 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n  报告已保存: {OUTPUT_DIR}/key_findings.txt")
    
    return findings


def main():
    """主函数"""
    print("="*60)
    print("100种子实验分析 - 增强版 (带统计显著性检验)")
    print("="*60)
    
    # 1. 加载数据
    df = load_all_results()
    
    # 2. 执行统计检验
    detailed_results, statistical_summary = perform_statistical_tests(df)
    
    # 3. 创建增强表格
    tables = create_enhanced_summary_table(df)
    
    # 4. 生成关键发现
    findings = create_key_findings_report(df, statistical_summary)
    
    print("\n" + "="*60)
    print(f"分析完成！结果保存在: {OUTPUT_DIR}/")
    print("="*60)
    print("\n生成的文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
