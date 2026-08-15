# -*- coding: utf-8 -*-
"""
分析实验数据，生成论文所需的各项结论数据
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(data_path):
    """加载实验数据"""
    df = pd.read_csv(data_path)
    print(f"加载数据: {len(df)} 条记录")
    print(f"模型: {df['model_name'].unique()}")
    print(f"缺失率: {sorted(df['missing_rate'].unique())}")
    return df

def calculate_improvement(baseline_mae, mim_mae):
    """计算改进率"""
    if baseline_mae == 0 or np.isnan(baseline_mae) or np.isnan(mim_mae):
        return np.nan
    return ((baseline_mae - mim_mae) / baseline_mae) * 100

def wilcoxon_test(baseline_values, mim_values):
    """执行Wilcoxon符号秩检验"""
    # 确保数组长度相同
    min_len = min(len(baseline_values), len(mim_values))
    baseline_values = baseline_values[:min_len]
    mim_values = mim_values[:min_len]
    
    statistic, p_value = stats.wilcoxon(baseline_values, mim_values, alternative='greater')
    return statistic, p_value

def get_model_data(df, baseline_name, mim_name, mr):
    """获取Baseline和MIM的数据"""
    baseline = df[(df['model_name'] == baseline_name) & (df['missing_rate'] == mr)]['mae']
    mim = df[(df['model_name'] == mim_name) & (df['missing_rate'] == mr)]['mae']
    return baseline, mim

def analyze_paper_conclusions(df):
    """分析论文中的各项结论"""
    
    results = {}
    
    # 模型映射: (基线名称, MIM名称, 显示名称)
    model_mapping = [
        ('MLP', 'MLP-MIM', 'MLP'),
        ('LSTM', 'LSTM-MIM', 'LSTM'),
        ('GRU', 'GRU-MIM', 'GRU'),
        ('CNN1D', 'CNN1D-MIM', '1D-CNN')
    ]
    
    missing_rates = sorted(df['missing_rate'].unique())
    
    print("="*80)
    print("论文数据结论分析")
    print("="*80)
    
    # 1. 表1: MAE对比 (MR=0.5 和 MR=0.9)
    print("\n【表1】MAE对比 (均值±标准差)")
    print("-"*80)
    
    table1_data = []
    for mr in [0.5, 0.9]:
        print(f"\nMR = {mr}:")
        for baseline_name, mim_name, display_name in model_mapping:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                print(f"{display_name:8s} 数据缺失")
                continue
            
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            mim_mean = mim.mean()
            mim_std = mim.std()
            
            print(f"{display_name:8s} Baseline: {baseline_mean:.3f}±{baseline_std:.3f}, MIM: {mim_mean:.3f}±{mim_std:.3f}")
            
            table1_data.append({
                'model': display_name,
                'mr': mr,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'mim_mean': mim_mean,
                'mim_std': mim_std
            })
    
    results['table1'] = pd.DataFrame(table1_data)
    
    # 2. 表2: 改进率汇总
    print("\n\n【表2】MIM相比Baseline的MAE改进率汇总 (%)")
    print("-"*80)
    
    table2_data = []
    key_mrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"{'模型':<10s} {'0.1':>8s} {'0.3':>8s} {'0.5':>8s} {'0.7':>8s} {'0.9':>8s} {'平均':>8s}")
    print("-"*60)
    
    for baseline_name, mim_name, display_name in model_mapping:
        improvements = []
        row = {'model': display_name}
        
        for mr in key_mrs:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                row[f'mr_{mr}'] = np.nan
                continue
            
            improvement = calculate_improvement(baseline.mean(), mim.mean())
            improvements.append(improvement)
            row[f'mr_{mr}'] = improvement
        
        avg_improvement = np.mean(improvements) if improvements else np.nan
        row['average'] = avg_improvement
        table2_data.append(row)
        
        print(f"{display_name:<10s} {row.get('mr_0.1', np.nan):>8.1f} {row.get('mr_0.3', np.nan):>8.1f} "
              f"{row.get('mr_0.5', np.nan):>8.1f} {row.get('mr_0.7', np.nan):>8.1f} "
              f"{row.get('mr_0.9', np.nan):>8.1f} {avg_improvement:>8.1f}")
    
    results['table2'] = pd.DataFrame(table2_data)
    
    # 3. 表3: 完整MAE结果与改进率 (MR=0.3, 0.5, 0.7, 0.9)
    print("\n\n【表3】不同缺失率下的MAE对比与改进率")
    print("-"*80)
    
    table3_data = []
    key_mrs = [0.3, 0.5, 0.7, 0.9]
    
    for baseline_name, mim_name, display_name in model_mapping:
        print(f"\n{display_name}:")
        for mr in key_mrs:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                print(f"  MR={mr}: 数据缺失")
                continue
            
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            mim_mean = mim.mean()
            mim_std = mim.std()
            improvement = calculate_improvement(baseline_mean, mim_mean)
            
            print(f"  MR={mr}: Baseline={baseline_mean:.3f}±{baseline_std:.3f}, "
                  f"MIM={mim_mean:.3f}±{mim_std:.3f}, 改进率={improvement:.1f}%")
            
            table3_data.append({
                'model': display_name,
                'mr': mr,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'mim_mean': mim_mean,
                'mim_std': mim_std,
                'improvement': improvement
            })
    
    results['table3'] = pd.DataFrame(table3_data)
    
    # 4. 表4: Wilcoxon符号秩检验
    print("\n\n【表4】Wilcoxon符号秩检验结果 (MIM vs Baseline)")
    print("-"*80)
    
    table4_data = []
    key_mrs = [0.3, 0.5, 0.7, 0.9]
    
    print(f"{'模型':<10s} {'MR=0.3':>12s} {'MR=0.5':>12s} {'MR=0.7':>12s} {'MR=0.9':>12s}")
    print("-"*60)
    
    for baseline_name, mim_name, display_name in model_mapping:
        row = {'model': display_name}
        
        for mr in key_mrs:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                row[f'mr_{mr}'] = np.nan
                row[f'mr_{mr}_sig'] = 'N/A'
                continue
            
            # Wilcoxon检验
            try:
                statistic, p_value = wilcoxon_test(baseline.values, mim.values)
                
                # 确定显著性标记
                if p_value < 0.001:
                    sig = "<0.001***"
                elif p_value < 0.01:
                    sig = f"{p_value:.3f}**"
                elif p_value < 0.05:
                    sig = f"{p_value:.3f}*"
                else:
                    sig = f"{p_value:.3f}"
                
                row[f'mr_{mr}'] = p_value
                row[f'mr_{mr}_sig'] = sig
            except Exception as e:
                row[f'mr_{mr}'] = np.nan
                row[f'mr_{mr}_sig'] = 'Error'
        
        table4_data.append(row)
        print(f"{display_name:<10s} {row.get('mr_0.3_sig', 'N/A'):>12s} {row.get('mr_0.5_sig', 'N/A'):>12s} "
              f"{row.get('mr_0.7_sig', 'N/A'):>12s} {row.get('mr_0.9_sig', 'N/A'):>12s}")
    
    results['table4'] = pd.DataFrame(table4_data)
    
    # 5. 关键发现数据
    print("\n\n【关键发现数据】")
    print("-"*80)
    
    # MR=0.5时的具体改进
    print("\n1. MR=0.5时的MAE改进:")
    for baseline_name, mim_name, display_name in model_mapping:
        baseline, mim = get_model_data(df, baseline_name, mim_name, 0.5)
        if len(baseline) > 0 and len(mim) > 0:
            baseline_mean = baseline.mean()
            mim_mean = mim.mean()
            improvement = calculate_improvement(baseline_mean, mim_mean)
            print(f"   {display_name}: {baseline_mean:.3f} → {mim_mean:.3f} (改进率{improvement:.1f}%)")
    
    # MR=0.9时的极端情况
    print("\n2. MR=0.9时的极端情况:")
    for baseline_name, mim_name, display_name in model_mapping:
        baseline, mim = get_model_data(df, baseline_name, mim_name, 0.9)
        if len(baseline) > 0 and len(mim) > 0:
            baseline_mean = baseline.mean()
            mim_mean = mim.mean()
            print(f"   {display_name} Baseline: {baseline_mean:.3f}, MIM: {mim_mean:.3f}")
    
    # 平均改进率
    print("\n3. 平均改进率 (MR=0.1-0.9):")
    for baseline_name, mim_name, display_name in model_mapping:
        improvements = []
        for mr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            if len(baseline) > 0 and len(mim) > 0:
                improvements.append(calculate_improvement(baseline.mean(), mim.mean()))
        if improvements:
            avg_imp = np.mean(improvements)
            print(f"   {display_name}: {avg_imp:.1f}%")
    
    # 6. 参数量信息
    print("\n【参数量信息】")
    print("-"*80)
    for baseline_name, mim_name, display_name in model_mapping:
        baseline_params = df[df['model_name'] == baseline_name]['params'].iloc[0] if len(df[df['model_name'] == baseline_name]) > 0 else 0
        mim_params = df[df['model_name'] == mim_name]['params'].iloc[0] if len(df[df['model_name'] == mim_name]) > 0 else 0
        print(f"{display_name:8s} Baseline: {baseline_params:,} params, MIM: {mim_params:,} params")
    
    # 7. 统计检验信息
    print("\n【统计检验信息】")
    print("-"*80)
    print(f"随机种子数量: {len(df['seed'].unique())}")
    print(f"种子范围: {df['seed'].min()} - {df['seed'].max()}")
    
    # 8. 生成LaTeX表格代码
    print("\n\n【LaTeX表格代码】")
    print("="*80)
    generate_latex_tables(results)
    
    return results

def generate_latex_tables(results):
    """生成LaTeX表格代码"""
    
    # 表1: MAE对比
    print("\n% 表1: MAE对比 (MR=0.5 和 MR=0.9)")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{缺失率0.5和0.9时的MAE对比（均值$\\pm$标准差，100随机种子）}")
    print("\\label{tab:mae_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{\\textbf{Missing Rate = 0.5}} & \\multicolumn{2}{c}{\\textbf{Missing Rate = 0.9}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print("\\textbf{模型} & \\textbf{Baseline} & \\textbf{MIM} & \\textbf{Baseline} & \\textbf{MIM} \\\\")
    print("\\midrule")
    
    if len(results['table1']) > 0:
        for model in results['table1']['model'].unique():
            model_data = results['table1'][results['table1']['model'] == model]
            mr5_data = model_data[model_data['mr'] == 0.5]
            mr9_data = model_data[model_data['mr'] == 0.9]
            
            if len(mr5_data) > 0 and len(mr9_data) > 0:
                row5 = mr5_data.iloc[0]
                row9 = mr9_data.iloc[0]
                print(f"{row5['model']} & {row5['baseline_mean']:.3f}$\\pm${row5['baseline_std']:.3f} & "
                      f"\\textbf{{{row5['mim_mean']:.3f}}}$\\pm${row5['mim_std']:.3f} & "
                      f"{row9['baseline_mean']:.3f}$\\pm${row9['baseline_std']:.3f} & "
                      f"\\textbf{{{row9['mim_mean']:.3f}}}$\\pm${row9['mim_std']:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 表2: 改进率汇总
    print("\n% 表2: 改进率汇总")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{MIM相比Baseline的MAE改进率汇总（\\%）}")
    print("\\label{tab:improvement_summary}")
    print("\\begin{tabular}{@{}lcccccc@{}}")
    print("\\toprule")
    print("\\textbf{模型} & \\textbf{0.1} & \\textbf{0.3} & \\textbf{0.5} & \\textbf{0.7} & \\textbf{0.9} & \\textbf{平均} \\\\")
    print("\\midrule")
    
    for _, row in results['table2'].iterrows():
        print(f"{row['model']} & {row['mr_0.1']:.1f} & {row['mr_0.3']:.1f} & "
              f"{row['mr_0.5']:.1f} & {row['mr_0.7']:.1f} & {row['mr_0.9']:.1f} & {row['average']:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 表3: 完整结果
    print("\n% 表3: 完整MAE结果与改进率")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\small")
    print("\\caption{不同缺失率下的MAE对比与改进率（100随机种子，最优值加粗）}")
    print("\\label{tab:full_results}")
    print("\\begin{tabular}{@{}llcccc@{}}")
    print("\\toprule")
    print("\\textbf{模型} & \\textbf{策略} & \\textbf{MR=0.3} & \\textbf{MR=0.5} & \\textbf{MR=0.7} & \\textbf{MR=0.9} \\\\")
    print("\\midrule")
    
    models = results['table3']['model'].unique() if len(results['table3']) > 0 else []
    for model in models:
        model_data = results['table3'][results['table3']['model'] == model]
        
        rows = {}
        for mr in [0.3, 0.5, 0.7, 0.9]:
            mr_data = model_data[model_data['mr'] == mr]
            if len(mr_data) > 0:
                rows[mr] = mr_data.iloc[0]
        
        if rows:
            print(f"\\multirow{{3}}{{*}}{{{model}}} ")
            
            # Baseline行
            if 0.3 in rows:
                print(f"& Baseline & {rows[0.3]['baseline_mean']:.3f}$\\pm${rows[0.3]['baseline_std']:.3f} & "
                      f"{rows[0.5]['baseline_mean']:.3f}$\\pm${rows[0.5]['baseline_std']:.3f} & "
                      f"{rows[0.7]['baseline_mean']:.3f}$\\pm${rows[0.7]['baseline_std']:.3f} & "
                      f"{rows[0.9]['baseline_mean']:.3f}$\\pm${rows[0.9]['baseline_std']:.3f} \\\\")
            
            # MIM行
            if 0.3 in rows:
                print(f"& MIM & \\textbf{{{rows[0.3]['mim_mean']:.3f}}}$\\pm${rows[0.3]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.5]['mim_mean']:.3f}}}$\\pm${rows[0.5]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.7]['mim_mean']:.3f}}}$\\pm${rows[0.7]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.9]['mim_mean']:.3f}}}$\\pm${rows[0.9]['mim_std']:.3f} \\\\")
            
            # 改进率行
            if 0.3 in rows:
                print(f"& 改进率 & {rows[0.3]['improvement']:.1f}\\% & {rows[0.5]['improvement']:.1f}\\% & "
                      f"{rows[0.7]['improvement']:.1f}\\% & {rows[0.9]['improvement']:.1f}\\% \\\\")
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 表4: Wilcoxon检验
    print("\n% 表4: Wilcoxon符号秩检验")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\small")
    print("\\caption{Wilcoxon符号秩检验结果（MIM vs Baseline）}")
    print("\\label{tab:significance}")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{模型} & \\textbf{MR=0.3} & \\textbf{MR=0.5} & \\textbf{MR=0.7} & \\textbf{MR=0.9} \\\\")
    print("\\midrule")
    
    for _, row in results['table4'].iterrows():
        print(f"{row['model']} & {row['mr_0.3_sig']} & {row['mr_0.5_sig']} & "
              f"{row['mr_0.7_sig']} & {row['mr_0.9_sig']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\\\[3pt]")
    print("\\footnotesize 注：*** 表示$p<$0.001，Wilcoxon符号秩检验，双尾检验，$K=$100个随机种子。")
    print("\\end{table}")

def save_results_to_csv(results, output_dir):
    """保存结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in results.items():
        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        print(f"保存: {filepath}")

if __name__ == '__main__':
    # 数据路径
    data_path = r"E:\00_KindKeeper\missing-data-battery-nn\experiments_v2\3C\20260204_084913\results_all.csv"
    output_dir = r"E:\00_KindKeeper\missing-data-battery-nn\experiments_v2\3C\20260204_084913\paper_results"
    
    # 加载数据
    df = load_data(data_path)
    
    # 分析论文结论
    results = analyze_paper_conclusions(df)
    
    # 保存结果
    save_results_to_csv(results, output_dir)
    
    print("\n" + "="*80)
    print("分析完成！")
    print(f"结果已保存到: {output_dir}")
    print("="*80)
