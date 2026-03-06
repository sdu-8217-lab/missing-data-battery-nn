# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
# -*- coding: utf-8 -*-
"""
鍒嗘瀽瀹為獙鏁版嵁锛岀敓鎴愯鏂囨墍闇€鐨勫悇椤圭粨璁烘暟鎹?
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(data_path):
    """鍔犺浇瀹為獙鏁版嵁"""
    df = pd.read_csv(data_path)
    print(f"鍔犺浇鏁版嵁: {len(df)} 鏉¤褰?)
    print(f"妯″瀷: {df['model_name'].unique()}")
    print(f"缂哄け鐜? {sorted(df['missing_rate'].unique())}")
    return df

def calculate_improvement(baseline_mae, mim_mae):
    """璁＄畻鏀硅繘鐜?""
    if baseline_mae == 0 or np.isnan(baseline_mae) or np.isnan(mim_mae):
        return np.nan
    return ((baseline_mae - mim_mae) / baseline_mae) * 100

def wilcoxon_test(baseline_values, mim_values):
    """鎵цWilcoxon绗﹀彿绉╂楠?""
    # 纭繚鏁扮粍闀垮害鐩稿悓
    min_len = min(len(baseline_values), len(mim_values))
    baseline_values = baseline_values[:min_len]
    mim_values = mim_values[:min_len]
    
    statistic, p_value = stats.wilcoxon(baseline_values, mim_values, alternative='greater')
    return statistic, p_value

def get_model_data(df, baseline_name, mim_name, mr):
    """鑾峰彇Baseline鍜孧IM鐨勬暟鎹?""
    baseline = df[(df['model_name'] == baseline_name) & (df['missing_rate'] == mr)]['mae']
    mim = df[(df['model_name'] == mim_name) & (df['missing_rate'] == mr)]['mae']
    return baseline, mim

def analyze_paper_conclusions(df):
    """鍒嗘瀽璁烘枃涓殑鍚勯」缁撹"""
    
    results = {}
    
    # 妯″瀷鏄犲皠: (鍩虹嚎鍚嶇О, MIM鍚嶇О, 鏄剧ず鍚嶇О)
    model_mapping = [
        ('MLP', 'MLP-MIM', 'MLP'),
        ('LSTM', 'LSTM-MIM', 'LSTM'),
        ('GRU', 'GRU-MIM', 'GRU'),
        ('CNN1D', 'CNN1D-MIM', '1D-CNN')
    ]
    
    missing_rates = sorted(df['missing_rate'].unique())
    
    print("="*80)
    print("璁烘枃鏁版嵁缁撹鍒嗘瀽")
    print("="*80)
    
    # 1. 琛?: MAE瀵规瘮 (MR=0.5 鍜?MR=0.9)
    print("\n銆愯〃1銆慚AE瀵规瘮 (鍧囧€悸辨爣鍑嗗樊)")
    print("-"*80)
    
    table1_data = []
    for mr in [0.5, 0.9]:
        print(f"\nMR = {mr}:")
        for baseline_name, mim_name, display_name in model_mapping:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                print(f"{display_name:8s} 鏁版嵁缂哄け")
                continue
            
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            mim_mean = mim.mean()
            mim_std = mim.std()
            
            print(f"{display_name:8s} Baseline: {baseline_mean:.3f}卤{baseline_std:.3f}, MIM: {mim_mean:.3f}卤{mim_std:.3f}")
            
            table1_data.append({
                'model': display_name,
                'mr': mr,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'mim_mean': mim_mean,
                'mim_std': mim_std
            })
    
    results['table1'] = pd.DataFrame(table1_data)
    
    # 2. 琛?: 鏀硅繘鐜囨眹鎬?
    print("\n\n銆愯〃2銆慚IM鐩告瘮Baseline鐨凪AE鏀硅繘鐜囨眹鎬?(%)")
    print("-"*80)
    
    table2_data = []
    key_mrs = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"{'妯″瀷':<10s} {'0.1':>8s} {'0.3':>8s} {'0.5':>8s} {'0.7':>8s} {'0.9':>8s} {'骞冲潎':>8s}")
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
    
    # 3. 琛?: 瀹屾暣MAE缁撴灉涓庢敼杩涚巼 (MR=0.3, 0.5, 0.7, 0.9)
    print("\n\n銆愯〃3銆戜笉鍚岀己澶辩巼涓嬬殑MAE瀵规瘮涓庢敼杩涚巼")
    print("-"*80)
    
    table3_data = []
    key_mrs = [0.3, 0.5, 0.7, 0.9]
    
    for baseline_name, mim_name, display_name in model_mapping:
        print(f"\n{display_name}:")
        for mr in key_mrs:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                print(f"  MR={mr}: 鏁版嵁缂哄け")
                continue
            
            baseline_mean = baseline.mean()
            baseline_std = baseline.std()
            mim_mean = mim.mean()
            mim_std = mim.std()
            improvement = calculate_improvement(baseline_mean, mim_mean)
            
            print(f"  MR={mr}: Baseline={baseline_mean:.3f}卤{baseline_std:.3f}, "
                  f"MIM={mim_mean:.3f}卤{mim_std:.3f}, 鏀硅繘鐜?{improvement:.1f}%")
            
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
    
    # 4. 琛?: Wilcoxon绗﹀彿绉╂楠?
    print("\n\n銆愯〃4銆慦ilcoxon绗﹀彿绉╂楠岀粨鏋?(MIM vs Baseline)")
    print("-"*80)
    
    table4_data = []
    key_mrs = [0.3, 0.5, 0.7, 0.9]
    
    print(f"{'妯″瀷':<10s} {'MR=0.3':>12s} {'MR=0.5':>12s} {'MR=0.7':>12s} {'MR=0.9':>12s}")
    print("-"*60)
    
    for baseline_name, mim_name, display_name in model_mapping:
        row = {'model': display_name}
        
        for mr in key_mrs:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            
            if len(baseline) == 0 or len(mim) == 0:
                row[f'mr_{mr}'] = np.nan
                row[f'mr_{mr}_sig'] = 'N/A'
                continue
            
            # Wilcoxon妫€楠?
            try:
                statistic, p_value = wilcoxon_test(baseline.values, mim.values)
                
                # 纭畾鏄捐憲鎬ф爣璁?
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
    
    # 5. 鍏抽敭鍙戠幇鏁版嵁
    print("\n\n銆愬叧閿彂鐜版暟鎹€?)
    print("-"*80)
    
    # MR=0.5鏃剁殑鍏蜂綋鏀硅繘
    print("\n1. MR=0.5鏃剁殑MAE鏀硅繘:")
    for baseline_name, mim_name, display_name in model_mapping:
        baseline, mim = get_model_data(df, baseline_name, mim_name, 0.5)
        if len(baseline) > 0 and len(mim) > 0:
            baseline_mean = baseline.mean()
            mim_mean = mim.mean()
            improvement = calculate_improvement(baseline_mean, mim_mean)
            print(f"   {display_name}: {baseline_mean:.3f} 鈫?{mim_mean:.3f} (鏀硅繘鐜噞improvement:.1f}%)")
    
    # MR=0.9鏃剁殑鏋佺鎯呭喌
    print("\n2. MR=0.9鏃剁殑鏋佺鎯呭喌:")
    for baseline_name, mim_name, display_name in model_mapping:
        baseline, mim = get_model_data(df, baseline_name, mim_name, 0.9)
        if len(baseline) > 0 and len(mim) > 0:
            baseline_mean = baseline.mean()
            mim_mean = mim.mean()
            print(f"   {display_name} Baseline: {baseline_mean:.3f}, MIM: {mim_mean:.3f}")
    
    # 骞冲潎鏀硅繘鐜?
    print("\n3. 骞冲潎鏀硅繘鐜?(MR=0.1-0.9):")
    for baseline_name, mim_name, display_name in model_mapping:
        improvements = []
        for mr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            baseline, mim = get_model_data(df, baseline_name, mim_name, mr)
            if len(baseline) > 0 and len(mim) > 0:
                improvements.append(calculate_improvement(baseline.mean(), mim.mean()))
        if improvements:
            avg_imp = np.mean(improvements)
            print(f"   {display_name}: {avg_imp:.1f}%")
    
    # 6. 鍙傛暟閲忎俊鎭?
    print("\n銆愬弬鏁伴噺淇℃伅銆?)
    print("-"*80)
    for baseline_name, mim_name, display_name in model_mapping:
        baseline_params = df[df['model_name'] == baseline_name]['params'].iloc[0] if len(df[df['model_name'] == baseline_name]) > 0 else 0
        mim_params = df[df['model_name'] == mim_name]['params'].iloc[0] if len(df[df['model_name'] == mim_name]) > 0 else 0
        print(f"{display_name:8s} Baseline: {baseline_params:,} params, MIM: {mim_params:,} params")
    
    # 7. 缁熻妫€楠屼俊鎭?
    print("\n銆愮粺璁℃楠屼俊鎭€?)
    print("-"*80)
    print(f"闅忔満绉嶅瓙鏁伴噺: {len(df['seed'].unique())}")
    print(f"绉嶅瓙鑼冨洿: {df['seed'].min()} - {df['seed'].max()}")
    
    # 8. 鐢熸垚LaTeX琛ㄦ牸浠ｇ爜
    print("\n\n銆怢aTeX琛ㄦ牸浠ｇ爜銆?)
    print("="*80)
    generate_latex_tables(results)
    
    return results

def generate_latex_tables(results):
    """鐢熸垚LaTeX琛ㄦ牸浠ｇ爜"""
    
    # 琛?: MAE瀵规瘮
    print("\n% 琛?: MAE瀵规瘮 (MR=0.5 鍜?MR=0.9)")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{缂哄け鐜?.5鍜?.9鏃剁殑MAE瀵规瘮锛堝潎鍊?\\pm$鏍囧噯宸紝100闅忔満绉嶅瓙锛墋")
    print("\\label{tab:mae_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{\\textbf{Missing Rate = 0.5}} & \\multicolumn{2}{c}{\\textbf{Missing Rate = 0.9}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print("\\textbf{妯″瀷} & \\textbf{Baseline} & \\textbf{MIM} & \\textbf{Baseline} & \\textbf{MIM} \\\\")
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
    
    # 琛?: 鏀硅繘鐜囨眹鎬?
    print("\n% 琛?: 鏀硅繘鐜囨眹鎬?)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{MIM鐩告瘮Baseline鐨凪AE鏀硅繘鐜囨眹鎬伙紙\\%锛墋")
    print("\\label{tab:improvement_summary}")
    print("\\begin{tabular}{@{}lcccccc@{}}")
    print("\\toprule")
    print("\\textbf{妯″瀷} & \\textbf{0.1} & \\textbf{0.3} & \\textbf{0.5} & \\textbf{0.7} & \\textbf{0.9} & \\textbf{骞冲潎} \\\\")
    print("\\midrule")
    
    for _, row in results['table2'].iterrows():
        print(f"{row['model']} & {row['mr_0.1']:.1f} & {row['mr_0.3']:.1f} & "
              f"{row['mr_0.5']:.1f} & {row['mr_0.7']:.1f} & {row['mr_0.9']:.1f} & {row['average']:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 琛?: 瀹屾暣缁撴灉
    print("\n% 琛?: 瀹屾暣MAE缁撴灉涓庢敼杩涚巼")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\small")
    print("\\caption{涓嶅悓缂哄け鐜囦笅鐨凪AE瀵规瘮涓庢敼杩涚巼锛?00闅忔満绉嶅瓙锛屾渶浼樺€煎姞绮楋級}")
    print("\\label{tab:full_results}")
    print("\\begin{tabular}{@{}llcccc@{}}")
    print("\\toprule")
    print("\\textbf{妯″瀷} & \\textbf{绛栫暐} & \\textbf{MR=0.3} & \\textbf{MR=0.5} & \\textbf{MR=0.7} & \\textbf{MR=0.9} \\\\")
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
            
            # Baseline琛?
            if 0.3 in rows:
                print(f"& Baseline & {rows[0.3]['baseline_mean']:.3f}$\\pm${rows[0.3]['baseline_std']:.3f} & "
                      f"{rows[0.5]['baseline_mean']:.3f}$\\pm${rows[0.5]['baseline_std']:.3f} & "
                      f"{rows[0.7]['baseline_mean']:.3f}$\\pm${rows[0.7]['baseline_std']:.3f} & "
                      f"{rows[0.9]['baseline_mean']:.3f}$\\pm${rows[0.9]['baseline_std']:.3f} \\\\")
            
            # MIM琛?
            if 0.3 in rows:
                print(f"& MIM & \\textbf{{{rows[0.3]['mim_mean']:.3f}}}$\\pm${rows[0.3]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.5]['mim_mean']:.3f}}}$\\pm${rows[0.5]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.7]['mim_mean']:.3f}}}$\\pm${rows[0.7]['mim_std']:.3f} & "
                      f"\\textbf{{{rows[0.9]['mim_mean']:.3f}}}$\\pm${rows[0.9]['mim_std']:.3f} \\\\")
            
            # 鏀硅繘鐜囪
            if 0.3 in rows:
                print(f"& 鏀硅繘鐜?& {rows[0.3]['improvement']:.1f}\\% & {rows[0.5]['improvement']:.1f}\\% & "
                      f"{rows[0.7]['improvement']:.1f}\\% & {rows[0.9]['improvement']:.1f}\\% \\\\")
            print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 琛?: Wilcoxon妫€楠?
    print("\n% 琛?: Wilcoxon绗﹀彿绉╂楠?)
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\small")
    print("\\caption{Wilcoxon绗﹀彿绉╂楠岀粨鏋滐紙MIM vs Baseline锛墋")
    print("\\label{tab:significance}")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{妯″瀷} & \\textbf{MR=0.3} & \\textbf{MR=0.5} & \\textbf{MR=0.7} & \\textbf{MR=0.9} \\\\")
    print("\\midrule")
    
    for _, row in results['table4'].iterrows():
        print(f"{row['model']} & {row['mr_0.3_sig']} & {row['mr_0.5_sig']} & "
              f"{row['mr_0.7_sig']} & {row['mr_0.9_sig']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\\\[3pt]")
    print("\\footnotesize 娉細*** 琛ㄧず$p<$0.001锛學ilcoxon绗﹀彿绉╂楠岋紝鍙屽熬妫€楠岋紝$K=$100涓殢鏈虹瀛愩€?)
    print("\\end{table}")

def save_results_to_csv(results, output_dir):
    """淇濆瓨缁撴灉鍒癈SV鏂囦欢"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in results.items():
        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        print(f"淇濆瓨: {filepath}")

if __name__ == '__main__':
    # 鏁版嵁璺緞
    data_path = r"E:\00_KindKeeper\missing-data-battery-nn\experiments_v2\3C\20260204_084913\results_all.csv"
    output_dir = r"E:\00_KindKeeper\missing-data-battery-nn\experiments_v2\3C\20260204_084913\paper_results"
    
    # 鍔犺浇鏁版嵁
    df = load_data(data_path)
    
    # 鍒嗘瀽璁烘枃缁撹
    results = analyze_paper_conclusions(df)
    
    # 淇濆瓨缁撴灉
    save_results_to_csv(results, output_dir)
    
    print("\n" + "="*80)
    print("鍒嗘瀽瀹屾垚锛?)
    print(f"缁撴灉宸蹭繚瀛樺埌: {output_dir}")
    print("="*80)

