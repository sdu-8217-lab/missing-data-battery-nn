# legacy script for reference
# 此脚本为历史版本，仅作参考用
# 当前项目使用 experiments/train.py 作为统一入口
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def load_all_csv_files(base_dir="batch_experiment_results"):
    """
    浠庣洰褰曚腑鍔犺浇鎵€鏈塁SV鏂囦欢
    """
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'missing_combinations' in file:
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError(f"鍦ㄧ洰褰?{base_dir} 涓湭鎵惧埌CSV鏂囦欢")
    
    print(f"鎵惧埌 {len(csv_files)} 涓狢SV鏂囦欢")
    
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # 娣诲姞瀹為獙ID淇℃伅
            exp_folder = Path(file_path).parent.name
            df['experiment_id'] = exp_folder
            all_data.append(df)
            print(f"鍔犺浇: {Path(file_path).name} - 褰㈢姸: {df.shape}")
        except Exception as e:
            print(f"鍔犺浇鏂囦欢澶辫触 {file_path}: {e}")
    
    if not all_data:
        raise ValueError("娌℃湁鎴愬姛鍔犺浇浠讳綍CSV鏂囦欢")
    
    # 鍚堝苟鎵€鏈夋暟鎹?
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 瑙ｆ瀽缂哄け缁勫悎瀛楃涓蹭负鍏冪粍
    combined_df['missing_tuple'] = combined_df['Missing_Combination'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple()
    )
    
    # 娣诲姞缂哄け鐗瑰緛鏁伴噺
    combined_df['num_missing_features'] = combined_df['missing_tuple'].apply(len)
    
    # 璁＄畻鎬ц兘宸紓
    combined_df['mae_diff'] = combined_df['Indicator_MAE'] - combined_df['Reduced_MAE']
    combined_df['rmse_diff'] = combined_df['Indicator_RMSE'] - combined_df['Reduced_RMSE']
    combined_df['r2_diff'] = combined_df['Indicator_R2'] - combined_df['Reduced_R2']
    
    # 娣诲姞鎬ц兘鍒嗙被
    combined_df['indicator_better'] = combined_df['mae_diff'] < 0  # 缂哄け鎸囩ず鍣ㄦ柟娉曟洿濂?
    combined_df['reduced_better'] = combined_df['mae_diff'] > 0    # 缂╁噺妯″瀷鏇村ソ
    combined_df['similar'] = combined_df['mae_diff'] == 0          # 鎬ц兘鐩镐技
    
    return combined_df

def plot_performance_comparison(df, save_dir="visualizations"):
    """
    缁樺埗鎬ц兘瀵规瘮鍥撅紙瀛愬浘鍒嗗埆淇濆瓨锛?
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. MAE瀵规瘮鏁ｇ偣鍥?
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Indicator_MAE'], df['Reduced_MAE'], alpha=0.6, s=30)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect Agreement')
    ax.set_xlabel('Indicator Model MAE', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reduced Model MAE', fontsize=12, fontweight='bold')
    ax.set_title('MAE Comparison Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE瀵规瘮鏁ｇ偣鍥?
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Indicator_RMSE'], df['Reduced_RMSE'], alpha=0.6, s=30, color='orange')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect Agreement')
    ax.set_xlabel('Indicator Model RMSE', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reduced Model RMSE', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Comparison Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rmse_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. R虏瀵规瘮鏁ｇ偣鍥?
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Indicator_R2'], df['Reduced_R2'], alpha=0.6, s=30, color='green')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect Agreement')
    ax.set_xlabel('Indicator Model R虏', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reduced Model R虏', fontsize=12, fontweight='bold')
    ax.set_title('R虏 Comparison Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/r2_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 鎬ц兘宸紓鍒嗗竷锛圡AE锛?
    fig, ax = plt.subplots(figsize=(8, 6))
    mae_diff = df['Indicator_MAE'] - df['Reduced_MAE']
    ax.hist(mae_diff, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero Difference Line')
    ax.set_xlabel('MAE Difference (Indicator - Reduced)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of MAE Differences', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_difference_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 鏀硅繘鐧惧垎姣斿垎甯?
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['Improvement_Percent'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='Zero Improvement Line')
    ax.set_xlabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Improvement Percentages', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_percentage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 鎸夌己澶辩壒寰佹暟閲忓垎缁勭殑绠辩嚎鍥?
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.sort_values('num_missing_features')
    sns.boxplot(data=df_sorted, x='num_missing_features', y='Indicator_MAE', ax=ax)
    ax.set_xlabel('Number of Missing Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('MAE Distribution by Number of Missing Features', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_by_missing_features_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_detailed_comparison(df, save_dir="visualizations"):
    """
    缁樺埗璇︾粏鐨勬€ц兘瀵规瘮鍥撅紙瀛愬浘鍒嗗埆淇濆瓨锛?
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 鎵€鏈夋寚鏍囩殑绠辩嚎鍥惧姣?
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_data = []
    for metric in ['MAE', 'RMSE', 'R2']:
        indicator_vals = df[f'Indicator_{metric}']
        reduced_vals = df[f'Reduced_{metric}']
        temp_df = pd.DataFrame({
            'Value': list(indicator_vals) + list(reduced_vals),
            'Model': ['Indicator Model'] * len(indicator_vals) + ['Reduced Model'] * len(reduced_vals),
            'Metric': [metric] * (len(indicator_vals) + len(reduced_vals))
        })
        metrics_data.append(temp_df)
    
    metrics_df = pd.concat(metrics_data, ignore_index=True)
    
    sns.boxplot(data=metrics_df, x='Metric', y='Value', hue='Model', ax=ax)
    ax.set_title('Model Performance Boxplot Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_performance_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 鎸夌己澶辩壒寰佹暟閲忕殑鎬ц兘瀵规瘮锛堝弻杞村浘锛?
    fig, ax1 = plt.subplots(figsize=(10, 6))
    grouped = df.groupby('num_missing_features').agg({
        'Indicator_MAE': ['mean', 'std'],
        'Reduced_MAE': ['mean', 'std'],
        'Improvement_Percent': ['mean', 'std']
    }).round(6)
    
    # 灞曞紑澶氱骇绱㈠紩
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    
    # 宸杞达細MAE鍧囧€悸辨爣鍑嗗樊锛堝甫璇樊妫掞級
    ax1.errorbar(grouped.index, grouped['Indicator_MAE_mean'], 
                 yerr=grouped['Indicator_MAE_std'], fmt='o-', color='#1f77b4', 
                 label='Indicator MAE', capsize=3, linewidth=2)
    ax1.errorbar(grouped.index, grouped['Reduced_MAE_mean'], 
                 yerr=grouped['Reduced_MAE_std'], fmt='s--', color='#ff7f0e', 
                 label='Reduced MAE', capsize=3, linewidth=2)
    
    # 鍙砓杞达細鏀硅繘鐧惧垎姣斿潎鍊硷紙甯︽樉钁楁€ф爣璁帮級
    ax2 = ax1.twinx()
    bars = ax2.bar(grouped.index, grouped['Improvement_Percent_mean'], 
                   yerr=grouped['Improvement_Percent_std'], alpha=0.3, color='gray', width=0.6, 
                   label='Avg Improvement %', capsize=3)
    
    # 鏄捐憲鎬ф爣璁?
    for x in grouped.index:
        if abs(grouped.loc[x, 'Improvement_Percent_mean']) > 5:  # 闃堝€煎彲璋?
            try:
                subset_df = df[df['num_missing_features'] == x]
                if len(subset_df) > 1:  # 闇€瑕佽嚦灏?涓牱鏈墠鑳借绠梩妫€楠?
                    t_stat, p_val = stats.ttest_ind(
                        subset_df['Indicator_MAE'],
                        subset_df['Reduced_MAE']
                    )
                    y_pos = grouped.loc[x, 'Improvement_Percent_mean']
                    if p_val < 0.001: 
                        ax2.text(x, y_pos + 2, '***', ha='center', fontsize=14, color='red')
                    elif p_val < 0.01: 
                        ax2.text(x, y_pos + 2, '**', ha='center', fontsize=14, color='red')
                    elif p_val < 0.05: 
                        ax2.text(x, y_pos + 2, '*', ha='center', fontsize=12, color='darkorange')
            except:
                pass  # 濡傛灉璁＄畻澶辫触鍒欒烦杩囨爣娉?
    
    ax1.set_xlabel('Number of Missing Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (Mean 卤 Std)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold', color='gray')
    ax1.set_title('Missing Degree Impact: Indicator Method Advantage Increases with More Missing Features', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/missing_degree_impact_double_axis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 鏀硅繘鐧惧垎姣斾笌缂哄け鐗瑰緛鏁伴噺鐨勫叧绯伙紙淇瓒嬪娍绾匡級
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['num_missing_features'], df['Improvement_Percent'], alpha=0.6)
    ax.set_xlabel('Number of Missing Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement Percentage vs Number of Missing Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 淇瓒嬪娍绾匡細x涓虹己澶辩壒寰佹暟閲忥紝y涓烘敼杩涚櫨鍒嗘瘮
    if len(df) > 1:  # 纭繚鏈夎冻澶熺殑鏁版嵁鐐?
        z = np.polyfit(df['num_missing_features'], df['Improvement_Percent'], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(df['num_missing_features'].min(), df['num_missing_features'].max(), 100)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2, label=f'Trend Line (slope={z[0]:.2f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_vs_missing_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 鏀硅繘鏂瑰悜缁熻楗煎浘
    fig, ax = plt.subplots(figsize=(8, 6))
    better_indicator = (df['Indicator_MAE'] < df['Reduced_MAE']).sum()
    better_reduced = (df['Indicator_MAE'] > df['Reduced_MAE']).sum()
    equal = (df['Indicator_MAE'] == df['Reduced_MAE']).sum()
    
    labels = ['Indicator Model Better', 'Reduced Model Better', 'Essentially Equal']
    sizes = [better_indicator, better_reduced, equal]
    colors = ['lightblue', 'lightcoral', 'lightgray']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_performance_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_single_feature_analysis(df, save_dir="visualizations"):
    """
    鍗曠壒寰佺己澶卞垎鏋愶紙瀛愬浘鍒嗗埆淇濆瓨锛?
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 鍙€冭檻鍗曠壒寰佺己澶辩殑鎯呭喌
    single_missing_df = df[df['num_missing_features'] == 1].copy()
    if len(single_missing_df) == 0:
        print("娌℃湁鍗曠壒寰佺己澶辩殑鏁版嵁")
        return
    
    single_missing_df['missing_feature'] = single_missing_df['missing_tuple'].apply(lambda x: x[0])
    
    # 1. 鍗曠壒寰佺己澶辩殑MAE瀵规瘮
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_data = []
    for _, row in single_missing_df.iterrows():
        comparison_data.append({
            'Feature': f'F{row["missing_feature"]}',
            'Model': 'Indicator',
            'MAE': row['Indicator_MAE']
        })
        comparison_data.append({
            'Feature': f'F{row["missing_feature"]}',
            'Model': 'Reduced',
            'MAE': row['Reduced_MAE']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    sns.boxplot(data=comparison_df, x='Feature', y='MAE', hue='Model', ax=ax)
    ax.set_title('MAE Comparison for Single Feature Missing', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/single_feature_mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 鍗曠壒寰佺己澶辩殑鏀硅繘鐧惧垎姣?
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(single_missing_df)), single_missing_df['Improvement_Percent'])
    ax.set_xlabel('Single Missing Feature Combinations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement by Single Missing Feature', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/single_feature_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 缂哄け鐗瑰緛鐨凪AE宸紓鍒嗗竷
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(single_missing_df['missing_feature'], single_missing_df['mae_diff'])
    ax.set_xlabel('Missing Feature Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE Difference (Indicator - Reduced)', fontsize=12, fontweight='bold')
    ax.set_title('MAE Difference by Missing Feature', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_difference_by_feature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 缂哄け鐗瑰緛鐨勯噸瑕佹€ф帓搴忥紙闆疯揪鍥撅級
    feature_impact = single_missing_df.groupby('missing_feature')['mae_diff'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2*np.pi, len(feature_impact), endpoint=False).tolist()
    angles += angles[:1]  # 闂悎
    
    values = feature_impact.values.tolist() + [feature_impact.values[0]]
    ax.plot(angles, values, 'o-', linewidth=2, label='MAE Difference (Indicator - Reduced)')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'F{idx}' for idx in feature_impact.index], fontsize=10)
    ax.set_title('Feature Importance in Single Feature Missing Scenarios', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 闆剁嚎
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_missing_pattern_heatmap(df, save_dir="visualizations"):
    """
    鏂板锛氱己澶辩粍鍚堢儹鍔涘浘
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 鍑嗗鐑姏鍥炬暟鎹?
    # 涓轰簡绠€鍖栵紝鎴戜滑鍙樉绀哄墠鍑犱釜缂哄け鐗瑰緛鐨勬暟閲忕骇瀵规瘮
    df_subset = df.head(100)  # 鍙栧墠100涓牱鏈潵鍑忓皯澶嶆潅鎬?
    df_subset['missing_feature_str'] = df_subset['missing_tuple'].apply(
        lambda x: f"F{x[0]}" if len(x) == 1 else f"Multi-{len(x)}"
    )
    
    # 鍒涘缓閫忚琛?
    pivot_data = df_subset.groupby(['missing_feature_str', 'num_missing_features']).agg({
        'Improvement_Percent': 'mean'
    }).unstack(fill_value=0)
    
    # 閲嶅鏁版嵁浠ヤ究缁樺埗鐑姏鍥?
    if len(pivot_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=True, fmt='.1f', 
                    cbar_kws={'label': 'Improvement Percentage (%)'}, ax=ax)
        ax.set_title('Joint Impact of Missing Feature Types and Quantity on Method Advantages', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Missing Features', fontsize=12)
        ax.set_ylabel('Missing Feature Type', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/missing_pattern_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_statistics(df):
    """
    鎵撳嵃缁熻鏁版嵁鎽樿
    """
    print("="*80)
    print("鎵归噺瀹為獙缁撴灉缁熻鎽樿")
    print("="*80)
    print(f"鎬诲疄楠屾暟閲? {len(df)}")
    print(f"缂哄け鐗瑰緛鏁伴噺鑼冨洿: {df['num_missing_features'].min()} - {df['num_missing_features'].max()}")
    
    # 缂哄け鎸囩ず鍣ㄦā鍨嬬粺璁?
    print(f"\nIndicator Model Performance:")
    print(f"  MAE: Mean={df['Indicator_MAE'].mean():.6f}, Std={df['Indicator_MAE'].std():.6f}")
    print(f"  RMSE: Mean={df['Indicator_RMSE'].mean():.6f}, Std={df['Indicator_RMSE'].std():.6f}")
    print(f"  R虏: Mean={df['Indicator_R2'].mean():.6f}, Std={df['Indicator_R2'].std():.6f}")
    
    # 缂╁噺妯″瀷缁熻
    print(f"\nReduced Model Performance:")
    print(f"  MAE: Mean={df['Reduced_MAE'].mean():.6f}, Std={df['Reduced_MAE'].std():.6f}")
    print(f"  RMSE: Mean={df['Reduced_RMSE'].mean():.6f}, Std={df['Reduced_RMSE'].std():.6f}")
    print(f"  R虏: Mean={df['Reduced_R2'].mean():.6f}, Std={df['Reduced_R2'].std():.6f}")
    
    # 鏀硅繘缁熻
    positive_improvement = (df['Improvement_Percent'] > 0).sum()
    negative_improvement = (df['Improvement_Percent'] < 0).sum()
    avg_improvement = df['Improvement_Percent'].mean()
    
    print(f"\nImprovement Statistics:")
    print(f"  Indicator Model Better: {positive_improvement} ({positive_improvement/len(df)*100:.1f}%)")
    print(f"  Reduced Model Better: {negative_improvement} ({negative_improvement/len(df)*100:.1f}%)")
    print(f"  Average Improvement: {avg_improvement:.2f}%")
    
    # 鎸夌己澶辩壒寰佹暟閲忓垎缁勭殑缁熻
    print(f"\nPerformance by Number of Missing Features:")
    grouped_stats = df.groupby('num_missing_features').agg({
        'Indicator_MAE': ['mean', 'std'],
        'Reduced_MAE': ['mean', 'std'],
        'Improvement_Percent': ['mean', 'std']
    }).round(6)
    
    print(grouped_stats)

def main():
    """
    涓诲嚱鏁帮細鍔犺浇鏁版嵁骞剁敓鎴愬彲瑙嗗寲
    """
    # 鍔犺浇鏁版嵁
    try:
        df = load_all_csv_files()
        print(f"\n鎴愬姛鍔犺浇鏁版嵁锛屾€昏褰曟暟: {len(df)}")
    except Exception as e:
        print(f"鏁版嵁鍔犺浇澶辫触: {e}")
        return
    
    # 鎵撳嵃缁熻鏁版嵁
    print_statistics(df)
    
    # 璁剧疆淇濆瓨鐩綍
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # 鐢熸垚鍙鍖栵紙姣忎釜瀛愬浘鍗曠嫭淇濆瓨锛?
    print(f"\n姝ｅ湪鐢熸垚鎬ц兘瀵规瘮鍥?..")
    plot_performance_comparison(df, save_dir)
    
    print(f"\n姝ｅ湪鐢熸垚璇︾粏瀵规瘮鍥?..")
    plot_detailed_comparison(df, save_dir)
    
    print(f"\n姝ｅ湪鐢熸垚鍗曠壒寰佸垎鏋愬浘...")
    plot_single_feature_analysis(df, save_dir)
    
    print(f"\n姝ｅ湪鐢熸垚缂哄け妯″紡鐑姏鍥?..")
    plot_missing_pattern_heatmap(df, save_dir)
    
    print(f"\n鏁版嵁鍔犺浇鍜屽彲瑙嗗寲瀹屾垚锛佹墍鏈夊浘琛ㄥ凡淇濆瓨鑷?'{save_dir}' 鐩綍")

if __name__ == "__main__":
    main()

