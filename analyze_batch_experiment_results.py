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
    从目录中加载所有CSV文件
    """
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'missing_combinations' in file:
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError(f"在目录 {base_dir} 中未找到CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # 添加实验ID信息
            exp_folder = Path(file_path).parent.name
            df['experiment_id'] = exp_folder
            all_data.append(df)
            print(f"加载: {Path(file_path).name} - 形状: {df.shape}")
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    if not all_data:
        raise ValueError("没有成功加载任何CSV文件")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 解析缺失组合字符串为元组
    combined_df['missing_tuple'] = combined_df['Missing_Combination'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple()
    )
    
    # 添加缺失特征数量
    combined_df['num_missing_features'] = combined_df['missing_tuple'].apply(len)
    
    # 计算性能差异
    combined_df['mae_diff'] = combined_df['Indicator_MAE'] - combined_df['Reduced_MAE']
    combined_df['rmse_diff'] = combined_df['Indicator_RMSE'] - combined_df['Reduced_RMSE']
    combined_df['r2_diff'] = combined_df['Indicator_R2'] - combined_df['Reduced_R2']
    
    # 添加性能分类
    combined_df['indicator_better'] = combined_df['mae_diff'] < 0  # 缺失指示器方法更好
    combined_df['reduced_better'] = combined_df['mae_diff'] > 0    # 缩减模型更好
    combined_df['similar'] = combined_df['mae_diff'] == 0          # 性能相似
    
    return combined_df

def plot_performance_comparison(df, save_dir="visualizations"):
    """
    绘制性能对比图（子图分别保存）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. MAE对比散点图
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
    
    # 2. RMSE对比散点图
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
    
    # 3. R²对比散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['Indicator_R2'], df['Reduced_R2'], alpha=0.6, s=30, color='green')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.8, label='Perfect Agreement')
    ax.set_xlabel('Indicator Model R²', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reduced Model R²', fontsize=12, fontweight='bold')
    ax.set_title('R² Comparison Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/r2_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 性能差异分布（MAE）
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
    
    # 5. 改进百分比分布
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
    
    # 6. 按缺失特征数量分组的箱线图
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
    绘制详细的性能对比图（子图分别保存）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 所有指标的箱线图对比
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
    
    # 2. 按缺失特征数量的性能对比（双轴图）
    fig, ax1 = plt.subplots(figsize=(10, 6))
    grouped = df.groupby('num_missing_features').agg({
        'Indicator_MAE': ['mean', 'std'],
        'Reduced_MAE': ['mean', 'std'],
        'Improvement_Percent': ['mean', 'std']
    }).round(6)
    
    # 展开多级索引
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    
    # 左Y轴：MAE均值±标准差（带误差棒）
    ax1.errorbar(grouped.index, grouped['Indicator_MAE_mean'], 
                 yerr=grouped['Indicator_MAE_std'], fmt='o-', color='#1f77b4', 
                 label='Indicator MAE', capsize=3, linewidth=2)
    ax1.errorbar(grouped.index, grouped['Reduced_MAE_mean'], 
                 yerr=grouped['Reduced_MAE_std'], fmt='s--', color='#ff7f0e', 
                 label='Reduced MAE', capsize=3, linewidth=2)
    
    # 右Y轴：改进百分比均值（带显著性标记）
    ax2 = ax1.twinx()
    bars = ax2.bar(grouped.index, grouped['Improvement_Percent_mean'], 
                   yerr=grouped['Improvement_Percent_std'], alpha=0.3, color='gray', width=0.6, 
                   label='Avg Improvement %', capsize=3)
    
    # 显著性标记
    for x in grouped.index:
        if abs(grouped.loc[x, 'Improvement_Percent_mean']) > 5:  # 阈值可调
            try:
                subset_df = df[df['num_missing_features'] == x]
                if len(subset_df) > 1:  # 需要至少2个样本才能计算t检验
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
                pass  # 如果计算失败则跳过标注
    
    ax1.set_xlabel('Number of Missing Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold', color='gray')
    ax1.set_title('Missing Degree Impact: Indicator Method Advantage Increases with More Missing Features', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/missing_degree_impact_double_axis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 改进百分比与缺失特征数量的关系（修正趋势线）
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['num_missing_features'], df['Improvement_Percent'], alpha=0.6)
    ax.set_xlabel('Number of Missing Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement Percentage vs Number of Missing Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 修正趋势线：x为缺失特征数量，y为改进百分比
    if len(df) > 1:  # 确保有足够的数据点
        z = np.polyfit(df['num_missing_features'], df['Improvement_Percent'], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(df['num_missing_features'].min(), df['num_missing_features'].max(), 100)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2, label=f'Trend Line (slope={z[0]:.2f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_vs_missing_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 改进方向统计饼图
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
    单特征缺失分析（子图分别保存）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 只考虑单特征缺失的情况
    single_missing_df = df[df['num_missing_features'] == 1].copy()
    if len(single_missing_df) == 0:
        print("没有单特征缺失的数据")
        return
    
    single_missing_df['missing_feature'] = single_missing_df['missing_tuple'].apply(lambda x: x[0])
    
    # 1. 单特征缺失的MAE对比
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
    
    # 2. 单特征缺失的改进百分比
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(single_missing_df)), single_missing_df['Improvement_Percent'])
    ax.set_xlabel('Single Missing Feature Combinations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement by Single Missing Feature', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/single_feature_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 缺失特征的MAE差异分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(single_missing_df['missing_feature'], single_missing_df['mae_diff'])
    ax.set_xlabel('Missing Feature Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE Difference (Indicator - Reduced)', fontsize=12, fontweight='bold')
    ax.set_title('MAE Difference by Missing Feature', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/mae_difference_by_feature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 缺失特征的重要性排序（雷达图）
    feature_impact = single_missing_df.groupby('missing_feature')['mae_diff'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2*np.pi, len(feature_impact), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    values = feature_impact.values.tolist() + [feature_impact.values[0]]
    ax.plot(angles, values, 'o-', linewidth=2, label='MAE Difference (Indicator - Reduced)')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'F{idx}' for idx in feature_impact.index], fontsize=10)
    ax.set_title('Feature Importance in Single Feature Missing Scenarios', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)  # 零线
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_missing_pattern_heatmap(df, save_dir="visualizations"):
    """
    新增：缺失组合热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备热力图数据
    # 为了简化，我们只显示前几个缺失特征的数量级对比
    df_subset = df.head(100)  # 取前100个样本来减少复杂性
    df_subset['missing_feature_str'] = df_subset['missing_tuple'].apply(
        lambda x: f"F{x[0]}" if len(x) == 1 else f"Multi-{len(x)}"
    )
    
    # 创建透视表
    pivot_data = df_subset.groupby(['missing_feature_str', 'num_missing_features']).agg({
        'Improvement_Percent': 'mean'
    }).unstack(fill_value=0)
    
    # 重塑数据以便绘制热力图
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
    打印统计数据摘要
    """
    print("="*80)
    print("批量实验结果统计摘要")
    print("="*80)
    print(f"总实验数量: {len(df)}")
    print(f"缺失特征数量范围: {df['num_missing_features'].min()} - {df['num_missing_features'].max()}")
    
    # 缺失指示器模型统计
    print(f"\nIndicator Model Performance:")
    print(f"  MAE: Mean={df['Indicator_MAE'].mean():.6f}, Std={df['Indicator_MAE'].std():.6f}")
    print(f"  RMSE: Mean={df['Indicator_RMSE'].mean():.6f}, Std={df['Indicator_RMSE'].std():.6f}")
    print(f"  R²: Mean={df['Indicator_R2'].mean():.6f}, Std={df['Indicator_R2'].std():.6f}")
    
    # 缩减模型统计
    print(f"\nReduced Model Performance:")
    print(f"  MAE: Mean={df['Reduced_MAE'].mean():.6f}, Std={df['Reduced_MAE'].std():.6f}")
    print(f"  RMSE: Mean={df['Reduced_RMSE'].mean():.6f}, Std={df['Reduced_RMSE'].std():.6f}")
    print(f"  R²: Mean={df['Reduced_R2'].mean():.6f}, Std={df['Reduced_R2'].std():.6f}")
    
    # 改进统计
    positive_improvement = (df['Improvement_Percent'] > 0).sum()
    negative_improvement = (df['Improvement_Percent'] < 0).sum()
    avg_improvement = df['Improvement_Percent'].mean()
    
    print(f"\nImprovement Statistics:")
    print(f"  Indicator Model Better: {positive_improvement} ({positive_improvement/len(df)*100:.1f}%)")
    print(f"  Reduced Model Better: {negative_improvement} ({negative_improvement/len(df)*100:.1f}%)")
    print(f"  Average Improvement: {avg_improvement:.2f}%")
    
    # 按缺失特征数量分组的统计
    print(f"\nPerformance by Number of Missing Features:")
    grouped_stats = df.groupby('num_missing_features').agg({
        'Indicator_MAE': ['mean', 'std'],
        'Reduced_MAE': ['mean', 'std'],
        'Improvement_Percent': ['mean', 'std']
    }).round(6)
    
    print(grouped_stats)

def main():
    """
    主函数：加载数据并生成可视化
    """
    # 加载数据
    try:
        df = load_all_csv_files()
        print(f"\n成功加载数据，总记录数: {len(df)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 打印统计数据
    print_statistics(df)
    
    # 设置保存目录
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成可视化（每个子图单独保存）
    print(f"\n正在生成性能对比图...")
    plot_performance_comparison(df, save_dir)
    
    print(f"\n正在生成详细对比图...")
    plot_detailed_comparison(df, save_dir)
    
    print(f"\n正在生成单特征分析图...")
    plot_single_feature_analysis(df, save_dir)
    
    print(f"\n正在生成缺失模式热力图...")
    plot_missing_pattern_heatmap(df, save_dir)
    
    print(f"\n数据加载和可视化完成！所有图表已保存至 '{save_dir}' 目录")

if __name__ == "__main__":
    main()
