"""汇总并对比所有当前实验结果

整合来源：
- battle_royale_*：端到端方法（MIM/MultiMR/GroupMIM/GNN/FMG/GraphMIM）
- two_stage_n5：两阶段插补 + MLP
- linear_baseline_n30：线性基线

输出：
- 每个缺失模式下的 MAE/RMSE/R² 汇总表
- 跨方法排名
- 可视化图片
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_battle_royale(results_root='./results'):
    """加载所有 battle_royale 结果。"""
    records = []
    root = Path(results_root)
    for pattern_dir in root.glob('battle_royale_*'):
        if not pattern_dir.is_dir():
            continue
        pattern = pattern_dir.name.replace('battle_royale_', '')
        for subdir in pattern_dir.iterdir():
            if not subdir.is_dir():
                continue
            csv_files = list(subdir.glob('results/*.csv'))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['pattern'] = pattern
                df['source'] = 'end2end'
                records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def load_graphmim(results_root='./results'):
    """加载 GraphMIM n=5 / n=30 结果；n=30 优先覆盖同 pattern。"""
    records = []
    root = Path(results_root)
    for prefix in ['graphmim_n30_', 'graphmim_n5_']:
        for pattern_dir in root.glob(f'{prefix}*'):
            if not pattern_dir.is_dir():
                continue
            pattern = pattern_dir.name.replace(prefix, '')
            for subdir in pattern_dir.iterdir():
                if not subdir.is_dir():
                    continue
                csv_files = list(subdir.glob('results/*.csv'))
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df['pattern'] = pattern
                    df['source'] = 'end2end'
                    df['_graphmim_scale'] = prefix
                    records.append(df)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    # 同一 pattern 若同时存在 n=30 与 n=5，保留 n=30
    has_n30 = set(df[df['_graphmim_scale'] == 'graphmim_n30_']['pattern'].unique())
    df = df[~((df['_graphmim_scale'] == 'graphmim_n5_') & (df['pattern'].isin(has_n30)))].copy()
    df = df.drop(columns=['_graphmim_scale'])
    return df


def load_two_stage(results_root='./results'):
    """加载两阶段基线结果。"""
    records = []
    root = Path(results_root) / 'two_stage_n5'
    if not root.exists():
        return pd.DataFrame()
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        csv_file = subdir / 'results.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df = df.rename(columns={'MAE': 'mae', 'RMSE': 'rmse', 'R2': 'r2', 'missing_pattern': 'pattern'})
            df['model'] = 'TwoStage-' + df['imputer'].str.capitalize()
            df['source'] = 'two_stage'
            records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def load_linear_baseline(results_root='./results'):
    """加载线性基线结果（含 mean 和 knn/iterative 变体）。"""
    records = []
    for root_name in ['linear_baseline_n30', 'linear_imputer_baseline']:
        root = Path(results_root) / root_name
        if not root.exists():
            continue
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            for csv_file in subdir.glob('*.csv'):
                df = pd.read_csv(csv_file)
                # 目录名格式：XJTU_3C_road_course 或 XJTU_3C_road_course_knn
                parts = subdir.name.split('_')
                if len(parts) > 3 and parts[-1] in ['knn', 'iterative']:
                    pattern = '_'.join(parts[2:-1])
                else:
                    pattern = '_'.join(parts[2:]) if len(parts) > 2 else parts[-1]
                df['pattern'] = pattern
                df['source'] = 'linear'
                records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def summarize(df, metric='mae'):
    """按 pattern 和 model 汇总中位数与标准差。"""
    summary = df.groupby(['pattern', 'model'])[metric].agg(['median', 'mean', 'std', 'count']).reset_index()
    return summary


def rank_methods(df, metric='mae', higher_is_better=False):
    """每个 (pattern, missing_rate) 下给方法排名。"""
    asc = not higher_is_better
    df = df.copy()
    df['rank'] = df.groupby(['pattern', 'missing_rate'])[metric].rank(ascending=asc, method='min')
    rank_summary = df.groupby(['pattern', 'model'])['rank'].mean().reset_index()
    return rank_summary


def plot_curves(df, output_dir, metric='mae'):
    """绘制每个 pattern 下各方法随缺失率变化的曲线。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = df['pattern'].unique()
    for pattern in patterns:
        plt.figure(figsize=(10, 6))
        pdf = df[df['pattern'] == pattern]
        summary = pdf.groupby(['missing_rate', 'model'])[metric].median().reset_index()
        for model in summary['model'].unique():
            mdf = summary[summary['model'] == model].sort_values('missing_rate')
            plt.plot(mdf['missing_rate'], mdf[metric], marker='o', label=model, alpha=0.8)
        plt.xlabel('Missing Rate')
        plt.ylabel(metric.upper())
        plt.title(f'{pattern}: {metric.upper()} vs Missing Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{pattern}_{metric}_curves.png', dpi=150)
        plt.close()


def plot_rank_heatmap(rank_summary, output_dir):
    """绘制方法排名热图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = rank_summary.pivot(index='model', columns='pattern', values='rank')
    plt.figure(figsize=(8, max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r')
    plt.title('Average Rank per Pattern (lower is better)')
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_heatmap.png', dpi=150)
    plt.close()


def statistical_test(df, method_a, method_b, metric='mae'):
    """对两个方法做配对 Wilcoxon 检验。"""
    results = []
    for pattern in df['pattern'].unique():
        for mr in df['missing_rate'].unique():
            a = df[(df['pattern'] == pattern) & (df['missing_rate'] == mr) & (df['model'] == method_a)][metric]
            b = df[(df['pattern'] == pattern) & (df['missing_rate'] == mr) & (df['model'] == method_b)][metric]
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                stat, p = stats.wilcoxon(a.values, b.values)
                results.append({
                    'pattern': pattern,
                    'missing_rate': mr,
                    'method_a': method_a,
                    'method_b': method_b,
                    'median_a': a.median(),
                    'median_b': b.median(),
                    'pvalue': p,
                })
            except ValueError:
                continue
    return pd.DataFrame(results)


def main():
    results_root = './results'
    output_dir = Path(results_root) / 'comparison_summary'
    output_dir.mkdir(parents=True, exist_ok=True)

    df_end2end = load_battle_royale(results_root)
    df_graphmim = load_graphmim(results_root)
    df_two_stage = load_two_stage(results_root)
    df_linear = load_linear_baseline(results_root)

    all_records = []
    if not df_end2end.empty:
        all_records.append(df_end2end[['pattern', 'missing_rate', 'model', 'mae', 'rmse', 'r2', 'seed']])
    if not df_graphmim.empty:
        all_records.append(df_graphmim[['pattern', 'missing_rate', 'model', 'mae', 'rmse', 'r2', 'seed']])
    if not df_two_stage.empty:
        all_records.append(df_two_stage[['pattern', 'missing_rate', 'model', 'mae', 'rmse', 'r2', 'seed']])
    if not df_linear.empty:
        all_records.append(df_linear[['pattern', 'missing_rate', 'model', 'mae', 'rmse', 'r2', 'seed']])

    if not all_records:
        print('未找到任何结果。')
        return

    df_all = pd.concat(all_records, ignore_index=True)

    # 保存长表
    df_all.to_csv(output_dir / 'all_results_long.csv', index=False)

    # 汇总
    summary = summarize(df_all, metric='mae')
    summary.to_csv(output_dir / 'summary_mae.csv', index=False)

    # 排名
    rank_summary = rank_methods(df_all, metric='mae')
    rank_summary.to_csv(output_dir / 'rank_summary.csv', index=False)

    # 绘图
    plot_curves(df_all, output_dir, metric='mae')
    plot_rank_heatmap(rank_summary, output_dir)

    # 打印快速摘要
    print('\n=== 每个 pattern 下各方法 MAE 中位数 ===')
    pivot = summary.pivot(index='model', columns='pattern', values='median')
    print(pivot.to_string())

    print('\n=== 平均排名（越低越好）===')
    print(rank_summary.sort_values(['pattern', 'rank']).to_string(index=False))

    # 统计检验：Top 方法两两对比
    models = set(df_all['model'].unique())
    top_candidates = [
        'MLP-GraphMIM-Uniform', 'MLP-GNN-Uniform', 'LSTM-GNN-Uniform',
        'LSTM-MIM-Uniform', 'LSTM-GroupMIM-Uniform', 'MLP-MIM-Uniform',
        'MLP-GroupMIM-Uniform', 'TwoStage-Mean'
    ]
    top_methods = [m for m in top_candidates if m in models]
    print('\n=== Wilcoxon 配对检验 (top 方法) ===')
    test_records = []
    for i, ma in enumerate(top_methods):
        for mb in top_methods[i+1:]:
            test_df = statistical_test(df_all, ma, mb)
            if not test_df.empty:
                # 汇总：按 pattern 报告显著性比例
                sig = (test_df['pvalue'] < 0.05).mean()
                better_a = (test_df['median_a'] < test_df['median_b']).mean()
                for pattern in test_df['pattern'].unique():
                    pt = test_df[test_df['pattern'] == pattern]
                    test_records.append({
                        'pattern': pattern,
                        'method_a': ma,
                        'method_b': mb,
                        'n_comparisons': len(pt),
                        'sig_ratio': (pt['pvalue'] < 0.05).mean(),
                        'a_better_ratio': (pt['median_a'] < pt['median_b']).mean(),
                        'median_a': pt['median_a'].median(),
                        'median_b': pt['median_b'].median(),
                    })
    if test_records:
        test_summary = pd.DataFrame(test_records)
        test_summary.to_csv(output_dir / 'wilcoxon_pairwise_summary.csv', index=False)
        print(test_summary.to_string(index=False))
    else:
        print('样本不足，未进行统计检验。')

    print(f'\n汇总结果保存至: {output_dir}')


if __name__ == '__main__':
    main()
