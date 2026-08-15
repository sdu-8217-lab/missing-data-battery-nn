"""可视化跨数据集线性基线结果"""
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


def main():
    csv_path = Path('./results/cross_dataset_linear/channel.csv')
    if not csv_path.exists():
        print(f'{csv_path} 不存在')
        return

    df = pd.read_csv(csv_path)

    # 1. 热力图：train vs test 平均 MAE
    pivot = df.groupby(['train_dataset', 'test_dataset'])['mae'].mean().reset_index()
    heatmap_data = pivot.pivot(index='train_dataset', columns='test_dataset', values='mae')

    out_dir = Path('./results/cross_dataset_linear')
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Cross-Dataset Linear Baseline: MAE (channel pattern, mean over MR)')
    plt.tight_layout()
    plt.savefig(out_dir / 'cross_dataset_mae_heatmap.png', dpi=150)
    plt.close()

    # 2. 每个 MR 下的同数据集 vs 跨数据集箱线图
    df['type'] = df.apply(lambda r: 'Same' if r['train_dataset'] == r['test_dataset'] else 'Cross', axis=1)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='missing_rate', y='mae', hue='type')
    plt.title('Same vs Cross-Dataset MAE Distribution')
    plt.tight_layout()
    plt.savefig(out_dir / 'same_vs_cross_boxplot.png', dpi=150)
    plt.close()

    # 3. 对角线（同数据集）结果
    same_df = df[df['train_dataset'] == df['test_dataset']]
    print('\n=== 同数据集 MAE ===')
    print(same_df.groupby('train_dataset')['mae'].mean().to_string())

    print('\n=== 跨数据集平均 MAE（按训练集）===')
    cross_df = df[df['train_dataset'] != df['test_dataset']]
    print(cross_df.groupby('train_dataset')['mae'].mean().to_string())

    print(f'\n图表保存至 {out_dir}')


if __name__ == '__main__':
    main()
