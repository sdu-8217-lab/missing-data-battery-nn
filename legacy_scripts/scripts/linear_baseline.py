"""线性基线：在零值插补（标准化后均值）和 MIM 指示器下评估 SOH 预测。

用于回答：
1. 该任务在缺失数据下是否仍可由线性模型解决？
2. MIM 指示器对线性模型是否有帮助？
3. 与神经网络相比，线性基线的天花板在哪里？
"""
import argparse
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.missing_patterns import create_missing_pattern
from src.config.experiment_config import ExperimentConfig


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def evaluate_linear(data, missing_rates, pattern_name, seed, use_mim=False):
    """对单个 seed 的划分评估线性模型。"""
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    pattern = create_missing_pattern(pattern_name)
    results = []

    for mr in missing_rates:
        # 训练集与测试集使用不同 seed 偏移，避免泄漏
        mask_train = pattern.generate(X_train_s, y_train, mr, seed=seed + 10000)
        mask_test = pattern.generate(X_test_s, y_test, mr, seed=seed + 20000)

        X_tr = np.where(mask_train, X_train_s, 0.0)
        X_te = np.where(mask_test, X_test_s, 0.0)

        if use_mim:
            ind_tr = (~mask_train).astype(np.float32)
            ind_te = (~mask_test).astype(np.float32)
            X_tr = np.concatenate([X_tr, ind_tr], axis=1)
            X_te = np.concatenate([X_te, ind_te], axis=1)

        model = LinearRegression().fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        mae, rmse, r2 = compute_metrics(y_test, y_pred)

        results.append({
            'model': 'Linear-MIM' if use_mim else 'Linear',
            'model_type': 'linear',
            'use_mim': use_mim,
            'strategy': 'mim' if use_mim else 'baseline',
            'missing_rate': mr,
            'seed': seed,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='线性基线缺失数据实验')
    parser.add_argument('--dataset', type=str, default='XJTU')
    parser.add_argument('--batch', type=str, default='3C')
    parser.add_argument('--data_dir', type=str, default='./data/XJTU data')
    parser.add_argument('--missing_pattern', type=str, default='bernoulli')
    parser.add_argument('--n_repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='./results/linear_baseline')
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_name=args.dataset,
        batch=args.batch,
        data_dir=args.data_dir,
        missing_pattern=args.missing_pattern,
        n_repeats=args.n_repeats,
        random_seed=args.seed,
    )

    out_dir = Path(args.results_dir) / f"{args.dataset}_{args.batch}_{args.missing_pattern}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    csv_path = out_dir / f"linear_results_{timestamp}.csv"

    loader = DatasetLoaderFactory.create_loader(config.data_dir, config.dataset_name, config.batch)
    all_results = []

    for i in range(config.n_repeats):
        seed = config.random_seed + i
        data = loader.prepare_data(
            config.feature_cols,
            config.target_col,
            config.test_size,
            config.val_size,
            seed,
        )
        all_results.extend(evaluate_linear(data, config.missing_rates, config.missing_pattern, seed, use_mim=False))
        all_results.extend(evaluate_linear(data, config.missing_rates, config.missing_pattern, seed, use_mim=True))
        print(f"Seed {seed} done")

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"结果保存至: {csv_path}")

    # 打印快速摘要
    summary = df.groupby(['model', 'missing_rate'])['mae'].agg(['mean', 'std']).reset_index()
    print('\n=== 线性基线 MAE (mean ± std) ===')
    print(summary.pivot(index='missing_rate', columns='model', values=['mean', 'std']).to_string())


if __name__ == '__main__':
    main()
