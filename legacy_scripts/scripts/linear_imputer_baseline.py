"""线性基线 + 不同插补策略：mean / knn / iterative"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.missing_patterns import create_missing_pattern
from src.config.experiment_config import ExperimentConfig


IMPUTERS = {
    'mean': SimpleImputer(strategy='mean'),
    'knn': KNNImputer(n_neighbors=5),
    'iterative': IterativeImputer(max_iter=10, random_state=42),
}


def compute_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='XJTU')
    parser.add_argument('--batch', type=str, default='3C')
    parser.add_argument('--data_dir', type=str, default='./data/XJTU data')
    parser.add_argument('--missing_pattern', type=str, default='bernoulli')
    parser.add_argument('--imputer', type=str, default='knn', choices=list(IMPUTERS.keys()))
    parser.add_argument('--n_repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='./results/linear_imputer_baseline')
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_name=args.dataset,
        batch=args.batch,
        data_dir=args.data_dir,
        missing_pattern=args.missing_pattern,
        n_repeats=args.n_repeats,
        random_seed=args.seed,
    )

    out_dir = Path(args.results_dir) / f"{args.dataset}_{args.batch}_{args.missing_pattern}_{args.imputer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    csv_path = out_dir / f"results_{timestamp}.csv"

    loader = DatasetLoaderFactory.create_loader(config.data_dir, config.dataset_name, config.batch)
    all_results = []

    for i in range(config.n_repeats):
        seed = config.random_seed + i
        data = loader.prepare_data(config.feature_cols, config.target_col, config.test_size, config.val_size, seed)

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        imputer = IMPUTERS[args.imputer]
        pattern = create_missing_pattern(config.missing_pattern)

        for mr in config.missing_rates:
            mask_train = pattern.generate(X_train, y_train, mr, seed=seed + 10000)
            mask_test = pattern.generate(X_test, y_test, mr, seed=seed + 20000)

            X_tr = np.where(mask_train, X_train, np.nan).astype(np.float32)
            X_te = np.where(mask_test, X_test, np.nan).astype(np.float32)

            X_tr_imp = imputer.fit_transform(X_tr)
            X_te_imp = imputer.transform(X_te)

            model = LinearRegression().fit(X_tr_imp, y_train)
            y_pred = model.predict(X_te_imp)
            metrics = compute_metrics(y_test, y_pred)

            all_results.append({
                'model': f'Linear-{args.imputer.capitalize()}',
                'missing_rate': mr,
                'seed': seed,
                **metrics,
            })

        print(f"Seed {seed} done")

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"结果保存至: {csv_path}")

    summary = df.groupby('missing_rate')['mae'].agg(['mean', 'std']).reset_index()
    print('\n=== MAE (mean ± std) ===')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
