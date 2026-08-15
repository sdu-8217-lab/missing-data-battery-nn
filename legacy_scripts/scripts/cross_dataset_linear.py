"""跨数据集线性基线：在一个数据集上训练，在另一个数据集上测试。

快速评估跨数据集泛化能力。对缺失数据使用 mean imputation。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.dataset_registry import get_default_feature_cols, get_target_col
from src.data.missing_patterns import create_missing_pattern


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def load_dataset(dataset_name, batch):
    data_dir_map = {
        'XJTU': './data/XJTU data',
        'HUST': './data/HUST data',
        'MIT': './data/MIT data',
        'TJU': './data/TJU data',
        'NASA': './data/NASA data',
    }
    data_dir = data_dir_map[dataset_name]
    loader = DatasetLoaderFactory.create_loader(data_dir, dataset_name, batch)
    feature_cols = get_default_feature_cols(dataset_name)
    target_col = get_target_col(dataset_name)
    data = loader.prepare_data(feature_cols, target_col, random_seed=42)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def apply_missing(X, y, pattern_name, mr, seed):
    pattern = create_missing_pattern(pattern_name)
    mask = pattern.generate(X, y, mr, seed=seed)
    return np.where(mask, X, 0.0).astype(np.float32)


def main():
    datasets = {
        'XJTU': '3C',
        'HUST': '1',
        'MIT': '2017-05-12',
        'TJU': 'Dataset_1_NCA_battery',
        'NASA': 'all',
    }
    pattern_name = 'channel'
    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for train_ds, train_batch in datasets.items():
        print(f'Loading {train_ds}...')
        X_tr_full, y_tr_full, _, _ = load_dataset(train_ds, train_batch)
        scaler = StandardScaler().fit(X_tr_full)
        X_tr_s = scaler.transform(X_tr_full)

        for test_ds, test_batch in datasets.items():
            print(f'  Testing on {test_ds}...')
            _, _, X_te_full, y_te = load_dataset(test_ds, test_batch)
            X_te_s = scaler.transform(X_te_full)

            for mr in missing_rates:
                X_tr = apply_missing(X_tr_s, y_tr_full, pattern_name, mr, seed=42)
                X_te = apply_missing(X_te_s, y_te, pattern_name, mr, seed=43)

                model = LinearRegression().fit(X_tr, y_tr_full)
                y_pred = model.predict(X_te)
                mae, rmse, r2 = compute_metrics(y_te, y_pred)

                results.append({
                    'train_dataset': train_ds,
                    'test_dataset': test_ds,
                    'missing_rate': mr,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                })

    df = pd.DataFrame(results)
    out_dir = Path('./results/cross_dataset_linear')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'{pattern_name}.csv', index=False)
    print(f'\n结果保存至 {out_dir / f"{pattern_name}.csv"}')

    # 打印同数据集 vs 跨数据集平均 MAE
    same = df[df['train_dataset'] == df['test_dataset']]['mae'].mean()
    cross = df[df['train_dataset'] != df['test_dataset']]['mae'].mean()
    print(f'同数据集平均 MAE: {same:.4f}')
    print(f'跨数据集平均 MAE: {cross:.4f}')


if __name__ == '__main__':
    main()
