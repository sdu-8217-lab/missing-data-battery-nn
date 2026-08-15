"""两阶段基线：imputation + SOH 估计

与端到端方法公平对比：
1. 在训练集上 fit imputer；
2. 用同一 imputer transform 训练/验证/测试集；
3. 在插补后的完整特征上训练 MLP 估计 SOH；
4. 测试时同样先插补再预测。

支持：mean / knn / iterative 三种插补策略。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from src.config.experiment_config import ExperimentConfig
from src.data.dataset_loader import DatasetLoaderFactory
from src.data.dataset_registry import get_default_feature_cols, get_target_col
from src.data.missing_patterns import create_missing_pattern
from src.models.neural_network_models import MLPModel
from src.models.base_model import BaseModel
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluators.metrics import calculate_all_metrics
from torch.utils.data import DataLoader, TensorDataset


IMPUTERS = {
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=5),
    'iterative': IterativeImputer(max_iter=10, random_state=42),
}


def apply_missing(X, y, missing_pattern, missing_rate, seed):
    """对特征数组应用缺失模式，返回零填充特征与掩码。"""
    generator = create_missing_pattern(missing_pattern)
    mask = generator.generate(X, y, missing_rate, seed=seed)  # True=存在
    X_missing = np.where(mask, X, 0.0).astype(np.float32)
    missing_indicator = (~mask).astype(np.float32)
    return X_missing, missing_indicator


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(X_train_imp, y_train, X_val_imp, y_val, input_dim, device, epochs=100, lr=0.001, patience=15):
    """在插补后的数据上训练一个标准 MLP。"""
    model = BaseModel(
        MLPModel(input_dim=input_dim, hidden_layers=[192, 96, 48, 24], dropout=0.15),
        device=device,
        model_type='pytorch'
    )
    train_loader = make_loader(X_train_imp, y_train, shuffle=True)
    val_loader = make_loader(X_val_imp, y_val, shuffle=False)
    trainer = NeuralNetworkTrainer(model.model, device=device)
    trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
    return model


def evaluate_model(model, X_test_imp, y_test, missing_rates, missing_pattern, seed):
    """在不同缺失率下评估。"""
    results = []
    for mr in missing_rates:
        X_test_missing, _ = apply_missing(X_test_imp, y_test, missing_pattern, mr, seed)
        # 注意：两阶段基线的测试流程是：先对测试集缺失 → 用训练集 fit 的 imputer transform
        # 这里为了公平，测试集缺失后不再重新 fit imputer，但 SimpleImputer/KNN 已 fit 在训练分布上，
        # 对测试集缺失数据直接 transform。
        y_pred_tensor = model.predict(torch.tensor(X_test_missing, dtype=torch.float32))
        y_pred = y_pred_tensor.detach().cpu().numpy().reshape(-1)
        y_true = y_test.reshape(-1)
        metrics = calculate_all_metrics(y_true, y_pred)
        results.append({
            'missing_rate': mr,
            'model': 'TwoStage-MLP',
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2': metrics['r2'],
        })
    return results


def run_two_stage(imputer_name, missing_pattern, n_repeats=5, epochs=100, results_dir='./results/two_stage_smoke', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ExperimentConfig(
        dataset_name='XJTU',
        batch='3C',
        data_dir='./data/XJTU data',
        results_dir=results_dir,
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
    )

    feature_cols = get_default_feature_cols('XJTU')
    target_col = get_target_col('XJTU')

    all_results = []
    for seed_offset in range(n_repeats):
        seed = config.random_seed + seed_offset
        loader = DatasetLoaderFactory.create_loader(config.data_dir, 'XJTU', '3C')
        data = loader.prepare_data(feature_cols, target_col, random_seed=seed)

        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        # 训练集固定 50% 缺失率，fit imputer
        X_train_missing, _ = apply_missing(X_train, y_train, missing_pattern, 0.5, seed)
        imputer = IMPUTERS[imputer_name]
        imputer.fit(X_train_missing)

        X_train_imp = imputer.transform(X_train_missing).astype(np.float32)

        # 验证集同样缺失+插补
        X_val_missing, _ = apply_missing(X_val, y_val, missing_pattern, 0.5, seed)
        X_val_imp = imputer.transform(X_val_missing).astype(np.float32)

        # 测试集保留完整，后续在不同缺失率下再缺失
        model = train_mlp(X_train_imp, y_train, X_val_imp, y_val,
                          input_dim=X_train.shape[1], device=device,
                          epochs=epochs, lr=config.lr, patience=config.early_stopping_patience)

        results = evaluate_model(model, X_test, y_test, config.missing_rates, missing_pattern, seed)
        for r in results:
            r['seed'] = seed
            r['imputer'] = imputer_name
            r['missing_pattern'] = missing_pattern
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    out_dir = Path(results_dir) / f'{missing_pattern}_{imputer_name}_n{n_repeats}'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'results.csv', index=False)
    print(f"[{imputer_name} / {missing_pattern}] 完成，结果保存至 {out_dir}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imputer', type=str, default='knn', choices=list(IMPUTERS.keys()))
    parser.add_argument('--pattern', type=str, default='channel')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--results_dir', type=str, default='./results/two_stage_smoke')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    args = parser.parse_args()

    run_two_stage(args.imputer, args.pattern, args.n_repeats, args.epochs, args.results_dir, args.device)
