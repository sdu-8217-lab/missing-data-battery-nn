"""时间泛化实验：在每个电池内部按时间顺序划分训练/测试。

训练集：每个电池前 70% 循环
测试集：每个电池后 30% 循环

这测试模型是否真正预测未来退化，而非在电池间插值。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.dataset_registry import get_default_feature_cols, get_target_col
from src.data.missing_patterns import create_missing_pattern
from src.models.neural_network_models import MLPModel
from src.models.model_factory import ModelFactory
from src.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.evaluators.metrics import calculate_all_metrics


def load_all_cycles(dataset_name='XJTU', batch='3C'):
    """加载原始数据并保留每个电池的循环顺序。"""
    data_dir_map = {
        'XJTU': './data/XJTU data',
        'HUST': './data/HUST data',
        'MIT': './data/MIT data',
        'TJU': './data/TJU data',
        'NASA': './data/NASA data',
    }
    loader = DatasetLoaderFactory.create_loader(data_dir_map[dataset_name], dataset_name, batch)
    feature_cols = get_default_feature_cols(dataset_name)
    target_col = get_target_col(dataset_name)

    files = loader.get_battery_files()
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []

    for f in files:
        X, y = loader.load_battery(f, feature_cols, target_col)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        n = len(X)
        split_idx = int(n * 0.7)
        all_X_train.append(X[:split_idx])
        all_y_train.append(y[:split_idx])
        all_X_test.append(X[split_idx:])
        all_y_test.append(y[split_idx:])

    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_test = np.vstack(all_X_test)
    y_test = np.concatenate(all_y_test)
    return X_train, y_train, X_test, y_test


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp_mim(X_train_mim, y_train, X_val_mim, y_val, input_dim, device, epochs=100, lr=0.001, patience=15):
    model = ModelFactory.create_model(
        'mlp', input_dim=input_dim, use_mim=True,
        device=device, hidden_layers=[192, 96, 48, 24], dropout=0.15
    )
    train_loader = make_loader(X_train_mim, y_train, shuffle=True)
    val_loader = make_loader(X_val_mim, y_val, shuffle=False)
    trainer = NeuralNetworkTrainer(model.model, device=device)
    trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)
    return model


def evaluate(model, X_test, y_test, pattern_name, missing_rates, seed):
    results = []
    pattern = create_missing_pattern(pattern_name)
    for mr in missing_rates:
        mask = pattern.generate(X_test, y_test, mr, seed=seed)
        X_missing = np.where(mask, X_test, 0.0).astype(np.float32)
        missing_ind = (~mask).astype(np.float32)
        X_input = np.hstack([X_missing, missing_ind])
        y_pred = model.predict(torch.tensor(X_input, dtype=torch.float32))
        y_pred = y_pred.detach().cpu().numpy().reshape(-1)
        metrics = calculate_all_metrics(y_test, y_pred)
        results.append({
            'model': 'MLP-MIM-TimeSplit',
            'missing_rate': mr,
            'pattern': pattern_name,
            **metrics,
        })
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    X_train, y_train, X_test, y_test = load_all_cycles('XJTU', '3C')
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # 训练时混合缺失率
    pattern = create_missing_pattern('bernoulli')
    X_tr_list, y_tr_list = [], []
    for mr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = pattern.generate(X_train, y_train, mr, seed=42 + int(mr * 10))
        X_missing = np.where(mask, X_train, 0.0).astype(np.float32)
        missing_ind = (~mask).astype(np.float32)
        X_tr_list.append(np.hstack([X_missing, missing_ind]))
        y_tr_list.append(y_train)
    X_train_mim = np.vstack(X_tr_list)
    y_train_mim = np.concatenate(y_tr_list)

    # 简单验证集：从训练集末尾取 10%
    n_val = int(len(X_train_mim) * 0.1)
    X_val_mim = X_train_mim[-n_val:]
    y_val_mim = y_train_mim[-n_val:]
    X_train_mim = X_train_mim[:-n_val]
    y_train_mim = y_train_mim[:-n_val]

    model = train_mlp_mim(X_train_mim, y_train_mim, X_val_mim, y_val_mim,
                          X_train.shape[1], device, epochs=100, lr=0.001, patience=15)

    results = []
    for pattern_name in ['bernoulli', 'channel', 'road_course']:
        results.extend(evaluate(model, X_test, y_test, pattern_name,
                                [0.1, 0.3, 0.5, 0.7, 0.9], seed=999))

    df = pd.DataFrame(results)
    out_dir = Path('./results/time_split')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'mlp_mim_results.csv', index=False)
    print('\n=== Time Split MAE ===')
    print(df.pivot(index='missing_rate', columns='pattern', values='mae').to_string())


if __name__ == '__main__':
    main()
