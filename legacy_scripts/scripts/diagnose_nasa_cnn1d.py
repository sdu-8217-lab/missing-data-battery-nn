#!/usr/bin/env python3
"""诊断 NASA 数据集上 CNN1D 基线失败的原因。"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoaderFactory
from src.data.datasets import SequenceDataset, BatteryDataset, MIMDataset
from src.models.model_factory import ModelFactory
from src.evaluators.model_evaluator import ModelEvaluator
from torch.utils.data import DataLoader


def train_and_eval(model_type: str, use_mim: bool, n_epochs: int = 30, seed: int = 42):
    """训练并返回测试集预测。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DatasetLoaderFactory.create_loader('data/NASA data', 'NASA', 'all')
    data = loader.prepare_data(feature_cols=[], target_col='capacity', random_seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extra = {}
    if model_type in ('lstm', 'gru'):
        extra = dict(hidden_size=48 if model_type == 'lstm' else 64, num_layers=2, dropout=0.2, seq_len=5)
    elif model_type in ('cnn1d', 'cnn'):
        extra = dict(channels=[72, 32], kernel_size=4, dropout=0.1, seq_len=5)

    model = ModelFactory.create_model(
        model_type,
        input_dim=16,
        use_mim=use_mim,
        device=device,
        **extra
    )

    # 训练数据
    if model_type in ('lstm', 'gru', 'cnn1d', 'cnn'):
        train_ds = SequenceDataset(
            data['X_train'], data['y_train'], seq_len=5,
            missing_rate=0.0, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )
        val_ds = SequenceDataset(
            data['X_val'], data['y_val'], seq_len=5,
            missing_rate=0.0, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )
    else:
        train_ds = BatteryDataset(
            data['X_train'], data['y_train'],
            missing_rate=0.0, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )
        val_ds = BatteryDataset(
            data['X_val'], data['y_val'],
            missing_rate=0.0, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    hist = model.fit(train_loader, val_loader, epochs=n_epochs, lr=0.001, patience=15)

    # 在 MR=0.5 测试
    if model_type in ('lstm', 'gru', 'cnn1d', 'cnn'):
        test_ds = SequenceDataset(
            data['X_test'], data['y_test'], seq_len=5,
            missing_rate=0.5, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )
    else:
        test_ds = BatteryDataset(
            data['X_test'], data['y_test'],
            missing_rate=0.5, use_mim=use_mim, seed=seed, missing_pattern='bernoulli'
        )
    test_loader = DataLoader(test_ds, batch_size=32)
    evaluator = ModelEvaluator(device)
    metrics, preds, targets = evaluator.evaluate(model, test_loader)

    return metrics, np.array(preds), np.array(targets), hist


def main():
    out_dir = Path('results/diagnose_nasa_cnn1d')
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ('cnn1d', False, 'CNN1D-Baseline'),
        ('cnn1d', True, 'CNN1D-MIM'),
        ('mlp', False, 'MLP-Baseline'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (mtype, mim, label) in enumerate(cases):
        print(f"\n=== {label} ===")
        metrics, preds, targets, hist = train_and_eval(mtype, mim, n_epochs=30)
        print(f"Test MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
        print(f"Pred range: [{preds.min():.4f}, {preds.max():.4f}], Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"Pred mean={preds.mean():.4f}, std={preds.std():.4f}")

        ax = axes[idx]
        ax.scatter(targets, preds, alpha=0.3, s=10)
        ax.plot([0, 1.2], [0, 1.2], 'r--')
        ax.set_xlabel('True SOH')
        ax.set_ylabel('Predicted SOH')
        ax.set_title(f'{label}\nMAE={metrics["mae"]:.4f}, R2={metrics["r2"]:.3f}')
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(out_dir / 'pred_vs_true.png', dpi=150)
    print(f"\n诊断图已保存到 {out_dir / 'pred_vs_true.png'}")


if __name__ == '__main__':
    main()
