"""跨数据集大乱斗（Cross-Dataset Battle Royale）

在 NASA / HUST / MIT / TJU 上验证 XJTU 3C 的结论是否泛化。
覆盖 6 种缺失模式 × 8 端到端方法 (含 GraphMIM) × n=30 seeds。

设计考虑：
- 每个数据集选一个代表性 batch（避免组合爆炸）
- 每个数据集运行 6 种缺失模式全套
- 复用 ExperimentRunner，与 XJTU 3C 保持完全一致的训练/评估流程
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner
from src.data.dataset_registry import (
    get_default_feature_cols, get_target_col, validate_feature_columns,
)


METHODS = [
    # 端到端 8 方法（与 battle royale 保持一致）
    'MLP-MIM-Uniform',
    'LSTM-MIM-Uniform',
    'LSTM-MultiMR-Uniform',
    'LSTM-FMG-Uniform',
    'MLP-GroupMIM-Uniform',
    'LSTM-GroupMIM-Uniform',
    'MLP-GNN-Uniform',
    'LSTM-GNN-Uniform',
    # GraphMIM 2 方法
    'MLP-GraphMIM-Uniform',
    'LSTM-GraphMIM-Uniform',
]

# 每个数据集的代表性 batch
DATASETS = [
    ('NASA', 'all', './data/NASA data'),
    ('HUST', '1', './data/HUST data'),
    ('MIT', '2017-05-12', './data/MIT data'),
    ('TJU', 'Dataset_1_NCA_battery', './data/TJU data'),
]

PATTERNS = ['bernoulli', 'block', 'channel', 'group', 'mixed', 'road_course']


def make_config(dataset: str, batch: str, data_dir: str,
                missing_pattern: str, n_repeats: int, epochs: int):
    tag = f"{dataset}_{batch.replace('/', '_').replace(' ', '_')}"
    config = ExperimentConfig(
        dataset_name=dataset,
        batch=batch,
        data_dir=data_dir,
        results_dir=f'./results/cross_dataset/{tag}/{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        include_fmg=True,
        model_filter=METHODS,
    )
    # 关键：按数据集覆盖默认特征列/目标列
    config.feature_cols = get_default_feature_cols(dataset)
    config.target_col = get_target_col(dataset)
    return config


def run_one(dataset, batch, data_dir, missing_pattern, n_repeats, epochs):
    config = make_config(dataset, batch, data_dir, missing_pattern, n_repeats, epochs)
    runner = ExperimentRunner(config)
    runner.run_all_experiments()
    print(f"\n[{dataset}/{batch}/{missing_pattern}] done -> {runner.exp_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=[d[0] for d in DATASETS],
                        help='要跑的数据集列表')
    parser.add_argument('--patterns', type=str, nargs='+', default=PATTERNS)
    parser.add_argument('--n_repeats', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    for dataset, batch, data_dir in DATASETS:
        if dataset not in args.datasets:
            continue
        for pattern in args.patterns:
            print(f"\n===== [{dataset}/{batch}] pattern={pattern} =====")
            try:
                run_one(dataset, batch, data_dir, pattern,
                        args.n_repeats, args.epochs)
            except Exception as e:
                print(f"[ERROR] {dataset}/{batch}/{pattern}: {e}")
                import traceback
                traceback.print_exc()
