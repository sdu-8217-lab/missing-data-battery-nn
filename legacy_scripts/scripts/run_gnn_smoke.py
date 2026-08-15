"""Feature Graph GAT smoke 实验

验证显式特征图关系建模（GAT）在结构化缺失下能否超过 uniform multi-rate 基线。
聚焦：
- MLP-GNN-Uniform
- LSTM-GNN-Uniform
- LSTM-MIM-Uniform（公平基线）
- LSTM-MultiMR-Uniform（公平基线）
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


def make_config(missing_pattern: str, n_repeats: int = 5, epochs: int = 100):
    config = ExperimentConfig(
        dataset_name='XJTU',
        batch='3C',
        data_dir='./data/XJTU data',
        results_dir=f'./results/gnn_smoke_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        model_filter=[
            'MLP-GNN-Uniform',
            'LSTM-GNN-Uniform',
            'LSTM-MIM-Uniform',
            'LSTM-MultiMR-Uniform',
        ]
    )
    return config


def run_pattern(missing_pattern: str, n_repeats: int = 5, epochs: int = 100):
    config = make_config(missing_pattern, n_repeats, epochs)
    runner = ExperimentRunner(config)
    runner.run_all_experiments()
    print(f"\n[{missing_pattern}] 实验完成，结果保存在: {runner.exp_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Feature Graph GAT smoke')
    parser.add_argument('--patterns', type=str, nargs='+',
                       default=['bernoulli', 'block', 'channel', 'group'],
                       help='要测试的缺失模式')
    parser.add_argument('--n_repeats', type=int, default=5,
                       help='每个配置重复次数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    args = parser.parse_args()

    for pattern in args.patterns:
        run_pattern(pattern, args.n_repeats, args.epochs)
