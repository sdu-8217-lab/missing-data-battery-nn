"""FMG 门控容量消融实验

基于 uniform_multi_rate 训练，比较不同 gate 容量 / 深度：
- LSTM-MultiMR-Uniform
- LSTM-FMG-Uniform (默认 hidden_dim=16, 2层)
- LSTM-FMG-Hid8 / Hid32 / Hid64
- LSTM-FMG-Linear (1层)
- LSTM-FMG-Deep3 (3层)

聚焦结构化缺失模式 block / group。
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
        results_dir=f'./results/gate_ablation_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        include_fmg=True,
        model_filter=[
            'LSTM-MultiMR-Uniform',
            'LSTM-FMG-Uniform',
            'LSTM-FMG-Hid8',
            'LSTM-FMG-Hid32',
            'LSTM-FMG-Hid64',
            'LSTM-FMG-Linear',
            'LSTM-FMG-Deep3',
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
    parser = argparse.ArgumentParser(description='FMG gate capacity ablation')
    parser.add_argument('--patterns', type=str, nargs='+',
                       default=['block', 'group'],
                       help='要测试的缺失模式')
    parser.add_argument('--n_repeats', type=int, default=5,
                       help='每个配置重复次数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    args = parser.parse_args()

    for pattern in args.patterns:
        run_pattern(pattern, args.n_repeats, args.epochs)
