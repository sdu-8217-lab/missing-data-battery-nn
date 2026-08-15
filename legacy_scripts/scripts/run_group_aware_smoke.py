"""组感知 MIM smoke 实验

验证：利用电压/电流分组先验的缺失处理是否能提升结构化缺失下的 SOH 预测稳定性。

对比：
- LSTM-MIM-Uniform / MLP-MIM-Uniform
- LSTM-GroupMIM-Uniform / MLP-GroupMIM-Uniform
- LSTM-MultiMR-Uniform
- LSTM-FMG-Uniform
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
        results_dir=f'./results/group_aware_smoke_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        include_fmg=True,
        model_filter=[
            'LSTM-MIM-Uniform',
            'LSTM-GroupMIM-Uniform',
            'MLP-MIM-Uniform',
            'MLP-GroupMIM-Uniform',
            'LSTM-MultiMR-Uniform',
            'LSTM-FMG-Uniform',
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
    parser = argparse.ArgumentParser(description='Group-aware MIM smoke')
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
