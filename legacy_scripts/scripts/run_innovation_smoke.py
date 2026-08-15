"""创新点快速对比实验

对比：
- MLP-MIM / LSTM-MIM（已有基线）
- MLP-FMG / LSTM-FMG（特征级缺失门控）
- MLP-FMG-CMR / LSTM-FMG-CMR（FMG + 课程学习）

在多种缺失模式（bernoulli / block / channel / group）下跑小规模重复实验，
用于快速判断哪个创新点更有价值。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig, ModelConfig
from src.experiments.experiment_runner import ExperimentRunner


def make_config(missing_pattern: str, n_repeats: int = 5, epochs: int = 100):
    """创建一个专门用于创新点对比的实验配置"""
    config = ExperimentConfig(
        dataset_name='XJTU',
        batch='3C',
        data_dir='./data/XJTU data',
        results_dir=f'./results/innovation_smoke_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        include_fmg=True,
        model_filter=[
            # 核心基线
            'MLP-MIM',
            'LSTM-MIM',
            # Reviewer 关心的最强基线
            'MLP-MultiMR',
            'LSTM-MultiMR',
            # 我们的方法
            'MLP-FMG',
            'LSTM-FMG',
            'MLP-FMG-CMR',
            'LSTM-FMG-CMR',
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
    parser = argparse.ArgumentParser(description='创新点快速对比实验')
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
