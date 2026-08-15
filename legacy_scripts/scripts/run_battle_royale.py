"""电池 SOH 缺失处理大乱斗（Battle Royale）

在完全公平的 uniform multi-rate 训练下，把当前所有候选方法拉到同一起跑线，
在多种缺失模式（含“超强模式” road_course）上决出胜者。

对比方法：
- MLP-MIM-Uniform
- LSTM-MIM-Uniform
- LSTM-MultiMR-Uniform
- LSTM-FMG-Uniform
- MLP-GroupMIM-Uniform
- LSTM-GroupMIM-Uniform
- MLP-GNN-Uniform
- LSTM-GNN-Uniform

缺失模式：
- bernoulli / block / channel / group / mixed / road_course
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


METHODS = [
    'MLP-MIM-Uniform',
    'LSTM-MIM-Uniform',
    'LSTM-MultiMR-Uniform',
    'LSTM-FMG-Uniform',
    'MLP-GroupMIM-Uniform',
    'LSTM-GroupMIM-Uniform',
    'MLP-GNN-Uniform',
    'LSTM-GNN-Uniform',
]

PATTERNS = ['bernoulli', 'block', 'channel', 'group', 'mixed', 'road_course']


def make_config(missing_pattern: str, n_repeats: int = 5, epochs: int = 100):
    return ExperimentConfig(
        dataset_name='XJTU',
        batch='3C',
        data_dir='./data/XJTU data',
        results_dir=f'./results/battle_royale_{missing_pattern}',
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


def run_pattern(missing_pattern: str, n_repeats: int = 5, epochs: int = 100):
    config = make_config(missing_pattern, n_repeats, epochs)
    runner = ExperimentRunner(config)
    runner.run_all_experiments()
    print(f"\n[{missing_pattern}] 大乱斗完成，结果保存在: {runner.exp_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SOH Missing-Data Battle Royale')
    parser.add_argument('--patterns', type=str, nargs='+', default=PATTERNS,
                       help='要测试的缺失模式')
    parser.add_argument('--n_repeats', type=int, default=5,
                       help='每个配置重复次数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    args = parser.parse_args()

    for pattern in args.patterns:
        run_pattern(pattern, args.n_repeats, args.epochs)
