"""新基线模型 smoke 实验

验证 4 个新基线模型能否在训练流程中正常工作：
- iTransformer-MIM-Uniform
- SAITS-MIM-Uniform
- NeuralCDE-MIM-Uniform
- GRIN-MIM-Uniform

与现有最强基线 LSTM-MIM-Uniform、MLP-GraphMIM-Uniform 做快速对比。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


def make_config(missing_pattern: str, n_repeats: int = 3, epochs: int = 50):
    config = ExperimentConfig(
        dataset_name='XJTU',
        batch='3C',
        data_dir='./data/XJTU data',
        results_dir=f'./results/new_baselines_smoke_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=10,
        include_fmg=False,
        model_filter=[
            'LSTM-MIM-Uniform',
            'MLP-GraphMIM-Uniform',
            'iTransformer-MIM-Uniform',
            'SAITS-MIM-Uniform',
            'NeuralCDE-MIM-Uniform',
            'GRIN-MIM-Uniform',
        ]
    )
    return config


def run_pattern(missing_pattern: str, n_repeats: int = 3, epochs: int = 50):
    config = make_config(missing_pattern, n_repeats, epochs)
    runner = ExperimentRunner(config)
    runner.run_all_experiments()
    print(f"\n[{missing_pattern}] 实验完成，结果保存在: {runner.exp_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='New baselines smoke test')
    parser.add_argument('--patterns', type=str, nargs='+',
                       default=['channel'],
                       help='要测试的缺失模式')
    parser.add_argument('--n_repeats', type=int, default=3,
                       help='每个配置重复次数')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    args = parser.parse_args()

    for pattern in args.patterns:
        run_pattern(pattern, args.n_repeats, args.epochs)
