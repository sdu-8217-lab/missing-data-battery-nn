"""Mask-Only 基线实验

验证核心问题：缺失掩码本身是否携带 SOH 信息？

对比：
- LSTM-MaskOnly-Uniform：只看缺失掩码，不看特征值
- MLP-MaskOnly-Uniform：同上
- LSTM-MIM-Uniform：特征 + mask 拼接（标准基线）
- LSTM-MissingAware-Uniform：特征分支 + 独立缺失分支
- LSTM-MultiMR-Uniform：无 mask 基线

测试模式：
- bernoulli：MCAR，mask-only 应接近随机
- state_dependent：MNAR，mask-only 应能预测 SOH
- group：结构化缺失
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
        results_dir=f'./results/mask_only_smoke_{missing_pattern}',
        missing_pattern=missing_pattern,
        n_repeats=n_repeats,
        random_seed=42,
        epochs=epochs,
        batch_size=32,
        lr=0.001,
        early_stopping_patience=15,
        include_fmg=True,
        model_filter=[
            'LSTM-MaskOnly-Uniform',
            'MLP-MaskOnly-Uniform',
            'LSTM-MIM-Uniform',
            'LSTM-MissingAware-Uniform',
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
    parser = argparse.ArgumentParser(description='Mask-only baseline')
    parser.add_argument('--patterns', type=str, nargs='+',
                       default=['bernoulli', 'state_dependent', 'group'],
                       help='要测试的缺失模式')
    parser.add_argument('--n_repeats', type=int, default=5,
                       help='每个配置重复次数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    args = parser.parse_args()

    for pattern in args.patterns:
        run_pattern(pattern, args.n_repeats, args.epochs)
