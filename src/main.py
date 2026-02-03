"""主入口脚本"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='SOH预测缺失数据处理实验')
    
    # 基本参数
    parser.add_argument('--batch', type=str, default='3C',
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='电池批次')
    parser.add_argument('--data_dir', type=str, default='./data/XJTU data',
                       help='数据目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果保存目录')
    
    # 实验参数
    parser.add_argument('--n_repeats', type=int, default=100,
                       help='重复实验次数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    
    # 模型选择
    parser.add_argument('--models', type=str, nargs='+',
                       default=['all'],
                       help='要运行的模型 (mlp, lstm, gru, cnn1d, xgboost, all)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        batch=args.batch,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        n_repeats=args.n_repeats,
        random_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience
    )
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    print(f"\n实验完成！结果保存在: {runner.exp_dir}")


if __name__ == '__main__':
    main()
