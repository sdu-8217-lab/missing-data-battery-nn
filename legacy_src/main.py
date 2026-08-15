"""主入口脚本"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.experiment_config import ExperimentConfig
from src.data.dataset_registry import (
    get_default_data_dir,
    get_default_feature_cols,
    get_target_col,
    validate_feature_columns,
)
from src.experiments.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='SOH预测缺失数据处理实验')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='XJTU',
                       choices=['XJTU', 'HUST', 'MIT', 'TJU', 'NASA'],
                       help='数据集名称 (XJTU, HUST, MIT, TJU, NASA)')
    parser.add_argument('--batch', type=str, default='3C',
                       help='电池批次/子集。'
                            'XJTU: 2C,3C,R2.5,R3,RW,Sim_satellite; '
                            'HUST: 1-10; MIT: 2017-05-12,2017-06-30,2018-04-12; '
                            'TJU: Dataset_1_NCA_battery,Dataset_2_NCM_battery,Dataset_3_NCM_NCA_battery; '
                            'NASA: all')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录；默认根据数据集自动选择')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果保存目录')
    
    # 实验参数
    parser.add_argument('--n_repeats', type=int, default=100,
                       help='重复实验次数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--missing_pattern', type=str, default='bernoulli',
                       choices=['bernoulli', 'block', 'channel', 'group', 'state_dependent', 'mixed', 'road_course'],
                       help='缺失模式 (bernoulli, block, channel, group, state_dependent, mixed, road_course)')
    
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
                       help='要运行的模型或模型类型 (mlp, lstm, gru, cnn1d, '
                            'mlp-mim, mlp-multimr 等, all=全部)')
    parser.add_argument('--physics_mono', type=float, default=None,
                       help='PINN monotonicity 损失权重（覆盖配置默认值）')
    parser.add_argument('--physics_smooth', type=float, default=None,
                       help='PINN smoothness 损失权重（覆盖配置默认值）')
    
    args = parser.parse_args()
    
    # 未指定数据目录时使用注册表默认值
    data_dir = args.data_dir or get_default_data_dir(args.dataset)

    # 创建配置
    config = ExperimentConfig(
        dataset_name=args.dataset,
        batch=args.batch,
        data_dir=data_dir,
        results_dir=args.results_dir,
        n_repeats=args.n_repeats,
        random_seed=args.seed,
        missing_pattern=args.missing_pattern,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        model_filter=args.models,
        physics_mono=args.physics_mono,
        physics_smooth=args.physics_smooth
    )

    # 从注册表自动设置默认特征列与目标列，避免为每个数据集写 if/else
    config.feature_cols = get_default_feature_cols(args.dataset)
    config.target_col = get_target_col(args.dataset)

    # 校验列名是否存在，提前给出明确错误
    validate_feature_columns(
        data_dir=config.data_dir,
        dataset_name=args.dataset,
        batch=args.batch,
        feature_cols=config.feature_cols,
        target_col=config.target_col,
    )

    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()
    
    print(f"\n实验完成！结果保存在: {runner.exp_dir}")


if __name__ == '__main__':
    main()
