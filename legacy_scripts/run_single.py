#!/usr/bin/env python3
"""
运行单个小实验（指定随机种子）
用于调试或单独运行特定种子
"""
import argparse
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.single_experiment import SingleExperimentRunner, SingleExperimentConfig


def main():
    parser = argparse.ArgumentParser(description='运行单个小实验')
    parser.add_argument('--batch', type=str, default='3C', help='数据批次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("单个小实验")
    print("=" * 70)
    print(f"批次: {args.batch}")
    print(f"种子: {args.seed}")
    print(f"轮数: {args.epochs}")
    print("=" * 70)
    
    # 创建配置
    config = SingleExperimentConfig(
        batch_name=args.batch,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        device=args.device
    )
    
    # 确定输出目录
    if args.output:
        base_dir = Path(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = Path("experiments") / args.batch / timestamp
    
    # 运行实验
    runner = SingleExperimentRunner(config, base_dir=base_dir)
    results = runner.run()
    
    print("\n实验完成!")
    print(f"结果保存至: {runner.seed_dir}")
    print(f"共 {len(results)} 条评估记录")


if __name__ == '__main__':
    main()
