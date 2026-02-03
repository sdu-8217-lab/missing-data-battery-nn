#!/usr/bin/env python3
"""
运行大实验（批量小实验）
支持中断恢复
"""
import argparse
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.batch_experiment import BatchExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='运行大实验（批量小实验）')
    parser.add_argument('--batch', type=str, default='3C', help='数据批次')
    parser.add_argument('--n_repeats', type=int, default=100, help='重复次数（种子数）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("大实验（批量小实验）")
    print("=" * 70)
    print(f"批次: {args.batch}")
    print(f"重复次数: {args.n_repeats}")
    print(f"种子范围: 42 ~ {42 + args.n_repeats - 1}")
    print(f"训练轮数: {args.epochs}")
    print(f"恢复模式: {args.resume}")
    print("=" * 70)
    print()
    
    # 运行大实验
    runner = BatchExperimentRunner(
        batch_name=args.batch,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        device=args.device,
        resume=args.resume
    )
    
    experiment_dir = runner.run()
    
    print("\n" + "=" * 70)
    print("大实验完成!")
    print(f"实验目录: {experiment_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
