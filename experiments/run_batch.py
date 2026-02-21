#!/usr/bin/env python3
"""
批量实验运行脚本 V2
支持多次重复实验、自动聚合结果、生成统计图表
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.batch_experiment_runner_v2 import BatchExperimentRunnerV2


def main():
    """运行 2C 批次 10 次重复实验"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量实验 V2 - 支持多次重复实验和自动绘图'
    )
    parser.add_argument(
        '--batch', 
        type=str, 
        default='2C',
        choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
        help='数据批次 (默认: 2C)'
    )
    parser.add_argument(
        '--n-repeats', 
        type=int, 
        default=10,
        help='重复实验次数 (默认: 10)'
    )
    parser.add_argument(
        '--start-seed', 
        type=int, 
        default=42,
        help='起始随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='训练轮数 (默认: 50)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='从检查点恢复（如果之前中断）'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("批量实验 V2")
    print("=" * 70)
    print(f"批次: {args.batch}")
    print(f"重复次数: {args.n_repeats}")
    print(f"种子范围: {args.start_seed} ~ {args.start_seed + args.n_repeats - 1}")
    print(f"训练轮数: {args.epochs}")
    print(f"恢复模式: {args.resume}")
    print("=" * 70)
    print()
    
    # 创建运行器并运行
    runner = BatchExperimentRunnerV2(
        batch_name=args.batch,
        n_repeats=args.n_repeats,
        start_seed=args.start_seed,
        epochs=args.epochs,
        resume=args.resume
    )
    
    exp_dir = runner.run()
    
    print("\n" + "=" * 70)
    print("批量实验完成!")
    print("=" * 70)
    print(f"实验目录: {exp_dir}")
    print(f"\n输出文件:")
    print(f"  - aggregate/results_all.csv: 所有原始结果")
    print(f"  - aggregate/summary_statistics.csv: 统计摘要")
    print(f"  - aggregate/figures/: 统计图表（置信区间、改进率等）")
    print(f"  - seed_*/figures/: 各单实验图表")
    print("=" * 70)


if __name__ == '__main__':
    main()
