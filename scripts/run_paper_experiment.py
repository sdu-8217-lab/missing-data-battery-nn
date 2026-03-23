#!/usr/bin/env python3
"""
Paper 实验运行脚本 - 简化版

使用方法:
    # MLP Baseline, 单个 seed, 单个缺失率 (快速测试)
    python scripts/run_paper_experiment.py --model paper_mlp --method baseline --seed 42 --mr 0.5
    
    # MLP MIM, 单个 seed, 单个缺失率 (快速测试)
    python scripts/run_paper_experiment.py --model paper_mlp --method mim --seed 42 --mr 0.5
    
    # MLP MIM, 多个 seeds
    python scripts/run_paper_experiment.py --model paper_mlp --method mim --seeds 42 43 44

参数说明:
    --model: paper_mlp, paper_lstm, paper_gru, paper_cnn1d
    --method: baseline, mim
    --seed/--seeds: 随机种子
    --mr: 缺失率 (0.1-0.9)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(model: str, method: str, seeds: list[int], missing_rates: list[float], 
                   dry_run: bool = False):
    """运行实验"""
    
    # 构建命令 - 直接使用基础配置，通过命令行覆盖参数
    cmd = [
        "python", "src/main.py",
        f"model={model}",
        f"method={method}",
        f"experiment.seeds={seeds}",
        f"missing.missing_rates_eval={missing_rates}",
    ]
    
    cmd_str = " ".join(cmd)
    print("=" * 70)
    print("实验命令:")
    print(cmd_str)
    print("=" * 70)
    
    if dry_run:
        print("\n[干运行模式] 不实际执行")
        return 0
    
    # 执行
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="运行 Paper 实验")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["paper_mlp", "paper_lstm", "paper_gru", "paper_cnn1d"],
                       help="模型类型 (使用 paper_ 前缀的配置)")
    parser.add_argument("--method", type=str, required=True,
                       choices=["baseline", "mim"],
                       help="方法类型")
    
    # 种子参数 (互斥)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="单个随机种子")
    seed_group.add_argument("--seeds", type=int, nargs="+", help="多个随机种子")
    
    # 缺失率参数
    parser.add_argument("--mr", type=float, nargs="+", 
                       default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                       help="缺失率列表 (默认: 0.0-0.9 共10个缺失率)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="只打印命令，不实际执行")
    
    args = parser.parse_args()
    
    # 处理种子
    seeds = [args.seed] if args.seed else args.seeds
    
    # 运行
    return run_experiment(
        model=args.model,
        method=args.method,
        seeds=seeds,
        missing_rates=args.mr,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
