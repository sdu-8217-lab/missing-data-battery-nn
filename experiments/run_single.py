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

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """主函数"""
    print("=" * 70)
    print("单个小实验")
    print("=" * 70)
    print(f"批次: {cfg.data.get('batch_id', '3C')}")
    print(f"种子: {cfg.experiment.seeds[0] if hasattr(cfg.experiment, 'seeds') else 42}")
    print(f"轮数: {cfg.training.epochs}")
    print("=" * 70)
    
    # TODO: 实现单实验运行逻辑
    # 可以使用 src/experiments/runner.py 中的 ExperimentRunner
    print("\n提示: 此脚本需要进一步实现")
    print("建议使用: python src/main.py data=xjtu model=paper_mlp method=mim")


if __name__ == '__main__':
    main()
