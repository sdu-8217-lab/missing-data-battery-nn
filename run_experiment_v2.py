#!/usr/bin/env python3
"""
使用新版代码（Pydantic + Loguru + Lightning）运行实验
"""
import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config.pydantic_config import ExperimentConfig as ExperimentConfigV2, ModelConfig as ModelConfigV2
from src.utils.logger_v2 import setup_logger
from loguru import logger


def run_single_experiment_v2(batch_name: str = "3C", seed: int = 42):
    """运行单个小实验（使用新版代码）"""
    # 设置日志
    logger = setup_logger(
        name="single_experiment",
        log_file=f"./experiments_v2/{batch_name}/seed_{seed}/experiment.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("启动单个小实验 (使用新版代码)")
    logger.info("=" * 60)
    
    # 使用Pydantic配置
    try:
        config = ExperimentConfigV2(
            name=f"soh_exp_{batch_name}",
            batch=batch_name,
            seed=seed,
            n_repeats=1,  # 单次实验
            epochs=50,    # 减少轮数以便快速测试
            early_stopping_patience=10
        )
        
        logger.info(f"实验配置:")
        logger.info(f"  名称: {config.name}")
        logger.info(f"  批次: {config.batch}")
        logger.info(f"  随机种子: {config.random_seed}")
        logger.info(f"  训练轮数: {config.epochs}")
        logger.info(f"  缺失率: {config.missing_rates}")
        
        # 获取模型配置
        model_configs = config.get_model_configs()
        logger.info(f"模型配置数量: {len(model_configs)}")
        for mc in model_configs:
            logger.info(f"  - {mc.name} ({mc.model_type}, MIM={mc.use_mim})")
        
        # TODO: 使用Lightning训练器运行实验
        # 这里可以替换为实际的新版训练代码
        
        logger.info("[OK] 单个小实验配置验证通过!")
        logger.info(f"实验目录: ./experiments_v2/{batch_name}/seed_{seed}/")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_batch_experiment_v2(batch_name: str = "3C", n_repeats: int = 10):
    """运行大实验（使用新版代码）"""
    # 设置日志
    logger = setup_logger(
        name="batch_experiment",
        log_file=f"./experiments_v2/{batch_name}/batch_experiment.log",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info(f"启动大实验 (使用新版代码, n_repeats={n_repeats})")
    logger.info("=" * 60)
    
    try:
        config = ExperimentConfigV2(
            name=f"soh_exp_{batch_name}_batch",
            batch=batch_name,
            n_repeats=n_repeats,
            epochs=50,
            early_stopping_patience=10
        )
        
        logger.info(f"实验配置:")
        logger.info(f"  名称: {config.name}")
        logger.info(f"  批次: {config.batch}")
        logger.info(f"  重复次数: {config.n_repeats}")
        logger.info(f"  随机种子范围: {config.random_seed} ~ {config.random_seed + n_repeats - 1}")
        
        # 显示种子列表
        seeds = list(range(config.random_seed, config.random_seed + n_repeats))
        logger.info(f"种子列表: {seeds}")
        
        # TODO: 循环运行多个种子实验
        for i, seed in enumerate(seeds):
            logger.info(f"\n[{i+1}/{n_repeats}] 运行种子 {seed}...")
            # 这里可以调用单实验函数
            
        logger.info("[OK] 大实验配置验证通过!")
        logger.info(f"实验目录: ./experiments_v2/{batch_name}/")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description="使用新版代码运行实验")
    parser.add_argument(
        "mode",
        choices=["single", "batch"],
        help="运行模式: single=单实验, batch=大实验"
    )
    parser.add_argument(
        "--batch-name",
        default="3C",
        choices=["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"],
        help="数据批次"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="单实验的随机种子"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="大实验的重复次数"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single_experiment_v2(args.batch_name, args.seed)
    else:
        run_batch_experiment_v2(args.batch_name, args.n_repeats)


if __name__ == "__main__":
    main()
