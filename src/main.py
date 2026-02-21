"""
Missing Data Battery SOH Prediction - Main Entry Point

使用 Hydra 进行配置管理，PyTorch Lightning 进行训练。

Usage:
    # 运行默认实验 (configs/config.yaml 中指定)
    python src/main.py
    
    # 指定实验配置
    python src/main.py experiment=mim_mar_0.3
    
    # 指定模型
    python src/main.py model=cnn1d
    
    # 指定数据集
    python src/main.py data=hust
    
    # 组合配置
    python src/main.py experiment=mim_mar_0.6 model=lstm data=xjtu
    
    # 覆盖参数
    python src/main.py training.epochs=100 training.batch_size=32
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trainers.neural_network_trainer import run_experiment
from src.utils.logger import setup_logger
from src.utils.wandb_utils import init_wandb, finish_wandb


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    主函数：根据 Hydra 配置运行实验
    
    Args:
        cfg: Hydra 自动解析的配置字典
    """
    # 设置日志
    logger = setup_logger(
        name="battery_soh",
        log_file=Path(cfg.logging.log_dir) / f"{cfg.experiment.name}.log",
        level=cfg.logging.level,
    )
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("Starting Battery SOH Prediction Experiment")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # 验证关键配置
    try:
        _validate_config(cfg)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # 初始化 WandB
    try:
        init_wandb(cfg)
        logger.info("Weights & Biases initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
    
    # 运行实验
    try:
        run_experiment(cfg)
        logger.info("Experiment completed successfully!")
    except Exception as e:
        logger.exception("Experiment failed with error:")
        raise
    finally:
        # 确保 WandB 会话正确关闭
        finish_wandb()


def _validate_config(cfg: DictConfig):
    """
    验证配置是否有效
    
    Raises:
        ValueError: 如果配置无效
    """
    # 检查必需的配置项
    required_keys = [
        "data.dataset",
        "model.name",
        "missing.mode",
        "experiment.name",
    ]
    
    for key in required_keys:
        parts = key.split(".")
        value = cfg
        for part in parts:
            if not hasattr(value, part):
                raise ValueError(f"Missing required config key: {key}")
            value = getattr(value, part)
    
    # 检查 missing mode 是否有效
    valid_modes = ["mcar", "mar"]
    if cfg.missing.mode not in valid_modes:
        raise ValueError(f"Invalid missing mode: {cfg.missing.mode}. Must be one of {valid_modes}")
    
    # 检查模型名称是否有效
    valid_models = ["mlp", "lstm", "gru", "cnn1d"]
    if cfg.model.name not in valid_models:
        raise ValueError(f"Invalid model name: {cfg.model.name}. Must be one of {valid_models}")
    
    # 检查缺失率范围
    if hasattr(cfg.experiment, 'missing_rate'):
        mr = cfg.experiment.missing_rate
        if not 0 <= mr <= 1:
            raise ValueError(f"Missing rate must be in [0, 1], got {mr}")
    
    # 检查训练参数
    if cfg.training.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {cfg.training.batch_size}")
    
    if cfg.training.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {cfg.training.epochs}")
    
    if cfg.training.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {cfg.training.learning_rate}")


if __name__ == "__main__":
    main()
