#!/usr/bin/env python3
"""
模型训练脚本
负责分界线以上的训练阶段（L1-L6）

用法:
    python train_models.py --config configs/experiments/phase1_mlp_mim.yaml
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig, load_model_config
from src.data.loader import XJTUDatasetLoader, prepare_datasets_for_training
from src.models.factory import create_model, count_parameters, get_model_summary
from src.trainers.trainer import Trainer
from src.utils.logger import setup_logger


def set_seed(seed: int):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_single_model(
    config: ExperimentConfig,
    model_arch_config: dict,
    seed: int,
    output_dir: Path,
    logger
) -> dict:
    """训练单个模型"""
    
    # 断点续跑：检查模型是否已存在
    model_type = model_arch_config['model_type']
    use_mim = config.training.use_mim
    model_filename = f"model_{config.data.batch}_{model_type}_mim{use_mim}_seed{seed}.pt"
    
    # 检查任何已存在的模型目录
    existing_models = list(output_dir.rglob(f"*{model_filename}"))
    if existing_models:
        logger.info(f"[{seed}] 模型已存在，跳过: {existing_models[0]}")
        return {
            'seed': seed,
            'model_path': str(existing_models[0]),
            'status': 'skipped',
            'message': 'Model already exists'
        }
    
    set_seed(seed)
    
    # 加载数据
    logger.info(f"[{seed}] 加载数据 - 批次: {config.data.batch}")
    data_loader = XJTUDatasetLoader(
        data_dir=config.data.data_dir,
        batch=config.data.batch
    )
    
    data = data_loader.prepare_data(
        feature_cols=config.data.feature_cols,
        target_col=config.data.target_col,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_seed=seed
    )
    
    # 准备数据集
    use_mim = config.training.use_mim
    model_type = model_arch_config['model_type']
    
    train_dataset, val_datasets = prepare_datasets_for_training(
        data=data,
        use_mim=use_mim,
        model_type=model_type,
        training_config={
            'seq_len': config.training.seq_len,
            'seed': seed,
            'training_missing_rates': config.training.training_missing_rates,
            'val_mr_subset': config.training.validation.val_mr_subset,
            'imputation_method': config.training.imputation_method
        }
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loaders = {
        name: DataLoader(ds, batch_size=config.training.batch_size, num_workers=4)
        for name, ds in val_datasets.items()
    }
    
    # 创建模型
    device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从 model_arch_config 中提取参数，避免重复传递 model_type
    model_kwargs = {k: v for k, v in model_arch_config.items() if k not in ['model_type', 'name', 'level', 'target_params_no_mim', 'target_params_mim']}
    
    model = create_model(
        model_type=model_type,
        input_dim=len(config.data.feature_cols),
        use_mim=use_mim,
        device=device,
        **model_kwargs
    )
    
    param_count = count_parameters(model)
    logger.info(f"[{seed}] 模型: {model_type}, MIM={use_mim}, 参数={param_count:,}")
    
    # 训练
    trainer = Trainer(model, device=device)
    
    logger.info(f"[{seed}] 开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loaders=val_loaders,
        epochs=config.training.epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        patience=config.training.early_stopping_patience,
        min_epochs=config.training.min_epochs,
        use_scheduler=config.training.use_scheduler,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_factor=config.training.scheduler_factor,
        base_metric=config.training.validation.base_metric,
        aggregation=config.training.validation.aggregation,
        verbose=False
    )
    
    logger.info(f"[{seed}] 训练完成 - best_epoch={history['best_epoch']}, "
                f"best_val_loss={history['best_val_loss']:.6f}")
    
    # 保存模型
    model_filename = f"model_{config.data.batch}_{model_type}_mim{use_mim}_seed{seed}.pt"
    model_path = output_dir / "models" / model_filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'model_arch': model_arch_config,
        'history': history,
        'seed': seed,
        'param_count': param_count
    }, model_path)
    
    logger.info(f"[{seed}] 模型已保存: {model_path}")
    
    return {
        'seed': seed,
        'model_path': str(model_path),
        'best_epoch': history['best_epoch'],
        'best_val_loss': history['best_val_loss'],
        'training_time': history['training_time'],
        'param_count': param_count
    }


import fcntl

def acquire_lock(lock_file):
    """获取文件锁，确保单进程训练"""
    try:
        fd = open(lock_file, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except IOError:
        return None

def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--config", required=True, help="实验配置文件")
    parser.add_argument("--model-config", help="模型架构配置文件（覆盖实验配置）")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--seeds", type=int, nargs="+", help="随机种子列表")
    parser.add_argument("--imputations", type=str, nargs="+", 
                       default=None,
                       help="多插补方法训练（如: zero mean knn iterative）")
    
    args = parser.parse_args()
    
    # 单进程锁
    lock_file = '/tmp/train_models.lock'
    lock_fd = acquire_lock(lock_file)
    if lock_fd is None:
        print("错误: 另一个训练实例正在运行，请等待完成")
        return 1
    
    # 加载配置
    config = ExperimentConfig.from_yaml(args.config)
    if args.model_config:
        config.model_config_path = args.model_config
    if args.output_dir:
        config.output_dir = args.output_dir
    
    model_arch_config = load_model_config(config.model_config_path)
    
    # 确定种子
    seeds = args.seeds if args.seeds else list(range(config.seed, config.seed + config.n_repeats))
    
    # 确定插补方法
    imputations = args.imputations if args.imputations else [config.training.imputation_method]
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.output_dir) / config.name / config.data.batch / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="train",
        log_file=str(output_dir / "train.log"),
        level="INFO"
    )
    
    logger.info("="*60)
    logger.info("模型训练阶段 (分界线以上)")
    logger.info("="*60)
    logger.info(f"实验: {config.name}")
    logger.info(f"批次: {config.data.batch}")
    logger.info(f"模型: {model_arch_config['model_type']}")
    logger.info(f"MIM: {config.training.use_mim}")
    logger.info(f"种子数: {len(seeds)}")
    logger.info(f"插补方法: {imputations}")
    logger.info("="*60)
    
    # 保存配置
    config.to_yaml(str(output_dir / "config.yaml"))
    
    # 训练所有模型（支持多插补方法）
    all_results = []
    total = len(seeds) * len(imputations)
    current = 0
    
    for seed in seeds:
        for imp_method in imputations:
            current += 1
            logger.info(f"\n[{current}/{total}] 训练: seed={seed}, imputation={imp_method}")
            
            # 临时修改配置中的插补方法
            original_imp = config.training.imputation_method
            config.training.imputation_method = imp_method
            
            try:
                result = train_single_model(config, model_arch_config, seed, output_dir, logger)
                result['imputation_method'] = imp_method
                all_results.append(result)
            except Exception as e:
                logger.error(f"[{seed}/{imp_method}] 训练失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                # 恢复原始插补方法
                config.training.imputation_method = original_imp
    
    # 保存训练结果
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "train_results.csv", index=False)
    
    logger.info("="*60)
    logger.info(f"训练阶段完成 - 成功: {len(all_results)}/{total}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
