#!/usr/bin/env python3
"""
大实验V2 - 整合改进版本
使用: PyTorch Lightning + Hydra + Loguru + configs_v3最佳参数
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.trainers.lightning_module import SOHLightningModule
from src.trainers.lightning_trainer import LightningTrainer
from src.utils.logger import setup_logger

# 数据加载
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset, MIMDataset, SequenceMIMDataset

# 模型
from src.models.model_factory import ModelFactory


def get_model_configs(cfg: DictConfig) -> List[Dict]:
    """获取所有模型配置 - configs_v3最佳参数 (4模型 x 2策略 = 8配置)"""
    base_configs = [
        {
            'name': 'MLP',
            'model_type': 'mlp',
            'hidden_layers': [192, 96, 48, 24],  # configs_v3最佳
            'dropout': 0.15,
            'use_mim': False,
        },
        {
            'name': 'MLP-MIM',
            'model_type': 'mlp',
            'hidden_layers': [192, 96, 48, 24],
            'dropout': 0.15,
            'use_mim': True,
        },
        {
            'name': 'LSTM',
            'model_type': 'lstm',
            'hidden_size': 48,  # configs_v3最佳
            'num_layers': 2,
            'dropout': 0.2,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': False,
        },
        {
            'name': 'LSTM-MIM',
            'model_type': 'lstm',
            'hidden_size': 48,
            'num_layers': 2,
            'dropout': 0.2,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': True,
        },
        {
            'name': 'GRU',
            'model_type': 'gru',
            'hidden_size': 64,  # configs_v3最佳
            'num_layers': 2,
            'dropout': 0.2,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': False,
        },
        {
            'name': 'GRU-MIM',
            'model_type': 'gru',
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': True,
        },
        {
            'name': 'CNN1D',
            'model_type': 'cnn1d',
            'channels': [72, 32],  # configs_v3最佳
            'kernel_size': 4,
            'dropout': 0.1,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': False,
        },
        {
            'name': 'CNN1D-MIM',
            'model_type': 'cnn1d',
            'channels': [72, 32],
            'kernel_size': 4,
            'dropout': 0.1,
            'seq_len': cfg.data.get('seq_len', 5),
            'use_mim': True,
        },
    ]
    return base_configs


def prepare_data(cfg: DictConfig, seed: int):
    """准备数据"""
    loader = XJTUDatasetLoader(
        data_dir=cfg.data.data_dir,
        batch=cfg.batch
    )
    
    data = loader.prepare_data(
        feature_cols=cfg.data.features,
        target_col=cfg.data.target,
        test_size=cfg.data.split.test_size,
        val_size=cfg.data.split.val_size,
        random_seed=seed
    )
    
    return data


def run_single_experiment(
    cfg: DictConfig,
    model_config: Dict,
    data: dict,
    seed: int,
    logger
) -> List[Dict]:
    """运行单个模型实验"""
    results = []
    model_name = model_config['name']
    use_mim = model_config['use_mim']
    
    logger.info(f"Training {model_name} (MIM={use_mim})")
    
    # 创建模型
    model_kwargs = {
        'hidden_layers': model_config.get('hidden_layers'),
        'dropout': model_config.get('dropout'),
        'hidden_size': model_config.get('hidden_size'),
        'num_layers': model_config.get('num_layers'),
        'channels': model_config.get('channels'),
        'kernel_size': model_config.get('kernel_size'),
    }
    # 过滤None值
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    
    model = ModelFactory.create_model(
        model_config['model_type'],
        input_dim=16,
        use_mim=use_mim,
        device='cpu',
        **model_kwargs
    )
    
    param_count = ModelFactory.count_parameters(model)
    logger.info(f"  Parameters: {param_count:,}")
    
    # 准备训练数据
    if use_mim:
        train_dataset, val_dataset = prepare_mim_data(cfg, model_config, data, seed)
    else:
        train_dataset, val_dataset = prepare_baseline_data(cfg, model_config, data, seed)
    
    # 使用PyTorch Lightning训练
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=0
    )
    
    # 包装为Lightning模块
    lightning_model = SOHLightningModule(
        model.model if hasattr(model, 'model') else model,
        learning_rate=cfg.training.learning_rate
    )
    
    # 训练
    trainer = LightningTrainer(
        max_epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        device='cpu',
        enable_progress_bar=False
    )
    
    start_time = time.time()
    try:
        history = trainer.train(lightning_model, train_loader, val_loader)
        training_time = time.time() - start_time
        logger.info(f"  Training completed in {training_time:.1f}s")
    except Exception as e:
        logger.error(f"  Training failed: {e}")
        return []
    
    # 评估所有缺失率
    for missing_rate in cfg.missing.missing_rates:
        metrics = evaluate_model(
            model, model_config, data, missing_rate, 
            cfg.training.batch_size, seed
        )
        metrics.update({
            'seed': seed,
            'model_name': model_name,
            'model_type': model_config['model_type'],
            'use_mim': use_mim,
            'missing_rate': missing_rate,
            'training_time': training_time,
            'params': param_count,
        })
        results.append(metrics)
        logger.info(f"  MR={missing_rate:.1f}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
    
    return results


def prepare_baseline_data(cfg: DictConfig, model_config: Dict, 
                          data: dict, seed: int):
    """准备Baseline训练数据"""
    model_type = model_config['model_type']
    seq_len = model_config.get('seq_len', 5)
    
    if model_type in ['lstm', 'gru', 'cnn1d']:
        train_dataset = SequenceDataset(
            data['X_train'], data['y_train'],
            seq_len=seq_len,
            missing_rate=0.0,
            use_mim=False,
            seed=seed
        )
        val_dataset = SequenceDataset(
            data['X_val'], data['y_val'],
            seq_len=seq_len,
            missing_rate=0.0,
            use_mim=False,
            seed=seed
        )
    else:
        train_dataset = BatteryDataset(
            data['X_train'], data['y_train'],
            missing_rate=0.0,
            use_mim=False,
            seed=seed
        )
        val_dataset = BatteryDataset(
            data['X_val'], data['y_val'],
            missing_rate=0.0,
            use_mim=False,
            seed=seed
        )
    
    return train_dataset, val_dataset


def prepare_mim_data(cfg: DictConfig, model_config: Dict,
                     data: dict, seed: int):
    """准备MIM训练数据"""
    model_type = model_config['model_type']
    seq_len = model_config.get('seq_len', 5)
    
    if model_type in ['lstm', 'gru', 'cnn1d']:
        train_dataset = SequenceMIMDataset(
            data['X_train'], data['y_train'],
            seq_len=seq_len,
            missing_rates=cfg.missing.training_missing_rates,
            use_mim=True,
            base_seed=seed
        )
        # 验证集使用50%缺失率
        val_dataset = SequenceDataset(
            data['X_val'], data['y_val'],
            seq_len=seq_len,
            missing_rate=0.5,
            use_mim=True,
            seed=seed
        )
    else:
        train_dataset = MIMDataset(
            data['X_train'], data['y_train'],
            missing_rates=cfg.missing.training_missing_rates,
            use_mim=True,
            base_seed=seed
        )
        val_dataset = BatteryDataset(
            data['X_val'], data['y_val'],
            missing_rate=0.5,
            use_mim=True,
            seed=seed
        )
    
    return train_dataset, val_dataset


def evaluate_model(model, model_config: Dict, data: dict,
                   missing_rate: float, batch_size: int, seed: int) -> Dict:
    """评估模型"""
    from torch.utils.data import DataLoader
    from src.evaluation.model_evaluator import ModelEvaluator
    
    model_type = model_config['model_type']
    use_mim = model_config['use_mim']
    seq_len = model_config.get('seq_len', 5)
    
    # 准备测试数据
    if model_type in ['lstm', 'gru', 'cnn1d']:
        test_dataset = SequenceDataset(
            data['X_test'], data['y_test'],
            seq_len=seq_len,
            missing_rate=missing_rate,
            use_mim=use_mim,
            seed=seed
        )
    else:
        test_dataset = BatteryDataset(
            data['X_test'], data['y_test'],
            missing_rate=missing_rate,
            use_mim=use_mim,
            seed=seed
        )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # 评估
    evaluator = ModelEvaluator('cpu')
    metrics, _, _ = evaluator.evaluate(model, test_loader)
    
    return metrics


def run_big_experiment(cfg: DictConfig, logger):
    """运行大实验"""
    logger.info("=" * 70)
    logger.info("Big Experiment V2 - PyTorch Lightning + Hydra + Loguru")
    logger.info("=" * 70)
    logger.info(f"Config: {cfg.experiment.n_repeats} repeats, {cfg.training.epochs} epochs")
    logger.info(f"Models: 4 (MLP, LSTM, GRU, CNN1D) x 2 strategies = 8 configs")
    logger.info(f"Missing rates: {cfg.missing.missing_rates}")
    logger.info("=" * 70)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(cfg.experiment.output_dir) / cfg.batch / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # 获取模型配置
    model_configs = get_model_configs(cfg)
    
    all_results = []
    
    for repeat_idx in range(cfg.experiment.n_repeats):
        seed = cfg.experiment.seeds[repeat_idx] if repeat_idx < len(cfg.experiment.seeds) else cfg.experiment.seeds[0] + repeat_idx
        logger.info(f"\n{'='*70}")
        logger.info(f"Repeat {repeat_idx + 1}/{cfg.experiment.n_repeats} (seed={seed})")
        logger.info(f"{'='*70}")
        
        # 准备数据（每个种子重新划分）
        data = prepare_data(cfg, seed)
        
        # 运行每个模型配置
        for model_config in model_configs:
            try:
                results = run_single_experiment(
                    cfg, model_config, data, seed, logger
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed: {model_config['name']} - {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 定期保存结果
        if (repeat_idx + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df.to_csv(output_dir / "results_partial.csv", index=False)
            logger.info(f"Saved partial results: {len(df)} rows")
    
    # 保存最终结果
    df = pd.DataFrame(all_results)
    result_file = output_dir / "results_all.csv"
    df.to_csv(result_file, index=False)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Experiment completed!")
    logger.info(f"Results: {result_file}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"{'='*70}")
    
    # 生成图表
    if len(df) > 0:
        try:
            logger.info("Generating plots...")
            from src.visualization.batch_plots import BatchExperimentPlots
            plotter = BatchExperimentPlots(output_dir / "figures")
            plotter.plot_all(df)
            logger.info(f"Plots saved to: {output_dir / 'figures'}")
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return output_dir


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """主函数"""
    # 设置日志
    output_dir = Path(cfg.get('experiment', {}).get('output_dir', 'experiments_v2'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / cfg.batch / timestamp / "experiment.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("BigExperimentV2", str(log_file))
    
    # 运行实验
    output_dir = run_big_experiment(cfg, logger)
    
    print(f"\nExperiment completed: {output_dir}")


if __name__ == '__main__':
    main()
