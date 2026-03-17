#!/usr/bin/env python3
"""
大实验V2 - 整合改进版本
使用: PyTorch Lightning + Pydantic + Loguru + configs_v3最佳参数
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

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

# V2改进组件
from src.config.pydantic_config import ExperimentConfig, ModelConfig
from src.trainers.lightning_module import SOHLightningModule
from src.trainers.lightning_trainer import LightningTrainer
from src.utils.logger import setup_logger

# 数据加载
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset, MIMDataset, SequenceMIMDataset

# 模型
from src.models.model_factory import ModelFactory


def prepare_data(config: ExperimentConfig, seed: int):
    """准备数据"""
    loader = XJTUDatasetLoader(
        data_dir=config.data_dir,
        batch=config.batch
    )
    
    data = loader.prepare_data(
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        test_size=config.test_size,
        val_size=config.val_size,
        random_seed=seed
    )
    
    return data


def run_single_experiment(
    config: ExperimentConfig,
    model_config: ModelConfig,
    data: dict,
    seed: int,
    logger
) -> List[Dict]:
    """运行单个模型实验"""
    results = []
    model_name = model_config.name
    use_mim = model_config.use_mim
    
    logger.info(f"Training {model_name} (MIM={use_mim})")
    
    # 创建模型
    # BaseModel会自动处理MIM的输入维度翻倍，所以这里传递原始维度16
    model_kwargs = {
        'hidden_layers': model_config.hidden_layers,
        'dropout': model_config.dropout,
        'hidden_size': model_config.hidden_size,
        'num_layers': model_config.num_layers,
        'channels': model_config.channels,
        'kernel_size': model_config.kernel_size,
    }
    # 过滤None值
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    
    model = ModelFactory.create_model(
        model_config.model_type,
        input_dim=16,  # BaseModel会自动翻倍当use_mim=True
        use_mim=use_mim,
        device='cpu',
        **model_kwargs
    )
    
    param_count = ModelFactory.count_parameters(model)
    logger.info(f"  Parameters: {param_count:,}")
    
    # 准备训练数据
    if use_mim:
        train_dataset, val_dataset = prepare_mim_data(config, model_config, data, seed)
    else:
        train_dataset, val_dataset = prepare_baseline_data(config, model_config, data, seed)
    
    # 使用PyTorch Lightning训练
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0
    )
    
    # 包装为Lightning模块
    lightning_model = SOHLightningModule(
        model.model if hasattr(model, 'model') else model,
        learning_rate=config.lr
    )
    
    # 训练
    trainer = LightningTrainer(
        max_epochs=config.epochs,
        patience=config.early_stopping_patience,
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
    for missing_rate in config.missing_rates:
        metrics = evaluate_model(
            model, model_config, data, missing_rate, 
            config.batch_size, seed
        )
        metrics.update({
            'seed': seed,
            'model_name': model_name,
            'model_type': model_config.model_type,
            'use_mim': use_mim,
            'missing_rate': missing_rate,
            'training_time': training_time,
            'params': param_count,
        })
        results.append(metrics)
        logger.info(f"  MR={missing_rate:.1f}: MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
    
    return results


def prepare_baseline_data(config: ExperimentConfig, model_config: ModelConfig, 
                          data: dict, seed: int):
    """准备Baseline训练数据"""
    model_type = model_config.model_type
    
    if model_type in ['lstm', 'gru', 'cnn1d']:
        train_dataset = SequenceDataset(
            data['X_train'], data['y_train'],
            seq_len=model_config.seq_len,
            missing_rate=0.0,
            use_mim=False,
            seed=seed
        )
        val_dataset = SequenceDataset(
            data['X_val'], data['y_val'],
            seq_len=model_config.seq_len,
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


def prepare_mim_data(config: ExperimentConfig, model_config: ModelConfig,
                     data: dict, seed: int):
    """准备MIM训练数据"""
    model_type = model_config.model_type
    
    if model_type in ['lstm', 'gru', 'cnn1d']:
        train_dataset = SequenceMIMDataset(
            data['X_train'], data['y_train'],
            seq_len=model_config.seq_len,
            missing_rates=config.training_missing_rates,
            use_mim=True,
            base_seed=seed
        )
        # 验证集使用50%缺失率
        val_dataset = SequenceDataset(
            data['X_val'], data['y_val'],
            seq_len=model_config.seq_len,
            missing_rate=0.5,
            use_mim=True,
            seed=seed
        )
    else:
        train_dataset = MIMDataset(
            data['X_train'], data['y_train'],
            missing_rates=config.training_missing_rates,
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


def evaluate_model(model, model_config: ModelConfig, data: dict,
                   missing_rate: float, batch_size: int, seed: int) -> Dict:
    """评估模型"""
    from torch.utils.data import DataLoader
    from src.evaluation.model_evaluator import ModelEvaluator
    
    model_type = model_config.model_type
    use_mim = model_config.use_mim
    
    # 准备测试数据
    if model_type in ['lstm', 'gru', 'cnn1d']:
        test_dataset = SequenceDataset(
            data['X_test'], data['y_test'],
            seq_len=model_config.seq_len,
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


def run_big_experiment(config: ExperimentConfig, logger):
    """运行大实验"""
    logger.info("=" * 70)
    logger.info("Big Experiment V2 - PyTorch Lightning + Pydantic + Loguru")
    logger.info("=" * 70)
    logger.info(f"Config: {config.n_repeats} repeats, {config.epochs} epochs")
    logger.info(f"Models: 4 (MLP, LSTM, GRU, CNN1D) x 2 strategies = 8 configs")
    logger.info(f"Missing rates: {config.missing_rates}")
    logger.info("=" * 70)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("experiments_v2") / config.batch / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config.save_json(output_dir / "config.json")
    
    # 获取模型配置
    model_configs = config.get_model_configs()
    
    all_results = []
    
    for repeat_idx in range(config.n_repeats):
        seed = config.random_seed + repeat_idx
        logger.info(f"\n{'='*70}")
        logger.info(f"Repeat {repeat_idx + 1}/{config.n_repeats} (seed={seed})")
        logger.info(f"{'='*70}")
        
        # 准备数据（每个种子重新划分）
        data = prepare_data(config, seed)
        
        # 运行每个模型配置
        for model_config in model_configs:
            try:
                results = run_single_experiment(
                    config, model_config, data, seed, logger
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed: {model_config.name} - {e}")
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Big Experiment V2')
    parser.add_argument('--batch', type=str, default='3C', 
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'])
    parser.add_argument('--n_repeats', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        batch=args.batch,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        random_seed=args.seed
    )
    
    # 设置日志
    log_file = f"experiments_v2/{args.batch}/experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("BigExperimentV2", log_file)
    
    # 运行实验
    output_dir = run_big_experiment(config, logger)
    
    print(f"\nExperiment completed: {output_dir}")


if __name__ == '__main__':
    main()
