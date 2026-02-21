#!/usr/bin/env python3
"""
使用新版代码（Pydantic + Loguru + Lightning）运行完整实验
包括：数据加载、模型训练、评估
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

# 新版组件
from src.config.pydantic_config import ExperimentConfig, ModelConfig
from src.utils.logger_v2 import setup_logger
from loguru import logger

# 数据加载
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset

# 模型
from src.models.model_factory import ModelFactory

# 新版训练器（可选）
try:
    from src.trainers import SOHLightningModule, LightningTrainer
    USE_LIGHTNING = True
except ImportError:
    USE_LIGHTNING = False
    from src.trainers import NeuralNetworkTrainer

# 评估
from src.evaluators.model_evaluator import ModelEvaluator


def load_experiment_data(config: ExperimentConfig, logger):
    """加载实验数据"""
    logger.info("=" * 60)
    logger.info("加载数据...")
    logger.info("=" * 60)
    
    loader = XJTUDatasetLoader(
        data_dir=config.data_dir,
        batch=config.batch
    )
    
    data = loader.prepare_data(
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        test_size=config.test_size,
        val_size=config.val_size,
        random_seed=config.random_seed,
        logger=logger
    )
    
    logger.info(f"数据加载完成:")
    logger.info(f"  训练集: {data['X_train'].shape}")
    logger.info(f"  验证集: {data['X_val'].shape}")
    logger.info(f"  测试集: {data['X_test'].shape}")
    
    return data


def train_and_evaluate_model(model_config: ModelConfig, data: dict, 
                              exp_config: ExperimentConfig, logger, device: str):
    """训练和评估单个模型"""
    import platform
    
    model_name = model_config.name
    model_type = model_config.model_type
    use_mim = model_config.use_mim
    
    # 设置数据加载器workers（Windows使用0避免多进程问题）
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    logger.info("-" * 60)
    logger.info(f"训练模型: {model_name} (MIM={use_mim})")
    logger.info("-" * 60)
    
    # 创建模型
    config_dict = {k: v for k, v in model_config.__dict__.items() 
                   if k not in ['model_type', 'use_mim', 'name']}
    
    model = ModelFactory.create_model(
        model_type,
        input_dim=16,
        use_mim=use_mim,
        device=device,
        **config_dict
    )
    
    param_count = ModelFactory.count_parameters(model)
    logger.info(f"模型参数量: {param_count:,}")
    
    # 准备数据
    from src.data.datasets import MIMDataset, SequenceMIMDataset
    
    if model_type in ['lstm', 'gru', 'cnn1d']:
        # 序列模型
        if use_mim:
            # MIM模型：使用多个缺失率训练
            train_dataset = SequenceMIMDataset(
                data['X_train'], data['y_train'],
                seq_len=model_config.seq_len,
                missing_rates=exp_config.training_missing_rates,  # [0.0, 0.1, ..., 0.9]
                use_mim=True,
                base_seed=exp_config.random_seed
            )
        else:
            # Baseline：使用完整数据训练
            train_dataset = SequenceDataset(
                data['X_train'], data['y_train'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=False,
                seed=exp_config.random_seed
            )
        # 验证集使用固定缺失率
        val_dataset = SequenceDataset(
            data['X_val'], data['y_val'],
            seq_len=model_config.seq_len,
            missing_rate=0.5,
            use_mim=use_mim,
            seed=exp_config.random_seed
        )
    else:
        # 非序列模型
        if use_mim:
            # MIM模型：使用多个缺失率训练
            train_dataset = MIMDataset(
                data['X_train'], data['y_train'],
                missing_rates=exp_config.training_missing_rates,  # [0.0, 0.1, ..., 0.9]
                use_mim=True,
                base_seed=exp_config.random_seed
            )
        else:
            # Baseline：使用完整数据训练
            train_dataset = BatteryDataset(
                data['X_train'], data['y_train'],
                missing_rate=0.0,
                use_mim=False,
                seed=exp_config.random_seed
            )
        # 验证集使用固定缺失率
        val_dataset = BatteryDataset(
            data['X_val'], data['y_val'],
            missing_rate=0.5,
            use_mim=use_mim,
            seed=exp_config.random_seed
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=exp_config.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=exp_config.batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # 训练
    start_time = time.time()
    
    if model_type == 'xgboost':
        # XGBoost特殊处理：从Dataset中提取numpy数组
        # 注意：不能直接访问 .X，因为对于MIM需要获取拼接后的特征
        if use_mim:
            # MIM模式下，通过遍历Dataset获取拼接后的特征
            X_train_list, y_train_list = [], []
            for x, y in train_dataset:
                X_train_list.append(x.numpy())
                y_train_list.append(y.numpy())
            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)
            
            X_val_list, y_val_list = [], []
            for x, y in val_dataset:
                X_val_list.append(x.numpy())
                y_val_list.append(y.numpy())
            X_val = np.array(X_val_list)
            y_val = np.array(y_val_list)
        else:
            # Baseline模式，直接使用X_missing
            X_train = train_dataset.X_missing
            y_train = train_dataset.y
            X_val = val_dataset.X_missing
            y_val = val_dataset.y
        
        model.fit(X_train, y_train, X_val, y_val)
    else:
        # 使用新版Lightning或旧版训练器
        if USE_LIGHTNING:
            logger.info("使用 PyTorch Lightning 训练")
            pl_module = SOHLightningModule(
                model.model if hasattr(model, 'model') else model,
                learning_rate=exp_config.lr
            )
            trainer = LightningTrainer(
                max_epochs=exp_config.epochs,
                patience=exp_config.early_stopping_patience,
                device=device
            )
            history = trainer.train(pl_module, train_loader, val_loader)
        else:
            logger.info("使用传统训练器")
            from src.trainers.neural_network_trainer import NeuralNetworkTrainer
            nn_trainer = NeuralNetworkTrainer(
                model.model if hasattr(model, 'model') else model,
                device=device
            )
            history = nn_trainer.train(
                train_loader, val_loader,
                epochs=exp_config.epochs,
                lr=exp_config.lr,
                patience=exp_config.early_stopping_patience
            )
    
    training_time = time.time() - start_time
    logger.info(f"训练完成，耗时: {training_time:.2f}s")
    
    # 评估不同缺失率
    evaluator = ModelEvaluator(device)
    results = []
    
    for missing_rate in exp_config.missing_rates:
        logger.info(f"  评估缺失率: {missing_rate}")
        
        # 准备测试数据
        if model_type in ['lstm', 'gru', 'cnn1d']:
            test_dataset = SequenceDataset(
                data['X_test'], data['y_test'],
                seq_len=model_config.seq_len,
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=exp_config.random_seed
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=exp_config.batch_size,
                num_workers=num_workers,
                pin_memory=False
            )
            metrics, preds, targets = evaluator.evaluate(model, test_loader)
        else:
            test_dataset = BatteryDataset(
                data['X_test'], data['y_test'],
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=exp_config.random_seed
            )
            
            if model_type == 'xgboost':
                # XGBoost测试数据提取
                if use_mim:
                    X_test_list, y_test_list = [], []
                    for x, y in test_dataset:
                        X_test_list.append(x.numpy())
                        y_test_list.append(y.numpy())
                    X_test = np.array(X_test_list)
                    y_test = np.array(y_test_list)
                else:
                    X_test = test_dataset.X_missing
                    y_test = test_dataset.y
                
                metrics, preds, targets = evaluator.evaluate_xgboost(
                    model.model if hasattr(model, 'model') else model,
                    X_test, y_test
                )
            else:
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=exp_config.batch_size,
                    num_workers=num_workers,
                    pin_memory=False
                )
                metrics, preds, targets = evaluator.evaluate(model, test_loader)
        
        result = {
            'model': model_name,
            'model_type': model_type,
            'use_mim': use_mim,
            'missing_rate': missing_rate,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'training_time': training_time,
            'seed': exp_config.random_seed
        }
        results.append(result)
        
        logger.info(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
    
    return results


def run_single_experiment(batch_name: str, seed: int, epochs: int = 50):
    """运行单个小实验"""
    exp_dir = Path(f"./experiments_v2/{batch_name}/seed_{seed}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="single_experiment",
        log_file=str(exp_dir / "experiment.log"),
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("启动单个小实验 (使用新版代码)")
    logger.info("=" * 60)
    
    # 创建配置
    config = ExperimentConfig(
        name=f"soh_exp_{batch_name}",
        batch=batch_name,
        random_seed=seed,
        n_repeats=1,
        epochs=epochs,
        early_stopping_patience=10
    )
    
    logger.info(f"实验配置: {config.to_dict()}")
    
    # 确定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    if USE_LIGHTNING:
        logger.info("PyTorch Lightning: 可用")
    else:
        logger.info("PyTorch Lightning: 不可用，使用传统训练器")
    
    # 加载数据
    data = load_experiment_data(config, logger)
    
    # 运行所有模型
    all_results = []
    model_configs = config.get_model_configs()
    
    for model_config in model_configs:
        try:
            results = train_and_evaluate_model(
                model_config, data, config, logger, device
            )
            all_results.extend(results)
        except Exception as e:
            logger.error(f"模型 {model_config.name} 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 保存结果
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_file = exp_dir / "results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"结果已保存: {results_file}")
    
    # 生成可视化图表
    try:
        from src.visualization.single_plots import SingleExperimentPlots
        plots_dir = exp_dir / "figures"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plotter = SingleExperimentPlots(plots_dir)
        plotter.plot_all(results_df)
        logger.info(f"图表已保存: {plots_dir}")
    except Exception as e:
        logger.warning(f"图表生成失败: {e}")
    
    logger.info("=" * 60)
    logger.info("实验完成!")
    logger.info("=" * 60)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="使用新版代码运行完整实验")
    parser.add_argument("--batch-name", default="3C", 
                       choices=["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"],
                       help="数据批次")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_single_experiment(args.batch_name, args.seed, args.epochs)
    
    print(f"\n实验完成! 结果已保存到 ./experiments_v2/{args.batch_name}/seed_{args.seed}/")
    print(f"总共运行 {len(results)} 个模型配置 × {len(results['missing_rate'].unique())} 个缺失率")


if __name__ == "__main__":
    main()
