"""实验运行器"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import set_seeds, setup_logger
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import (
    BatteryDataset, SequenceDataset, 
    MIMDataset, SequenceMIMDataset
)
from src.models.model_factory import ModelFactory
from src.evaluators.model_evaluator import ModelEvaluator


class ExperimentRunner:
    """实验运行器 - 协调整个实验流程"""
    
    def __init__(self, config, timestamp: str = None):
        """
        Args:
            config: 实验配置
            timestamp: 时间戳（可选）
        """
        self.config = config
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建结果目录
        self.exp_dir = Path(config.results_dir) / f"{self.timestamp}_{config.batch}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.model_dir = self.exp_dir / "models"
        self.result_dir = self.exp_dir / "results"
        self.figure_dir = self.exp_dir / "figures"
        self.log_dir = self.exp_dir / "logs"
        
        for d in [self.model_dir, self.result_dir, self.figure_dir, self.log_dir]:
            d.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            'experiment',
            str(self.log_dir / f"experiment_{self.timestamp}.log")
        )
        
        self.logger.info(f"实验时间戳: {self.timestamp}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"批次: {config.batch}")
        
        # 数据容器
        self.data = None
        self.results = []
    
    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")
        
        loader = XJTUDatasetLoader(
            self.config.data_dir,
            self.config.batch
        )
        
        self.data = loader.prepare_data(
            self.config.feature_cols,
            self.config.target_col,
            self.config.test_size,
            self.config.val_size,
            self.config.random_seed
        )
        
        self.logger.info(f"训练集: {self.data['X_train'].shape}")
        self.logger.info(f"验证集: {self.data['X_val'].shape}")
        self.logger.info(f"测试集: {self.data['X_test'].shape}")
        
        return self.data
    
    def train_baseline_model(self, model_config, seed: int = 42):
        """训练基线模型"""
        set_seeds(seed)
        
        model_name = model_config.name
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        
        self.logger.info(f"训练基线模型: {model_name}")
        
        # 创建模型
        # 从model_config.__dict__中移除已明确传递的参数
        config_dict = {k: v for k, v in model_config.__dict__.items() 
                      if k not in ['model_type', 'use_mim', 'name']}
        model = ModelFactory.create_model(
            model_type,
            input_dim=16,
            use_mim=use_mim,
            device=self.device,
            **config_dict
        )
        
        param_count = ModelFactory.count_parameters(model)
        self.logger.info(f"模型参数量: {param_count:,}")
        
        # 准备数据
        if model_type in ['lstm', 'gru', 'cnn1d', 'CNN1D']:
            # 序列模型
            train_dataset = SequenceDataset(
                self.data['X_train'], self.data['y_train'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=use_mim,
                seed=seed
            )
            val_dataset = SequenceDataset(
                self.data['X_val'], self.data['y_val'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=use_mim,
                seed=seed
            )
        else:
            # 非序列模型
            train_dataset = BatteryDataset(
                self.data['X_train'], self.data['y_train'],
                missing_rate=0.0,
                use_mim=use_mim,
                seed=seed
            )
            val_dataset = BatteryDataset(
                self.data['X_val'], self.data['y_val'],
                missing_rate=0.0,
                use_mim=use_mim,
                seed=seed
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        # 训练
        start_time = time.time()
        
        if model_type == 'xgboost':
            # XGBoost特殊处理，注意MIM模式下需要拼接特征和指示器
            if use_mim:
                # MIM模式下，从dataset获取拼接好的数据 [n_samples, 32]
                X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
                X_val = np.array([val_dataset[i][0].numpy() for i in range(len(val_dataset))])
            else:
                X_train = train_dataset.X if hasattr(train_dataset, 'X') else train_dataset.sequences.reshape(len(train_dataset.sequences), -1)
                X_val = val_dataset.X if hasattr(val_dataset, 'X') else val_dataset.sequences.reshape(len(val_dataset.sequences), -1)
            history = model.fit(X_train, train_dataset.y, X_val, val_dataset.y)
        else:
            history = model.fit(
                train_loader, val_loader,
                epochs=self.config.epochs,
                lr=self.config.lr,
                patience=self.config.early_stopping_patience
            )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.model_dir / f"best_model_{model_name}_{self.timestamp}.pth"
        model.save(str(model_path))
        
        self.logger.info(f"模型保存至: {model_path}")
        self.logger.info(f"训练耗时: {training_time:.2f}s")
        
        return model, history
    
    def train_mim_model(self, model_config, seed: int = 42):
        """训练MIM模型"""
        set_seeds(seed)
        
        model_name = model_config.name
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        
        self.logger.info(f"训练MIM模型: {model_name}")
        
        # 创建模型
        config_dict = {k: v for k, v in model_config.__dict__.items() 
                      if k not in ['model_type', 'use_mim', 'name']}
        model = ModelFactory.create_model(
            model_type,
            input_dim=16,
            use_mim=use_mim,
            device=self.device,
            **config_dict
        )
        
        param_count = ModelFactory.count_parameters(model)
        self.logger.info(f"模型参数量: {param_count:,}")
        
        # 准备MIM训练数据（多缺失率混合）
        if model_type in ['lstm', 'gru', 'cnn1d', 'CNN1D']:
            train_dataset = SequenceMIMDataset(
                self.data['X_train'], self.data['y_train'],
                seq_len=model_config.seq_len,
                missing_rates=self.config.training_missing_rates,
                use_mim=use_mim,
                base_seed=seed
            )
            val_dataset = SequenceDataset(
                self.data['X_val'], self.data['y_val'],
                seq_len=model_config.seq_len,
                missing_rate=0.5,  # 验证集使用50%缺失率
                use_mim=use_mim,
                seed=seed
            )
        else:
            train_dataset = MIMDataset(
                self.data['X_train'], self.data['y_train'],
                missing_rates=self.config.training_missing_rates,
                use_mim=use_mim,
                base_seed=seed
            )
            val_dataset = BatteryDataset(
                self.data['X_val'], self.data['y_val'],
                missing_rate=0.5,
                use_mim=use_mim,
                seed=seed
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        # 训练
        start_time = time.time()
        
        if model_type == 'xgboost':
            # XGBoost需要numpy数组，处理MIM维度
            if use_mim:
                # MIM模式下，数据已经是拼接好的 [n_samples, 32]
                X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
                X_val = np.array([val_dataset[i][0].numpy() for i in range(len(val_dataset))])
            else:
                X_train = train_dataset.X
                X_val = val_dataset.X if hasattr(val_dataset, 'X') else val_dataset.sequences.reshape(len(val_dataset.sequences), -1)
            history = model.fit(X_train, train_dataset.y, X_val, val_dataset.y)
        else:
            history = model.fit(
                train_loader, val_loader,
                epochs=self.config.epochs,
                lr=self.config.lr,
                patience=self.config.early_stopping_patience
            )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.model_dir / f"best_model_{model_name}_{self.timestamp}.pth"
        model.save(str(model_path))
        
        self.logger.info(f"模型保存至: {model_path}")
        self.logger.info(f"训练耗时: {training_time:.2f}s")
        
        return model, history
    
    def evaluate_model(self, model, model_config, missing_rate: float, seed: int = 42):
        """在指定缺失率下评估模型"""
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        
        evaluator = ModelEvaluator(self.device)
        
        # 准备测试数据
        if model_type in ['lstm', 'gru', 'cnn1d', 'CNN1D']:
            test_dataset = SequenceDataset(
                self.data['X_test'], self.data['y_test'],
                seq_len=model_config.seq_len,
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=seed
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size
            )
            metrics, preds, targets = evaluator.evaluate(model, test_loader)
        else:
            test_dataset = BatteryDataset(
                self.data['X_test'], self.data['y_test'],
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=seed
            )
            
            if model_type == 'xgboost':
                # XGBoost评估，注意MIM模式下需要使用拼接后的数据
                if use_mim:
                    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
                else:
                    X_test = test_dataset.X
                metrics, preds, targets = evaluator.evaluate_xgboost(
                    model.model if hasattr(model, 'model') else model,
                    X_test,
                    test_dataset.y
                )
            else:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size
                )
                metrics, preds, targets = evaluator.evaluate(model, test_loader)
        
        return metrics, preds, targets
    
    def run_single_experiment(self, model_config, seed: int = 42):
        """运行单次实验"""
        model_name = model_config.name
        use_mim = model_config.use_mim
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"开始实验: {model_name}, Seed: {seed}")
        self.logger.info(f"=" * 60)
        
        # 训练模型
        if use_mim:
            model, history = self.train_mim_model(model_config, seed)
        else:
            model, history = self.train_baseline_model(model_config, seed)
        
        # 在不同缺失率下评估
        results = []
        for mr in self.config.missing_rates:
            metrics, preds, targets = self.evaluate_model(
                model, model_config, mr, seed
            )
            
            result = {
                'timestamp': self.timestamp,
                'model': model_config.name,
                'model_type': model_config.model_type,
                'use_mim': use_mim,
                'missing_rate': mr,
                'seed': seed,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'inference_time': metrics['inference_time'],
                'training_time': history.get('training_time', 0)
            }
            results.append(result)
            
            self.logger.info(
                f"Missing Rate {mr:.1f}: "
                f"MAE={metrics['mae']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, "
                f"R2={metrics['r2']:.4f}"
            )
        
        return results
    
    def run_all_experiments(self):
        """运行所有实验"""
        self.logger.info("=" * 60)
        self.logger.info("开始全部实验")
        self.logger.info("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 获取模型配置
        model_configs = self.config.get_model_configs()
        
        all_results = []
        
        # 对每个模型配置
        for model_config in model_configs:
            # 运行n_repeats次重复实验
            for i in range(self.config.n_repeats):
                seed = self.config.random_seed + i
                results = self.run_single_experiment(model_config, seed)
                all_results.extend(results)
        
        # 保存结果
        self.save_results(all_results)
        
        self.logger.info("=" * 60)
        self.logger.info("全部实验完成")
        self.logger.info("=" * 60)
        
        return all_results
    
    def save_results(self, results):
        """保存实验结果"""
        import pandas as pd
        
        df = pd.DataFrame(results)
        csv_path = self.result_dir / f"experiment_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"结果保存至: {csv_path}")
        
        # 保存配置
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'batch': self.config.batch,
                'random_seed': self.config.random_seed,
                'n_repeats': self.config.n_repeats,
                'missing_rates': self.config.missing_rates,
                'battery_info': self.data['battery_info'] if self.data else {}
            }, f, indent=2)
