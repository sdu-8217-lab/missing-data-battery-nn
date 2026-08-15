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
from src.data.dataset_loader import DatasetLoaderFactory

# 按架构固定 seed offset，保证同一 repeat 下 Baseline / MIM / MultiMR 共享同一数据划分
MODEL_SEED_OFFSETS = {
    'mlp': 0,
    'lstm': 100000,
    'gru': 200000,
    'cnn1d': 300000,
    'cnn': 300000,
}
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
        
        # 数据容器（需先于 save_config 初始化）
        self.data = None
        self.results = []
        
        # 实验开始即保存配置（后续加载数据后再更新 battery_info）
        self.save_config()
        
        # 设置日志
        self.logger = setup_logger(
            'experiment',
            str(self.log_dir / f"experiment_{self.timestamp}.log")
        )
        
        self.logger.info(f"实验时间戳: {self.timestamp}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"批次: {config.batch}")
    
    def load_data(self, seed: int = None):
        """加载数据（按指定 seed 划分电池）"""
        self.logger.info("加载数据...")
        
        loader = DatasetLoaderFactory.create_loader(
            self.config.data_dir,
            self.config.dataset_name,
            self.config.batch
        )
        
        split_seed = seed if seed is not None else self.config.random_seed
        self.data = loader.prepare_data(
            self.config.feature_cols,
            self.config.target_col,
            self.config.test_size,
            self.config.val_size,
            split_seed,
            logger=self.logger
        )
        
        self.logger.info(f"训练集: {self.data['X_train'].shape}")
        self.logger.info(f"验证集: {self.data['X_val'].shape}")
        self.logger.info(f"测试集: {self.data['X_test'].shape}")
        self.logger.info(f"序列边界: train={self.data.get('train_boundaries')}, val={self.data.get('val_boundaries')}, test={self.data.get('test_boundaries')}")
        
        return self.data
    
    def train_baseline_model(self, model_config, seed: int = 42):
        """训练基线模型"""
        set_seeds(seed)
        
        model_name = model_config.name
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        use_fmg = getattr(model_config, 'use_fmg', False)
        use_physics_loss = getattr(model_config, 'use_physics_loss', False)
        physics_loss_types = getattr(model_config, 'physics_loss_types', None)
        physics_loss_weights = getattr(model_config, 'physics_loss_weights', None)
        
        self.logger.info(f"训练基线模型: {model_name}")
        
        # 创建模型
        # 从model_config.__dict__中移除已明确传递的参数
        config_dict = {k: v for k, v in model_config.__dict__.items() 
                      if k not in ['model_type', 'use_mim', 'use_fmg', 'name']}
        # 从实际数据维度推导输入维度（兼容不同数据集的特征数）
        input_dim = self.data['X_train'].shape[1]
        model = ModelFactory.create_model(
            model_type,
            input_dim=input_dim,
            use_mim=use_mim,
            use_fmg=use_fmg,
            device=self.device,
            **config_dict
        )
        
        param_count = ModelFactory.count_parameters(model)
        self.logger.info(f"模型参数量: {param_count:,}")
        
        # 准备数据
        if model_type in ['lstm', 'gru', 'cnn1d', 'cnn', 'transformer', 'itransformer',
                            'saits', 'neural_cde', 'grin', 'CNN1D']:
            # 序列模型
            train_dataset = SequenceDataset(
                self.data['X_train'], self.data['y_train'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                boundaries=self.data.get('train_boundaries'),
                battery_ids=self.data.get('train_battery_ids'),
                return_battery_id=use_physics_loss
            )
            val_dataset = SequenceDataset(
                self.data['X_val'], self.data['y_val'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                boundaries=self.data.get('val_boundaries'),
                battery_ids=self.data.get('val_battery_ids'),
                return_battery_id=use_physics_loss
            )
        else:
            # 非序列模型
            train_dataset = BatteryDataset(
                self.data['X_train'], self.data['y_train'],
                missing_rate=0.0,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                battery_ids=self.data.get('train_battery_ids'),
                return_battery_id=use_physics_loss
            )
            val_dataset = BatteryDataset(
                self.data['X_val'], self.data['y_val'],
                missing_rate=0.0,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                battery_ids=self.data.get('val_battery_ids'),
                return_battery_id=use_physics_loss
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
            # XGBoost特殊处理，注意MIM/FMG模式下需要拼接特征和指示器
            if use_mim or use_fmg:
                # 从dataset获取拼接好的数据 [n_samples, 32]
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
                patience=self.config.early_stopping_patience,
                physics_loss_types=physics_loss_types,
                physics_loss_weights=physics_loss_weights
            )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.model_dir / f"best_model_{model_name}_seed{seed}_{self.timestamp}.pth"
        model.save(str(model_path))
        
        self.logger.info(f"模型保存至: {model_path}")
        self.logger.info(f"训练耗时: {training_time:.2f}s")
        
        return model, history
    
    def train_multi_rate_model(self, model_config, seed: int = 42):
        """训练多缺失率模型（MIM 或 不带指示器的多缺失率基线）"""
        set_seeds(seed)
        
        model_name = model_config.name
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        use_fmg = getattr(model_config, 'use_fmg', False)
        use_curriculum = getattr(model_config, 'use_curriculum', False)
        strategy = getattr(model_config, 'strategy', 'mim')
        use_physics_loss = getattr(model_config, 'use_physics_loss', False)
        physics_loss_types = getattr(model_config, 'physics_loss_types', None)
        physics_loss_weights = getattr(model_config, 'physics_loss_weights', None)
        
        self.logger.info(f"训练多缺失率模型: {model_name} (strategy={strategy}, MIM={use_mim}, FMG={use_fmg}, CMR={use_curriculum})")
        
        # 创建模型
        config_dict = {k: v for k, v in model_config.__dict__.items() 
                      if k not in ['model_type', 'use_mim', 'use_fmg', 'use_curriculum', 'name']}
        # 从实际数据维度推导输入维度（兼容不同数据集的特征数）
        input_dim = self.data['X_train'].shape[1]
        model = ModelFactory.create_model(
            model_type,
            input_dim=input_dim,
            use_mim=use_mim,
            use_fmg=use_fmg,
            device=self.device,
            **config_dict
        )
        
        param_count = ModelFactory.count_parameters(model)
        self.logger.info(f"模型参数量: {param_count:,}")
        
        # 准备多缺失率训练数据（MIM / FMG / 不带指示器）
        if model_type in ['lstm', 'gru', 'cnn1d', 'cnn', 'transformer', 'itransformer',
                            'saits', 'neural_cde', 'grin', 'CNN1D']:
            train_dataset = SequenceMIMDataset(
                self.data['X_train'], self.data['y_train'],
                seq_len=model_config.seq_len,
                missing_rates=self.config.training_missing_rates,
                use_mim=use_mim,
                use_fmg=use_fmg,
                use_curriculum=use_curriculum,
                total_epochs=self.config.epochs,
                base_seed=seed,
                missing_pattern=self.config.missing_pattern,
                boundaries=self.data.get('train_boundaries'),
                strategy=strategy,
                battery_ids=self.data.get('train_battery_ids'),
                return_battery_id=use_physics_loss
            )
            val_dataset = SequenceDataset(
                self.data['X_val'], self.data['y_val'],
                seq_len=model_config.seq_len,
                missing_rate=0.5,  # 验证集使用50%缺失率以保持与MIM一致
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                boundaries=self.data.get('val_boundaries'),
                battery_ids=self.data.get('val_battery_ids'),
                return_battery_id=use_physics_loss
            )
        else:
            train_dataset = MIMDataset(
                self.data['X_train'], self.data['y_train'],
                missing_rates=self.config.training_missing_rates,
                use_mim=use_mim,
                use_fmg=use_fmg,
                use_curriculum=use_curriculum,
                total_epochs=self.config.epochs,
                base_seed=seed,
                missing_pattern=self.config.missing_pattern,
                strategy=strategy,
                battery_ids=self.data.get('train_battery_ids'),
                return_battery_id=use_physics_loss
            )
            val_dataset = BatteryDataset(
                self.data['X_val'], self.data['y_val'],
                missing_rate=0.5,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                battery_ids=self.data.get('val_battery_ids'),
                return_battery_id=use_physics_loss
            )
        
        # 课程学习回调：每个 epoch 根据当前阶段重新生成 DataLoader
        epoch_callback = None
        if use_curriculum:
            def epoch_callback(epoch):
                train_dataset.set_epoch(epoch)
                return DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True
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
            # XGBoost需要numpy数组，处理MIM/FMG维度
            if use_mim or use_fmg:
                # 数据已经是拼接好的 [n_samples, 32]
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
                patience=self.config.early_stopping_patience,
                epoch_callback=epoch_callback,
                physics_loss_types=physics_loss_types,
                physics_loss_weights=physics_loss_weights
            )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.model_dir / f"best_model_{model_name}_seed{seed}_{self.timestamp}.pth"
        model.save(str(model_path))
        
        self.logger.info(f"模型保存至: {model_path}")
        self.logger.info(f"训练耗时: {training_time:.2f}s")
        
        return model, history
    
    def evaluate_model(self, model, model_config, missing_rate: float, seed: int = 42):
        """在指定缺失率下评估模型"""
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        use_fmg = getattr(model_config, 'use_fmg', False)
        
        evaluator = ModelEvaluator(self.device)
        
        # 准备测试数据
        if model_type in ['lstm', 'gru', 'cnn1d', 'cnn', 'transformer', 'itransformer',
                            'saits', 'neural_cde', 'grin', 'CNN1D']:
            test_dataset = SequenceDataset(
                self.data['X_test'], self.data['y_test'],
                seq_len=model_config.seq_len,
                missing_rate=missing_rate,
                use_mim=use_mim,
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern,
                boundaries=self.data.get('test_boundaries'),
                battery_ids=self.data.get('test_battery_ids'),
                return_battery_id=False
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
                use_fmg=use_fmg,
                seed=seed,
                missing_pattern=self.config.missing_pattern
            )
            
            if model_type == 'xgboost':
                # XGBoost评估，注意MIM/FMG模式下需要使用拼接后的数据
                if use_mim or use_fmg:
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
    
    def save_config(self):
        """保存实验配置到 config.json"""
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'dataset_name': self.config.dataset_name,
                'batch': self.config.batch,
                'missing_pattern': self.config.missing_pattern,
                'random_seed': self.config.random_seed,
                'n_repeats': self.config.n_repeats,
                'missing_rates': self.config.missing_rates,
                'training_missing_rates': self.config.training_missing_rates,
                'feature_cols': getattr(self.config, 'feature_cols', []),
                'input_dim': int(self.data['X_train'].shape[1]) if self.data else None,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'early_stopping_patience': self.config.early_stopping_patience,
                'battery_info': self.data['battery_info'] if self.data else {}
            }, f, indent=2)
    
    def save_split(self, seed: int):
        """保存当前 seed 的电池划分"""
        if self.data is None or 'battery_info' not in self.data:
            return
        split_path = self.result_dir / f"splits_seed{seed}.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(self.data['battery_info'], f, indent=2)
    
    def run_single_experiment(self, model_config, seed: int = 42):
        """运行单次实验"""
        model_name = model_config.name
        use_mim = model_config.use_mim
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"开始实验: {model_name}, Seed: {seed}")
        self.logger.info(f"=" * 60)
        
        # 每个 seed 独立进行数据划分，确保 100 次重复实验覆盖数据划分差异
        self.load_data(seed=seed)
        self.save_config()
        self.save_split(seed)
        
        # 训练模型
        if model_config.strategy in ('mim', 'multi_missing_rate', 'uniform_multi_rate'):
            model, history = self.train_multi_rate_model(model_config, seed)
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
                'use_fmg': getattr(model_config, 'use_fmg', False),
                'use_curriculum': getattr(model_config, 'use_curriculum', False),
                'strategy': getattr(model_config, 'strategy', 'baseline'),
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
        
        # 数据将在 run_single_experiment 中按 seed 分别加载
        
        # 获取模型配置
        model_configs = self.config.get_model_configs()
        
        all_results = []
        
        # 对每个模型配置使用固定、可复现的 seed 映射：
        # seed = base_seed + model_idx * n_repeats + repeat_idx
        base_seed = self.config.random_seed
        for model_config in model_configs:
            offset = MODEL_SEED_OFFSETS.get(model_config.model_type.lower(), 0)
            # 运行n_repeats次重复实验
            for i in range(self.config.n_repeats):
                # 同一架构的 Baseline / MIM / MultiMR 使用相同 seed，保证公平对比
                seed = base_seed + offset + i
                results = self.run_single_experiment(model_config, seed)
                all_results.extend(results)
                # 每完成一个模型/种子组合就增量保存，避免中断丢失全部结果
                self.save_results(results, append=True)
        
        self.logger.info("=" * 60)
        self.logger.info("全部实验完成")
        self.logger.info("=" * 60)
        
        return all_results
    
    def save_results(self, results, append: bool = False):
        """保存实验结果
        
        Args:
            results: 本次要保存的结果列表
            append: 是否追加到已有 CSV
        """
        import pandas as pd
        
        df = pd.DataFrame(results)
        csv_path = self.result_dir / f"experiment_results_{self.timestamp}.csv"
        
        if append and csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        
        self.logger.info(f"结果保存至: {csv_path}")
        
        # 保存配置
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'dataset_name': self.config.dataset_name,
                'batch': self.config.batch,
                'missing_pattern': self.config.missing_pattern,
                'random_seed': self.config.random_seed,
                'n_repeats': self.config.n_repeats,
                'missing_rates': self.config.missing_rates,
                'feature_cols': getattr(self.config, 'feature_cols', []),
                'input_dim': int(self.data['X_train'].shape[1]) if self.data else None,
                'battery_info': self.data['battery_info'] if self.data else {}
            }, f, indent=2)
