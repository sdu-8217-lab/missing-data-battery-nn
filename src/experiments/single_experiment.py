"""小实验运行器 - 单次实验（固定随机种子）"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from src.utils import set_seeds, setup_logger
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset
from src.models.model_factory import ModelFactory
from src.config.experiment_config import ModelConfig
from src.evaluators.model_evaluator import ModelEvaluator


class SingleExperimentConfig:
    """小实验配置"""
    def __init__(self, 
                 batch_name: str = "3C",
                 seed: int = 42,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 early_stopping_patience: int = 15,
                 missing_rates: List[float] = None,
                 device: str = 'cpu'):
        self.batch_name = batch_name
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.missing_rates = missing_rates or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.device = device
        
        # MIM训练用的缺失率列表（包含0.0）
        self.mim_training_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class SingleExperimentRunner:
    """小实验运行器 - 固定随机种子，训练所有模型配置"""
    
    def __init__(self, config: SingleExperimentConfig, base_dir: Path = None):
        self.config = config
        self.seed = config.seed
        
        # 实验目录结构: experiments/{batch_name}/{timestamp}/seed_{seed}/
        if base_dir is None:
            base_dir = Path("experiments") / config.batch_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.seed_dir = base_dir / f"seed_{self.seed}"
        self.model_dir = self.seed_dir / "models"
        self.figure_dir = self.seed_dir / "figures"
        
        # 创建目录
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger("SingleExperiment", self.seed_dir / f"seed_{self.seed}.log")
        
        # 数据缓存
        self.data = None
        
    def load_data(self):
        """加载数据"""
        if self.data is not None:
            return self.data
            
        loader = XJTUDatasetLoader(
            data_dir="./data/XJTU data",
            batch=self.config.batch_name
        )
        
        # 定义特征列（16个）
        feature_cols = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ]
        
        self.data = loader.prepare_data(
            feature_cols=feature_cols,
            target_col='capacity',
            test_size=0.25,
            val_size=0.25,
            random_seed=self.seed,
            logger=self.logger
        )
        
        self.logger.info(f"数据形状 - 训练集: {self.data['X_train'].shape}, "
                        f"验证集: {self.data['X_val'].shape}, 测试集: {self.data['X_test'].shape}")
        return self.data
    
    def run(self) -> pd.DataFrame:
        """运行完整的小实验"""
        self.logger.info("=" * 70)
        self.logger.info(f"开始小实验 - Seed: {self.seed}")
        self.logger.info("=" * 70)
        
        # 加载数据
        self.load_data()
        
        # 设置随机种子
        set_seeds(self.seed)
        
        # 所有模型配置 - 4模型 × 2策略 = 8配置 (已移除XGBoost)
        model_configs = [
            ModelConfig(model_type='mlp', use_mim=False, name='MLP', seq_len=5),
            ModelConfig(model_type='mlp', use_mim=True, name='MLP-MIM', seq_len=5),
            ModelConfig(model_type='lstm', use_mim=False, name='LSTM', seq_len=5),
            ModelConfig(model_type='lstm', use_mim=True, name='LSTM-MIM', seq_len=5),
            ModelConfig(model_type='gru', use_mim=False, name='GRU', seq_len=5),
            ModelConfig(model_type='gru', use_mim=True, name='GRU-MIM', seq_len=5),
            ModelConfig(model_type='cnn1d', use_mim=False, name='CNN1D', seq_len=5),
            ModelConfig(model_type='cnn1d', use_mim=True, name='CNN1D-MIM', seq_len=5),
        ]
        
        all_results = []
        
        # 逐个运行模型
        for model_config in model_configs:
            try:
                results = self.run_model_experiment(model_config)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"模型 {model_config.name} 实验失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 保存结果
        df_results = pd.DataFrame(all_results)
        result_file = self.seed_dir / "results.csv"
        df_results.to_csv(result_file, index=False)
        self.logger.info(f"结果已保存: {result_file}")
        
        # 生成图表
        self._generate_plots(df_results)
        
        self.logger.info(f"小实验完成 - Seed: {self.seed}")
        return df_results
    
    def run_model_experiment(self, model_config: ModelConfig) -> List[Dict]:
        """运行单个模型的实验"""
        model_name = model_config.name
        use_mim = model_config.use_mim
        
        self.logger.info(f"-" * 60)
        self.logger.info(f"训练模型: {model_name} (MIM={use_mim})")
        self.logger.info(f"-" * 60)
        
        # 训练模型
        model, training_time = self._train_model(model_config)
        
        # 评估所有缺失率
        results = []
        for missing_rate in self.config.missing_rates:
            metrics = self._evaluate_model(model, model_config, missing_rate)
            metrics.update({
                'seed': self.seed,
                'model_name': model_name,
                'model_type': model_config.model_type,
                'use_mim': use_mim,
                'missing_rate': missing_rate,
                'training_time': training_time
            })
            results.append(metrics)
            
            self.logger.info(f"  Missing Rate {missing_rate:.1f}: "
                           f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                           f"R2={metrics['r2']:.4f}")
        
        return results
    
    def _train_model(self, model_config: ModelConfig):
        """训练模型 - 区分Baseline和MIM策略"""
        from torch.utils.data import DataLoader
        from src.trainers.neural_network_trainer import NeuralNetworkTrainer
        
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        
        # 创建模型
        config_dict = {k: v for k, v in model_config.__dict__.items() 
                      if k not in ['model_type', 'use_mim', 'name']}
        model = ModelFactory.create_model(
            model_type,
            input_dim=16,
            use_mim=use_mim,
            device=self.config.device,
            **config_dict
        )
        
        param_count = ModelFactory.count_parameters(model)
        self.logger.info(f"  模型参数量: {param_count:,}")
        
        # 训练策略区分
        if use_mim:
            # MIM策略: 拼接多缺失率数据
            train_data, val_data = self._prepare_mim_data(model_config)
        else:
            # Baseline策略: 只用完整数据
            train_data, val_data = self._prepare_baseline_data(model_config)
        
        # 训练 - PyTorch模型
        start_time = time.time()
        
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size)
        
        trainer = NeuralNetworkTrainer(model.model if hasattr(model, 'model') else model, self.config.device)
        trainer.train(
            train_loader, val_loader,
            epochs=self.config.epochs,
            lr=self.config.lr,
            patience=self.config.early_stopping_patience
        )
        
        training_time = time.time() - start_time
        
        # 保存模型
        model_path = self.model_dir / f"{model_config.name}.pth"
        model.save(str(model_path))
        
        return model, training_time
    
    def _prepare_baseline_data(self, model_config: ModelConfig):
        """准备Baseline训练数据 - 只用完整数据"""
        model_type = model_config.model_type
        use_mim = model_config.use_mim  # False for baseline
        
        if model_type in ['lstm', 'gru', 'cnn1d']:
            # 序列模型
            train_dataset = SequenceDataset(
                self.data['X_train'], self.data['y_train'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,  # 完整数据
                use_mim=False,  # Baseline不用MIM
                seed=self.seed
            )
            val_dataset = SequenceDataset(
                self.data['X_val'], self.data['y_val'],
                seq_len=model_config.seq_len,
                missing_rate=0.0,
                use_mim=False,
                seed=self.seed
            )
        else:
            # 非序列模型
            train_dataset = BatteryDataset(
                self.data['X_train'], self.data['y_train'],
                missing_rate=0.0,
                use_mim=False,
                seed=self.seed
            )
            val_dataset = BatteryDataset(
                self.data['X_val'], self.data['y_val'],
                missing_rate=0.0,
                use_mim=False,
                seed=self.seed
            )
        
        return train_dataset, val_dataset
    
    def _prepare_mim_data(self, model_config: ModelConfig):
        """准备MIM训练数据 - 拼接多缺失率数据"""
        model_type = model_config.model_type
        seq_len = model_config.seq_len
        
        # 验证集使用50%缺失率（固定）
        if model_type in ['lstm', 'gru', 'cnn1d']:
            # 序列模型: 构建3D序列数据 [samples, seq_len, features]
            return self._prepare_mim_sequence_data(seq_len)
        else:
            # 非序列模型: 构建2D数据 [samples, features]
            return self._prepare_mim_flat_data()
    
    def _prepare_mim_flat_data(self):
        """为非序列模型准备MIM数据 (2D)"""
        X_train_list = []
        y_train_list = []
        
        for i, mr in enumerate(self.config.mim_training_rates):
            X_copy = self.data['X_train'].copy()
            y_copy = self.data['y_train'].copy()
            
            # 施加缺失
            np.random.seed(self.seed + i)
            mask = np.random.rand(*X_copy.shape) > mr
            X_missing = np.where(mask, X_copy, 0.0)
            missing_indicators = (~mask).astype(np.float32)
            
            # 拼接特征和缺失指示器
            X_combined = np.concatenate([X_missing, missing_indicators], axis=1)
            
            X_train_list.append(X_combined)
            y_train_list.append(y_copy)
        
        # 拼接所有副本
        X_train_mim = np.vstack(X_train_list)
        y_train_mim = np.concatenate(y_train_list)
        
        self.logger.info(f"  MIM训练集大小: {X_train_mim.shape} (10×原大小)")
        
        # 验证集
        X_val = self.data['X_val'].copy()
        y_val = self.data['y_val'].copy()
        
        np.random.seed(self.seed)
        mask = np.random.rand(*X_val.shape) > 0.5
        X_val_missing = np.where(mask, X_val, 0.0)
        val_indicators = (~mask).astype(np.float32)
        X_val = np.concatenate([X_val_missing, val_indicators], axis=1)
        
        # 创建TensorDataset
        from torch.utils.data import TensorDataset
        import torch
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_mim, dtype=torch.float32),
            torch.tensor(y_train_mim, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        return train_dataset, val_dataset
    
    def _prepare_mim_sequence_data(self, seq_len: int):
        """为序列模型准备MIM数据 (3D: [samples, seq_len, features])"""
        from torch.utils.data import TensorDataset
        import torch
        
        all_sequences = []
        all_targets = []
        
        # 对每个缺失率构建序列数据
        for i, mr in enumerate(self.config.mim_training_rates):
            X_copy = self.data['X_train'].copy()
            y_copy = self.data['y_train'].copy()
            
            # 施加缺失
            np.random.seed(self.seed + i)
            mask = np.random.rand(*X_copy.shape) > mr
            X_missing = np.where(mask, X_copy, 0.0)
            missing_indicators = (~mask).astype(np.float32)
            
            # 拼接特征和缺失指示器 (32维)
            X_combined = np.concatenate([X_missing, missing_indicators], axis=1)
            
            # 构建序列 [n_samples-seq_len+1, seq_len, 32]
            sequences, targets = self._create_sequences(X_combined, y_copy, seq_len)
            
            all_sequences.append(sequences)
            all_targets.append(targets)
        
        # 拼接所有缺失率版本的序列
        X_train_mim = np.vstack(all_sequences)
        y_train_mim = np.concatenate(all_targets)
        
        self.logger.info(f"  MIM序列训练集大小: {X_train_mim.shape} (10×原大小)")
        
        # 验证集 - 使用50%缺失率
        X_val = self.data['X_val'].copy()
        y_val = self.data['y_val'].copy()
        
        np.random.seed(self.seed)
        mask = np.random.rand(*X_val.shape) > 0.5
        X_val_missing = np.where(mask, X_val, 0.0)
        val_indicators = (~mask).astype(np.float32)
        X_val_combined = np.concatenate([X_val_missing, val_indicators], axis=1)
        
        X_val_seq, y_val_seq = self._create_sequences(X_val_combined, y_val, seq_len)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_mim, dtype=torch.float32),
            torch.tensor(y_train_mim, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_seq, dtype=torch.float32),
            torch.tensor(y_val_seq, dtype=torch.float32)
        )
        
        return train_dataset, val_dataset
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        """创建滑动窗口序列"""
        sequences = []
        targets = []
        
        for i in range(len(X) - seq_len + 1):
            seq = X[i:i+seq_len]
            target = y[i+seq_len-1]  # 使用序列最后一个时间步的标签
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def _evaluate_model(self, model, model_config: ModelConfig, missing_rate: float) -> Dict:
        """评估模型在特定缺失率下的性能"""
        model_type = model_config.model_type
        use_mim = model_config.use_mim
        
        evaluator = ModelEvaluator(self.config.device)
        
        # 准备测试数据
        if model_type in ['lstm', 'gru', 'cnn1d']:
            test_dataset = SequenceDataset(
                self.data['X_test'], self.data['y_test'],
                seq_len=model_config.seq_len,
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=self.seed
            )
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
            metrics, preds, targets = evaluator.evaluate(model, test_loader)
        else:
            test_dataset = BatteryDataset(
                self.data['X_test'], self.data['y_test'],
                missing_rate=missing_rate,
                use_mim=use_mim,
                seed=self.seed
            )
            
            if model_type == 'xgboost':
                # XGBoost需要numpy数组
                if use_mim:
                    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
                else:
                    X_test = test_dataset.X
                metrics, preds, targets = evaluator.evaluate_xgboost(
                    model.model if hasattr(model, 'model') else model,
                    X_test, test_dataset.y
                )
            else:
                from torch.utils.data import DataLoader
                test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
                metrics, preds, targets = evaluator.evaluate(model, test_loader)
        
        return metrics
    
    def _generate_plots(self, df: pd.DataFrame):
        """生成小实验图表"""
        try:
            from src.visualization.single_plots import SingleExperimentPlots
            plotter = SingleExperimentPlots(self.figure_dir)
            plotter.plot_all(df)
            self.logger.info(f"图表已保存到: {self.figure_dir}")
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
