#!/usr/bin/env python3
"""
基于CSV配置文件的架构搜索运行器
- 从CSV读取配置
- 运行实验（无缺失数据）
- 将结果写回CSV
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

from loguru import logger
from src.config.pydantic_config import ExperimentConfig
from src.utils.logger_v2 import setup_logger
from src.data.dataset_loader import XJTUDatasetLoader
from src.data.datasets import BatteryDataset, SequenceDataset
from src.models.model_factory import ModelFactory
from src.trainers.lightning_trainer import LightningTrainer, SOHLightningModule
from src.evaluators.model_evaluator import ModelEvaluator
from torch.utils.data import DataLoader
import torch


class CSVBasedSearchRunner:
    """基于CSV配置的搜索运行器"""
    
    def __init__(self, model_type: str, csv_path: str, batch_name: str = '3C', seed: int = 42):
        """
        Args:
            model_type: 模型类型 (mlp/lstm/gru/cnn1d/xgboost)
            csv_path: 配置文件CSV路径
            batch_name: 数据批次
            seed: 随机种子
        """
        self.model_type = model_type.lower()
        self.csv_path = Path(csv_path)
        self.batch_name = batch_name
        self.seed = seed
        
        # 确保CSV存在
        if not self.csv_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {csv_path}")
        
        # 读取配置
        self.df = pd.read_csv(csv_path)
        
        # 实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"./experiments_v2/csv_search_{model_type}_{batch_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name=f"csv_search_{model_type}",
            log_file=str(self.exp_dir / "search.log"),
            level="INFO"
        )
        
        # 加载数据（一次加载，重复使用）
        self._load_data()
    
    def _load_data(self):
        """加载数据集（无缺失）"""
        self.logger.info("加载数据...")
        loader = XJTUDatasetLoader(
            data_dir='./data/XJTU data',
            batch=self.batch_name
        )
        
        self.data = loader.prepare_data(
            feature_cols=ExperimentConfig().feature_cols,
            target_col='capacity',
            test_size=0.25,
            val_size=0.25,
            random_seed=self.seed
        )
        
        self.logger.info(f"数据加载完成:")
        self.logger.info(f"  训练集: {self.data['X_train'].shape}")
        self.logger.info(f"  验证集: {self.data['X_val'].shape}")
        self.logger.info(f"  测试集: {self.data['X_test'].shape}")
    
    def parse_config(self, row: pd.Series) -> dict:
        """从CSV行解析配置"""
        if self.model_type == 'mlp':
            hidden_str = row['hidden_config'].strip('[]')
            hidden_layers = [int(x) for x in hidden_str.split(',')]
            return {
                'hidden_layers': hidden_layers,
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type in ['lstm', 'gru']:
            config_str = row['hidden_config']  # e.g., "h=64,l=2"
            parts = config_str.split(',')
            hidden = int(parts[0].split('=')[1])
            layers = int(parts[1].split('=')[1])
            return {
                'hidden_size': hidden,
                'num_layers': layers,
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type == 'cnn1d':
            channels_str = row['hidden_config'].strip('[]')
            channels = [int(x) for x in channels_str.split(',')]
            return {
                'channels': channels,
                'kernel_size': int(row['kernel']),
                'dropout': float(row['dropout'])
            }
        
        elif self.model_type == 'xgboost':
            config_str = row['hidden_config']  # e.g., "n=200,d=5,lr=0.1"
            parts = config_str.split(',')
            n_est = int(parts[0].split('=')[1])
            depth = int(parts[1].split('=')[1])
            lr = float(parts[2].split('=')[1])
            return {
                'n_estimators': n_est,
                'max_depth': depth,
                'learning_rate': lr
            }
        
        return {}
    
    def run_single_config(self, idx: int) -> dict:
        """运行单个配置"""
        row = self.df.iloc[idx]
        config_id = row['config_id']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"运行配置: {config_id}")
        self.logger.info(f"层类型: {row['layer_type']}")
        self.logger.info(f"配置: {row['hidden_config']}")
        self.logger.info(f"估算参数量: {row['estimated_params']:,}")
        
        # 解析配置
        config = self.parse_config(row)
        
        try:
            # 创建模型
            model = ModelFactory.create_model(
                model_type=self.model_type,
                input_dim=16,
                use_mim=False,  # 无缺失数据
                device='cpu',
                **config
            )
            
            # 实际参数量
            actual_params = ModelFactory.count_parameters(model)
            self.logger.info(f"实际参数量: {actual_params:,}")
            
            # 准备数据（无缺失）
            if self.model_type in ['lstm', 'gru', 'cnn1d']:
                train_dataset = SequenceDataset(
                    self.data['X_train'], self.data['y_train'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                val_dataset = SequenceDataset(
                    self.data['X_val'], self.data['y_val'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
                test_dataset = SequenceDataset(
                    self.data['X_test'], self.data['y_test'],
                    seq_len=5,
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
            else:
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
                test_dataset = BatteryDataset(
                    self.data['X_test'], self.data['y_test'],
                    missing_rate=0.0,
                    use_mim=False,
                    seed=self.seed
                )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # 训练
            start_time = time.time()
            
            if self.model_type == 'xgboost':
                # XGBoost特殊处理
                X_train = train_dataset.X_missing
                y_train = train_dataset.y
                X_val = val_dataset.X_missing
                y_val = val_dataset.y
                X_test = test_dataset.X_missing
                y_test = test_dataset.y
                
                model.fit(X_train, y_train, X_val, y_val)
            else:
                pl_module = SOHLightningModule(
                    model.model if hasattr(model, 'model') else model,
                    learning_rate=0.001
                )
                trainer = LightningTrainer(
                    max_epochs=50,
                    patience=10,
                    device='cpu'
                )
                trainer.train(pl_module, train_loader, val_loader)
            
            training_time = time.time() - start_time
            
            # 评估
            evaluator = ModelEvaluator('cpu')
            
            if self.model_type == 'xgboost':
                metrics, _, _ = evaluator.evaluate_xgboost(
                    model.model if hasattr(model, 'model') else model,
                    X_test, y_test
                )
            else:
                metrics, _, _ = evaluator.evaluate(model, test_loader)
            
            results = {
                'actual_params': actual_params,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'training_time': training_time,
                'status': 'completed'
            }
            
            self.logger.info(f"结果: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"配置 {config_id} 运行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'actual_params': '',
                'mae': '',
                'rmse': '',
                'r2': '',
                'training_time': '',
                'status': 'failed'
            }
    
    def run(self, start_idx: int = 0, end_idx: int = None):
        """
        运行搜索
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引（None表示全部）
        """
        if end_idx is None:
            end_idx = len(self.df)
        
        self.logger.info("=" * 70)
        self.logger.info(f"CSV架构搜索 - {self.model_type.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"总配置数: {len(self.df)}")
        self.logger.info(f"运行范围: {start_idx} - {end_idx}")
        self.logger.info(f"数据批次: {self.batch_name}")
        self.logger.info("=" * 70)
        
        completed = 0
        failed = 0
        
        for idx in range(start_idx, end_idx):
            # 检查是否已完成
            if self.df.iloc[idx]['status'] == 'completed':
                self.logger.info(f"跳过已完成配置: {self.df.iloc[idx]['config_id']}")
                continue
            
            # 更新状态为运行中
            self.df.at[idx, 'status'] = 'running'
            self._save_csv()
            
            # 运行实验
            results = self.run_single_config(idx)
            
            # 更新结果
            for key, value in results.items():
                self.df.at[idx, key] = value
            
            # 保存CSV
            self._save_csv()
            
            if results['status'] == 'completed':
                completed += 1
            else:
                failed += 1
            
            # 每5个保存一次中间结果
            if idx % 5 == 0:
                self._save_intermediate_results()
        
        self.logger.info("=" * 70)
        self.logger.info("搜索完成!")
        self.logger.info(f"完成: {completed}, 失败: {failed}")
        self.logger.info(f"结果保存: {self.csv_path}")
        self.logger.info("=" * 70)
    
    def _save_csv(self):
        """保存CSV"""
        self.df.to_csv(self.csv_path, index=False)
    
    def _save_intermediate_results(self):
        """保存中间结果副本"""
        backup_path = self.exp_dir / f"results_backup_{datetime.now().strftime('%H%M%S')}.csv"
        self.df.to_csv(backup_path, index=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='基于CSV的架构搜索')
    parser.add_argument('--model', choices=['mlp', 'lstm', 'gru', 'cnn1d'],
                       help='模型类型')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件CSV路径（可选，默认使用configs/{model}_configs.csv）')
    parser.add_argument('--batch', default='3C', help='数据批次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--start', type=int, default=0, help='起始索引')
    parser.add_argument('--end', type=int, default=None, help='结束索引')
    
    args = parser.parse_args()
    
    # 优先使用--config指定的路径，否则使用默认路径
    if args.config:
        csv_path = args.config
    else:
        csv_path = f'configs/{args.model}_configs.csv'
    
    runner = CSVBasedSearchRunner(
        model_type=args.model,
        csv_path=csv_path,
        batch_name=args.batch,
        seed=args.seed
    )
    
    runner.run(start_idx=args.start, end_idx=args.end)


if __name__ == '__main__':
    main()
