"""细粒度架构搜索 - 在当前最佳配置附近密集探索"""
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.experiments.architecture_search import (
    ArchitectureSearchRunner, ArchitectureConfig
)


class FineGrainedSearchSpace:
    """细粒度搜索空间定义 - 基于当前最佳配置扩展"""
    
    @staticmethod
    def get_mlp_search_space() -> List[Dict]:
        """MLP细粒度搜索空间"""
        configs = []
        
        # 单层隐藏层精细搜索 [32, 48, 64, 80, 96, 128, 160, 192, 256]
        for hidden in [48, 80, 96, 160, 192]:
            for dropout in [0.0, 0.1, 0.2, 0.3]:
                configs.append({'hidden_layers': [hidden], 'dropout': dropout})
        
        # 双层精细搜索
        for h1 in [48, 64, 80, 96, 128]:
            for h2 in [32, 48, 64, 80]:
                if h2 < h1:  # 确保递减
                    for dropout in [0.1, 0.2, 0.3]:
                        configs.append({'hidden_layers': [h1, h2], 'dropout': dropout})
        
        # 三层精细搜索（当前最佳附近）
        for h1 in [128, 160, 192, 256]:
            for h2 in [64, 80, 96, 128]:
                for h3 in [32, 48, 64]:
                    if h2 < h1 and h3 < h2:
                        for dropout in [0.2, 0.3, 0.5]:
                            configs.append({'hidden_layers': [h1, h2, h3], 'dropout': dropout})
        
        return configs
    
    @staticmethod
    def get_lstm_search_space() -> List[Dict]:
        """LSTM细粒度搜索空间"""
        configs = []
        
        # hidden_size 精细搜索 [16, 24, 32, 48, 64, 80, 96, 128]
        for hidden in [24, 48, 80, 96]:
            for layers in [1, 2, 3]:
                for dropout in [0.0, 0.1, 0.2, 0.3, 0.5]:
                    configs.append({
                        'hidden_size': hidden,
                        'num_layers': layers,
                        'dropout': dropout
                    })
        
        # 当前最佳附近 (hidden=64, layers=2) 更密集搜索
        for hidden in [56, 64, 72, 80]:
            for layers in [1, 2, 3]:
                for dropout in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    configs.append({
                        'hidden_size': hidden,
                        'num_layers': layers,
                        'dropout': dropout
                    })
        
        return configs
    
    @staticmethod
    def get_gru_search_space() -> List[Dict]:
        """GRU细粒度搜索空间"""
        configs = []
        
        # 当前最佳附近 (hidden=128, layers=2) 密集搜索
        for hidden in [96, 112, 128, 144, 160]:
            for layers in [1, 2, 3]:
                for dropout in [0.1, 0.15, 0.2, 0.25, 0.3]:
                    configs.append({
                        'hidden_size': hidden,
                        'num_layers': layers,
                        'dropout': dropout
                    })
        
        # 小参数高效区域
        for hidden in [48, 64, 80, 96]:
            for layers in [1, 2]:
                for dropout in [0.0, 0.1, 0.2]:
                    configs.append({
                        'hidden_size': hidden,
                        'num_layers': layers,
                        'dropout': dropout
                    })
        
        return configs
    
    @staticmethod
    def get_xgboost_search_space() -> List[Dict]:
        """XGBoost细粒度搜索空间"""
        configs = []
        
        # 学习率精细搜索
        for lr in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
            for depth in [3, 4, 5, 6, 7]:
                for n_est in [50, 100, 150, 200]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr
                    })
        
        # 小学习率+多树组合
        for lr in [0.01, 0.05]:
            for depth in [3, 5, 7]:
                for n_est in [300, 500, 800]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr
                    })
        
        return configs
    
    @staticmethod
    def get_cnn_search_space() -> List[Dict]:
        """CNN1D细粒度搜索空间"""
        configs = []
        
        # 单卷积层精细搜索
        for ch in [32, 48, 64, 80, 96, 128]:
            for kernel in [2, 3, 4, 5]:
                for dropout in [0.0, 0.1, 0.2, 0.3]:
                    configs.append({
                        'channels': [ch],
                        'kernel_size': kernel,
                        'dropout': dropout
                    })
        
        # 双卷积层精细搜索（当前最佳 [128,64] 附近）
        for ch1 in [96, 112, 128, 144, 160]:
            for ch2 in [48, 64, 80, 96]:
                for kernel in [2, 3, 4]:
                    for dropout in [0.1, 0.2, 0.3]:
                        configs.append({
                            'channels': [ch1, ch2],
                            'kernel_size': kernel,
                            'dropout': dropout
                        })
        
        # 三层卷积
        for ch1 in [128, 160]:
            for ch2 in [64, 80, 96]:
                for ch3 in [32, 48, 64]:
                    for kernel in [3, 5]:
                        for dropout in [0.2, 0.3]:
                            configs.append({
                                'channels': [ch1, ch2, ch3],
                                'kernel_size': kernel,
                                'dropout': dropout
                            })
        
        return configs
    
    @staticmethod
    def get_search_space(model_type: str) -> List[Dict]:
        """获取指定模型的细粒度搜索空间"""
        spaces = {
            'mlp': FineGrainedSearchSpace.get_mlp_search_space(),
            'lstm': FineGrainedSearchSpace.get_lstm_search_space(),
            'gru': FineGrainedSearchSpace.get_gru_search_space(),
            'cnn1d': FineGrainedSearchSpace.get_cnn_search_space(),
            'xgboost': FineGrainedSearchSpace.get_xgboost_search_space(),
        }
        return spaces.get(model_type.lower(), [])


class FineGrainedSearchRunner(ArchitectureSearchRunner):
    """细粒度搜索运行器"""
    
    def __init__(self, model_type: str, batch_name: str = '3C', seed: int = 42):
        """
        Args:
            model_type: 指定模型类型 'mlp'/'lstm'/'gru'/'cnn1d'/'xgboost'
            batch_name: 数据批次
            seed: 随机种子
        """
        self.target_model = model_type.lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        super().__init__(
            batch_name=batch_name,
            seed=seed,
            stage='fine',
            exp_dir=Path(f"./experiments_v2/fine_grained_{model_type}_{batch_name}_{timestamp}")
        )
        
        # 细粒度配置：完整训练
        self.config = {
            'epochs': 50,  # 完整训练
            'missing_rates': [0.1, 0.5, 0.9],  # 低中高缺失率
            'patience': 10,
            'prune_threshold_mae': None,  # 不剪枝
            'prune_threshold_params': None,
        }
    
    def generate_architectures(self) -> List[ArchitectureConfig]:
        """生成细粒度候选架构"""
        architectures = []
        configs = FineGrainedSearchSpace.get_search_space(self.target_model)
        
        for i, config in enumerate(configs):
            from src.experiments.architecture_search import ArchitectureSearchSpace
            param_count = ArchitectureSearchSpace.estimate_params(
                self.target_model, config
            )
            
            arch = ArchitectureConfig(
                model_type=self.target_model,
                config_name=f"{self.target_model}_fine_{i+1}",
                config=config,
                param_count=param_count
            )
            architectures.append(arch)
        
        return architectures
    
    def run(self, start_idx: int = 0) -> pd.DataFrame:
        """运行细粒度搜索"""
        self.logger.info("=" * 70)
        self.logger.info(f"细粒度架构搜索 - {self.target_model.upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"完整搜索空间：{len(self.generate_architectures())} 个配置")
        self.logger.info(f"训练：50 epochs，缺失率 [0.1, 0.5, 0.9]")
        self.logger.info("=" * 70)
        
        return super().run(start_idx=start_idx)


def run_single_model_search(model_type: str, batch_name: str = '3C', seed: int = 42):
    """运行单个模型的细粒度搜索"""
    runner = FineGrainedSearchRunner(model_type, batch_name, seed)
    return runner.run()


def run_all_models_fine_grained(batch_name: str = '3C', seed: int = 42):
    """运行所有模型的细粒度搜索"""
    results = {}
    
    for model_type in ['mlp', 'lstm', 'gru', 'cnn1d', 'xgboost']:
        print(f"\n{'='*70}")
        print(f"开始 {model_type.upper()} 细粒度搜索")
        print(f"{'='*70}")
        
        runner = FineGrainedSearchRunner(model_type, batch_name, seed)
        df = runner.run()
        results[model_type] = df
        
        # 打印该模型最佳结果
        if len(df) > 0:
            best = df.loc[df['mae_mean'].idxmin()]
            print(f"\n{model_type.upper()} 最佳架构:")
            print(f"  配置: {best['config_name']}")
            print(f"  MAE: {best['mae_mean']:.4f}")
            print(f"  参数量: {best['param_count']:,}")
            print(f"  配置详情: {best['config']}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='细粒度架构搜索')
    parser.add_argument('--model', choices=['mlp', 'lstm', 'gru', 'cnn1d', 'all'],
                       default='all', help='要搜索的模型类型')
    parser.add_argument('--batch', default='3C', help='数据批次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--start-idx', type=int, default=0, help='起始索引')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        run_all_models_fine_grained(args.batch, args.seed)
    else:
        run_single_model_search(args.model, args.batch, args.seed)
