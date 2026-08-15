"""模型架构搜索实验 - 寻找最佳参数量与架构配置"""
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from src.config.pydantic_config import ExperimentConfig
from src.utils.logger_v2 import setup_logger


@dataclass
class ArchitectureConfig:
    """架构配置"""
    model_type: str
    config_name: str
    config: Dict[str, Any]
    param_count: int = 0
    
    def to_dict(self):
        return asdict(self)


class ArchitectureSearchSpace:
    """架构搜索空间定义"""
    
    # MLP 搜索空间
    MLP_CONFIGS = [
        {"hidden_layers": [64], "dropout": 0.0},
        {"hidden_layers": [128], "dropout": 0.0},
        {"hidden_layers": [256], "dropout": 0.0},
        {"hidden_layers": [64, 32], "dropout": 0.2},
        {"hidden_layers": [128, 64], "dropout": 0.2},
        {"hidden_layers": [256, 128], "dropout": 0.2},
        {"hidden_layers": [128, 64, 32], "dropout": 0.2},
        {"hidden_layers": [256, 128, 64], "dropout": 0.5},
    ]
    
    # LSTM/GRU 搜索空间
    RNN_CONFIGS = [
        {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        {"hidden_size": 32, "num_layers": 1, "dropout": 0.0},
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
        {"hidden_size": 128, "num_layers": 1, "dropout": 0.2},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
        {"hidden_size": 128, "num_layers": 3, "dropout": 0.5},
    ]
    
    # CNN1D 搜索空间
    CNN_CONFIGS = [
        {"channels": [32], "kernel_size": 3, "dropout": 0.0},
        {"channels": [64], "kernel_size": 3, "dropout": 0.0},
        {"channels": [32, 16], "kernel_size": 3, "dropout": 0.2},
        {"channels": [64, 32], "kernel_size": 3, "dropout": 0.2},
        {"channels": [128, 64], "kernel_size": 3, "dropout": 0.2},
        {"channels": [128, 64, 32], "kernel_size": 5, "dropout": 0.2},
    ]
    

    @classmethod
    def get_search_space(cls, model_type: str) -> List[Dict]:
        """获取指定模型的搜索空间"""
        spaces = {
            'mlp': cls.MLP_CONFIGS,
            'lstm': cls.RNN_CONFIGS,
            'gru': cls.RNN_CONFIGS,
            'cnn1d': cls.CNN_CONFIGS,
        }
        return spaces.get(model_type.lower(), [])
    
    @classmethod
    def estimate_params(cls, model_type: str, config: Dict, use_mim: bool = True) -> int:
        """
        精确计算参数量
        
        通过实际创建模型并使用PyTorch的numel()方法
        """
        from src.models.model_factory import ModelFactory
        
        try:
            model = ModelFactory.create_model(
                model_type=model_type,
                input_dim=16,
                use_mim=use_mim,
                device='cpu',
                **config
            )
            
            # 获取实际模型
            actual_model = model.model if hasattr(model, 'model') else model
            
            # 精确计算参数量
            total_params = sum(p.numel() for p in actual_model.parameters())
            return total_params
            
        except Exception as e:
            # 如果创建失败，使用粗略估算
            logger.warning(f"精确计算参数量失败: {e}，使用估算值")
            return cls._estimate_params_approximate(model_type, config)
    
    @classmethod
    def _estimate_params_approximate(cls, model_type: str, config: Dict) -> int:
        """粗略估算参数量（备用方法）"""
        input_dim = 16
        output_dim = 1
        
        if model_type == 'mlp':
            layers = [input_dim] + config['hidden_layers'] + [output_dim]
            # 包含bias: 每层的 (输入*输出) + 输出
            return sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        
        elif model_type in ['lstm', 'gru']:
            hidden = config['hidden_size']
            num_layers = config['num_layers']
            # LSTM: 4 * (input_size * hidden_size + hidden_size^2 + hidden_size)
            params_per_layer = 4 * (input_dim * hidden + hidden * hidden + hidden)
            return params_per_layer * num_layers + hidden * output_dim
        
        elif model_type == 'cnn1d':
            channels = [input_dim] + config['channels']
            kernel = config['kernel_size']
            # 简化的CNN参数估算
            conv_params = sum(channels[i] * channels[i+1] * kernel 
                            for i in range(len(channels)-1))
            fc_params = channels[-1] * output_dim
            return conv_params + fc_params
        
        return 0


class ArchitectureSearchRunner:
    """架构搜索运行器"""
    
    def __init__(
        self,
        batch_name: str = '3C',
        seed: int = 42,
        stage: str = 'coarse',  # 'coarse' or 'fine'
        exp_dir: Optional[Path] = None,
    ):
        self.batch_name = batch_name
        self.seed = seed
        self.stage = stage
        
        # 实验目录
        if exp_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exp_dir = Path(f"./experiments_v2/arch_search_{batch_name}_{stage}_{timestamp}")
        else:
            self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name="arch_search",
            log_file=str(self.exp_dir / "search.log"),
            level="INFO"
        )
        
        # 阶段配置
        self.config = self._get_stage_config(stage)
        
    def _get_stage_config(self, stage: str) -> Dict:
        """获取阶段配置"""
        configs = {
            'coarse': {
                'epochs': 10,
                'missing_rates': [0.5],  # 只测中等缺失
                'patience': 3,
                'prune_threshold_mae': 0.15,  # MAE > 0.15 剪枝
                'prune_threshold_params': 100000,  # 参数量 > 100K 剪枝
            },
            'fine': {
                'epochs': 50,
                'missing_rates': [0.1, 0.5, 0.9],  # 低中高
                'patience': 10,
                'prune_threshold_mae': None,  # 细粒度阶段不剪枝
                'prune_threshold_params': None,
            }
        }
        return configs.get(stage, configs['coarse'])
    
    def generate_architectures(self) -> List[ArchitectureConfig]:
        """生成所有候选架构"""
        architectures = []
        
        for model_type in ['mlp', 'lstm', 'gru', 'cnn1d']:
            configs = ArchitectureSearchSpace.get_search_space(model_type)
            
            for i, config in enumerate(configs):
                param_count = ArchitectureSearchSpace.estimate_params(model_type, config)
                
                arch = ArchitectureConfig(
                    model_type=model_type,
                    config_name=f"{model_type}_v{i+1}",
                    config=config,
                    param_count=param_count
                )
                architectures.append(arch)
        
        return architectures
    
    def run_single_architecture(
        self, 
        arch: ArchitectureConfig,
        data: Dict,
    ) -> Optional[Dict]:
        """运行单个架构实验"""
        from src.models.model_factory import ModelFactory
        from src.data.datasets import BatteryDataset, SequenceDataset, MIMDataset, SequenceMIMDataset
        from src.trainers.lightning_trainer import LightningTrainer, SOHLightningModule
        from torch.utils.data import DataLoader
        import torch
        
        self.logger.info(f"测试架构: {arch.config_name}")
        self.logger.info(f"  配置: {arch.config}")
        self.logger.info(f"  估算参数量: {arch.param_count:,}")
        
        model_type = arch.model_type
        use_mim = True  # 架构搜索聚焦MIM版本
        device = 'cpu'
        
        try:
            # 创建模型
            model = ModelFactory.create_model(
                model_type=model_type,
                input_dim=16,
                use_mim=use_mim,
                device=device,
                **arch.config
            )
            actual_params = ModelFactory.count_parameters(model)
            self.logger.info(f"  实际参数量: {actual_params:,}")
            
            # 准备数据
            if model_type in ['lstm', 'gru', 'cnn1d']:
                if use_mim:
                    train_dataset = SequenceMIMDataset(
                        data['X_train'], data['y_train'],
                        seq_len=5,
                        missing_rates=[0.0, 0.5],
                        use_mim=True,
                        base_seed=self.seed
                    )
                else:
                    train_dataset = SequenceDataset(
                        data['X_train'], data['y_train'],
                        seq_len=5,
                        missing_rate=0.0,
                        use_mim=False,
                        seed=self.seed
                    )
                val_dataset = SequenceDataset(
                    data['X_val'], data['y_val'],
                    seq_len=5,
                    missing_rate=0.5,
                    use_mim=use_mim,
                    seed=self.seed
                )
            else:
                if use_mim:
                    train_dataset = MIMDataset(
                        data['X_train'], data['y_train'],
                        missing_rates=[0.0, 0.5],
                        use_mim=True,
                        base_seed=self.seed
                    )
                else:
                    train_dataset = BatteryDataset(
                        data['X_train'], data['y_train'],
                        missing_rate=0.0,
                        use_mim=False,
                        seed=self.seed
                    )
                val_dataset = BatteryDataset(
                    data['X_val'], data['y_val'],
                    missing_rate=0.5,
                    use_mim=use_mim,
                    seed=self.seed
                )
            
            # 训练
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            start_time = time.time()
            

            pl_module = SOHLightningModule(
                model.model if hasattr(model, 'model') else model,
                learning_rate=0.001
            )
            trainer = LightningTrainer(
                max_epochs=self.config['epochs'],
                patience=self.config['patience'],
                device=device
            )
            history = trainer.train(pl_module, train_loader, val_loader)
            best_val_loss = min(history.get('val_loss', [float('inf')]))
            
            training_time = time.time() - start_time
            
            # 评估（获取实际性能）
            results = self._evaluate_architecture(
                model, model_type, use_mim, data, training_time, actual_params
            )
            
            # 阶段1：检查是否需要剪枝（基于实际MAE）
            if self.stage == 'coarse' and self._should_prune(arch, results):
                self.logger.warning(f"  架构 {arch.config_name} 被剪枝")
                return None
            
            return results
            
        except Exception as e:
            self.logger.error(f"架构 {arch.config_name} 运行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _should_prune(self, arch: ArchitectureConfig, results: Dict) -> bool:
        """判断是否剪枝（基于实际评估结果）"""
        mae_mean = results.get('mae_mean', float('inf'))
        
        threshold_mae = self.config.get('prune_threshold_mae')
        threshold_params = self.config.get('prune_threshold_params')
        
        if threshold_mae and mae_mean > threshold_mae:
            self.logger.info(f"  剪枝原因: MAE {mae_mean:.4f} > {threshold_mae}")
            return True
        
        if threshold_params and arch.param_count > threshold_params:
            self.logger.info(f"  剪枝原因: 参数量 {arch.param_count:,} > {threshold_params:,}")
            return True
        
        return False
    
    def _evaluate_architecture(
        self, 
        model, 
        model_type: str, 
        use_mim: bool, 
        data: Dict,
        training_time: float,
        param_count: int
    ) -> Dict:
        """评估架构性能"""
        from src.evaluators.model_evaluator import ModelEvaluator
        from src.data.datasets import BatteryDataset, SequenceDataset
        from torch.utils.data import DataLoader
        import torch
        import time as time_module
        
        evaluator = ModelEvaluator('cpu')
        device = 'cpu'
        
        results = {
            'model_type': model_type,
            'use_mim': use_mim,
            'param_count': param_count,
            'training_time': training_time,
        }
        
        # 评估多个缺失率
        mae_list = []
        for missing_rate in self.config['missing_rates']:
            if model_type in ['lstm', 'gru', 'cnn1d']:
                test_dataset = SequenceDataset(
                    data['X_test'], data['y_test'],
                    seq_len=5,
                    missing_rate=missing_rate,
                    use_mim=use_mim,
                    seed=self.seed
                )
                test_loader = DataLoader(test_dataset, batch_size=32)
                metrics, _, _ = evaluator.evaluate(model, test_loader)
            else:
                test_dataset = BatteryDataset(
                    data['X_test'], data['y_test'],
                    missing_rate=missing_rate,
                    use_mim=use_mim,
                    seed=self.seed
                )
                test_loader = DataLoader(test_dataset, batch_size=32)
                metrics, _, _ = evaluator.evaluate(model, test_loader)
            
            results[f'mae_{missing_rate}'] = metrics['mae']
            results[f'rmse_{missing_rate}'] = metrics['rmse']
            results[f'r2_{missing_rate}'] = metrics['r2']
            mae_list.append(metrics['mae'])
        
        results['mae_mean'] = np.mean(mae_list)
        results['mae_max'] = np.max(mae_list)
        
        # 推理时间（估算）
        sample_input = torch.randn(1, 32 if use_mim else 16)
        if model_type in ['lstm', 'gru', 'cnn1d']:
            sample_input = torch.randn(1, 5, 32 if use_mim else 16)
        
        start = time_module.time()
        for _ in range(100):
            _ = model.predict(sample_input.numpy())
        inference_time = (time_module.time() - start) / 100 * 1000  # ms
        results['inference_time_ms'] = inference_time
        
        return results
    
    def run(self, start_idx: int = 0) -> pd.DataFrame:
        """
        运行完整搜索
        
        Args:
            start_idx: 起始架构索引（用于从中断处恢复）
        """
        self.logger.info("=" * 70)
        self.logger.info(f"架构搜索实验 - {self.stage} 阶段")
        self.logger.info("=" * 70)
        self.logger.info(f"批次: {self.batch_name}")
        self.logger.info(f"种子: {self.seed}")
        self.logger.info(f"配置: {self.config}")
        self.logger.info("=" * 70)
        
        # 加载数据
        from src.data.dataset_loader import XJTUDatasetLoader
        loader = XJTUDatasetLoader(data_dir='./data/XJTU data', batch=self.batch_name)
        data = loader.prepare_data(
            feature_cols=ExperimentConfig().feature_cols,
            target_col='capacity',
            test_size=0.25,
            val_size=0.25,
            random_seed=self.seed
        )
        
        # 生成候选架构
        architectures = self.generate_architectures()
        self.logger.info(f"总共 {len(architectures)} 个候选架构")
        
        # 支持从指定索引开始
        if start_idx > 0:
            architectures = architectures[start_idx:]
            self.logger.info(f"从索引 {start_idx} 开始，剩余 {len(architectures)} 个架构")
        
        # 运行所有架构
        all_results = []
        pruned_count = 0
        
        for i, arch in enumerate(architectures, start_idx + 1):
            self.logger.info(f"\n[{i}/{start_idx + len(architectures)}] {arch.config_name}")
            
            result = self.run_single_architecture(arch, data)
            
            if result is None:
                pruned_count += 1
            else:
                result['config_name'] = arch.config_name
                result['config'] = json.dumps(arch.config)
                all_results.append(result)
                
                # 保存中间结果
                if i % 5 == 0:
                    self._save_results(all_results)
        
        # 保存最终结果
        df = self._save_results(all_results)
        
        self.logger.info("=" * 70)
        self.logger.info("搜索完成!")
        self.logger.info(f"测试架构: {len(architectures)}")
        self.logger.info(f"成功: {len(all_results)}")
        self.logger.info(f"剪枝: {pruned_count}")
        self.logger.info(f"结果保存: {self.exp_dir / 'results.csv'}")
        self.logger.info("=" * 70)
        
        return df
    
    def _save_results(self, results: List[Dict]) -> pd.DataFrame:
        """保存结果"""
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df.to_csv(self.exp_dir / "results.csv", index=False)
        
        return df


def run_coarse_search(batch_name: str = '3C', seed: int = 42, start_idx: int = 0) -> pd.DataFrame:
    """运行粗粒度搜索"""
    runner = ArchitectureSearchRunner(
        batch_name=batch_name,
        seed=seed,
        stage='coarse'
    )
    return runner.run(start_idx=start_idx)


def run_fine_search(
    coarse_results_path: str,
    batch_name: str = '3C',
    seed: int = 42
) -> pd.DataFrame:
    """运行细粒度搜索（基于粗粒度结果筛选）"""
    # 从粗粒度结果中筛选 top 架构
    coarse_df = pd.read_csv(coarse_results_path)
    
    # 每个模型选 top 3
    top_configs = []
    for model_type in coarse_df['model_type'].unique():
        model_df = coarse_df[coarse_df['model_type'] == model_type]
        top_3 = model_df.nsmallest(3, 'mae_mean')
        top_configs.extend(top_3['config_name'].tolist())
    
    print(f"细粒度阶段筛选出 {len(top_configs)} 个架构:")
    for cfg in top_configs:
        print(f"  - {cfg}")
    
    # 运行细粒度搜索
    runner = ArchitectureSearchRunner(
        batch_name=batch_name,
        seed=seed,
        stage='fine'
    )
    
    # 这里可以修改 runner 只运行选中的架构
    # 为简化，先运行完整搜索，实际使用时可优化
    return runner.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='模型架构搜索')
    parser.add_argument('--stage', choices=['coarse', 'fine'], default='coarse')
    parser.add_argument('--batch', default='3C', help='数据批次')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--start-idx', type=int, default=0, help='起始架构索引（用于恢复）')
    parser.add_argument('--coarse-results', help='粗粒度结果路径（细粒度阶段使用）')
    
    args = parser.parse_args()
    
    if args.stage == 'coarse':
        run_coarse_search(args.batch, args.seed, args.start_idx)
    else:
        if not args.coarse_results:
            print("错误: 细粒度阶段需要提供 --coarse-results 参数")
            exit(1)
        run_fine_search(args.coarse_results, args.batch, args.seed)
