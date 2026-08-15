"""大实验运行器 - 批量运行小实验"""
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from src.utils import setup_logger
from src.experiments.single_experiment import SingleExperimentRunner, SingleExperimentConfig
from src.experiments.checkpoint import CheckpointManager


class BatchExperimentRunner:
    """大实验运行器 - 管理多次小实验的执行"""
    
    def __init__(self,
                 batch_name: str = "3C",
                 n_repeats: int = 100,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 early_stopping_patience: int = 15,
                 device: str = 'cpu',
                 resume: bool = False):
        
        self.batch_name = batch_name
        self.n_repeats = n_repeats
        self.epochs = epochs
        self.resume = resume
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 实验目录: experiments/{batch_name}/{timestamp}/
        self.experiment_dir = Path("experiments") / batch_name / self.timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 聚合结果目录
        self.aggregate_dir = self.experiment_dir / "aggregate"
        self.aggregate_dir.mkdir(exist_ok=True)
        
        # 日志目录
        self.log_dir = self.experiment_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger("BatchExperiment", self.log_dir / "batch.log")
        
        # 生成种子列表
        self.seeds = list(range(42, 42 + n_repeats))
        
        # 检查点管理
        self.checkpoint_manager = CheckpointManager(self.experiment_dir, n_repeats)
        
        # 小实验共享配置
        self.base_config = {
            'batch_name': batch_name,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'early_stopping_patience': early_stopping_patience,
            'device': device
        }
        
    def run(self) -> Path:
        """运行大实验"""
        self.logger.info("=" * 70)
        self.logger.info("开始大实验")
        self.logger.info("=" * 70)
        self.logger.info(f"批次: {self.batch_name}")
        self.logger.info(f"重复次数: {self.n_repeats}")
        self.logger.info(f"随机种子: {self.seeds[0]} ~ {self.seeds[-1]}")
        self.logger.info(f"实验目录: {self.experiment_dir}")
        self.logger.info(f"恢复模式: {self.resume}")
        
        # 初始化或加载检查点
        if self.resume:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint is None:
                self.logger.warning("未找到检查点，创建新的实验")
                self.checkpoint_manager.create_checkpoint(self.batch_name, self.seeds)
            else:
                self.logger.info(f"恢复实验，已完成: {len(checkpoint.get('completed_seeds', []))}/{self.n_repeats}")
        else:
            self.checkpoint_manager.create_checkpoint(self.batch_name, self.seeds)
        
        # 获取待运行的种子
        if self.resume:
            remaining_seeds = self.checkpoint_manager.get_remaining_seeds()
        else:
            remaining_seeds = self.seeds
        
        if not remaining_seeds:
            self.logger.info("所有种子已完成，直接生成统计图表")
            self._generate_aggregate_plots()
            return self.experiment_dir
        
        self.logger.info(f"待运行种子数: {len(remaining_seeds)}")
        
        # 运行每个小实验
        start_time = time.time()
        
        for i, seed in enumerate(remaining_seeds):
            self.logger.info("")
            self.logger.info(f"进度: {i+1}/{len(remaining_seeds)} (总体: {self._get_overall_progress()})")
            self.logger.info(f"当前种子: {seed}")
            
            try:
                self._run_single_experiment(seed)
                self.checkpoint_manager.update_checkpoint(seed, "completed")
            except Exception as e:
                self.logger.error(f"种子 {seed} 实验失败: {e}")
                self.checkpoint_manager.update_checkpoint(seed, "failed")
                import traceback
                self.logger.error(traceback.format_exc())
                # 继续下一个，不中断整个实验
            
            # 定期保存聚合结果（每10个种子）
            if (i + 1) % 10 == 0:
                self._save_aggregate_results()
        
        elapsed = time.time() - start_time
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("大实验完成")
        self.logger.info(f"总耗时: {elapsed/3600:.2f} 小时")
        self.logger.info(f"实验目录: {self.experiment_dir}")
        self.logger.info("=" * 70)
        
        # 最终聚合结果
        self._save_aggregate_results()
        
        # 生成统计图表
        self._generate_aggregate_plots()
        
        # 标记完成
        self.checkpoint_manager.finalize_checkpoint()
        
        return self.experiment_dir
    
    def _run_single_experiment(self, seed: int):
        """运行单个小实验"""
        config = SingleExperimentConfig(seed=seed, **self.base_config)
        runner = SingleExperimentRunner(config, base_dir=self.experiment_dir)
        runner.run()
    
    def _get_overall_progress(self) -> str:
        """获取总体进度"""
        progress = self.checkpoint_manager.get_progress()
        return f"{progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)"
    
    def _save_aggregate_results(self):
        """合并所有小实验结果"""
        self.logger.info("保存聚合结果...")
        
        all_results = []
        for seed in self.seeds:
            result_file = self.experiment_dir / f"seed_{seed}" / "results.csv"
            if result_file.exists():
                df = pd.read_csv(result_file)
                all_results.append(df)
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            output_file = self.aggregate_dir / "results_all.csv"
            combined.to_csv(output_file, index=False)
            self.logger.info(f"聚合结果已保存: {output_file} ({len(combined)} 行)")
        else:
            self.logger.warning("没有可用的结果文件")
    
    def _generate_aggregate_plots(self):
        """生成统计性图表"""
        try:
            from src.visualization.batch_plots import BatchExperimentPlots
            
            result_file = self.aggregate_dir / "results_all.csv"
            if not result_file.exists():
                self.logger.warning("没有聚合结果，跳过绘图")
                return
            
            self.logger.info("生成统计性图表...")
            df = pd.read_csv(result_file)
            
            plotter = BatchExperimentPlots(self.aggregate_dir / "figures")
            plotter.plot_all(df)
            
            self.logger.info(f"统计图表已保存: {self.aggregate_dir / 'figures'}")
            
        except Exception as e:
            self.logger.error(f"生成统计图表失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
