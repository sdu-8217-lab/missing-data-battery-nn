"""Batch Experiment Runner V2 - 支持多次重复实验和自动绘图"""
import sys
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.pydantic_config import ExperimentConfig
from src.utils.logger_v2 import setup_logger


class BatchExperimentRunnerV2:
    """
    批量实验运行器 V2
    
    功能：
    1. 运行多次重复实验（不同随机种子）
    2. 自动聚合所有结果
    3. 生成统计图表（置信区间、改进率等）
    4. 支持中断恢复
    """
    
    def __init__(
        self,
        batch_name: str,
        n_repeats: int = 10,
        start_seed: int = 42,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        device: str = 'cpu',
        resume: bool = True,
    ):
        """
        初始化批量实验运行器
        
        Args:
            batch_name: 数据批次名称 (2C, 3C, R2.5, R3, RW, Sim_satellite)
            n_repeats: 重复实验次数
            start_seed: 起始随机种子
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
            early_stopping_patience: 早停耐心值
            device: 计算设备
            resume: 是否从检查点恢复
        """
        self.batch_name = batch_name
        self.n_repeats = n_repeats
        self.start_seed = start_seed
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.resume = resume
        
        # 实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"./experiments_v2/{batch_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name="batch_experiment",
            log_file=str(self.exp_dir / "batch_experiment.log"),
            level="INFO"
        )
        
        # 记录已完成种子的检查点文件
        self.checkpoint_file = self.exp_dir / "completed_seeds.txt"
        self.completed_seeds = self._load_checkpoint()
        
    def _load_checkpoint(self) -> set:
        """加载已完成的种子检查点"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()
    
    def _save_checkpoint(self, seed: int):
        """保存已完成的种子"""
        with open(self.checkpoint_file, 'a') as f:
            f.write(f"{seed}\n")
        self.completed_seeds.add(seed)
    
    def _run_single_seed(self, seed: int) -> Optional[pd.DataFrame]:
        """运行单个小实验"""
        try:
            # 延迟导入避免循环依赖
            from run_full_experiment_v2 import run_single_experiment
            
            self.logger.info(f"开始运行种子 {seed}")
            start_time = time.time()
            
            results_df = run_single_experiment(
                batch_name=self.batch_name,
                seed=seed,
                epochs=self.epochs
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"种子 {seed} 完成，耗时: {elapsed:.1f}s")
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"种子 {seed} 运行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run(self) -> Path:
        """
        运行批量实验
        
        Returns:
            实验目录路径
        """
        self.logger.info("=" * 70)
        self.logger.info("批量实验启动")
        self.logger.info("=" * 70)
        self.logger.info(f"批次: {self.batch_name}")
        self.logger.info(f"重复次数: {self.n_repeats}")
        self.logger.info(f"种子范围: {self.start_seed} ~ {self.start_seed + self.n_repeats - 1}")
        self.logger.info(f"训练轮数: {self.epochs}")
        self.logger.info(f"恢复模式: {self.resume}")
        self.logger.info(f"实验目录: {self.exp_dir}")
        self.logger.info("=" * 70)
        
        # 生成种子列表
        all_seeds = list(range(self.start_seed, self.start_seed + self.n_repeats))
        
        # 如果恢复模式，过滤已完成的种子
        if self.resume:
            pending_seeds = [s for s in all_seeds if s not in self.completed_seeds]
            if pending_seeds != all_seeds:
                self.logger.info(f"恢复模式: 跳过 {len(all_seeds) - len(pending_seeds)} 个已完成种子")
                all_seeds = pending_seeds
        
        # 运行所有种子
        all_results = []
        for i, seed in enumerate(all_seeds, 1):
            self.logger.info(f"\n[{i}/{len(all_seeds)}] 运行种子 {seed}")
            
            results_df = self._run_single_seed(seed)
            
            if results_df is not None:
                all_results.append(results_df)
                self._save_checkpoint(seed)
            
            # 每完成10%保存一次中间结果
            if i % max(1, len(all_seeds) // 10) == 0:
                self._save_intermediate_results(all_results)
        
        # 保存最终结果
        if all_results:
            self._save_final_results(all_results)
            self._generate_plots()
        
        self.logger.info("=" * 70)
        self.logger.info("批量实验完成!")
        self.logger.info(f"实验目录: {self.exp_dir}")
        self.logger.info("=" * 70)
        
        return self.exp_dir
    
    def _save_intermediate_results(self, all_results: List[pd.DataFrame]):
        """保存中间结果"""
        if not all_results:
            return
        
        combined = pd.concat(all_results, ignore_index=True)
        intermediate_file = self.exp_dir / "aggregate" / "results_intermediate.csv"
        intermediate_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(intermediate_file, index=False)
        self.logger.info(f"中间结果已保存: {intermediate_file}")
    
    def _save_final_results(self, all_results: List[pd.DataFrame]):
        """保存最终结果"""
        combined = pd.concat(all_results, ignore_index=True)
        
        # 保存到 aggregate 目录
        aggregate_dir = self.exp_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        final_file = aggregate_dir / "results_all.csv"
        combined.to_csv(final_file, index=False)
        self.logger.info(f"最终结果已保存: {final_file}")
        
        # 生成统计摘要
        self._generate_summary(combined, aggregate_dir)
    
    def _generate_summary(self, df: pd.DataFrame, output_dir: Path):
        """生成统计摘要"""
        summary = []
        
        for (model, use_mim, missing_rate), group in df.groupby(['model', 'use_mim', 'missing_rate']):
            summary.append({
                'model': model,
                'use_mim': use_mim,
                'missing_rate': missing_rate,
                'mae_mean': group['mae'].mean(),
                'mae_std': group['mae'].std(),
                'rmse_mean': group['rmse'].mean(),
                'rmse_std': group['rmse'].std(),
                'r2_mean': group['r2'].mean(),
                'r2_std': group['r2'].std(),
                'n_seeds': len(group)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_file = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"统计摘要已保存: {summary_file}")
    
    def _generate_plots(self):
        """生成批量实验图表"""
        try:
            from src.visualization.batch_plots import BatchExperimentPlots
            
            aggregate_dir = self.exp_dir / "aggregate"
            results_file = aggregate_dir / "results_all.csv"
            
            if not results_file.exists():
                self.logger.warning("结果文件不存在，跳过绘图")
                return
            
            df = pd.read_csv(results_file)
            
            # 创建图表目录
            figures_dir = aggregate_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成所有图表
            plotter = BatchExperimentPlots(figures_dir)
            plotter.plot_all(df)
            
            self.logger.info(f"批量实验图表已保存: {figures_dir}")
            
        except Exception as e:
            self.logger.error(f"批量实验图表生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行批量实验 V2')
    parser.add_argument('--batch', type=str, default='3C', 
                       choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'],
                       help='数据批次')
    parser.add_argument('--n-repeats', type=int, default=10, help='重复次数')
    parser.add_argument('--start-seed', type=int, default=42, help='起始随机种子')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复')
    
    args = parser.parse_args()
    
    runner = BatchExperimentRunnerV2(
        batch_name=args.batch,
        n_repeats=args.n_repeats,
        start_seed=args.start_seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        device=args.device,
        resume=args.resume
    )
    
    exp_dir = runner.run()
    print(f"\n实验完成! 结果保存在: {exp_dir}")


if __name__ == '__main__':
    main()
