"""
预实验主运行器

实现两阶段流程：
1. Stage 1: Optuna 搜索（单种子快速筛选）
2. Stage 2: 多种子验证（Top-K 候选的鲁棒性验证）
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.preexperiment.search_space import SearchSpace
from src.preexperiment.objective import (
    ObjectiveFactory, 
    create_model, 
    train_and_evaluate,
    count_parameters
)
from src.data.loader import load_dataset
from src.utils.seed_manager import set_seed


# 关闭 Optuna 的日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class BudgetConfig:
    """单个预算等级的配置"""
    budget: int
    n_trials: int
    epochs_search: int
    patience_search: int
    n_seeds_final: int
    epochs_final: int
    patience_final: int
    top_k: int = 3  # 进入第二阶段验证的候选数


class PreExperimentRunner:
    """
    预实验运行器
    
    为给定模型类型和参数量预算自动搜索最优架构
    
    特性：
    - 数据缓存：避免重复加载数据集
    - 并行搜索：支持多进程并行优化
    - 多样化探索：增加随机探索比例避免局部最优
    """
    
    # 类级数据缓存
    _data_cache = None
    _data_cache_key = None
    
    def __init__(
        self,
        model_type: str,
        budget_config: BudgetConfig,
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        output_dir: Path,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Args:
            model_type: 模型类型 (mlp, lstm, gru, cnn1d)
            budget_config: 预算配置
            data_config: 数据加载配置
            training_config: 训练超参数配置
            output_dir: 结果输出目录
            device: 计算设备
            seed: Optuna 搜索阶段的随机种子
        """
        self.model_type = model_type.lower()
        self.budget_config = budget_config
        self.data_config = data_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取搜索空间函数
        self.search_space_func = SearchSpace.get_suggest_func(self.model_type)
        if self.search_space_func is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 加载数据
        print(f"[PreExperiment] Loading XJTU dataset...")
        self.data = self._load_data()
        print(f"  Train: {len(self.data['X_train'])}, Val: {len(self.data['X_val'])}, Test: {len(self.data['X_test'])}")
    
    def _load_data(self) -> Dict[str, torch.Tensor]:
        """
        加载 XJTU 数据集（带缓存）
        
        使用类级缓存避免重复加载相同配置的数据
        """
        # 生成缓存键
        cache_key = f"{self.data_config.get('data', {}).get('dataset', 'xjtu')}_{self.data_config.get('data', {}).get('batch_id', '2C')}"
        
        # 检查缓存
        if PreExperimentRunner._data_cache is not None and PreExperimentRunner._data_cache_key == cache_key:
            print(f"[PreExperiment] Using cached dataset ({cache_key})")
            return PreExperimentRunner._data_cache
        
        # 加载数据
        from omegaconf import DictConfig
        cfg = DictConfig(self.data_config)
        data = load_dataset(cfg)
        
        # 存入缓存
        PreExperimentRunner._data_cache = data
        PreExperimentRunner._data_cache_key = cache_key
        
        return data
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整预实验流程
        
        Returns:
            包含最佳配置和性能指标的字典
        """
        print(f"\n{'='*70}")
        print(f"PreExperiment: {self.model_type.upper()} @ {self.budget_config.budget} params")
        print(f"{'='*70}")
        print(f"Stage 1: Optuna search ({self.budget_config.n_trials} trials)")
        print(f"Stage 2: Multi-seed validation ({self.budget_config.n_seeds_final} seeds)")
        
        # Stage 1: Optuna 搜索
        stage1_results = self._run_stage1_search()
        
        # Stage 2: 多种子验证
        final_results = self._run_stage2_validation(stage1_results)
        
        # 保存结果
        self._save_results(final_results)
        
        return final_results
    
    def _run_stage1_search(self) -> List[Dict[str, Any]]:
        """
        第一阶段：Optuna 智能搜索
        
        Returns:
            Top-K 候选列表
        """
        print(f"\n[Stage 1] Optuna Search")
        print(f"  Trials: {self.budget_config.n_trials}")
        print(f"  Epochs: {self.budget_config.epochs_search}, Patience: {self.budget_config.patience_search}")
        print(f"  Seed: {self.seed}")
        
        set_seed(self.seed)
        
        # 创建 Optuna study（优化配置）
        study_name = f"{self.model_type}_{self.budget_config.budget}"
        storage_path = f"sqlite:///{self.output_dir / f'{study_name}.db'}"
        
        # 计算随机探索次数：30% 用于多样性的随机探索
        n_startup = max(10, int(self.budget_config.n_trials * 0.3))
        
        print(f"  Sampler: TPE (multivariate)")
        print(f"  Random exploration: {n_startup} trials ({n_startup/self.budget_config.n_trials*100:.0f}%)")
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=n_startup,  # 30%随机探索
                multivariate=True,            # 考虑参数间相关性
            ),
            pruner=optuna.pruners.HyperbandPruner(),  # 更激进的剪枝
            load_if_exists=True,
        )
        
        # 准备训练配置
        search_training_config = {
            **self.training_config,
            "epochs": self.budget_config.epochs_search,
            "patience": self.budget_config.patience_search,
        }
        
        # 检测并行数：CPU模式使用多核，GPU模式保持串行避免显存冲突
        n_jobs = 4 if self.device == "cpu" else 1
        print(f"  Parallel jobs: {n_jobs}")
        
        # 收集 trial 结果
        trial_results = []
        
        # 创建目标函数
        # 注意：并行模式下不使用 trial_queue（避免线程安全问题）
        objective = ObjectiveFactory(
            model_type=self.model_type,
            input_dim=16,  # 无缺失 baseline
            param_budget=self.budget_config.budget,
            data=self.data,
            search_space_func=self.search_space_func,
            training_config=search_training_config,
            device=self.device,
            trial_queue=trial_results if n_jobs == 1 else None,
        )
        
        # 运行搜索（支持并行）
        start_time = time.time()
        
        study.optimize(
            objective, 
            n_trials=self.budget_config.n_trials, 
            show_progress_bar=True,
            n_jobs=n_jobs,  # 并行优化
        )
        search_time = time.time() - start_time
        
        print(f"  Search completed in {search_time:.1f}s")
        print(f"  Completed trials: {len(study.trials_dataframe())}")
        
        # 获取 Top-K 候选
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda t: t.value)
        
        top_k = self.budget_config.top_k
        top_trials = completed_trials[:top_k]
        
        print(f"\n  Top {top_k} candidates:")
        candidates = []
        for i, trial in enumerate(top_trials, 1):
            config = trial.user_attrs.get("model_config", {})
            n_params = trial.user_attrs.get("n_params", 0)
            test_mae = trial.user_attrs.get("test_mae", float("inf"))
            print(f"    {i}. Params: {n_params:,}, Test MAE: {test_mae:.4f}, Config: {config}")
            
            candidates.append({
                "rank": i,
                "trial_number": trial.number,
                "model_config": config,
                "n_params": n_params,
                "val_mae": trial.value,
                "test_mae": test_mae,
                "test_r2": trial.user_attrs.get("test_r2", 0),
            })
        
        # 保存 Stage 1 详细结果
        self._save_stage1_results(study, trial_results)
        
        return candidates
    
    def _run_stage2_validation(self, candidates: List[Dict]) -> Dict[str, Any]:
        """
        第二阶段：多种子验证
        
        对 Stage 1 选出的 Top-K 候选进行多轮训练验证鲁棒性
        
        Returns:
            最佳候选的详细结果
        """
        print(f"\n[Stage 2] Multi-seed Validation")
        print(f"  Candidates: {len(candidates)}")
        print(f"  Seeds: {self.budget_config.n_seeds_final}")
        print(f"  Epochs: {self.budget_config.epochs_final}, Patience: {self.budget_config.patience_final}")
        
        seeds = [42, 101, 102, 202, 303][:self.budget_config.n_seeds_final]
        
        validation_results = []
        
        for cand in candidates:
            config = cand["model_config"]
            print(f"\n  Validating candidate {cand['rank']} (params: {cand['n_params']:,})")
            
            seed_results = []
            for seed in seeds:
                set_seed(seed)
                
                # 创建新模型
                model = create_model(
                    model_type=self.model_type,
                    input_dim=16,
                    config=config,
                    device=self.device
                )
                
                # 完整训练
                final_training_config = {
                    **self.training_config,
                    "epochs": self.budget_config.epochs_final,
                    "patience": self.budget_config.patience_final,
                }
                
                results = train_and_evaluate(
                    model=model,
                    data=self.data,
                    config=final_training_config,
                    device=self.device,
                    epochs=self.budget_config.epochs_final,
                    patience=self.budget_config.patience_final,
                )
                
                seed_results.append(results)
                print(f"    Seed {seed}: MAE={results['test_mae']:.4f}, R2={results['test_r2']:.4f}")
            
            # 计算统计量
            # 使用已导入的numpy
            mae_values = [r["test_mae"] for r in seed_results]
            r2_values = [r["test_r2"] for r in seed_results]
            
            stats = {
                "candidate_rank": cand["rank"],
                "model_config": config,
                "n_params": cand["n_params"],
                "test_mae_mean": float(np.mean(mae_values)),
                "test_mae_std": float(np.std(mae_values)),
                "test_r2_mean": float(np.mean(r2_values)),
                "test_r2_std": float(np.std(r2_values)),
                "seed_results": seed_results,
            }
            
            print(f"    -> Mean MAE: {stats['test_mae_mean']:.4f} ± {stats['test_mae_std']:.4f}")
            
            validation_results.append(stats)
        
        # 选择最佳候选（基于 mean MAE）
        validation_results.sort(key=lambda x: x["test_mae_mean"])
        best_result = validation_results[0]
        
        print(f"\n  Best Candidate: Rank {best_result['candidate_rank']}")
        print(f"    Config: {best_result['model_config']}")
        print(f"    Params: {best_result['n_params']:,}")
        print(f"    Test MAE: {best_result['test_mae_mean']:.4f} ± {best_result['test_mae_std']:.4f}")
        print(f"    Test R2:  {best_result['test_r2_mean']:.4f} ± {best_result['test_r2_std']:.4f}")
        
        return {
            "best_config": best_result["model_config"],
            "n_params": best_result["n_params"],
            "performance": {
                "test_mae_mean": best_result["test_mae_mean"],
                "test_mae_std": best_result["test_mae_std"],
                "test_r2_mean": best_result["test_r2_mean"],
                "test_r2_std": best_result["test_r2_std"],
            },
            "all_candidates": validation_results,
        }
    
    def _save_stage1_results(self, study: optuna.Study, trial_results: List[Dict]):
        """保存 Stage 1 搜索结果"""
        # 保存 study
        study_path = self.output_dir / f"{self.model_type}_{self.budget_config.budget}_study.pkl"
        # 使用已导入的joblib
        joblib.dump(study, study_path)
        
        # 保存 trial 详情
        df = pd.DataFrame(trial_results)
        csv_path = self.output_dir / f"{self.model_type}_{self.budget_config.budget}_stage1.csv"
        df.to_csv(csv_path, index=False)
    
    def _save_results(self, results: Dict[str, Any]):
        """保存最终结果"""
        # YAML 配置
        yaml_path = self.output_dir / f"{self.model_type}_{self.budget_config.budget}.yaml"
        OmegaConf.save(OmegaConf.create(results), yaml_path)
        
        # JSON 详细结果
        json_path = self.output_dir / f"{self.model_type}_{self.budget_config.budget}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n[Results saved]")
        print(f"  YAML: {yaml_path}")
        print(f"  JSON: {json_path}")


def run_preexperiment_for_budget(
    model_type: str,
    param_budget: int,
    output_dir: Path,
    n_trials: int = 100,
    device: str = "cpu",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    便捷函数：为单个预算运行预实验
    
    Args:
        model_type: 模型类型
        param_budget: 参数量预算
        output_dir: 输出目录
        n_trials: Optuna 搜索次数
        device: 计算设备
        smoke_test: 是否快速测试模式
        
    Returns:
        实验结果字典
    """
    if smoke_test:
        # 快速测试模式：减少 trials 和 epochs
        budget_config = BudgetConfig(
            budget=param_budget,
            n_trials=min(n_trials, 5),
            epochs_search=10,
            patience_search=5,
            n_seeds_final=2,
            epochs_final=20,
            patience_final=5,
            top_k=2,
        )
    else:
        budget_config = BudgetConfig(
            budget=param_budget,
            n_trials=n_trials,
            epochs_search=50,
            patience_search=10,
            n_seeds_final=3,
            epochs_final=100,
            patience_final=15,
            top_k=3,
        )
    
    # XJTU 数据配置
    data_config = {
        "data": {
            "dataset": "xjtu",
            "data_dir": "data/XJTU data",
            "file_pattern": "{batch_id}_battery-*.csv",
            "batch_id": "2C",  # 默认使用 2C 数据
            "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            "features": [
                "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness",
                "current mean", "current std", "current kurtosis", "current skewness",
                "CC Q", "CC charge time", "CV Q", "CV charge time",
                "voltage slope", "current slope", "voltage entropy", "current entropy"
            ],
            "target": "capacity",
            "preprocessing": {
                "remove_outliers": True,
                "outlier_method": "3sigma",
                "standardize": True,
                "fill_na": True,
                "fill_method": "mean",
            }
        }
    }
    
    # 训练配置
    training_config = {
        "lr": 1e-3,
        "batch_size": 64,
        "weight_decay": 0.0,
    }
    
    runner = PreExperimentRunner(
        model_type=model_type,
        budget_config=budget_config,
        data_config=data_config,
        training_config=training_config,
        output_dir=output_dir,
        device=device,
    )
    
    return runner.run()
