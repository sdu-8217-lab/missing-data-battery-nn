"""
配置加载工具 - 支持批量实验配置生成
"""

from pathlib import Path
from typing import Iterator, List, Dict, Any
import itertools

from .pydantic_config import ExperimentConfig, SeedConfig, DataConfig, ModelArchitectureConfig, MIMConfig, HAS_YAML


def load_config(path: str) -> ExperimentConfig:
    """从文件加载配置（支持 YAML 和 JSON）"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    
    if path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        return ExperimentConfig.from_yaml(path)
    elif path.suffix == '.json':
        return ExperimentConfig.from_json(path)
    else:
        raise ValueError(f"不支持的配置文件格式: {path.suffix}")


def save_config(config: ExperimentConfig, path: str) -> None:
    """保存配置到文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix in ['.yaml', '.yml']:
        config.to_yaml(path)
    elif path.suffix == '.json':
        config.to_json(path)
    else:
        raise ValueError(f"不支持的配置文件格式: {path.suffix}")


def generate_experiment_matrix(
    seeds: List[int],
    batches: List[str],
    models: List[str],
    mim_settings: List[bool],
    base_config: ExperimentConfig = None
) -> Iterator[ExperimentConfig]:
    """
    生成实验矩阵的所有配置组合
    
    根据 meta.md 的9层架构，生成 L1-L5 的所有组合
    用于 run_100seeds_final.py 这类批量实验
    
    Args:
        seeds: L1 - 随机种子列表
        batches: L3 - 批次列表
        models: L4 - 模型架构列表
        mim_settings: L5 - MIM设置列表
        base_config: 基础配置模板
        
    Yields:
        ExperimentConfig - 每个实验的配置
    """
    if base_config is None:
        base_config = ExperimentConfig()
    
    for seed, batch, model_type, use_mim in itertools.product(
        seeds, batches, models, mim_settings
    ):
        # 根据是否使用MIM调整模型配置
        if model_type == 'mlp':
            hidden_dims = [128, 96] if use_mim else [192, 96]
            model_config = ModelArchitectureConfig(
                model_type='mlp',
                mlp_hidden_dims=hidden_dims,
                mlp_dropout=base_config.model.mlp_dropout
            )
        elif model_type == 'lstm':
            model_config = ModelArchitectureConfig(
                model_type='lstm',
                lstm_hidden_size=base_config.model.lstm_hidden_size,
                lstm_num_layers=base_config.model.lstm_num_layers,
                lstm_dropout=base_config.model.lstm_dropout
            )
        elif model_type == 'cnn':
            model_config = ModelArchitectureConfig(
                model_type='cnn',
                cnn_channels=base_config.model.cnn_channels,
                cnn_kernel_size=base_config.model.cnn_kernel_size,
                cnn_dropout=base_config.model.cnn_dropout
            )
        
        config = ExperimentConfig(
            experiment_name=f"seed{seed}_{batch}_{model_type}_{'mim' if use_mim else 'no_mim'}",
            seed=SeedConfig(seed=seed),
            data=DataConfig(batch=batch, data_dir=base_config.data.data_dir),
            model=model_config,
            mim=MIMConfig(
                use_mim=use_mim,
                train_mr_list=base_config.mim.train_mr_list,
                val_mr=-1 if use_mim else 0.5
            ),
            training=base_config.training,
            testing=base_config.testing,
            paths=base_config.paths,
            save_model=base_config.save_model,
            save_results=base_config.save_results
        )
        
        yield config


def generate_100seeds_configs(
    base_config_path: str = None
) -> Iterator[ExperimentConfig]:
    """
    生成100种子实验的完整配置矩阵
    
    总实验数: 100 × 6 × 3 × 2 = 3600
    """
    if base_config_path:
        base_config = load_config(base_config_path)
    else:
        # 使用默认配置作为基础
        base_config = ExperimentConfig()
    
    seeds = list(range(100))
    batches = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
    models = ["mlp", "lstm", "cnn"]
    mim_settings = [False, True]
    
    yield from generate_experiment_matrix(
        seeds=seeds,
        batches=batches,
        models=models,
        mim_settings=mim_settings,
        base_config=base_config
    )


def generate_ablation_configs(
    batch: str = "2C",
    model: str = "mlp",
    seed: int = 42
) -> List[ExperimentConfig]:
    """
    生成消融实验配置 (use_mim=true vs false)
    """
    configs = []
    for use_mim in [False, True]:
        config = ExperimentConfig(
            experiment_name=f"ablation_{batch}_{model}_{'mim' if use_mim else 'no_mim'}",
            seed=SeedConfig(seed=seed),
            data=DataConfig(batch=batch),
            model=ModelArchitectureConfig(model_type=model),
            mim=MIMConfig(use_mim=use_mim),
            tags=["ablation", batch, model, "mim" if use_mim else "no_mim"]
        )
        configs.append(config)
    return configs


def count_experiments(configs: List[ExperimentConfig]) -> Dict[str, Any]:
    """
    统计实验配置矩阵的信息
    """
    stats = {
        'total': len(configs),
        'by_seed': {},
        'by_batch': {},
        'by_model': {},
        'by_mim': {}
    }
    
    for cfg in configs:
        # 按种子统计
        seed = cfg.seed.seed
        stats['by_seed'][seed] = stats['by_seed'].get(seed, 0) + 1
        
        # 按批次统计
        batch = cfg.data.batch
        stats['by_batch'][batch] = stats['by_batch'].get(batch, 0) + 1
        
        # 按模型统计
        model = cfg.model.model_type
        stats['by_model'][model] = stats['by_model'].get(model, 0) + 1
        
        # 按MIM统计
        mim = cfg.mim.use_mim
        stats['by_mim'][mim] = stats['by_mim'].get(mim, 0) + 1
    
    return stats


def validate_configs(configs: List[ExperimentConfig]) -> Dict[str, Any]:
    """
    验证实验配置的完整性
    
    Returns:
        包含验证结果的字典
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # 检查必需的批次
    expected_batches = {"2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"}
    actual_batches = {cfg.data.batch for cfg in configs}
    missing_batches = expected_batches - actual_batches
    
    if missing_batches:
        result['warnings'].append(f"缺少批次: {missing_batches}")
    
    # 检查必需的模型
    expected_models = {"mlp", "lstm", "cnn"}
    actual_models = {cfg.model.model_type for cfg in configs}
    missing_models = expected_models - actual_models
    
    if missing_models:
        result['warnings'].append(f"缺少模型: {missing_models}")
    
    # 检查MIM配置
    mim_true = sum(1 for cfg in configs if cfg.mim.use_mim)
    mim_false = sum(1 for cfg in configs if not cfg.mim.use_mim)
    
    if mim_true == 0:
        result['errors'].append("缺少MIM=true的配置")
        result['valid'] = False
    if mim_false == 0:
        result['errors'].append("缺少MIM=false的配置")
        result['valid'] = False
    
    # 检查配置一致性
    for i, cfg in enumerate(configs):
        # 检查模型类型与配置一致性
        if cfg.model.model_type == 'mlp':
            if cfg.mim.use_mim:
                # MIM模式下，MLP应该使用较小的隐藏层
                if cfg.model.mlp_hidden_dims == [192, 96]:
                    result['warnings'].append(
                        f"配置{i}: MLP+MIM使用非MIM隐藏层配置 [192, 96]，建议改为 [128, 96]"
                    )
        
        # 检查路径设置
        if not cfg.save_model and cfg.save_results:
            result['warnings'].append(
                f"配置{i}: 不保存模型但保存结果，可能导致无法复现"
            )
    
    return result
