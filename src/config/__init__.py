"""
配置模块 - 基于 meta.md 9层架构的结构化配置系统

提供:
- 类型安全的配置类 (dataclasses)
- YAML/JSON 序列化支持
- 批量配置生成工具
- 配置验证

使用示例:
    from src.config import ExperimentConfig, load_config
    
    # 从文件加载
    config = load_config("configs/reference/default_mlp_2c.yaml")
    
    # 程序化创建
    config = ExperimentConfig(
        experiment_name="my_experiment",
        data=DataConfig(batch="2C"),
        model=ModelArchitectureConfig(model_type="mlp"),
        mim=MIMConfig(use_mim=True)
    )
    
    # 保存配置
    config.to_yaml("my_config.yaml")
"""

from .pydantic_config import (
    # 核心配置类
    ExperimentConfig,
    SeedConfig,
    DataConfig,
    ModelArchitectureConfig,
    MIMConfig,
    TrainingConfig,
    TestingConfig,
    PathConfig,
    # 工具函数
    get_default_config,
    get_100seeds_config,
    get_ablation_config,
    validate_experiment_matrix,
)

from .loader import (
    # 加载工具
    load_config,
    save_config,
    generate_experiment_matrix,
    generate_100seeds_configs,
    generate_ablation_configs,
    count_experiments,
    validate_configs,
)

__all__ = [
    # 配置类
    'ExperimentConfig',
    'SeedConfig',
    'DataConfig',
    'ModelArchitectureConfig',
    'MIMConfig',
    'TrainingConfig',
    'TestingConfig',
    'PathConfig',
    # 工具函数
    'get_default_config',
    'get_100seeds_config',
    'get_ablation_config',
    'validate_experiment_matrix',
    # 加载工具
    'load_config',
    'save_config',
    'generate_experiment_matrix',
    'generate_100seeds_configs',
    'generate_ablation_configs',
    'count_experiments',
    'validate_configs',
]
