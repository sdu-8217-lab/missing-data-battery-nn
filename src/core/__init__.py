"""
核心基础设施 - 接口定义与注册中心
"""

from .interfaces import (
    IImputer,
    IMissingGenerator,
    IModel,
    ITrainer,
    IDataLoader,
    ICallback,
    BatteryDataset,
    TrainingConfig,
    TrainingResult,
    ExperimentConfig,
)

from .registry import (
    Registry,
    IMPUTERS,
    MISSING_GENERATORS,
    MODELS,
    TRAINERS,
    DATA_LOADERS,
    create_imputer,
    create_missing_generator,
    create_model,
    create_trainer,
    create_data_loader,
    list_all_components,
)

__all__ = [
    # 接口
    'IImputer',
    'IMissingGenerator',
    'IModel',
    'ITrainer',
    'IDataLoader',
    'ICallback',
    'BatteryDataset',
    'TrainingConfig',
    'TrainingResult',
    'ExperimentConfig',
    # 注册中心
    'Registry',
    'IMPUTERS',
    'MISSING_GENERATORS',
    'MODELS',
    'TRAINERS',
    'DATA_LOADERS',
    # 便捷函数
    'create_imputer',
    'create_missing_generator',
    'create_model',
    'create_trainer',
    'create_data_loader',
    'list_all_components',
]
