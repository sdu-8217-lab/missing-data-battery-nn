"""
Hydra 工具函数 - 简化版

利用 Hydra 的 instantiate 功能，无需自己实现注册中心
"""

from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Any


def create_from_config(cfg: DictConfig, **override_kwargs) -> Any:
    """
    从配置创建对象
    
    利用 Hydra 的 instantiate，自动解析 _target_
    
    Example:
        >>> # config.yaml:
        >>> # model:
        >>> #   _target_: src.models.mlp.MLP
        >>> #   input_dim: 32
        >>> 
        >>> model = create_from_config(cfg.model)
    """
    if '_target_' not in cfg:
        raise ValueError(f"Config missing '_target_': {cfg}")
    
    # 合并覆盖参数
    cfg_dict = dict(cfg)
    cfg_dict.update(override_kwargs)
    
    return instantiate(cfg_dict)


def get_missing_rates(cfg: DictConfig) -> list[float]:
    """从配置获取缺失率列表"""
    return cfg.mim.get('train_mr_list', [0.0, 0.1, 0.2, 0.3, 0.4, 
                                          0.5, 0.6, 0.7, 0.8, 0.9])


def get_model_input_dim(cfg: DictConfig) -> int:
    """根据 MIM 配置获取模型输入维度"""
    return 32 if cfg.mim.get('use_mim', False) else 16
