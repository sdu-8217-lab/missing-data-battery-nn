"""
预实验模块：自动搜索给定参数量预算下的最优模型架构

使用 Optuna 进行智能搜索，支持四个预算等级：
- 8,192  (2^13)
- 16,384 (2^14)  
- 32,768 (2^15)
- 65,536 (2^16)
"""

from .search_space import SearchSpace
from .runner import PreExperimentRunner

__all__ = ['SearchSpace', 'PreExperimentRunner']
