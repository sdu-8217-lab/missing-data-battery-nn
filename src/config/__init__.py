"""配置模块"""
# 向后兼容：导出旧版配置
from .experiment_config import ExperimentConfig, ModelConfig

# 新版Pydantic配置（推荐）
try:
    from .pydantic_config import ExperimentConfig as ExperimentConfigV2, ModelConfig as ModelConfigV2
    __all__ = ['ExperimentConfig', 'ModelConfig', 'ExperimentConfigV2', 'ModelConfigV2']
except ImportError:
    # 如果pydantic未安装，仅导出旧版
    __all__ = ['ExperimentConfig', 'ModelConfig']
