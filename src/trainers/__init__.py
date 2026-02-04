"""训练器模块"""
# 旧版训练器（向后兼容）
from .neural_network_trainer import NeuralNetworkTrainer

# 新版Lightning训练器（推荐）
try:
    from .lightning_module import SOHLightningModule
    from .lightning_trainer import LightningTrainer, create_lightning_trainer
    __all__ = [
        'NeuralNetworkTrainer',
        'SOHLightningModule',
        'LightningTrainer', 'create_lightning_trainer'
    ]
except ImportError:
    # 如果pytorch-lightning未安装，仅导出旧版
    __all__ = ['NeuralNetworkTrainer']
