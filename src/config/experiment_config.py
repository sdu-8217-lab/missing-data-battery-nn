"""实验配置"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str
    use_mim: bool = False
    # MLP参数
    hidden_layers: List[int] = None
    # LSTM/GRU参数
    hidden_size: int = 64
    num_layers: int = 1
    seq_len: int = 5
    # CNN参数
    channels: List[int] = None
    kernel_size: int = 3
    # XGBoost参数
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验基本信息
    name: str = "soh_mim_xjtu"
    timestamp: str = None
    random_seed: int = 42
    
    # 重复实验次数
    n_repeats: int = 100
    
    # 缺失率设置
    missing_rates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    training_missing_rates: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    
    # 数据设置
    data_dir: str = "./data/XJTU data"
    batch: str = "3C"  # 2C, 3C, R2.5, R3, RW, Sim_satellite
    test_size: float = 0.25  # 按电池划分
    val_size: float = 0.25
    
    # 特征列（16个）
    feature_cols: List[str] = field(default_factory=lambda: [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ])
    target_col: str = 'capacity'
    seq_len: int = 5
    
    # 训练设置
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "Adam"
    weight_decay: float = 0.0
    early_stopping_patience: int = 15
    
    # 路径设置
    results_dir: str = "./results"
    
    # 模型配置
    def get_model_configs(self) -> List[ModelConfig]:
        """获取所有模型配置"""
        configs = [
            ModelConfig(
                name='MLP', 
                model_type='mlp',
                hidden_layers=[100, 64, 32],
                use_mim=False
            ),
            ModelConfig(
                name='MLP-MIM', 
                model_type='mlp',
                hidden_layers=[100, 64, 32],
                use_mim=True
            ),
            ModelConfig(
                name='XGBoost',
                model_type='xgboost',
                n_estimators=60,
                max_depth=5,
                use_mim=False
            ),
            ModelConfig(
                name='XGBoost-MIM',
                model_type='xgboost',
                n_estimators=60,
                max_depth=5,
                use_mim=True
            ),
            ModelConfig(
                name='LSTM',
                model_type='lstm',
                hidden_size=42,
                num_layers=1,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='LSTM-MIM',
                model_type='lstm',
                hidden_size=42,
                num_layers=1,
                seq_len=self.seq_len,
                use_mim=True
            ),
            ModelConfig(
                name='GRU',
                model_type='gru',
                hidden_size=48,
                num_layers=1,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='GRU-MIM',
                model_type='gru',
                hidden_size=48,
                num_layers=1,
                seq_len=self.seq_len,
                use_mim=True
            ),
            ModelConfig(
                name='CNN1D',
                model_type='cnn1d',
                channels=[48, 32],
                kernel_size=3,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='CNN1D-MIM',
                model_type='cnn1d',
                channels=[48, 32],
                kernel_size=3,
                seq_len=self.seq_len,
                use_mim=True
            ),
        ]
        return configs
