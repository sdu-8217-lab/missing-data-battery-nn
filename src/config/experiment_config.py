"""实验配置"""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    """模型配置 - 使用configs_v3最佳参数"""
    name: str
    model_type: str
    use_mim: bool = False
    # MLP参数
    hidden_layers: List[int] = None
    # LSTM/GRU参数
    hidden_size: int = 64
    num_layers: int = 2  # configs_v3最佳: 2层
    seq_len: int = 5
    # CNN参数
    channels: List[int] = None
    kernel_size: int = 3
    dropout: float = 0.1  # 添加dropout参数


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
    
    # 模型配置 - configs_v3最佳参数 (已移除XGBoost)
    def get_model_configs(self) -> List[ModelConfig]:
        """获取所有模型配置 - 使用configs_v3架构搜索最佳参数"""
        configs = [
            # MLP: [192,96,48,24] 4层, dropout=0.15, MAE=0.0128
            ModelConfig(
                name='MLP', 
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=False
            ),
            ModelConfig(
                name='MLP-MIM', 
                model_type='mlp',
                hidden_layers=[192, 96, 48, 24],
                dropout=0.15,
                use_mim=True
            ),
            # LSTM: h=48, l=2, dropout=0.2, MAE=0.0077
            ModelConfig(
                name='LSTM',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='LSTM-MIM',
                model_type='lstm',
                hidden_size=48,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True
            ),
            # GRU: h=64, l=2, dropout=0.2, MAE=0.0069
            ModelConfig(
                name='GRU',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='GRU-MIM',
                model_type='gru',
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                seq_len=self.seq_len,
                use_mim=True
            ),
            # CNN1D: [72,32], kernel=4, dropout=0.1, MAE=0.0057 (最佳)
            ModelConfig(
                name='CNN1D',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=False
            ),
            ModelConfig(
                name='CNN1D-MIM',
                model_type='cnn1d',
                channels=[72, 32],
                kernel_size=4,
                dropout=0.1,
                seq_len=self.seq_len,
                use_mim=True
            ),
        ]
        return configs
