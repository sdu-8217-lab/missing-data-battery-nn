"""
实验配置管理系统
使用Hydra进行配置组合，支持YAML配置文件
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data/raw/XJTU"
    batch: str = "3C"
    test_size: float = 0.25
    val_size: float = 0.25
    target_col: str = "capacity"
    feature_cols: List[str] = field(default_factory=lambda: [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ])


@dataclass
class ValidationConfig:
    """验证指标配置 (META §2.5)"""
    # 验证缺失率子集
    # [-1] 表示使用所有训练缺失率
    # [0.5] 单点优化模式
    # [0.3, 0.5, 0.7] 多MR关注模式
    val_mr_subset: List[float] = field(default_factory=lambda: [-1])
    
    # 基础指标类型
    base_metric: str = "mae"  # mae, mse, rmse
    
    # 聚合方式（多验证MR时）
    aggregation: str = "mean"  # mean, max, min, median, worst
    
    def get_validation_missing_rates(self, training_missing_rates: List[float]) -> List[float]:
        """获取实际使用的验证缺失率列表"""
        if self.val_mr_subset == [-1]:
            # 使用所有训练缺失率
            return training_missing_rates
        else:
            # 使用指定的子集
            return self.val_mr_subset


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 200
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "Adam"
    weight_decay: float = 0.0
    early_stopping_patience: int = 30  # META §6.2: 应大于scheduler_patience
    min_epochs: int = 10
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # MIM特定配置
    use_mim: bool = False
    training_missing_rates: List[float] = field(default_factory=lambda: [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ])
    
    # 训练插补方法 (META §1.3)
    imputation_method: str = "zero"  # zero, mean, knn, iterative
    
    # 序列模型配置
    seq_len: int = 5
    
    # 验证配置 (META §2.5)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class EvaluationConfig:
    """评估配置"""
    missing_rates: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    missing_modes: List[str] = field(default_factory=lambda: ["MCAR"])
    imputation_methods: List[str] = field(default_factory=lambda: ["zero"])
    use_cache: bool = True
    cache_size: int = 10000


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 实验标识
    name: str = "soh_experiment"
    seed: int = 42
    n_repeats: int = 1
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 模型架构配置（引用外部YAML）
    model_config_path: str = "configs/models/mlp_default.yaml"
    
    # 输出路径
    output_dir: str = "./results"
    
    # 计算设备
    device: str = "auto"  # auto, cuda, cpu
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 解析数据配置
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # 解析训练配置（包含验证配置嵌套）
        training_dict = config_dict.get('training', {})
        validation_dict = training_dict.pop('validation', {})
        validation_config = ValidationConfig(**validation_dict)
        training_config = TrainingConfig(**training_dict, validation=validation_config)
        
        # 解析评估配置
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # 创建完整配置
        return cls(
            name=config_dict.get('name', 'soh_experiment'),
            seed=config_dict.get('seed', 42),
            n_repeats=config_dict.get('n_repeats', 1),
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            model_config_path=config_dict.get('model_config_path', 'configs/models/mlp_default.yaml'),
            output_dir=config_dict.get('output_dir', './results'),
            device=config_dict.get('device', 'auto')
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'seed': self.seed,
            'n_repeats': self.n_repeats,
            'data': self.data.__dict__,
            'training': {
                **self.training.__dict__,
                'validation': self.training.validation.__dict__
            },
            'evaluation': self.evaluation.__dict__,
            'model_config_path': self.model_config_path,
            'output_dir': self.output_dir,
            'device': self.device
        }
    
    def to_yaml(self, path: str):
        """保存为YAML文件"""
        config_dict = self.to_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


@dataclass
class ModelArchitectureConfig:
    """模型架构配置（单独保存）"""
    name: str
    model_type: str  # mlp, lstm, gru, cnn1d
    
    # MLP参数
    hidden_layers: Optional[List[int]] = None
    dropout: float = 0.0
    
    # LSTM/GRU参数
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    
    # CNN参数
    channels: Optional[List[int]] = None
    kernel_size: Optional[int] = None
    
    # 序列模型通用
    seq_len: int = 5
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelArchitectureConfig":
        """从YAML文件加载模型配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """保存为YAML文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)


def load_model_config(path: str) -> Dict[str, Any]:
    """加载模型配置字典"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
