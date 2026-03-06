"""实验配置 - 使用Pydantic v2"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ModelConfig(BaseModel):
    """模型配置 - 使用configs_v3最佳参数"""
    model_config = ConfigDict(frozen=True)  # 不可变配置
    
    name: str = Field(..., description="模型名称")
    model_type: Literal['mlp', 'lstm', 'gru', 'cnn1d'] = Field(
        ..., description="模型类型"
    )
    use_mim: bool = Field(default=False, description="是否使用缺失指示器")
    
    # MLP参数 - configs_v3最佳: [192,96,48,24], dropout=0.15
    hidden_layers: List[int] = Field(
        default=[192, 96, 48, 24],
        description="MLP隐藏层结构"
    )
    dropout: float = Field(default=0.15, ge=0, le=1, description="Dropout比率")
    
    # LSTM/GRU参数 - configs_v3最佳: LSTM h=48,l=2,dropout=0.2; GRU h=64,l=2,dropout=0.2
    hidden_size: int = Field(default=64, ge=1, description="隐藏层大小")
    num_layers: int = Field(default=2, ge=1, le=5, description="层数")
    seq_len: int = Field(default=5, ge=1, description="序列长度")
    
    # CNN参数 - configs_v3最佳: [72,32], kernel=4, dropout=0.1
    channels: List[int] = Field(
        default=[72, 32],
        description="CNN通道数"
    )
    kernel_size: int = Field(default=4, ge=1, description="卷积核大小")
    
    @field_validator('hidden_layers')
    @classmethod
    def validate_hidden_layers(cls, v):
        if len(v) < 1:
            raise ValueError('hidden_layers 至少需要一个元素')
        return v
    
    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if len(v) < 1:
            raise ValueError('channels 至少需要一个元素')
        return v


class ExperimentConfig(BaseModel):
    """实验配置"""
    model_config = ConfigDict(validate_assignment=True)
    
    # 实验基本信息
    name: str = Field(default="soh_mim_xjtu", description="实验名称")
    timestamp: Optional[str] = Field(default=None, description="时间戳")
    random_seed: int = Field(default=42, ge=0, description="随机种子")
    
    # 重复实验次数
    n_repeats: int = Field(default=100, ge=1, le=1000, description="重复次数")
    
    # 缺失率设置
    missing_rates: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        description="测试缺失率列表"
    )
    training_missing_rates: List[float] = Field(
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        description="训练缺失率列表"
    )
    
    # 数据设置
    data_dir: str = Field(default="./data/XJTU data", description="数据目录")
    batch: Literal['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'] = Field(
        default="3C",
        description="数据批次"
    )
    test_size: float = Field(default=0.25, gt=0, lt=1, description="测试集比例")
    val_size: float = Field(default=0.25, gt=0, lt=1, description="验证集比例")
    
    # 特征列
    feature_cols: List[str] = Field(
        default=[
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ],
        description="特征列名"
    )
    target_col: str = Field(default='capacity', description="目标列名")
    seq_len: int = Field(default=5, ge=1, description="序列长度")
    
    # 训练设置
    epochs: int = Field(default=100, ge=1, le=1000, description="训练轮数")
    batch_size: int = Field(default=32, ge=1, le=1024, description="批量大小")
    lr: float = Field(default=0.001, gt=0, le=1, description="学习率")
    optimizer: Literal["Adam", "SGD", "AdamW"] = Field(default="Adam", description="优化器")
    weight_decay: float = Field(default=0.0, ge=0, description="权重衰减")
    early_stopping_patience: int = Field(default=15, ge=0, description="早停耐心值")
    
    # 路径设置
    results_dir: str = Field(default="./results", description="结果目录")
    
    @field_validator('missing_rates', 'training_missing_rates')
    @classmethod
    def validate_missing_rates(cls, v):
        for rate in v:
            if not 0 <= rate <= 1:
                raise ValueError(f'缺失率 {rate} 必须在 [0, 1] 范围内')
        return v
    
    @field_validator('val_size', 'test_size')
    @classmethod
    def validate_split_sizes(cls, v, info):
        if v <= 0 or v >= 1:
            raise ValueError(f'{info.field_name} 必须在 (0, 1) 范围内')
        return v
    
    def get_model_configs(self) -> List[ModelConfig]:
        """获取所有模型配置 - configs_v3最佳参数 (4模型 x 2策略 = 8配置)"""
        base_configs = [
            {
                'name': 'MLP',
                'model_type': 'mlp',
                'hidden_layers': [192, 96, 48, 24],  # configs_v3最佳
                'dropout': 0.15,
            },
            {
                'name': 'LSTM',
                'model_type': 'lstm',
                'hidden_size': 48,  # configs_v3最佳
                'num_layers': 2,
                'dropout': 0.2,
                'seq_len': self.seq_len,
            },
            {
                'name': 'GRU',
                'model_type': 'gru',
                'hidden_size': 64,  # configs_v3最佳
                'num_layers': 2,
                'dropout': 0.2,
                'seq_len': self.seq_len,
            },
            {
                'name': 'CNN1D',
                'model_type': 'cnn1d',
                'channels': [72, 32],  # configs_v3最佳
                'kernel_size': 4,
                'dropout': 0.1,
                'seq_len': self.seq_len,
            },
        ]
        
        configs = []
        for base in base_configs:
            # Baseline版本
            configs.append(ModelConfig(**base, use_mim=False))
            # MIM版本 - 复制base并修改name
            mim_config = base.copy()
            mim_config['name'] = f"{base['name']}-MIM"
            configs.append(ModelConfig(**mim_config, use_mim=True))
        
        return configs
    
    def to_dict(self) -> dict:
        """转换为字典（用于保存）"""
        return self.model_dump()
    
    def save_json(self, path: str):
        """保存为JSON文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, path: str) -> "ExperimentConfig":
        """从JSON文件加载"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
