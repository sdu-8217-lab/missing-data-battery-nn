"""
结构化配置系统 - 基于 meta.md 的9层实验架构

使用 Python dataclasses 实现配置验证和序列化
（不依赖 pydantic 以减少外部依赖）
"""

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Union
from pathlib import Path
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


# ============================================================================
# 分界线以上配置 (L1-L6): 影响模型训练
# ============================================================================

@dataclass(frozen=True)
class DataConfig:
    """L2-L3: 数据配置
    
    L2: Dataset - 固定为XJTU数据集
    L3: Battery Batch - 6个批次之一
    """
    batch: Literal['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']
    data_dir: str = "data/XJTU data"
    
    def __post_init__(self):
        valid_batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']
        if self.batch not in valid_batches:
            raise ValueError(f"batch必须是 {valid_batches} 之一， got {self.batch}")


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """L4: 模型架构配置
    
    所有模型参数量统一在 (2^14, 2^15) = (16384, 32768) 范围内
    """
    model_type: Literal['mlp', 'lstm', 'cnn']
    
    # MLP 特定配置
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [192, 96])
    mlp_dropout: float = 0.15
    
    # LSTM 特定配置
    lstm_hidden_size: int = 46
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # CNN 特定配置  
    cnn_channels: List[int] = field(default_factory=lambda: [64, 80])
    cnn_kernel_size: int = 3
    cnn_dropout: float = 0.1
    
    def __post_init__(self):
        if self.model_type not in ['mlp', 'lstm', 'cnn']:
            raise ValueError(f"model_type必须是 ['mlp', 'lstm', 'cnn'] 之一")


@dataclass(frozen=True)
class MIMConfig:
    """L5-L6: MIM方法配置
    
    L5: use_mim - 是否使用MIM指示器
    L6: 训练缺失率 - 由use_mim隐含决定
    """
    use_mim: bool = False
    
    # MIM训练时的缺失率范围
    train_mr_list: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    
    # 验证策略
    # 0.5 = 单MR验证, -1 = 多MR平均, -2 = 多MR独立监控
    val_mr: float = -1.0
    
    def __post_init__(self):
        if not all(0 <= mr <= 1 for mr in self.train_mr_list):
            raise ValueError("train_mr_list 中的所有值必须在 [0, 1] 范围内")
        if not (self.val_mr == -1 or self.val_mr == -2 or 0 <= self.val_mr <= 1):
            raise ValueError("val_mr 必须是 0-1 之间的值，或 -1/-2")


@dataclass(frozen=True)
class TrainingConfig:
    """训练超参数配置"""
    epochs: int = 200
    lr: float = 0.001
    batch_size: int = 32
    patience: int = 30
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs 必须 >= 1")
        if self.lr <= 0:
            raise ValueError("lr 必须 > 0")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        if self.patience < 1:
            raise ValueError("patience 必须 >= 1")


# ============================================================================
# 分界线以下配置 (L7-L9): 仅影响测试
# ============================================================================

@dataclass(frozen=True)
class TestingConfig:
    """L7-L9: 测试配置
    
    L7: Mode - 缺失模式 (MCAR/MAR/MNAR)
    L8: 测试MR - 缺失率 0.0-0.9
    L9: Imputation - 插补方法
    """
    mode: Literal['MCAR', 'MAR', 'MNAR'] = 'MCAR'
    test_mr: float = 0.3
    imputation: Literal['mean', 'knn', 'iterative', 'zero'] = 'zero'
    
    # 批处理测试时使用
    modes: List[str] = field(default_factory=lambda: ['MCAR', 'MAR', 'MNAR'])
    test_mrs: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    imputations: List[str] = field(
        default_factory=lambda: ['mean', 'knn', 'iterative', 'zero']
    )
    
    def __post_init__(self):
        if not 0 <= self.test_mr <= 1:
            raise ValueError("test_mr 必须在 [0, 1] 范围内")


# ============================================================================
# 实验全局配置
# ============================================================================

@dataclass(frozen=True)
class SeedConfig:
    """L1: 随机种子配置"""
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42])  # 多种子实验
    
    def __post_init__(self):
        if self.seed < 0:
            raise ValueError("seed 必须 >= 0")


@dataclass(frozen=True)
class PathConfig:
    """路径配置"""
    model_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    data_dir: str = "data/XJTU data"


@dataclass
class ExperimentConfig:
    """
    完整实验配置 - 基于 meta.md 9层架构
    
    分界线以上 (影响训练):
        - seed: L1
        - data: L2-L3
        - model: L4
        - mim: L5-L6
        - training: 训练超参数
    
    分界线以下 (仅测试):
        - testing: L7-L9
    
    其他:
        - paths: 路径配置
        - experiment_name: 实验名称
    """
    # 实验标识
    experiment_name: str = "battery_soh_experiment"
    
    # L1: 随机种子
    seed: SeedConfig = field(default_factory=SeedConfig)
    
    # L2-L3: 数据配置
    data: DataConfig = field(default_factory=lambda: DataConfig(batch='2C'))
    
    # L4: 模型架构
    model: ModelArchitectureConfig = field(default_factory=lambda: ModelArchitectureConfig(model_type='mlp'))
    
    # L5-L6: MIM配置
    mim: MIMConfig = field(default_factory=MIMConfig)
    
    # 训练超参数
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # L7-L9: 测试配置
    testing: TestingConfig = field(default_factory=TestingConfig)
    
    # 路径
    paths: PathConfig = field(default_factory=PathConfig)
    
    # 运行时选项
    save_model: bool = True
    save_results: bool = True
    verbose: bool = True
    
    # 元数据
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """配置验证"""
        # 验证MIM配置一致性
        if self.mim.use_mim:
            # MIM模式下，输入维度为32 (16特征 + 16掩码)
            pass  # 验证在模型创建时进行
        
    def to_dict(self) -> dict:
        """转换为字典"""
        def convert(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            return obj
        return convert(self)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """保存为JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """保存为YAML文件"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperimentConfig':
        """从字典创建配置"""
        return cls(
            experiment_name=data.get('experiment_name', 'battery_soh_experiment'),
            seed=SeedConfig(**data.get('seed', {})),
            data=DataConfig(**data.get('data', {'batch': '2C'})),
            model=ModelArchitectureConfig(**data.get('model', {'model_type': 'mlp'})),
            mim=MIMConfig(**data.get('mim', {})),
            training=TrainingConfig(**data.get('training', {})),
            testing=TestingConfig(**data.get('testing', {})),
            paths=PathConfig(**data.get('paths', {})),
            save_model=data.get('save_model', True),
            save_results=data.get('save_results', True),
            verbose=data.get('verbose', True),
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def get_model_path(self) -> str:
        """生成模型保存路径"""
        mim_str = "mim" if self.mim.use_mim else "no_mim"
        return str(Path(self.paths.model_dir) / 
                   f"seed{self.seed.seed}_batch{self.data.batch}_"
                   f"model{self.model.model_type}_{mim_str}.pt")
    
    def get_results_path(self, suffix: str = "results") -> str:
        """生成结果保存路径"""
        mim_str = "mim" if self.mim.use_mim else "no_mim"
        return str(Path(self.paths.results_dir) /
                   f"{self.experiment_name}_seed{self.seed.seed}_"
                   f"batch{self.data.batch}_model{self.model.model_type}_{mim_str}_{suffix}.json")
    
    def get_input_dim(self) -> int:
        """获取输入维度"""
        return 32 if self.mim.use_mim else 16


# ============================================================================
# 预定义配置模板
# ============================================================================

def get_default_config(batch: str = '2C', model: str = 'mlp', use_mim: bool = False) -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig(
        experiment_name=f"{batch}_{model}_{'mim' if use_mim else 'no_mim'}",
        data=DataConfig(batch=batch),
        model=ModelArchitectureConfig(model_type=model),
        mim=MIMConfig(use_mim=use_mim)
    )


def get_100seeds_config(batch: str = '2C', model: str = 'mlp', use_mim: bool = False) -> ExperimentConfig:
    """获取100种子实验配置"""
    return ExperimentConfig(
        experiment_name=f"100seeds_{batch}_{model}_{'mim' if use_mim else 'no_mim'}",
        seed=SeedConfig(seed=0, seeds=list(range(100))),
        data=DataConfig(batch=batch),
        model=ModelArchitectureConfig(model_type=model),
        mim=MIMConfig(use_mim=use_mim, val_mr=-1),
        training=TrainingConfig(epochs=200, patience=30),
        testing=TestingConfig(
            modes=['MCAR', 'MAR', 'MNAR'],
            test_mrs=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            imputations=['mean', 'knn', 'iterative', 'zero']
        )
    )


def get_ablation_config(batch: str = '2C', model: str = 'mlp') -> List[ExperimentConfig]:
    """获取消融实验配置列表 (use_mim=true/false)"""
    return [
        get_default_config(batch, model, use_mim=False),
        get_default_config(batch, model, use_mim=True)
    ]


# ============================================================================
# 配置验证工具
# ============================================================================

def validate_experiment_matrix(configs: List[ExperimentConfig]) -> dict:
    """验证实验矩阵的完整性
    
    根据meta.md，检查是否覆盖了所有必要的实验组合
    
    Returns:
        验证报告字典
    """
    report = {
        'total_configs': len(configs),
        'unique_batches': set(),
        'unique_models': set(),
        'unique_mim': set(),
        'unique_seeds': set(),
        'missing_combinations': [],
        'valid': True
    }
    
    for cfg in configs:
        report['unique_batches'].add(cfg.data.batch)
        report['unique_models'].add(cfg.model.model_type)
        report['unique_mim'].add(cfg.mim.use_mim)
        report['unique_seeds'].add(cfg.seed.seed)
    
    # 检查是否覆盖所有批次
    expected_batches = {'2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite'}
    if report['unique_batches'] != expected_batches:
        report['missing_combinations'].append(
            f"批次不完整: {expected_batches - report['unique_batches']}"
        )
        report['valid'] = False
    
    # 检查是否覆盖所有模型
    expected_models = {'mlp', 'lstm', 'cnn'}
    if not expected_models.issubset(report['unique_models']):
        report['missing_combinations'].append(
            f"模型不完整: {expected_models - report['unique_models']}"
        )
        report['valid'] = False
    
    # 检查MIM配置
    if True not in report['unique_mim'] or False not in report['unique_mim']:
        report['missing_combinations'].append("MIM配置不完整: 必须同时包含use_mim=True和False")
        report['valid'] = False
    
    return report
