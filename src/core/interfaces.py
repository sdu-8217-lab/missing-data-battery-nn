"""
抽象接口定义 - 所有组件的契约

设计原则：
1. 接口只定义契约，不实现逻辑
2. 使用 Protocol 支持结构子类型（鸭子类型）
3. 所有接口都有明确的输入输出类型注解
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Any
from dataclasses import dataclass
import numpy as np
import torch


# ============================================================================
# 数据相关
# ============================================================================

@dataclass
class BatteryDataset:
    """
    统一电池数据集格式
    
    Attributes:
        features: 特征矩阵 [n_samples, n_features]
        labels: 标签向量 [n_samples]
        battery_ids: 电池ID [n_samples]
        cycles: 循环次数 [n_samples] (可选)
        metadata: 额外元数据
    """
    features: np.ndarray
    labels: np.ndarray
    battery_ids: np.ndarray
    cycles: np.ndarray = None
    metadata: dict = None
    
    def __post_init__(self):
        # 验证维度一致性
        n_samples = len(self.features)
        assert len(self.labels) == n_samples, "labels 长度不匹配"
        assert len(self.battery_ids) == n_samples, "battery_ids 长度不匹配"
        if self.cycles is not None:
            assert len(self.cycles) == n_samples, "cycles 长度不匹配"


@runtime_checkable
class IDataLoader(Protocol):
    """数据加载器接口"""
    
    @abstractmethod
    def load(self, batch_id: str) -> BatteryDataset:
        """
        加载指定批次的数据
        
        Args:
            batch_id: 批次标识符 (如 '2C', '3C')
            
        Returns:
            BatteryDataset 对象
        """
        ...
    
    @abstractmethod
    def list_batches(self) -> list[str]:
        """返回可用的批次列表"""
        ...


# ============================================================================
# 缺失数据相关
# ============================================================================

@runtime_checkable
class IMissingGenerator(Protocol):
    """缺失数据生成器接口"""
    
    @abstractmethod
    def generate(self, X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
        """
        生成缺失掩码
        
        Args:
            X: 原始特征矩阵 [n_samples, n_features]
            missing_rate: 缺失率 (0-1)
            seed: 随机种子
            
        Returns:
            mask: 布尔掩码，True 表示缺失 [n_samples, n_features]
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """生成器名称标识"""
        ...


@runtime_checkable
class IImputer(Protocol):
    """插补策略接口"""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "IImputer":
        """
        基于训练数据学习插补参数
        
        Args:
            X: 完整训练数据 [n_samples, n_features]
            
        Returns:
            self (支持链式调用)
        """
        ...
    
    @abstractmethod
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """
        执行插补
        
        Args:
            X_missing: 带缺失的数据 (np.nan 表示缺失)
            
        Returns:
            X_imputed: 插补后的数据
        """
        ...
    
    def fit_transform(self, X: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        便捷方法：先 fit 再 transform
        
        Args:
            X: 数据（如果 mask 为 None，则 X 应包含 np.nan）
            mask: 缺失掩码（可选）
            
        Returns:
            X_imputed: 插补后的数据
        """
        if mask is not None:
            X_missing = X.copy()
            X_missing[mask] = np.nan
        else:
            X_missing = X
        return self.fit(X).transform(X_missing)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插补方法名称标识"""
        ...


# ============================================================================
# 模型相关
# ============================================================================

@runtime_checkable
class IModel(Protocol):
    """神经网络模型接口"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        ...
    
    def parameters(self):
        """返回模型参数（兼容 PyTorch）"""
        ...
    
    def to(self, device):
        """移动模型到设备"""
        ...
    
    def train(self, mode: bool = True):
        """设置训练/评估模式"""
        ...
    
    def eval(self):
        """设置评估模式"""
        ...
    
    def state_dict(self) -> dict:
        """获取模型状态"""
        ...
    
    def load_state_dict(self, state_dict: dict):
        """加载模型状态"""
        ...
    
    @property
    def input_dim(self) -> int:
        """输入维度"""
        ...


# ============================================================================
# 训练相关
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 200
    lr: float = 0.001
    batch_size: int = 32
    patience: int = 30
    weight_decay: float = 1e-5
    device: str = "auto"


@dataclass
class TrainingResult:
    """训练结果"""
    best_val_loss: float
    best_epoch: int
    history: dict  # 包含 train_loss, val_loss 等
    model_path: str = None


class ITrainer(Protocol):
    """训练引擎接口"""
    
    @abstractmethod
    def train(
        self,
        model: IModel,
        train_data: list,
        val_data: list,
        config: TrainingConfig = None
    ) -> TrainingResult:
        """
        执行训练
        
        Args:
            model: 模型实例
            train_data: 训练数据列表 [(X1, y1), (X2, y2), ...]
            val_data: 验证数据列表
            config: 训练配置
            
        Returns:
            TrainingResult 包含训练历史和最佳模型信息
        """
        ...


# ============================================================================
# 回调相关
# ============================================================================

class ICallback(Protocol):
    """训练回调接口"""
    
    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        """
        每个 epoch 结束时调用
        
        Args:
            epoch: 当前 epoch
            metrics: 包含 train_loss, val_loss 等的字典
            
        Returns:
            返回 False 则停止训练
        """
        return True
    
    def on_training_end(self, result: TrainingResult):
        """训练结束时调用"""
        pass


# ============================================================================
# 实验相关
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    完整实验配置 - 纯数据对象
    
    这是连接用户输入和系统组件的桥梁。
    所有字段都是基本类型（str, int, float, list），
    不包含任何对象实例。
    """
    # L1: 种子
    seed: int = 42
    
    # L3: 数据
    batch_id: str = "2C"
    data_loader: str = "xjtu"  # 注册中心中的名称
    
    # L4: 模型
    model_type: str = "mlp"
    model_kwargs: dict = None
    
    # L5-L6: MIM 配置
    use_mim: bool = False
    missing_generator: str = "mcar"
    train_imputation: str = "zero"
    missing_rates: list = None
    
    # 训练配置
    training_config: TrainingConfig = None
    
    # L7-L9: 测试配置
    test_mode: str = None
    test_missing_rate: float = None
    test_imputation: str = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.missing_rates is None:
            self.missing_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 
                                 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 
                                 0.8, 0.85, 0.9, 0.95]
        if self.training_config is None:
            self.training_config = TrainingConfig()
