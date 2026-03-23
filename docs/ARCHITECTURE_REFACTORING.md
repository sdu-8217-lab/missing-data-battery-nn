# 项目架构重构方案：低耦合、高内聚设计

## 一、当前架构问题诊断

### 1.1 高耦合问题

| 问题类别 | 具体表现 | 影响 |
|---------|---------|------|
| **脚本臃肿** | `run_experiment.py` (898行), `main.py` (345行) | 维护困难、测试复杂、复用性差 |
| **职责混杂** | 数据加载、模型创建、训练循环、评估逻辑混杂在一起 | 无法独立测试各个组件 |
| **硬编码依赖** | 模型类型硬编码在多处 (`if model_type == 'mlp': ...`) | 添加新模型需修改多处代码 |
| **配置分散** | 缺失率、路径等常量在多个文件中重复定义 | 修改配置易遗漏、不一致 |
| **流程控制混乱** | 训练、验证、测试逻辑交织 | 难以单独运行某个阶段 |

### 1.2 低内聚问题

| 模块 | 当前职责 | 问题 |
|-----|---------|------|
| `experiments/run_experiment.py` | 数据准备 + 模型训练 + 模型评估 + 结果保存 | 违反单一职责原则 |
| `src/main.py` | 配置解析 + 模型配置 + 训练流程 + 日志 | 入口函数过于复杂 |
| `src/data/` | 包含加载、预处理、分割、插补多个子模块 | 部分功能跨模块重复 |

### 1.3 依赖关系图（当前）

```
┌─────────────────────────────────────────────────────────────────┐
│                    experiments/run_experiment.py                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │数据加载  │ │缺失生成  │ │模型创建  │ │训练循环  │          │
│  │+标准化   │ │+插补     │ │+参数设置 │ │+早停     │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                      │
│  │模型保存  │ │结果评估  │                                      │
│  └──────────┘ └──────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 依赖: XJTUDataLoader, MLP/LSTM/CNN, set_seed, 多个工具函数      │
└─────────────────────────────────────────────────────────────────┘
```

**问题**: 这是一个"大泥球"(Big Ball of Mud)架构，所有功能交织在一起。

---

## 二、理想架构设计

### 2.1 核心原则

1. **单一职责原则 (SRP)**: 每个模块只有一个变化理由
2. **开闭原则 (OCP)**: 对扩展开放，对修改关闭
3. **依赖倒置 (DIP)**: 依赖抽象接口，而非具体实现
4. **策略模式**: 将可变算法（插补、缺失模式）封装为可替换策略

### 2.2 目标架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Experiment Pipeline                          │
│                    (仅负责流程编排，无业务逻辑)                      │
└─────────────────────────────────────────────────────────────────────┘
   │              │              │              │              │
   ▼              ▼              ▼              ▼              ▼
┌────────┐   ┌────────┐   ┌────────────┐   ┌────────┐   ┌──────────┐
│Data    │   │Missing │   │Imputation  │   │Model   │   │Trainer   │
│Pipeline│◄──│Generator│◄──│Strategy    │──►│Factory │──►│Engine    │
└────────┘   └────────┘   └────────────┘   └────────┘   └──────────┘
   │              │              │              │              │
   │              │              │              │              │
   ▼              ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Abstract Interfaces                             │
├─────────────────────────────────────────────────────────────────────┤
│  IDataLoader │ IMissingGenerator │ IImputer │ IModel │ ITrainer   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 模块划分

#### 核心层 (Core)
```
src/core/
├── __init__.py
├── interfaces.py          # 所有抽象接口定义
├── registry.py            # 组件注册中心（插件系统）
├── config.py              # 统一配置管理
└── types.py               # 共享类型定义
```

#### 数据层 (Data)
```
src/data/
├── __init__.py
├── loaders/
│   ├── __init__.py
│   ├── base.py           # IDataLoader 接口
│   ├── xjtu_loader.py    # XJTU 数据集实现
│   └── mock_loader.py    # 测试用 Mock
├── pipeline.py           # 数据处理管道（加载→清洗→标准化）
├── transforms.py         # 数据转换操作（标准化、填充等）
└── splits.py             # 数据集分割策略
```

#### 缺失数据层 (Missing)
```
src/missing/
├── __init__.py
├── generators/
│   ├── __init__.py
│   ├── base.py           # IMissingGenerator 接口
│   ├── mcar.py           # MCAR 实现
│   ├── mar.py            # MAR 实现
│   └── mnar.py           # MNAR 实现
├── imputers/
│   ├── __init__.py
│   ├── base.py           # IImputer 接口
│   ├── zero.py
│   ├── mean.py
│   ├── knn.py
│   └── iterative.py
└── mim_builder.py        # MIM 输入构建器
```

#### 模型层 (Models)
```
src/models/
├── __init__.py
├── base.py               # IModel 接口
├── registry.py           # 模型注册中心
├── mlp.py
├── lstm.py
├── cnn1d.py
└── factory.py            # 统一模型创建
```

#### 训练层 (Training)
```
src/training/
├── __init__.py
├── base.py               # ITrainer 接口
├── engines/
│   ├── __init__.py
│   ├── standard.py       # 标准训练引擎
│   └── mim.py            # MIM 多 MR 训练引擎
├── callbacks.py          # 回调机制（早停、保存等）
└── metrics.py            # 评估指标
```

#### 实验层 (Experiments)
```
src/experiments/
├── __init__.py
├── builder.py            # 实验构建器（组装各个组件）
├── runner.py             # 实验运行器
└── phases/
    ├── __init__.py
    ├── base.py           # IExperimentPhase 接口
    ├── training.py       # 训练阶段
    ├── validation.py     # 验证阶段
    └── testing.py        # 测试阶段
```

---

## 三、具体重构步骤

### 步骤 1: 定义抽象接口

```python
# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import numpy as np
import torch

# ========== 数据加载接口 ==========
@runtime_checkable
class IDataLoader(Protocol):
    """数据加载器接口"""
    
    @abstractmethod
    def load(self, batch_id: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 (features, labels)"""
        ...

# ========== 缺失数据生成接口 ==========
class IMissingGenerator(ABC):
    """缺失数据生成器抽象基类"""
    
    @abstractmethod
    def generate(self, X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
        """生成缺失掩码 (mask)，True 表示缺失"""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

# ========== 插补策略接口 ==========
class IImputer(ABC):
    """插补器抽象基类"""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> "IImputer":
        """基于训练数据学习插补参数"""
        ...
    
    @abstractmethod
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        """执行插补"""
        ...
    
    def fit_transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """便捷方法：先 fit 再 transform"""
        return self.fit(X[~mask]).transform(X)
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

# ========== 模型接口 ==========
class IModel(ABC):
    """神经网络模型接口"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        ...

# ========== 训练引擎接口 ==========
class ITrainer(ABC):
    """训练引擎接口"""
    
    @abstractmethod
    def train(
        self, 
        model: IModel, 
        train_data: list, 
        val_data: list,
        config: dict
    ) -> dict:
        """训练并返回训练历史"""
        ...
```

### 步骤 2: 实现注册中心（插件系统）

```python
# src/core/registry.py
from typing import Type, TypeVar, Generic, Callable
from functools import wraps

T = TypeVar('T')

class Registry(Generic[T]):
    """通用注册中心"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Type[T]] = {}
    
    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """装饰器：注册组件"""
        def decorator(cls: Type[T]) -> Type[T]:
            self._registry[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        if name not in self._registry:
            raise KeyError(f"Unknown {self.name}: {name}. "
                          f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs) -> T:
        """创建组件实例"""
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def list_available(self) -> list[str]:
        return list(self._registry.keys())

# 全局注册中心
MISSING_GENERATORS = Registry("missing_generator")
IMPUTERS = Registry("imputer")
MODELS = Registry("model")
TRAINERS = Registry("trainer")
```

### 步骤 3: 重构数据加载

```python
# src/data/loaders/base.py
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class BatteryDataset:
    """统一数据集格式"""
    features: np.ndarray      # [n_samples, n_features]
    labels: np.ndarray        # [n_samples]
    battery_ids: np.ndarray   # [n_samples]
    metadata: dict            # 额外信息
    
    def split(self, train_ratio=0.5, val_ratio=0.25, seed=42) -> tuple:
        """返回 (train, val, test) 三个 Dataset 对象"""
        ...

class BaseDataLoader(ABC):
    """数据加载器基类"""
    
    @abstractmethod
    def load(self, batch_id: str) -> BatteryDataset:
        ...
    
    @abstractmethod
    def list_batches(self) -> list[str]:
        ...
```

### 步骤 4: 重构缺失数据生成

```python
# src/missing/generators/base.py
import numpy as np
from src.core.interfaces import IMissingGenerator
from src.core.registry import MISSING_GENERATORS

@MISSING_GENERATORS.register("mcar")
class MCARGenerator(IMissingGenerator):
    """完全随机缺失生成器"""
    
    @property
    def name(self) -> str:
        return "MCAR"
    
    def generate(self, X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mask = rng.random(X.shape) < missing_rate
        return mask

@MISSING_GENERATORS.register("mar")
class MARGenerator(IMissingGenerator):
    """随机缺失生成器"""
    
    @property
    def name(self) -> str:
        return "MAR"
    
    def generate(self, X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
        # MAR 逻辑：缺失概率与其他观测值相关
        rng = np.random.default_rng(seed)
        # 实现略...
        return mask

# 使用示例
def create_missing_data(
    X: np.ndarray, 
    mode: str, 
    missing_rate: float, 
    seed: int
) -> np.ndarray:
    """工厂函数：通过名称创建缺失数据"""
    generator = MISSING_GENERATORS.create(mode)
    return generator.generate(X, missing_rate, seed)
```

### 步骤 5: 重构插补策略

```python
# src/missing/imputers/base.py
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from src.core.interfaces import IImputer
from src.core.registry import IMPUTERS

@IMPUTERS.register("zero")
class ZeroImputer(IImputer):
    """零值填充"""
    
    def fit(self, X: np.ndarray) -> "ZeroImputer":
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        X = X_missing.copy()
        X[np.isnan(X)] = 0
        return X
    
    @property
    def name(self) -> str:
        return "zero"

@IMPUTERS.register("mean")
class MeanImputer(IImputer):
    """均值填充"""
    
    def __init__(self):
        self._means = None
    
    def fit(self, X: np.ndarray) -> "MeanImputer":
        self._means = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        X = X_missing.copy()
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self._means[i]
        return X
    
    @property
    def name(self) -> str:
        return "mean"

@IMPUTERS.register("knn")
class KNNImputerWrapper(IImputer):
    """KNN 插补包装器"""
    
    def __init__(self, n_neighbors=5):
        self._imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def fit(self, X: np.ndarray) -> "KNNImputerWrapper":
        self._imputer.fit(X)
        return self
    
    def transform(self, X_missing: np.ndarray) -> np.ndarray:
        return self._imputer.transform(X_missing)
    
    @property
    def name(self) -> str:
        return "knn"
```

### 步骤 6: 重构模型工厂

```python
# src/models/base.py
import torch.nn as nn
from src.core.registry import MODELS

class SOHBaseModel(nn.Module):
    """SOH 预测模型基类"""
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# src/models/mlp.py
from src.models.base import SOHBaseModel
from src.core.registry import MODELS

@MODELS.register("mlp")
class MLPModel(SOHBaseModel):
    """MLP 模型"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.15,
        **kwargs
    ):
        super().__init__(input_dim)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# 使用示例
def create_model(
    model_type: str, 
    input_dim: int, 
    **kwargs
) -> SOHBaseModel:
    """统一模型创建接口"""
    return MODELS.create(model_type, input_dim=input_dim, **kwargs)
```

### 步骤 7: 重构训练引擎

```python
# src/training/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

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
    history: dict
    model_path: Optional[str] = None

class Callback(ABC):
    """训练回调基类"""
    
    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        """返回 False 则停止训练"""
        return True
    
    def on_training_end(self, result: TrainingResult):
        pass

class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter < self.patience

# src/training/engines/standard.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.base import Callback, TrainingConfig, TrainingResult

class StandardTrainer:
    """标准训练引擎"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
    
    def train(
        self,
        model,
        train_data: list[tuple],
        val_data: list[tuple],
        callbacks: Optional[list[Callback]] = None
    ) -> TrainingResult:
        """执行训练"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            model.train()
            train_loss = self._run_epoch(model, train_data, criterion, optimizer)
            
            # 验证阶段
            model.eval()
            val_loss = self._run_epoch(model, val_data, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # 更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            
            # 执行回调
            metrics = {'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch}
            if callbacks:
                should_continue = all(cb.on_epoch_end(epoch, metrics) for cb in callbacks)
                if not should_continue:
                    break
        
        return TrainingResult(
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            history=history
        )
    
    def _run_epoch(self, model, data, criterion, optimizer=None):
        """运行一个 epoch"""
        total_loss = 0
        for X, y in data:
            X, y = X.to(self.device), y.to(self.device)
            
            if optimizer:
                optimizer.zero_grad()
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            if optimizer:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data)
    
    def _get_device(self):
        if self.config.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)

# src/training/engines/mim.py
class MIMTrainer(StandardTrainer):
    """MIM 多 MR 训练引擎"""
    
    def train(
        self,
        model,
        multi_mr_data: list[list[tuple]],  # [mr_level][batch]
        val_data: list[tuple],
        callbacks: Optional[list[Callback]] = None
    ) -> TrainingResult:
        """多缺失率混合训练"""
        # 展平所有 MR 的数据
        flattened_data = []
        for mr_data in multi_mr_data:
            flattened_data.extend(mr_data)
        
        # 使用标准训练流程
        return super().train(model, flattened_data, val_data, callbacks)
```

### 步骤 8: 重构实验构建器（依赖注入）

```python
# src/experiments/builder.py
from dataclasses import dataclass
from typing import Optional
from src.core.registry import MISSING_GENERATORS, IMPUTERS, MODELS, TRAINERS
from src.data.loaders.base import BaseDataLoader
from src.training.base import TrainingConfig

@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # L1: 种子
    seed: int
    
    # L3: 数据
    batch_id: str
    data_loader: str  # 数据加载器名称
    
    # L4: 模型
    model_type: str
    model_kwargs: dict
    
    # L5-L6: MIM 配置
    use_mim: bool
    missing_generator: str  # "mcar", "mar", "mnar"
    train_imputation: str   # "zero", "mean", "knn", "iterative"
    missing_rates: list[float]
    
    # 训练配置
    training_config: TrainingConfig
    
    # L7-L9: 测试配置（可选，训练阶段不需要）
    test_mode: Optional[str] = None
    test_missing_rate: Optional[float] = None
    test_imputation: Optional[str] = None

class ExperimentBuilder:
    """实验构建器 - 组装各个组件"""
    
    def __init__(self):
        self._components = {}
    
    def build_data_pipeline(self, config: ExperimentConfig):
        """构建数据管道"""
        # 1. 加载原始数据
        loader = self._get_data_loader(config.data_loader)
        dataset = loader.load(config.batch_id)
        
        # 2. 分割数据集
        train_ds, val_ds, test_ds = dataset.split(seed=config.seed)
        
        # 3. 标准化
        train_ds = self._normalize(train_ds)
        val_ds = self._normalize(val_ds, stats_from=train_ds)
        
        return train_ds, val_ds, test_ds
    
    def build_missing_data_pipeline(
        self, 
        train_data,
        config: ExperimentConfig
    ):
        """构建缺失数据处理管道"""
        # 1. 获取缺失生成器
        generator = MISSING_GENERATORS.create(config.missing_generator)
        
        # 2. 获取插补器
        imputer = IMPUTERS.create(config.train_imputation)
        
        # 3. 为每个 MR 生成训练数据
        multi_mr_data = []
        for mr in config.missing_rates:
            if mr == 0:
                data = train_data
            else:
                # 生成缺失
                mask = generator.generate(train_data.features, mr, config.seed)
                X_missing = train_data.features.copy()
                X_missing[mask] = np.nan
                
                # 插补
                X_imputed = imputer.fit(train_data.features).transform(X_missing)
                
                # 构建 MIM 输入
                data = self._build_mim_input(X_imputed, mask)
            
            multi_mr_data.append(data)
        
        return multi_mr_data, generator, imputer
    
    def build_model(self, config: ExperimentConfig):
        """构建模型"""
        input_dim = 32 if config.use_mim else 16
        return MODELS.create(
            config.model_type,
            input_dim=input_dim,
            **config.model_kwargs
        )
    
    def build_trainer(self, config: ExperimentConfig):
        """构建训练器"""
        if config.use_mim:
            return MIMTrainer(config.training_config)
        else:
            return StandardTrainer(config.training_config)
    
    def _get_data_loader(self, name: str) -> BaseDataLoader:
        # 数据加载器注册逻辑略...
        pass
    
    def _normalize(self, dataset, stats_from=None):
        # 标准化逻辑略...
        pass
    
    def _build_mim_input(self, X_imputed, mask):
        # MIM 输入构建逻辑略...
        pass
```

### 步骤 9: 简化的实验运行器

```python
# src/experiments/runner.py
import json
from pathlib import Path
from src.experiments.builder import ExperimentBuilder, ExperimentConfig

class ExperimentRunner:
    """实验运行器 - 只负责流程编排"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.builder = ExperimentBuilder()
    
    def run_training(self, config: ExperimentConfig) -> dict:
        """运行训练阶段"""
        print(f"Training: seed={config.seed}, batch={config.batch_id}, "
              f"model={config.model_type}, mim={config.use_mim}")
        
        # 1. 数据准备
        train_ds, val_ds, _ = self.builder.build_data_pipeline(config)
        
        # 2. 构建模型
        model = self.builder.build_model(config)
        
        # 3. 准备训练数据
        if config.use_mim:
            train_data, _, _ = self.builder.build_missing_data_pipeline(train_ds, config)
        else:
            train_data = [(train_ds.features, train_ds.labels)]
        
        # 4. 训练
        trainer = self.builder.build_trainer(config)
        result = trainer.train(model, train_data, [(val_ds.features, val_ds.labels)])
        
        # 5. 保存
        model_path = self._save_model(model, config)
        
        return {
            'best_val_loss': result.best_val_loss,
            'best_epoch': result.best_epoch,
            'model_path': str(model_path)
        }
    
    def run_testing(self, config: ExperimentConfig) -> dict:
        """运行测试阶段"""
        # 1. 加载模型
        model = self._load_model(config)
        
        # 2. 准备测试数据
        _, _, test_ds = self.builder.build_data_pipeline(config)
        
        # 3. 生成缺失并测试
        generator = MISSING_GENERATORS.create(config.test_mode)
        imputer = IMPUTERS.create(config.test_imputation)
        
        mask = generator.generate(test_ds.features, config.test_missing_rate, config.seed)
        X_missing = test_ds.features.copy()
        X_missing[mask] = np.nan
        X_imputed = imputer.fit(test_ds.features).transform(X_missing)
        
        # 4. 评估
        metrics = self._evaluate(model, X_imputed, test_ds.labels)
        
        return metrics
    
    def _save_model(self, model, config: ExperimentConfig) -> Path:
        """保存模型"""
        model_dir = self.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"seed{config.seed}_{config.batch_id}_{config.model_type}"
        if config.use_mim:
            filename += f"_mim_{config.train_imputation}"
        else:
            filename += "_no_mim"
        
        path = model_dir / f"{filename}.pt"
        torch.save({'model': model.state_dict(), 'config': config}, path)
        return path
    
    def _load_model(self, config: ExperimentConfig):
        """加载模型"""
        # 略...
        pass
    
    def _evaluate(self, model, X, y):
        """评估模型"""
        # 略...
        pass
```

### 步骤 10: 主入口简化

```python
# src/main.py (重构后)
import argparse
from src.experiments.runner import ExperimentRunner
from src.experiments.builder import ExperimentConfig, TrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'test', 'batch-test'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=str, required=True)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--use-mim', type=bool, default=False)
    parser.add_argument('--train-imputation', type=str, default='zero')
    # ... 其他参数
    
    args = parser.parse_args()
    
    # 构建配置
    config = ExperimentConfig(
        seed=args.seed,
        batch_id=args.batch,
        data_loader='xjtu',
        model_type=args.model,
        model_kwargs={},
        use_mim=args.use_mim,
        missing_generator='mcar',
        train_imputation=args.train_imputation,
        missing_rates=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        training_config=TrainingConfig(epochs=200, patience=30)
    )
    
    # 运行实验
    runner = ExperimentRunner()
    
    if args.phase == 'train':
        result = runner.run_training(config)
        print(f"Training completed: best_val_loss={result['best_val_loss']:.4f}")
    elif args.phase == 'test':
        result = runner.run_testing(config)
        print(f"Testing completed: MAE={result['mae']:.4f}")

if __name__ == '__main__':
    main()
```

---

## 四、重构前后对比

### 代码行数

| 模块 | 重构前 | 重构后 | 变化 |
|-----|-------|-------|-----|
| `run_experiment.py` | 898 行 | ~100 行 (runner) | -78% |
| `main.py` | 345 行 | ~50 行 | -85% |
| 新增接口/基类 | 0 | ~200 行 | +200 |
| 分散的小模块 | 少 | 多 | 职责更清晰 |

### 耦合度对比

```
重构前：
run_experiment.py ─────┬─────► XJTUDataLoader
                       ├─────► MLP/LSTM/CNN (直接导入)
                       ├─────► set_seed
                       ├─────► numpy/pytorch
                       └─────► sklearn imputers
                       
重构后：
ExperimentRunner ─────► ExperimentBuilder ─────┬─────► IDataLoader (接口)
                                               ├─────► IModel (接口)
                                               ├─────► IImputer (接口)
                                               └─────► ITrainer (接口)
                                               
具体实现通过 Registry 动态加载，Runner 不直接依赖任何具体类
```

### 扩展性对比

| 场景 | 重构前 | 重构后 |
|-----|-------|-------|
| 添加新模型 | 修改 `create_model()`, `parse_args()` 等多处 | 只需 `@MODELS.register("new_model")` |
| 添加新插补方法 | 修改 `impute_missing_values()` | 只需 `@IMPUTERS.register("new_imputer")` |
| 添加新缺失模式 | 修改多处 if-else | 只需 `@MISSING_GENERATORS.register("new_mode")` |
| 更换数据加载器 | 修改多处导入和调用 | 修改配置 `data_loader="new_loader"` |

---

## 五、重构实施建议

### 阶段 1: 建立基础设施（1-2天）
1. 创建 `src/core/interfaces.py` 定义所有接口
2. 创建 `src/core/registry.py` 实现注册中心
3. 编写测试确保基础设施正确

### 阶段 2: 重构数据层（2-3天）
1. 实现 `IDataLoader` 接口
2. 重构 `XJTUDataLoader` 实现接口
3. 创建数据管道类
4. 迁移并测试数据加载逻辑

### 阶段 3: 重构缺失数据层（2-3天）
1. 实现 `IMissingGenerator` 和 `IImputer` 接口
2. 将现有缺失生成器、插补器改造为插件
3. 创建 MIM 构建器
4. 测试各策略正确性

### 阶段 4: 重构模型层（1-2天）
1. 实现 `IModel` 接口
2. 将 MLP/LSTM/CNN 注册为插件
3. 简化模型工厂

### 阶段 5: 重构训练层（2-3天）
1. 实现 `ITrainer` 接口
2. 创建标准训练引擎和 MIM 训练引擎
3. 实现回调机制
4. 测试训练流程

### 阶段 6: 重构实验层（2-3天）
1. 实现 `ExperimentBuilder`
2. 重构 `ExperimentRunner`
3. 简化 `main.py`
4. 端到端测试

### 总计: 10-16 天

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 重构引入 bug | 高 | 1. 保持原有测试通过<br>2. 添加新测试覆盖重构代码<br>3. 小步提交，方便回滚 |
| 开发时间超预期 | 中 | 1. 分阶段实施，每阶段可独立交付<br>2. 先重构最迫切需要改的部分 |
| 团队学习成本 | 中 | 1. 编写详细文档<br>2. 代码示例<br>3. 结对编程 |
| 与旧代码兼容 | 低 | 1. 保持旧接口作为适配器<br>2. 逐步迁移，而非一次性替换 |

---

## 七、总结

这份重构方案的核心是：**通过抽象接口和插件机制，将紧耦合的大脚本拆分为松散耦合的小模块**。

**关键收益**:
1. **可测试性**: 每个模块可独立单元测试
2. **可扩展性**: 添加新功能只需新增插件，无需修改现有代码
3. **可维护性**: 每个模块职责单一，代码量减少 70-80%
4. **可复用性**: 组件可在不同实验间复用

**设计模式应用**:
- 策略模式: `IMputer`, `IMissingGenerator`
- 工厂模式: `Registry`, `ExperimentBuilder`
- 模板方法: `BaseTrainer`
- 依赖注入: `ExperimentBuilder` 组装组件
