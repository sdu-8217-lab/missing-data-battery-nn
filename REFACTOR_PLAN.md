# 重构实施计划

## 执行概要

**目标**: 将代码库从74个文件15,486行缩减至35个文件约5,000行，提升可维护性。  
**时间预估**: 8-10个工作日  
**分支**: `refactor-code`  
**验证标准**: 所有现有测试通过 + mypy strict + 覆盖率>80%

---

## 阶段一: 清理与准备

### 1.1 删除冗余文件

```bash
# 创建清理提交
git checkout -b refactor/cleanup

# 1. 删除临时/实验目录
rm -rf experiments_new/
rm -rf .rework/
rm -rf src/analysis/  # 如果分析脚本已过期

# 2. 删除重复工厂
rm src/models/factory_simple.py
rm src/models/model_factory.py

# 3. 删除简化版文件
rm src/missing/missing_data_simple.py
rm src/experiments/simple_runner.py

# 4. 删除未使用的大文件
rm src/missing_data/fixed_feature_missing.py  # 715行，29KB

# 5. 删除重复数据加载
git rm src/data/dataset_loader.py  # 与loader.py重复
# 保留: loader.py (统一入口), xjtu_loader.py (简化版), loader_dataloaders.py (需要合并)

# 6. 删除重复训练器
git rm src/trainers/neural_network_trainer.py  # 与Lightning重复

# 7. 清理测试
git rm tests/test_simplified.py
git rm tests/test_p1_mock.py

# 提交
git add .
git commit -m "refactor(cleanup): 删除冗余和重复文件

- 删除3个重复工厂文件，保留factory.py
- 删除简化版临时文件
- 删除未使用的fixed_feature_missing.py (715行)
- 删除重复的数据加载器和训练器
- 清理临时测试文件

减少约2000行代码"
```

### 1.2 统一缺失数据处理

**目标**: 合并 `missing/` 和 `missing_data/`

```bash
mkdir -p src/missing/generators
```

**文件移动和重构**:

```python
# src/missing/generators/base.py (新建)
from abc import ABC, abstractmethod
import numpy as np

class MissingGenerator(ABC):
    """缺失模式生成器基类."""
    
    @abstractmethod
    def generate(
        self, 
        X: np.ndarray, 
        missing_rate: float, 
        seed: int,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成缺失掩码.
        
        Returns:
            (X_missing, mask): 缺失数据和掩码(1=观测,0=缺失)
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...

# src/missing/generators/mcar.py (从missing_data/mcar.py重构)
# src/missing/generators/mar.py (从missing_data/mar.py重构)
# src/missing/generators/mnar.py (从missing_data/mnar.py重构)

# src/missing/imputation.py (合并missing_data/imputation.py)
# 保留简单函数接口，内部使用imputers/
```

**删除旧目录**:
```bash
rm -rf src/missing_data/
# src/missing/imputers/ 已存在，保持不变
```

### 1.3 统一数据加载

**合并策略**:

```python
# src/data/loader.py (重构，统一入口)
"""统一数据加载接口."""

from pathlib import Path
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class DataLoader(Protocol):
    """数据加载器协议."""
    
    def load(self, batch_id: str) -> tuple[NDArray, NDArray, list[str]]:
        """加载数据.
        
        Returns:
            (X, y, battery_ids): 特征、标签、电池ID列表
        """
        ...

class XJTULoader:
    """XJTU数据集加载器."""
    
    DATA_DIR = Path("data/XJTU data")
    BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]
    
    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
    
    def load(self, batch_id: str) -> tuple[NDArray, NDArray, list[str]]:
        if batch_id not in self.BATCHES:
            raise ValueError(f"Unknown batch: {batch_id}")
        # 实现...
    
    def list_batches(self) -> list[str]:
        return self.BATCHES.copy()

# 删除 xjtu_loader.py, datasets.py
# 保留 splits.py, transforms.py
```

**提交**:
```bash
git add .
git commit -m "refactor(data): 统一数据加载接口

- 合并3个数据加载器为1个统一接口
- 使用Protocol定义抽象接口
- 删除重复的数据集定义
- 添加类型注解"
```

---

## 阶段二: 核心架构重构

### 2.1 重构核心接口

```python
# src/core/types.py (新建)
"""核心类型定义."""

from typing import TypeVar, NewType
from numpy.typing import NDArray

# 数组类型
Features = NDArray[np.float32]
Labels = NDArray[np.float32]
Mask = NDArray[np.bool_]  # 缺失掩码

# 标识类型
Seed = NewType("Seed", int)
BatchId = NewType("BatchId", str)
MissingRate = NewType("MissingRate", float)

# 泛型
T = TypeVar("T")
```

```python
# src/core/interfaces.py (重构)
"""纯接口定义，无实现."""

from typing import Protocol, runtime_checkable
from abc import abstractmethod
from .types import Features, Labels, Mask

@runtime_checkable
class Model(Protocol):
    """模型协议."""
    
    def __call__(self, x: Features) -> Labels: ...
    def parameters(self): ...
    def to(self, device): ...
    def eval(self): ...

@runtime_checkable  
class Trainer(Protocol):
    """训练器协议."""
    
    @abstractmethod
    def fit(
        self, 
        model: Model, 
        train_data: tuple[Features, Labels],
        val_data: tuple[Features, Labels] | None = None
    ) -> "TrainingResult": ...

@runtime_checkable
class Imputer(Protocol):
    """插补器协议."""
    
    @abstractmethod
    def fit(self, X: Features) -> "Imputer": ...
    
    @abstractmethod
    def transform(self, X_missing: Features) -> Features: ...
    
    def fit_transform(self, X: Features, mask: Mask | None = None) -> Features:
        if mask is not None:
            X_missing = X.copy()
            X_missing[~mask] = np.nan
        else:
            X_missing = X
        return self.fit(X).transform(X_missing)
```

### 2.2 重构模型工厂

```python
# src/models/factory.py (重构，约150行)
"""模型创建工厂."""

from typing import Literal
import torch.nn as nn
from .mlp import MLP
from .lstm import LSTM
from .cnn import CNN1D  # 重命名cnn1d.py -> cnn.py

ModelType = Literal["mlp", "lstm", "cnn"]

def create_model(
    model_type: ModelType,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """创建模型.
    
    Args:
        model_type: 模型类型
        input_dim: 输入维度 (16 for non-MIM, 32 for MIM)
        **kwargs: 模型特定参数
        
    Raises:
        ValueError: 未知模型类型
    """
    constructors = {
        "mlp": _create_mlp,
        "lstm": _create_lstm,
        "cnn": _create_cnn,
    }
    
    if model_type not in constructors:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return constructors[model_type](input_dim, **kwargs)

def _create_mlp(input_dim: int, **kwargs) -> MLP:
    defaults = {"hidden_dims": [128, 96] if input_dim == 32 else [192, 96]}
    defaults.update(kwargs)
    return MLP(input_dim=input_dim, **defaults)

def _create_lstm(input_dim: int, **kwargs) -> LSTM:
    defaults = {"hidden_size": 46, "num_layers": 2}
    defaults.update(kwargs)
    return LSTM(input_dim=input_dim, **defaults)

def _create_cnn(input_dim: int, **kwargs) -> CNN1D:
    defaults = {"channels": [64, 80], "kernel_size": 3}
    defaults.update(kwargs)
    return CNN1D(input_dim=input_dim, **defaults)

def count_parameters(model: nn.Module) -> int:
    """计算可训练参数数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_parameter_budget(
    model: nn.Module,
    min_params: int = 16_384,
    max_params: int = 32_768
) -> bool:
    """检查参数量是否在预算范围内."""
    n = count_parameters(model)
    return min_params <= n <= max_params
```

### 2.3 重构训练模块

```python
# src/training/trainer.py (新建，约100行)
"""训练器实现."""

from dataclasses import dataclass
from typing import Callable
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from ..core.types import Features, Labels
from ..core.interfaces import Model

@dataclass(frozen=True)
class TrainingConfig:
    """训练配置."""
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 15
    device: str = "auto"

@dataclass(frozen=True)
class TrainingResult:
    """训练结果."""
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    history: dict[str, list[float]]

class LightningTrainer:
    """PyTorch Lightning训练器包装."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._trainer: pl.Trainer | None = None
    
    def fit(
        self,
        model: pl.LightningModule,
        train_data: tuple[Features, Labels],
        val_data: tuple[Features, Labels] | None = None
    ) -> TrainingResult:
        """训练模型."""
        X_train, y_train = train_data
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # 配置Lightning Trainer
        callbacks = [
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min"
            )
        ]
        
        self._trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=self.config.device,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        
        self._trainer.fit(model, train_loader, val_loader)
        
        return TrainingResult(
            best_val_loss=self._trainer.callback_metrics["val_loss"].item(),
            best_epoch=self._trainer.current_epoch,
            total_epochs=self._trainer.current_epoch,
            history={}  # 可从trainer获取
        )
```

### 2.4 重构实验入口

```python
# experiments/train.py (重构，约150行)
"""训练入口 - 分界线以上(L1-L6)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_config
from src.data.loader import XJTULoader
from src.data.splits import split_batteries
from src.data.transforms import build_features
from src.models.factory import create_model, check_parameter_budget
from src.training.trainer import LightningTrainer, TrainingConfig
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOH prediction model")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "lstm", "cnn"])
    parser.add_argument("--use-mim", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", type=str, default="models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # L1: 设置种子
    set_seed(args.seed)
    
    # L2-L3: 加载数据
    loader = XJTULoader()
    df = loader.load_batch(args.batch)
    
    # 特征工程和数据划分
    X, y = build_features(df)
    splits = split_batteries(df["battery_id"].unique(), seed=args.seed)
    
    # L4-L6: 创建和训练模型
    input_dim = 32 if args.use_mim else 16
    model = create_model(args.model, input_dim=input_dim)
    
    if not check_parameter_budget(model):
        print("Warning: Model parameter count outside budget")
    
    # 训练
    config = TrainingConfig(epochs=args.epochs)
    trainer = LightningTrainer(config)
    
    # TODO: 处理MIM训练逻辑
    result = trainer.fit(model, (X_train, y_train), (X_val, y_val))
    
    # 保存模型
    output_path = Path(args.output) / f"{args.model}_{args.batch}_{args.seed}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ... 保存逻辑
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# experiments/evaluate.py (重构，约150行)
"""评估入口 - 分界线以下(L7-L9)."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="MCAR", choices=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--imputation", type=str, default="mean", 
                       choices=["mean", "knn", "iterative", "zero"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # 加载模型
    # 加载测试数据
    # 生成缺失 (L7)
    # 应用插补 (L8)
    # 评估 (L9)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

## 阶段三: 配置系统统一

### 3.1 配置结构定义

```python
# src/config/schema.py (重构pydantic_config.py)
"""配置结构定义."""

from dataclasses import dataclass, field
from typing import Literal, List


@dataclass(frozen=True)
class DataConfig:
    """数据配置."""
    dataset: Literal["xjtu"] = "xjtu"
    batch: str = "2C"
    data_dir: str = "data/XJTU data"


@dataclass(frozen=True)
class ModelConfig:
    """模型配置."""
    type: Literal["mlp", "lstm", "cnn"] = "mlp"
    # 通用参数
    dropout: float = 0.15
    # MLP参数
    hidden_dims: List[int] = field(default_factory=lambda: [192, 96])
    # LSTM参数
    hidden_size: int = 46
    num_layers: int = 2
    # CNN参数
    channels: List[int] = field(default_factory=lambda: [64, 80])
    kernel_size: int = 3


@dataclass(frozen=True)
class MIMConfig:
    """MIM配置."""
    enabled: bool = False
    train_missing_rates: List[float] = field(
        default_factory=lambda: [i * 0.05 for i in range(20)]
    )


@dataclass(frozen=True)
class TrainingConfig:
    """训练配置."""
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 15


@dataclass(frozen=True)
class Config:
    """完整配置."""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mim: MIMConfig = field(default_factory=MIMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

### 3.2 配置加载

```python
# src/config/loader.py (新建)
"""配置加载."""

from pathlib import Path
from omegaconf import OmegaConf
from .schema import Config


def load_config(path: Path | str) -> Config:
    """从YAML加载配置."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_object(cfg, Config)


def merge_configs(base: Config, overrides: dict) -> Config:
    """合并配置覆盖."""
    base_dict = OmegaConf.structured(base)
    merged = OmegaConf.merge(base_dict, OmegaConf.create(overrides))
    return OmegaConf.to_object(merged, Config)
```

---

## 阶段四: 测试重构

### 4.1 测试结构

```python
# tests/conftest.py (新建)
"""测试共享fixture."""

import pytest
import numpy as np


@pytest.fixture
def sample_features() -> np.ndarray:
    """样本特征数据."""
    return np.random.randn(100, 16).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """样本标签数据."""
    return np.random.randn(100).astype(np.float32)


@pytest.fixture
def sample_missing_mask() -> np.ndarray:
    """样本缺失掩码."""
    return np.random.rand(100, 16) > 0.3
```

```python
# tests/unit/test_models.py (合并模型测试)
"""模型单元测试."""

import torch
import pytest
from src.models.factory import create_model, count_parameters


class TestModelFactory:
    """测试模型工厂."""
    
    @pytest.mark.parametrize("model_type", ["mlp", "lstm", "cnn"])
    @pytest.mark.parametrize("input_dim", [16, 32])
    def test_create_model(self, model_type: str, input_dim: int):
        """测试模型创建."""
        model = create_model(model_type, input_dim=input_dim)
        assert model is not None
        
        # 测试前向传播
        if model_type == "mlp":
            x = torch.randn(10, input_dim)
        else:
            x = torch.randn(10, 1, input_dim)
        
        y = model(x)
        assert y.shape == (10,)
    
    def test_parameter_budget(self):
        """测试参数量预算."""
        for input_dim in [16, 32]:
            for model_type in ["mlp", "lstm", "cnn"]:
                model = create_model(model_type, input_dim=input_dim)
                n_params = count_parameters(model)
                assert 16_384 <= n_params <= 32_768, \
                    f"{model_type} with input_dim={input_dim} has {n_params} params"
```

```python
# tests/integration/test_pipeline.py (新建)
"""集成测试 - 端到端流程."""

import tempfile
from pathlib import Path
import numpy as np
from src.data.loader import XJTULoader
from src.models.factory import create_model
from src.training.trainer import LightningTrainer, TrainingConfig


class TestPipeline:
    """测试完整流程."""
    
    def test_train_and_evaluate(self):
        """测试训练和评估流程."""
        # 1. 加载数据
        # 2. 创建模型
        # 3. 训练
        # 4. 评估
        pass
```

---

## 阶段五: 文档更新

### 5.1 文档结构

```markdown
# docs/ARCHITECTURE.md

## 架构概述
...

## 核心概念
### Battery-wise Split
...

### MIM (Missing Indicator Method)
...

## 模块说明
### data - 数据层
...

### models - 模型层
...

### training - 训练层
...

## 扩展指南
...
```

### 5.2 README精简

```markdown
# Battery SOH Prediction with Missing Data

[![Tests](...)](...)

## Quick Start

```bash
# 训练
python experiments/train.py --seed 42 --batch 2C --model mlp

# 评估
python experiments/evaluate.py --model-path models/mlp_2C_42.pt
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Contributing](docs/CONTRIBUTING.md)

## License
MIT
```

---

## 验证清单

每个阶段完成后检查:

- [ ] `python -m pytest tests/` 全部通过
- [ ] `python -m mypy src/ --strict` 无错误
- [ ] `python -m pytest --cov=src --cov-report=term-missing` 覆盖率>80%
- [ ] 代码行数符合目标
- [ ] 文件数量符合目标
- [ ] 文档已更新

---

*计划制定时间: 2026-04-10*
