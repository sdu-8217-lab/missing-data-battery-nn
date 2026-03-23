# 简化架构方案：利用现有工具，删除冗余代码

## 核心原则

1. **不要重新造轮子** - 使用 Hydra + PyTorch Lightning
2. **函数优于类** - 简单逻辑用函数，复杂逻辑用类
3. **配置优于代码** - 可变参数放 YAML，不变逻辑放代码
4. **合并相似文件** - 减少文件数量

---

## 方案对比

### 我的过度设计（不好）

```python
# src/core/interfaces.py - 200+行定义接口
class IImputer(Protocol):
    @abstractmethod
    def fit(self, X): ...
    @abstractmethod
    def transform(self, X): ...

# src/core/registry.py - 300+行实现注册中心
class Registry(Generic[T]):
    def register(self, name): ...
    def create(self, name, **kwargs): ...

# src/missing/imputers/zero.py - 50行一个简单类
@IMPUTERS.register("zero")
class ZeroImputer(BaseImputer):
    def fit(self, X): return self
    def transform(self, X): return np.nan_to_num(X, nan=0.0)

# 使用
imputer = IMPUTERS.create("zero")
```

**问题**: 30+ 行配置代码，只为封装 1 行实际逻辑

---

### 简化方案（好）

```python
# src/missing/imputation.py - 100行包含所有方法
import numpy as np
from typing import Literal

def impute(
    X: np.ndarray, 
    method: Literal['zero', 'mean', 'knn', 'iterative'],
    fit_data: np.ndarray = None
) -> np.ndarray:
    '''统一的插补函数'''
    if method == 'zero':
        return np.nan_to_num(X, nan=0.0)
    
    elif method == 'mean':
        means = np.nanmean(fit_data, axis=0)
        return _fill_with_value(X, means)
    
    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer().fit(fit_data)
        return imputer.transform(X)
    
    elif method == 'iterative':
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer().fit(fit_data)
        return imputer.transform(X)

# 使用
X_filled = impute(X_missing, method='mean', fit_data=X_train)
```

**优势**: 
- 1 个文件 vs 5 个文件
- 1 个函数 vs 4 个类
- 100 行 vs 600+ 行

---

## 完整简化架构

### 目录结构（从 50+ 文件简化到 15 文件）

```
# 删除这些冗余目录
❌ src/core/           # 删除 - 用 Hydra 替代
❌ src/missing/imputers/  # 删除 - 合并为单个文件
❌ src/missing/generators/ # 删除 - 合并为单个文件

# 保留并简化
✅ src/
├── missing/
│   ├── __init__.py
│   ├── missing_data.py     # 100行: MCAR/MAR/MNAR生成 + 插补
│   └── mim.py              # 50行: MIM输入构建
│
├── data/
│   ├── __init__.py
│   ├── loader.py           # 100行: XJTU数据加载
│   └── pipeline.py         # 100行: 标准化 + 分割
│
├── models/
│   ├── __init__.py
│   ├── factory.py          # 50行: 简单的模型创建函数
│   └── models.py           # 150行: MLP/LSTM/CNN定义
│
├── training/
│   ├── __init__.py
│   └── lightning_module.py # 200行: Lightning模块
│
└── experiments/
    ├── __init__.py
    ├── config.py           # 50行: 配置数据结构
    └── runner.py           # 150行: 简化版运行器

# 配置文件（已有，直接使用）
configs/
├── experiment.yaml         # 实验配置
├── model/
│   ├── mlp.yaml
│   ├── lstm.yaml
│   └── cnn.yaml
└── data/
    └── xjtu.yaml
```

---

## 具体实现

### 1. 简化插补（100行替代600行）

```python
# src/missing/missing_data.py
"""缺失数据生成与插补 - 简化版"""

import numpy as np
from typing import Literal
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def generate_missing(
    X: np.ndarray, 
    mode: Literal['mcar', 'mar', 'mnar'], 
    missing_rate: float, 
    seed: int
) -> np.ndarray:
    """生成缺失掩码"""
    rng = np.random.default_rng(seed)
    
    if mode == 'mcar':
        return rng.random(X.shape) < missing_rate
    
    elif mode == 'mar':
        # 缺失与观测值相关
        # 简化版: 基于第一个特征的值决定缺失概率
        feature = X[:, 0]
        probs = missing_rate * (0.5 + 0.5 * (feature - feature.min()) / 
                                (feature.max() - feature.min() + 1e-8))
        return rng.random(X.shape[0]) < probs
    
    elif mode == 'mnar':
        # 缺失与缺失值本身相关
        # 简化版: 基于目标值决定缺失概率
        raise NotImplementedError("MNAR not implemented in simplified version")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def impute(
    X: np.ndarray,
    method: Literal['zero', 'mean', 'knn', 'iterative'],
    fit_data: np.ndarray = None
) -> np.ndarray:
    """插补缺失值"""
    
    if method == 'zero':
        return np.nan_to_num(X, nan=0.0)
    
    elif method == 'mean':
        if fit_data is None:
            raise ValueError("fit_data required for mean imputation")
        means = np.nanmean(fit_data, axis=0)
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X_filled[mask, i] = means[i]
        return X_filled
    
    elif method == 'knn':
        if fit_data is None:
            raise ValueError("fit_data required for knn imputation")
        imputer = KNNImputer(n_neighbors=5).fit(fit_data)
        return imputer.transform(X)
    
    elif method == 'iterative':
        if fit_data is None:
            raise ValueError("fit_data required for iterative imputation")
        imputer = IterativeImputer(max_iter=10, random_state=42).fit(fit_data)
        return imputer.transform(X)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def build_mim_input(X_imputed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """构建 MIM 输入: [X_imputed | mask]"""
    return np.concatenate([X_imputed, mask.astype(float)], axis=1)
```

### 2. 简化模型工厂（50行替代300行）

```python
# src/models/factory.py
"""模型创建 - 简化版"""

import torch.nn as nn
from typing import Literal


def create_model(
    model_type: Literal['mlp', 'lstm', 'cnn'],
    input_dim: int,
    **kwargs
) -> nn.Module:
    """创建模型"""
    
    if model_type == 'mlp':
        from src.models.models import MLP
        hidden_dims = kwargs.get('hidden_dims', [128, 96] if input_dim == 32 else [192, 96])
        return MLP(input_dim, hidden_dims)
    
    elif model_type == 'lstm':
        from src.models.models import LSTM
        hidden_size = kwargs.get('hidden_size', 46)
        return LSTM(input_dim, hidden_size)
    
    elif model_type == 'cnn':
        from src.models.models import CNN1D
        channels = kwargs.get('channels', [64, 80])
        return CNN1D(input_dim, channels)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 3. 简化实验运行器（150行替代898行）

```python
# src/experiments/runner.py
"""实验运行器 - 简化版，使用 Hydra + Lightning"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from omegaconf import DictConfig
import numpy as np

from src.data.loader import load_xjtu_data
from src.models.factory import create_model
from src.missing.missing_data import generate_missing, impute, build_mim_input
from src.training.lightning_module import SOHLightningModule


def run_experiment(cfg: DictConfig):
    """运行完整实验"""
    
    # 1. 加载数据
    data = load_xjtu_data(cfg.data.batch, cfg.data.dir)
    X_train, y_train, X_val, y_val = split_data(data, cfg.seed)
    
    # 2. 标准化
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # 3. 准备训练数据
    if cfg.model.use_mim:
        # MIM: 多 MR 训练
        train_data = prepare_mim_data(
            X_train, y_train,
            missing_rates=cfg.missing.rates,
            imputation=cfg.missing.imputation,
            seed=cfg.seed
        )
        input_dim = 32
    else:
        # Baseline: 完整数据
        train_data = [(X_train, y_train)]
        input_dim = 16
    
    # 4. 创建模型
    model = create_model(cfg.model.type, input_dim, **cfg.model.get('kwargs', {}))
    
    # 5. 创建 Lightning 模块
    pl_module = SOHLightningModule(model, cfg.training)
    
    # 6. 训练
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=cfg.training.patience)],
        accelerator='gpu' if cfg.training.get('use_gpu', False) else 'cpu'
    )
    trainer.fit(pl_module, train_dataloader(train_data), val_dataloader(X_val, y_val))
    
    # 7. 测试（如果配置）
    if cfg.get('test'):
        results = test_model(model, X_val, y_val, cfg.test)
        return results
    
    return trainer


def prepare_mim_data(X, y, missing_rates, imputation, seed):
    """准备 MIM 多 MR 训练数据"""
    data = []
    for mr in missing_rates:
        if mr == 0:
            X_mr = X
            mask = np.zeros_like(X)
        else:
            mask = generate_missing(X, 'mcar', mr, seed + int(mr * 100))
            X_missing = X.copy()
            X_missing[mask] = np.nan
            X_mr = impute(X_missing, imputation, fit_data=X)
        
        X_mim = build_mim_input(X_mr, mask)
        data.append((X_mim, y))
    
    return data


# 辅助函数
def split_data(data, seed): ...
def train_dataloader(data): ...
def val_dataloader(X, y): ...
def test_model(model, X, y, test_cfg): ...
```

### 4. Hydra 配置（已有，直接使用）

```yaml
# configs/experiment.yaml
defaults:
  - model: mlp
  - data: xjtu

seed: 42
batch: 2C

model:
  use_mim: true
  
missing:
  rates: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
          0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
  imputation: mean  # zero, mean, knn, iterative

training:
  epochs: 200
  patience: 30
  lr: 0.001
```

---

## 代码量对比

| 方案 | 文件数 | 代码行数 | 复杂度 |
|-----|-------|---------|-------|
| **当前（我的过度设计）** | 50+ | 3000+ | 高 |
| **简化方案** | 15 | ~1000 | 低 |
| **原始脚本** | 20 | 2000+ | 中 |

---

## 关键决策

### 何时使用类？何时使用函数？

```python
# ❌ 过度：简单逻辑用类
class ZeroImputer(BaseImputer):
    def fit(self, X): return self
    def transform(self, X): return np.nan_to_num(X, nan=0.0)

# ✅ 恰当：简单逻辑用函数
def impute_zero(X):
    return np.nan_to_num(X, nan=0.0)

# ✅ 恰当：复杂状态用类
class MIMTrainer:
    def __init__(self, multi_mr_data):
        self.multi_mr_data = multi_mr_data
    
    def train_epoch(self, model):
        # 复杂的多 MR 训练逻辑
        ...
```

### 何时使用 Hydra？何时使用 Registry？

```python
# ❌ 过度：自己实现 Registry
@IMPUTERS.register("mean")
class MeanImputer: ...
imputer = IMPUTERS.create("mean")

# ✅ 恰当：使用 Hydra
# config.yaml:
# imputer:
#   _target_: src.missing.missing_data.impute
#   method: mean

imputer = hydra.utils.instantiate(cfg.imputer)
```

---

## 实施建议

### 阶段 1: 删除冗余代码（1天）
```bash
# 删除我创建的过度设计代码
rm -rf src/core/
rm -rf src/missing/imputers/
rm -f src/missing/imputation_utils.py
```

### 阶段 2: 创建简化版核心（2-3天）
```bash
# 创建合并后的简化文件
touch src/missing/missing_data.py      # 100行
touch src/models/factory.py             # 50行
touch src/experiments/runner.py         # 150行
```

### 阶段 3: 迁移到 Hydra + Lightning（2-3天）
- 将 `run_experiment.py` 功能整合到 Lightning 模块
- 使用 Hydra 配置替代硬编码参数
- 保持现有实验脚本作为适配器

### 总计: 5-7天（vs 原来计划的10-16天）

---

## 总结

**好的设计**:
- 利用成熟工具（Hydra, Lightning）而非自己实现
- 函数优先，类仅在需要状态时使用
- 配置优先，代码保持简洁

**我的错误**:
- 过度抽象，为这个规模的项目创建复杂架构
- 重新实现 Hydra 已提供的功能
- 文件过度拆分，增加维护负担

**推荐路径**:
1. **删除**我创建的 `src/core/` 和 `src/missing/imputers/`
2. **合并**功能到少量文件
3. **利用**已有的 Hydra + Lightning 框架
4. **简化**配置和运行流程
