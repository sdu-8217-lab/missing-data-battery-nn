# 架构重构快速指南

## 当前 vs 目标架构对比

### 当前架构（紧耦合）

```python
# run_experiment.py - 898行
import numpy as np
import torch
from src.data.xjtu_loader import XJTUDataLoader
from src.models.mlp import MLP  # 硬编码导入

def prepare_training_data(batch_id, seed, use_mim, model_type):
    # 数据加载 + 标准化 + 缺失生成 + 插补 + MIM构建 - 全部混在一起
    loader = XJTUDataLoader()
    df = loader.load_data()
    # ... 200+行代码处理所有逻辑
    
    if use_mim:
        for mr in [0.0, 0.1, ...]:  # 硬编码缺失率
            mask = np.random.rand(...) < mr
            X_mr = X_train.copy()
            X_mr[mask] = 0  # 硬编码zero插补
            # ...

def create_model(model_type, input_dim, use_mim):
    # 硬编码模型创建
    if model_type == 'mlp':
        return MLP(input_dim, [128, 96])
    elif model_type == 'lstm':
        return LSTM(input_dim, 46)
    # ... 添加新模型需修改此处

def train_model(args):
    # 数据准备 + 模型创建 + 训练循环 + 评估 + 保存 - 全部混在一起
    data = prepare_training_data(...)
    model = create_model(...)
    # ... 300+行代码
```

**问题**: 
- 添加新模型需要修改 `create_model()`
- 添加新插补方法需要修改 `prepare_training_data()`
- 无法单独测试数据准备逻辑
- 无法复用训练逻辑到其他数据集

---

### 目标架构（松耦合）

```python
# src/core/interfaces.py
from typing import Protocol
import numpy as np

class IImputer(Protocol):
    def fit(self, X: np.ndarray) -> "IImputer": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class IMissingGenerator(Protocol):
    def generate(self, X: np.ndarray, mr: float, seed: int) -> np.ndarray: ...

# src/core/registry.py
class Registry:
    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator
    def create(self, name: str, **kwargs):
        return self._registry[name](**kwargs)

IMPUTERS = Registry("imputer")
MISSING_GENERATORS = Registry("missing_generator")
MODELS = Registry("model")

# src/missing/imputers/mean.py
from src.core.registry import IMPUTERS

@IMPUTERS.register("mean")
class MeanImputer:
    def fit(self, X): ...
    def transform(self, X): ...

# src/missing/imputers/knn.py
@IMPUTERS.register("knn")
class KNNImputer:
    ...

# src/experiments/builder.py
class ExperimentBuilder:
    def build_missing_pipeline(self, config):
        generator = MISSING_GENERATORS.create(config.missing_mode)
        imputer = IMPUTERS.create(config.train_imputation)
        # 两者通过接口交互，互不依赖具体实现
        return generator, imputer

# src/experiments/runner.py
class ExperimentRunner:
    def __init__(self):
        self.builder = ExperimentBuilder()
    
    def run(self, config):
        # 流程编排，无业务逻辑
        generator, imputer = self.builder.build_missing_pipeline(config)
        model = self.builder.build_model(config)
        trainer = self.builder.build_trainer(config)
        return trainer.train(model, ...)

# 使用
from src.experiments.runner import ExperimentRunner
from src.config import ExperimentConfig

config = ExperimentConfig(
    missing_mode="mcar",      # 可替换为 "mar", "mnar"
    train_imputation="mean",  # 可替换为 "knn", "iterative", "zero"
    model_type="mlp",         # 可替换为 "lstm", "cnn"
)
runner = ExperimentRunner()
result = runner.run(config)
```

**优势**:
- 添加新插补方法只需创建新类并 `@IMPUTERS.register("new")`
- 添加新模型只需 `@MODELS.register("new_model")`
- 每个组件可独立测试
- 组件可自由组合

---

## 核心重构技巧

### 1. 识别变化点，封装抽象

```python
# ❌ 坏：变化点硬编码
if imputation == 'mean':
    X = mean_impute(X)
elif imputation == 'knn':
    X = knn_impute(X)
# 添加新方法需要修改这里

# ✅ 好：变化点封装为策略
imputer = IMPUTERS.create(imputation)  # 运行时决定
X = imputer.fit_transform(X)
# 添加新方法只需注册，无需修改此处
```

### 2. 依赖注入，而非硬编码

```python
# ❌ 坏：直接实例化依赖
def train():
    loader = XJTUDataLoader()  # 硬编码
    model = MLP()              # 硬编码

# ✅ 好：依赖注入
def train(loader: IDataLoader, model: IModel):
    # 不依赖具体实现，只依赖接口
    data = loader.load()
    model.fit(data)
```

### 3. 单一职责，拆分大函数

```python
# ❌ 坏：一个函数做所有事（200+行）
def run_experiment(args):
    # 加载数据
    # 预处理
    # 生成缺失
    # 插补
    # 创建模型
    # 训练
    # 评估
    # 保存结果

# ✅ 好：每个函数只做一件事
def load_data(...) -> Dataset: ...
def generate_missing(...) -> Mask: ...
def impute(...) -> ImputedData: ...
def train_model(...) -> TrainedModel: ...
def evaluate(...) -> Metrics: ...

def run_experiment(args):
    # 流程编排，无具体逻辑
    data = load_data(...)
    mask = generate_missing(...)
    imputed = impute(data, mask)
    model = train_model(imputed)
    return evaluate(model)
```

---

## 迁移路径建议

### 阶段 1: 先创建新接口（不破坏旧代码）

```bash
# 1. 创建新目录结构
mkdir -p src/core src/missing/generators src/missing/imputers

# 2. 定义接口（不修改旧代码）
# src/core/interfaces.py

# 3. 创建适配器让旧代码兼容新接口
# src/adapters/legacy_adapter.py
```

### 阶段 2: 逐步迁移组件

```bash
# 每次迁移一个组件，保持测试通过

# 第1周：迁移插补方法
# - 创建 IImputer 接口
# - 将 ZeroImputer, MeanImputer 改造为插件
# - 修改一处使用点，测试通过后再修改下一处

# 第2周：迁移缺失生成器
# - 创建 IMissingGenerator 接口
# - 改造 MCAR, MAR, MNAR

# 第3周：迁移模型
# - 创建 IModel 接口
# - 改造 MLP, LSTM, CNN

# 第4周：迁移训练逻辑
# - 创建 ITrainer 接口
# - 重构训练流程
```

### 阶段 3: 移除旧代码

```bash
# 当所有代码都使用新接口后
# 1. 删除旧的 monolithic 脚本
# 2. 更新入口函数
# 3. 全面测试
```

---

## 检查清单

重构一个模块前问自己：

- [ ] 这个模块是否有单一、明确的职责？
- [ ] 它是否依赖抽象接口而非具体实现？
- [ ] 添加新功能是否需要修改这个模块？
- [ ] 是否可以单独测试这个模块？
- [ ] 这个模块是否可以在其他上下文中复用？

如果答案都是"是"，恭喜，这是一个高内聚、低耦合的模块！
