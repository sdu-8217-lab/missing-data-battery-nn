# 架构可视化对比

## 依赖关系图

### 当前架构（紧耦合）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     experiments/run_experiment.py                        │
│                              898 lines                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│   │ Data Loading │  │ Missing Gen  │  │ Imputation   │  │ Training  │  │
│   │   150 lines  │  │   100 lines  │  │   80 lines   │  │ 200 lines │  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│          │                 │                 │                │        │
│          ▼                 ▼                 ▼                ▼        │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                        Direct Dependencies                       │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │  │
│   │  │XJTULoader│  │np.random │  │  sklearn │  │  torch.nn     │  │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │  │
│   │  │  MLP()   │  │  LSTM()  │  │  CNN()   │  <-- Hardcoded      │  │
│   │  └──────────┘  └──────────┘  └──────────┘                       │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Problem: Any change requires modifying this file                       │
│            Cannot test components independently                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 目标架构（松耦合）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    src/experiments/runner.py                             │
│                            ~100 lines                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    ExperimentBuilder                             │   │
│   │         (Assembles components, knows interfaces only)            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│          │                │                │                │            │
│          ▼                ▼                ▼                ▼            │
│   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐    │
│   │  Data      │   │  Missing   │   │ Imputation │   │  Training  │    │
│   │  Pipeline  │   │ Generator  │   │  Strategy  │   │   Engine   │    │
│   │  ~50 lines │   │  ~30 lines │   │  ~40 lines │   │  ~60 lines │    │
│   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘    │
│         │                │                │                │           │
│         ▼                ▼                ▼                ▼           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Abstract Interfaces                         │   │
│   │  ┌─────────────────────────────────────────────────────────┐   │   │
│   │  │  IDataLoader   IMissingGenerator   IImputer   ITrainer │   │   │
│   │  └─────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Registry (Plugin System)                      │   │
│   │                                                                  │   │
│   │   @IMPUTERS.register("mean")    @MODELS.register("mlp")         │   │
│   │   class MeanImputer: ...        class MLPModel: ...             │   │
│   │                                                                  │   │
│   │   @MISSING.register("mcar")                                      │   │
│   │   class MCARGenerator: ...                                       │   │
│   │                                                                  │   │
│   │   Add new: Just create class with decorator                      │   │
│   │   No need to modify existing code                                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Benefit: Components are independent and testable                       │
│            Add new features without changing existing code               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 调用流程对比

### 当前流程

```
User
 │
 │  python run_experiment.py --model mlp --imputation mean
 ▼
run_experiment.py
 │
 ├─▶ parse_args() ──▶ 解析参数
 │
 ├─▶ prepare_training_data()
 │   │
 │   ├─▶ XJTUDataLoader().load()  # 硬编码
 │   ├─▶ normalize()
 │   ├─▶ for mr in [0.0, 0.1, ...]:  # 硬编码
 │   │   ├─▶ generate_missing()  # MCAR 逻辑硬编码
 │   │   └─▶ if imputation == 'mean':  # if-else 链
 │   │           mean_impute()
 │   │       elif imputation == 'knn':
 │   │           knn_impute()
 │   │       # 添加新方法需修改此处
 │   │
 │   └─▶ build_mim_input()
 │
 ├─▶ create_model()
 │   └─▶ if model_type == 'mlp':  # if-else 链
 │           return MLP(...)      # 硬编码类
 │       elif model_type == 'lstm':
 │           return LSTM(...)
 │       # 添加新模型需修改此处
 │
 ├─▶ train_model()
 │   ├─▶ for epoch in range(epochs):
 │   │   ├─▶ forward()
 │   │   ├─▶ backward()
 │   │   └─▶ validate()
 │   │
 │   └─▶ save_model()
 │
 └─▶ evaluate_model()
     └─▶ ...
```

### 目标流程

```
User
 │
 │  python main.py --model mlp --imputation mean
 ▼
main.py (50 lines)
 │
 ├─▶ parse_args()
 │
 ├─▶ config = ExperimentConfig(...)  # 纯数据对象
 │
 └─▶ ExperimentRunner().run(config)
     │
     ├─▶ ExperimentBuilder.build_pipeline(config)
     │   │
     │   ├─▶ DataLoaderRegistry.create(config.loader_name)
     │   │   └─▶ XJTUDataLoader().load()  # 通过接口调用
     │   │
     │   ├─▶ MissingGeneratorRegistry.create(config.missing_mode)
     │   │   └─▶ MCARGenerator()  # 运行时决定
     │   │
     │   ├─▶ ImputerRegistry.create(config.imputation)
     │   │   └─▶ MeanImputer()  # 运行时决定
     │   │
     │   └─▶ ModelRegistry.create(config.model_type)
     │       └─▶ MLPModel()  # 运行时决定
     │
     ├─▶ TrainerRegistry.create(config.trainer_type)
     │   └─▶ StandardTrainer().train(model, data)
     │       # 或 MIMTrainer().train(model, multi_mr_data)
     │
     └─▶ save_results()

Add New Component:
   1. Create class implementing interface
   2. Add @Registry.register("name") decorator
   3. Use via config: name
   
   No changes to existing code!
```

---

## 代码量对比

### 当前（紧耦合）

```
experiments/
├── run_experiment.py          898 lines  ████████████████████
├── run_100seeds_final.py      152 lines  ███
├── run_batch_experiments_v2.py 200 lines  ████
└── ... (other scripts)

Total: ~3000+ lines in experiments/
```

### 目标（松耦合）

```
src/
├── core/
│   ├── interfaces.py           50 lines  █
│   └── registry.py             40 lines  █
│
├── data/
│   ├── pipeline.py             60 lines  █
│   └── loaders/
│       └── xjtu.py             50 lines  █
│
├── missing/
│   ├── generators/
│   │   ├── base.py             20 lines  ▏
│   │   ├── mcar.py             15 lines  ▏
│   │   ├── mar.py              20 lines  ▏
│   │   └── mnar.py             25 lines  ▏
│   └── imputers/
│       ├── base.py             15 lines  ▏
│       ├── zero.py             10 lines  ▏
│       ├── mean.py             15 lines  ▏
│       ├── knn.py              15 lines  ▏
│       └── iterative.py        20 lines  ▏
│
├── models/
│   ├── base.py                 20 lines  ▏
│   ├── mlp.py                  30 lines  ▏
│   ├── lstm.py                 35 lines  ▏
│   └── cnn1d.py                40 lines  ▏
│
├── training/
│   ├── base.py                 30 lines  ▏
│   ├── standard.py             60 lines  █
│   └── mim.py                  40 lines  ▏
│
└── experiments/
    ├── runner.py               80 lines  ██
    └── builder.py              70 lines  ██

Total: ~700 lines in src/
       + much more maintainable
```

---

## 扩展性对比

### 添加新插补方法

#### 当前架构
```python
# 需要修改的文件：
# 1. run_experiment.py
# 2. run_batch_experiments_v2.py
# 3. main.py
# 4. 所有使用 imputation 的地方

# 修改内容（多文件）:
if imputation == 'mean':
    X = mean_impute(X)
elif imputation == 'knn':
    X = knn_impute(X)
elif imputation == 'new_method':  # <-- 添加
    X = new_method_impute(X)       # <-- 添加
# Risk: Missing a file causes bugs
```

#### 目标架构
```python
# 只需创建一个新文件：
# src/missing/imputers/new_method.py

from src.core.registry import IMPUTERS

@IMPUTERS.register("new_method")  # <-- 仅此一处
class NewMethodImputer(IImputer):
    def fit(self, X): ...
    def transform(self, X): ...

# 使用：
config.train_imputation = "new_method"  # 自动生效
# No other files need changes!
```

---

## 测试策略对比

### 当前架构

```python
# 难以测试：所有逻辑耦合在一起
def test_prepare_training_data():
    # 必须提供完整的 args 对象
    args = Args(batch="2C", model="mlp", use_mim=True, ...)
    
    # 会实际加载数据、创建模型、执行训练
    result = prepare_training_data(args)
    
    # 无法单独测试插补逻辑
    # 无法单独测试缺失生成逻辑
```

### 目标架构

```python
# 每个组件可独立测试

def test_mean_imputer():
    imputer = MeanImputer()
    X = np.array([[1, np.nan], [2, 3]])
    result = imputer.fit(X).transform(X)
    assert result[0, 1] == 2.0  # 填充均值

def test_mcar_generator():
    gen = MCARGenerator()
    X = np.random.randn(100, 16)
    mask = gen.generate(X, missing_rate=0.3, seed=42)
    assert mask.mean() ≈ 0.3

def test_mlp_model():
    model = MLPModel(input_dim=32, hidden_dims=[64, 32])
    x = torch.randn(10, 32)
    y = model(x)
    assert y.shape == (10,)

# 集成测试
def test_experiment_pipeline():
    # 可以使用 mock 组件
    config = ExperimentConfig(
        data_loader="mock",
        imputation="mean",
        model_type="mlp"
    )
    result = ExperimentRunner().run(config)
    assert result['mae'] < 0.1
```

---

## 总结

| 维度 | 当前架构 | 目标架构 |
|-----|---------|---------|
| **耦合度** | 高（紧耦合） | 低（松耦合） |
| **代码行数** | ~3000 | ~700 |
| **平均文件大小** | 300+ 行 | 30-50 行 |
| **添加新功能** | 修改 N 个文件 | 创建 1 个新文件 |
| **单元测试** | 困难 | 容易 |
| **复用性** | 低 | 高 |
| **维护成本** | 高 | 低 |
| **学习曲线** | 陡峭（需理解整体） | 平缓（只需理解接口） |

**核心转变**: 从 "过程式编程" 转向 "策略模式 + 依赖注入"
