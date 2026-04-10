# 项目代码功能架构系统梳理汇总

> 文档版本: 1.0  
> 基于分支: `refactor-code`  
> 最后更新: 2026-04-10

---

## 一、项目概述

### 1.1 核心目标

本项目是一个基于神经网络的电池健康状态(SOH)预测研究框架，核心研究问题是：

> **在相同插补策略下，添加MIM (Missing Indicator Method) 缺失指示器是否能提升电池SOH预测性能？**

### 1.2 9层实验架构

项目遵循严格的9层层次化实验架构，核心是"分界线原则"：

```
╔══════════════════════════════════════════════════════════════╗
║  分界线以上（影响模型训练）                                   ║
╠══════════════════════════════════════════════════════════════╣
L1: Seed (随机种子)          - 控制可复现性
L2: Dataset (数据集)         - 固定使用XJTU数据集
L3: Batch (电池批次)         - 2C, 3C, R2.5, R3, RW, Sim_satellite
L4: Model (模型架构)         - MLP, LSTM, CNN (3种)
L5: use_mim (MIM使用)        - false/true
L6: Train MR (训练缺失率)    - use_mim=false时为0.0; use_mim=true时多MR训练
╠══════════════════════════════════════════════════════════════╣
║  分界线：训练完成，模型参数固定                              ║
╠══════════════════════════════════════════════════════════════╣
L7: Mode (缺失模式)          - MCAR, MAR, MNAR (仅测试阶段)
L8: Test MR (测试缺失率)     - 0.0-0.95 (仅测试阶段)
L9: Imputation (插补方法)    - mean, knn, iterative, zero
╚══════════════════════════════════════════════════════════════╝
```

**关键理解**:
- 分界线以上 (L1-L6): 每个组合需独立训练模型
- 分界线以下 (L7-L9): 同一模型可复用测试所有组合
- 总实验规模: 100种子 × 6批次 × 3模型 × 2 MIM × 3模式 × 20 MR × 4插补

---

## 二、代码架构总览

### 2.1 目录结构

```
.
├── src/                        # 源代码
│   ├── core/                   # 核心接口定义
│   │   ├── interfaces.py       # 抽象接口 (IDataLoader, IModel, etc.)
│   │   └── registry.py         # 组件注册中心 (插件系统)
│   ├── data/                   # 数据层
│   │   ├── xjtu_loader.py      # XJTU数据加载器
│   │   ├── loader.py           # 通用数据加载
│   │   ├── features.py         # 特征工程
│   │   ├── preprocessing.py    # 数据预处理
│   │   ├── splits.py           # Battery-wise数据划分
│   │   ├── imputation_utils.py # 插补工具函数
│   │   └── loader_dataloaders.py # DataLoader创建
│   ├── models/                 # 模型层
│   │   ├── mlp.py              # MLP模型
│   │   ├── lstm.py             # LSTM模型
│   │   ├── cnn1d.py            # 1D-CNN模型
│   │   ├── gru.py              # GRU模型
│   │   ├── factory.py          # 模型工厂
│   │   └── model_factory.py    # 模型工厂(旧版)
│   ├── trainers/               # 训练器
│   │   ├── lightning_module.py # PyTorch Lightning模块
│   │   ├── lightning_trainer.py # Lightning训练器封装
│   │   └── neural_network_trainer.py # 基础NN训练器
│   ├── missing_data/           # 缺失数据处理(传统方法)
│   │   ├── mcar.py             # MCAR缺失模拟
│   │   ├── mar.py              # MAR缺失模拟
│   │   ├── mnar.py             # MNAR缺失模拟
│   │   └── imputation.py       # 传统插补方法实现
│   ├── missing/imputers/       # 插补器(基于注册中心)
│   │   ├── base.py             # 插补器基类
│   │   ├── mean.py             # 均值插补
│   │   ├── knn.py              # KNN插补
│   │   ├── iterative.py        # 迭代插补
│   │   └── zero.py             # 零值插补
│   ├── evaluation/             # 评估模块
│   │   ├── metrics.py          # 评估指标计算
│   │   ├── model_evaluator.py  # 模型评估器
│   │   └── result_writer.py    # 结果写入器
│   ├── experiments/            # 实验执行模块
│   │   ├── runner.py           # 单实验运行器
│   │   ├── scheduler.py        # 实验调度器(并行)
│   │   ├── database.py         # 实验数据库
│   │   └── simple_runner.py    # 简化版运行器
│   ├── preexperiment/          # 预实验(超参搜索)
│   │   ├── runner.py           # 预实验主运行器
│   │   ├── search_space.py     # 搜索空间定义
│   │   └── objective.py        # Optuna目标函数
│   ├── config/                 # 配置管理
│   │   └── pydantic_config.py  # 结构化配置(dataclass)
│   ├── utils/                  # 工具函数
│   │   ├── seed_manager.py     # 随机种子管理
│   │   ├── logger.py           # 日志工具
│   │   └── param_counter.py    # 参数量统计
│   └── visualization/          # 可视化
│       ├── heatmaps.py         # 热力图
│       ├── missing_rate_curves.py # 缺失率曲线
│       └── batch_plots.py      # 批次对比图
│
├── experiments/                # 实验入口脚本
│   ├── run_experiment.py       # 主实验运行器(支持9层架构)
│   ├── run_batch_experiments_v2.py # 批量实验(多进程)
│   ├── run_100seeds_final.py   # 100种子实验
│   ├── architecture_search.py  # 架构搜索
│   ├── evaluate.py             # 结果评估
│   └── visualization_*.py      # 可视化脚本
│
├── tests/                      # 测试
│   ├── test_critical_path.py   # 关键路径测试
│   ├── test_imputers.py        # 插补器测试
│   └── test_missing_generators.py # 缺失生成器测试
│
└── configs/                    # 配置文件目录
```

---

## 三、核心模块功能详解

### 3.1 核心接口层 (src/core/)

#### 3.1.1 interfaces.py - 抽象接口定义

定义所有组件的契约，使用 Protocol 支持结构子类型（鸭子类型）：

| 接口 | 功能 | 关键方法 |
|------|------|----------|
| `BatteryDataset` | 统一电池数据集格式 | dataclass，包含features/labels/battery_ids |
| `IDataLoader` | 数据加载器接口 | `load()`, `list_batches()` |
| `IMissingGenerator` | 缺失数据生成器 | `generate()`, `name` |
| `IImputer` | 插补策略接口 | `fit()`, `transform()`, `fit_transform()` |
| `IModel` | 神经网络模型接口 | PyTorch模型标准方法 |
| `ITrainer` | 训练引擎接口 | `train()` |
| `ICallback` | 训练回调接口 | `on_epoch_end()`, `on_training_end()` |
| `ExperimentConfig` | 完整实验配置 | 包含L1-L9所有配置参数 |

#### 3.1.2 registry.py - 插件系统

提供通用的组件注册和发现机制：

```python
# 全局注册中心实例
IMPUTERS = Registry["IImputer"]("imputer")
MISSING_GENERATORS = Registry["IMissingGenerator"]("missing_generator")
MODELS = Registry["IModel"]("model")
TRAINERS = Registry["ITrainer"]("trainer")
DATA_LOADERS = Registry["IDataLoader"]("data_loader")

# 使用示例
@IMPUTERS.register("mean")
class MeanImputer:
    pass

imputer = IMPUTERS.create("mean")
```

**功能特性**:
- 装饰器注册: `@registry.register("name")`
- 延迟加载: `register_lazy()`
- 类型约束: 可指定基类验证
- 覆盖保护: `override=True`参数

---

### 3.2 数据层 (src/data/)

#### 3.2.1 xjtu_loader.py - XJTU数据加载

```python
class XJTUDataLoader:
    def __init__(self, batch_id: str, data_dir: str = "data/XJTU data")
    def load_data(self) -> pd.DataFrame  # 加载批次所有电池
    def get_battery_ids(self) -> List[str]
```

**支持批次**: 2C, 3C, R2.5, R3, RW, Sim_satellite

#### 3.2.2 loader.py - 通用数据加载

支持多数据集: XJTU, TJU, HUST, MIT

```python
def load_batteries(data_dir, pattern, cfg, dataset_type) -> Dict[str, Tuple[X, y]]
def load_dataset(cfg: DictConfig) -> Dict[str, torch.Tensor]
```

#### 3.2.3 features.py - 特征工程

```python
def build_features(df: pd.DataFrame, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """从清洗后的数据构建特征X和目标y (SOH)"""
    # 计算 SOH = capacity / nominal_capacity
```

**16维特征列表**:
- voltage mean/std/kurtosis/skewness
- current mean/std/kurtosis/skewness
- CC Q, CC charge time
- CV Q, CV charge time
- voltage slope, current slope
- voltage entropy, current entropy

#### 3.2.4 splits.py - Battery-wise划分

**关键设计**: 同一电池的所有循环只属于train/val/test中的一个，避免数据泄漏

```python
def split_batteries(battery_ids, test_size, val_size, random_state) -> Tuple[train_ids, val_ids, test_ids]
def train_val_test_split_by_battery(battery_data, cfg) -> Dict[str, torch.Tensor]
```

#### 3.2.5 imputation_utils.py - 插补工具

```python
def impute_missing_values(X_missing, method, train_stats, seed) -> np.ndarray
def compute_train_statistics(X_train) -> dict  # mean/std/median/min/max
def generate_mcar_missing(X, missing_rate, seed) -> tuple
def prepare_mim_training_data(X_train, y_train, train_stats, imputation_method, missing_rates, seed, model_type) -> list
def prepare_validation_data(X_val, train_stats, imputation_method, missing_rates, seed, model_type) -> list
```

---

### 3.3 模型层 (src/models/)

#### 3.3.1 MLP (mlp.py)

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [192, 96], dropout: float = 0.15)
    def forward(self, x: [batch, input_dim]) -> [batch]
```

**参数量控制**:
- non-MIM (16维输入): [192, 96] → ~21,889 params
- MIM (32维输入): [128, 96] → ~16,705 params

#### 3.3.2 LSTM (lstm.py)

```python
class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 46, num_layers: int = 2, dropout: float = 0.2)
    def forward(self, x: [batch, seq_len, input_dim]) -> [batch]
```

**参数量**: 输入16维→29,119; 输入32维→32,063

#### 3.3.3 CNN1D (cnn1d.py)

```python
class CNN1D(nn.Module):
    def __init__(self, input_dim: int, channels: List[int] = [64, 80], kernel_size: int = 3, dropout: float = 0.1)
    def forward(self, x: [batch, seq_len, input_dim]) -> [batch]
```

**参数量**: 输入16维→18,657; 输入32维→21,729

#### 3.3.4 模型工厂 (factory.py)

```python
def create_model(model_type, input_dim, use_mim, **override_kwargs) -> nn.Module
def count_parameters(model) -> int
def verify_parameter_budget(model, min_params=16384, max_params=32768) -> bool
def get_all_model_variants() -> Dict[str, nn.Module]
```

**参数量预算**: 所有模型控制在 (2^14, 2^15) = (16,384, 32,768) 范围内

---

### 3.4 缺失数据处理层

#### 3.4.1 MCAR模拟 (missing_data/mcar.py)

完全随机缺失，与观测值和未观测值均无关:

```python
def simulate_mcar(X, missing_rate, seed, impute_method='mean') -> Tuple[X_imputed, mask, mim_input]
```

#### 3.4.2 MAR模拟 (missing_data/mar.py)

缺失概率依赖SOH: `p = alpha * SOH^beta + gamma`

```python
def simulate_mar(X, sohs, missing_rate, alpha=None, beta=2.0, gamma=0.05, seed=42, impute_method='mean')
```

**物理意义**: SOH高(新电池)→缺失率低; SOH低(老电池)→缺失率高

#### 3.4.3 MNAR模拟 (missing_data/mnar.py)

缺失概率依赖特征值本身（极端值更容易缺失）:

```python
def simulate_mnar(X, y, missing_rate, alpha=None, beta=0.05, gamma=1.0, feature_index=0, seed=42, impute_method='mean')
# 以及基于SOH的变体
def simulate_mnar_by_soh(X, y, missing_rate, alpha=None, beta=0.05, seed=42, impute_method='mean')
```

#### 3.4.4 传统插补方法 (missing_data/imputation.py)

| 方法 | 函数 | 说明 |
|------|------|------|
| Mean | `mean_imputation()` | 列均值填充 |
| Median | `median_imputation()` | 列中位数填充 |
| KNN | `knn_imputation()` | K近邻加权均值 |
| Zero | `zero_imputation()` | 零值填充 |
| Iterative | `iterative_imputation()` | MICE-style迭代回归 |
| Forward Fill | `forward_fill_imputation()` | 时序前向填充 |

#### 3.4.5 插补器实现 (missing/imputers/)

基于注册中心的标准化实现:

```python
@IMPUTERS.register("mean")
class MeanImputer(BaseImputer):
    def fit(self, X) -> "MeanImputer"
    def transform(self, X_missing) -> np.ndarray
```

---

### 3.5 训练层 (src/trainers/)

#### 3.5.1 Lightning模块 (lightning_module.py)

```python
class SOHLightningModule(pl.LightningModule):
    def __init__(self, model=None, model_type=None, input_dim=None, 
                 learning_rate=1e-3, optimizer_name="Adam", weight_decay=0.0, ...)
    def training_step(batch, batch_idx) -> loss
    def validation_step(batch, batch_idx) -> dict
    def test_step(batch, batch_idx) -> dict
    def configure_optimizers() -> dict  # Adam + ReduceLROnPlateau
```

#### 3.5.2 Lightning训练器 (lightning_trainer.py)

```python
class LightningTrainer:
    def __init__(self, max_epochs=100, patience=15, device='auto', ...)
    def train(model, train_loader, val_loader) -> Dict[str, Any]  # history
    def save_model(path)
    def load_model(path, model)

def get_trainer(cfg: DictConfig) -> pl.Trainer  # 从配置创建
```

**回调配置**:
- EarlyStopping (监控val_loss)
- LearningRateMonitor
- ModelCheckpoint (可选)
- WandBLogger (可选)

---

### 3.6 评估层 (src/evaluation/)

#### 3.6.1 指标计算 (metrics.py)

```python
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    # 返回: mae, rmse, r2
    
def compute_additional_metrics(y_true, y_pred) -> Dict[str, float]:
    # 返回: max_error, median_ae, mape, explained_variance
```

#### 3.6.2 模型评估器 (model_evaluator.py)

```python
class ModelEvaluator:
    def __init__(self, device='cpu')
    def evaluate(model, test_loader) -> Tuple[metrics, predictions, targets]
```

---

### 3.7 实验执行层 (src/experiments/)

#### 3.7.1 单实验运行器 (runner.py)

```python
class ExperimentRunner:
    def __init__(self, output_dir, use_gpu=True, gpu_id=0)
    def run(record: ExperimentRecord) -> Dict[str, Any]
    def _build_command(record) -> list  # 构建命令行
    def _extract_metrics(log_file) -> Dict[str, Any]

class GPUExperimentRunner(ExperimentRunner):
    def get_gpu_memory() -> Dict[str, float]
    def is_memory_available(required_gb=2.0) -> bool
```

#### 3.7.2 实验调度器 (scheduler.py)

```python
class ExperimentScheduler:
    def __init__(self, config: SchedulerConfig)
    def initialize_experiments(seeds, models, methods, eval_missing_rates, ...) -> int
    def run_batch(batch_size=None) -> int  # 运行一批实验
    def run_all(continuous=True) -> int  # 运行所有待处理实验
    def retry_failed() -> int  # 重试失败的实验
    def get_status() -> Dict[str, Any]
    def recover() -> int  # 恢复卡住的实验
```

**特性**:
- 多进程并行执行 (1 GPU + N CPU workers)
- 自动重试机制 (max_retries=3)
- 进度报告 (interval=60s)
- 优雅关闭 (signal处理)

#### 3.7.3 实验数据库 (database.py)

```python
class ExperimentRecord:  # 实验记录dataclass
    exp_id, seed, model, method, status, metrics, ...

class ExperimentDatabase:
    def __init__(self, db_path)
    def add_experiment(record) -> bool
    def get_experiment(exp_id) -> Optional[ExperimentRecord]
    def get_by_status(status) -> List[ExperimentRecord]
    def mark_completed(exp_id, metrics, duration, log_file)
    def mark_failed(exp_id, error, log_file)
    def mark_running(exp_id)
    def reset_running() -> int  # 重置卡住的running状态
    def get_statistics() -> Dict[str, int]
    def generate_experiment_id(seed, model, method) -> str
```

---

### 3.8 预实验层 (src/preexperiment/)

用于模型架构超参数搜索（两阶段流程）：

```python
# Stage 1: Optuna搜索（单种子快速筛选）
class PreExperimentRunner:
    def _run_stage1_search() -> List[Dict]  # Top-K候选
    def _run_stage2_validation(candidates) -> Dict  # 多种子验证

# 搜索空间
class SearchSpace:
    @staticmethod
    def get_suggest_func(model_type) -> Callable
    # MLP: hidden_dims, dropout
    # LSTM: hidden_size, num_layers, dropout
    # CNN: channels, kernel_size, dropout
```

**预算配置**:
```python
@dataclass
class BudgetConfig:
    budget: int                    # 参数量预算
    n_trials: int                  # Optuna搜索次数
    epochs_search: int             # 搜索阶段epochs
    patience_search: int           # 搜索阶段patience
    n_seeds_final: int             # 验证阶段种子数
    epochs_final: int              # 验证阶段epochs
    patience_final: int            # 验证阶段patience
    top_k: int = 3                 # 进入验证的候选数
```

---

### 3.9 配置管理 (src/config/)

```python
@dataclass(frozen=True)
class DataConfig:  # L2-L3
    batch: Literal['2C', '3C', 'R2.5', 'R3', 'RW', 'Sim_satellite']

@dataclass(frozen=True)
class ModelArchitectureConfig:  # L4
    model_type: Literal['mlp', 'lstm', 'cnn']
    # MLP: mlp_hidden_dims, mlp_dropout
    # LSTM: lstm_hidden_size, lstm_num_layers, lstm_dropout
    # CNN: cnn_channels, cnn_kernel_size, cnn_dropout

@dataclass(frozen=True)
class MIMConfig:  # L5-L6
    use_mim: bool
    train_mr_list: List[float]  # 20档: 0.0-0.95
    val_mr: float  # 0.5(单MR), -1(多MR平均), -2(多MR独立)

@dataclass(frozen=True)
class TestingConfig:  # L7-L9
    mode: Literal['MCAR', 'MAR', 'MNAR']
    test_mr: float
    imputation: Literal['mean', 'knn', 'iterative', 'zero']
    modes: List[str]  # 批量测试用
    test_mrs: List[float]
    imputations: List[str]

@dataclass
class ExperimentConfig:  # 完整配置
    seed: SeedConfig
    data: DataConfig
    model: ModelArchitectureConfig
    mim: MIMConfig
    training: TrainingConfig
    testing: TestingConfig
    paths: PathConfig
    # 方法: to_dict(), to_json(), to_yaml(), from_dict(), from_json(), from_yaml()
    # 工具: get_model_path(), get_results_path(), get_input_dim()
```

---

### 3.10 工具层 (src/utils/)

#### 3.10.1 随机种子管理 (seed_manager.py)

```python
def set_seed(seed: int)  # 设置Python/NumPy/PyTorch/CUDA种子
def create_worker_init_fn(seed) -> Callable  # DataLoader worker初始化
def create_dataloader_generator(seed) -> torch.Generator
def get_random_state() -> dict  # 获取状态快照
def set_random_state(state: dict)  # 恢复状态
class DeterministicContext:  # 确定性上下文管理器

def verify_seed_setting(seed=42) -> bool  # 验证可复现性
```

---

## 四、实验入口脚本详解

### 4.1 run_experiment.py - 主实验运行器

**支持三种运行模式**:

```bash
# 1. 训练阶段 (分界线以上: L1-L6)
python experiments/run_experiment.py --phase train \
    --seed 42 --batch 2C --model mlp --use-mim false \
    --epochs 50 --save-model

# 2. 单配置测试 (分界线以下: L7-L9)
python experiments/run_experiment.py --phase test \
    --seed 42 --batch 2C --model mlp --use-mim false \
    --mode MCAR --test-mr 0.3 --imputation mean

# 3. 批量测试 (分界线以下: 所有组合)
python experiments/run_experiment.py --phase batch-test \
    --seed 42 --batch 2C --model mlp --use-mim false
# 自动测试: 3 modes × 20 MRs × 4 imputations = 120 tests
```

**核心函数**:
```python
def prepare_training_data(batch_id, seed, use_mim, model_type, train_imputation) -> Tuple
    # non-MIM: 使用完整数据训练
    # MIM: 生成多MR训练数据（MCAR），使用指定插补方法填充

def train_model(args) -> Dict
    # non-MIM: 完整数据训练
    # MIM: 多MR混合训练（每轮遍历所有MR数据）
    # 早停: 基于验证集多MR平均loss

def test_model(args) -> Dict
    # 生成缺失数据（根据mode和test_mr）
    # 应用插补方法
    # 预测并计算MAE/R2

def batch_test_model(args) -> List[Dict]
    # 一次加载模型，测试所有L7-L9组合
    # 3 modes × 20 MRs × 4 imputations = 120 tests per model
```

### 4.2 run_batch_experiments_v2.py - 批量实验(多进程)

```bash
python experiments/run_batch_experiments_v2.py \
    --model-dir models/100seeds \
    --results-dir results/100seeds \
    --seeds 0 1 2 ... 99 \
    --workers 15  # 默认CPU count - 1
```

**并行策略**:
```python
def batch_test_single_model(args_tuple) -> Tuple[List[Dict], float]
    # 每个模型在一个独立进程中测试
    # 使用imap_unordered并行执行
    # 进程安全的CSV写入
```

**实验规模**:
- 100 seeds × 6 batches × 3 models × 2 MIM = 3,600 模型
- 每模型120测试 = 432,000 测试结果

---

## 五、关键设计模式

### 5.1 Battery-wise Split

```python
# ✅ 正确：整个电池只属于一个集合
np.random.seed(seed)
batteries = feature_df['battery_id'].unique()
np.random.shuffle(batteries)

train_batteries = batteries[:n_train]
val_batteries = batteries[n_train:n_train + n_val]
test_batteries = batteries[n_train + n_val:]
```

**目的**: 避免数据泄漏，模拟真实部署场景（未见过的电池）

### 5.2 MIM维度处理

```python
# 传统方法 (16维输入)
input = [feature_values]  # shape: [batch, 16]

# MIM方法 (32维输入)
input = [feature_values] + [missing_mask]  # shape: [batch, 32]
# missing_mask: 1表示缺失，0表示存在
```

### 5.3 分界线原则

```python
# 分界线以上 (L1-L6): 影响训练
if use_mim:
    # MIM模型: 32维输入，多MR训练
    multi_mr_data = prepare_mim_training_data(...)
else:
    # non-MIM模型: 16维输入，完整数据训练
    X_train, y_train = load_full_data(...)

# 分界线以下 (L7-L9): 仅测试
test_results = []
for mode in ['MCAR', 'MAR', 'MNAR']:
    for test_mr in [0.0, 0.05, ..., 0.95]:
        for imputation in ['mean', 'knn', 'iterative', 'zero']:
            # 复用已训练模型进行测试
            result = test_model(model, mode, test_mr, imputation)
            test_results.append(result)
```

### 5.4 插补方法标准化

```python
# 通过注册中心统一接口
@IMPUTERS.register("mean")
class MeanImputer(BaseImputer):
    def fit(self, X): ...
    def transform(self, X_missing): ...

# 创建插补器
imputer = IMPUTERS.create("mean")
X_imputed = imputer.fit(X_train).transform(X_missing)
```

---

## 六、测试体系

### 6.1 关键路径测试 (test_critical_path.py)

| 测试函数 | 测试内容 |
|----------|----------|
| `test_seed_reproducibility()` | 随机种子可复现性 |
| `test_model_forward_dimensions()` | 模型输出维度正确性 |
| `test_model_save_load()` | 模型保存/加载一致性 |
| `test_standardization()` | 标准化逻辑正确性 |
| `test_data_split_no_overlap()` | 数据划分无重叠 |
| `test_mim_dimension()` | MIM维度拼接正确性 |

### 6.2 其他测试

- `test_imputers.py`: 插补器功能测试
- `test_missing_generators.py`: 缺失生成器测试
- `test_core_infrastructure.py`: 核心基础设施测试

---

## 七、可视化模块

### 7.1 热力图 (heatmaps.py)

绘制MIM相对Baseline的MAE改善百分比热力图:

```python
def compute_improvement(baseline_df, mim_df) -> Dict[str, float]:
    # improvement = (MAE_baseline - MAE_mim) / MAE_baseline * 100%

def plot_improvement_heatmap(results, output_dir, model_name, missing_mode)
```

### 7.2 缺失率曲线 (missing_rate_curves.py)

绘制不同缺失率下的性能曲线

### 7.3 批次对比图 (batch_plots.py)

不同电池批次的性能对比

---

## 八、项目配置

### 8.1 Hydra配置结构 (configs/)

```yaml
config.yaml          # 主配置
model/               # 模型配置
  - paper_mlp.yaml
  - paper_lstm.yaml
  - paper_cnn1d.yaml
method/              # 方法配置
  - mim.yaml
  - mean.yaml
  - knn.yaml
  - iterative.yaml
  - zero.yaml
experiment/          # 实验配置
  - 100seeds.yaml
```

### 8.2 项目元数据

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | Python项目配置、依赖管理 |
| `environment.yml` | Conda环境配置 |
| `model_configs.yaml` | 12种模型架构配置 (3模型×4参数量级别) |
| `meta.md` | 9层架构权威定义（只读） |
| `README.md` | 项目主文档 |

---

## 九、依赖栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 语言 | Python | >=3.12 |
| 深度学习 | PyTorch | >=2.10.0 |
| 训练框架 | PyTorch Lightning | >=2.6.0 |
| 数据处理 | pandas, numpy | >=2.0.0, >=1.26.0 |
| 配置管理 | OmegaConf, Hydra | >=2.3.0, >=1.3.0 |
| 超参优化 | Optuna | >=4.0.0 |
| 实验追踪 | Weights & Biases | >=0.19.0 |
| 代码质量 | Black, isort, pre-commit | - |

---

## 十、扩展指南

### 10.1 添加新模型

1. 创建模型文件 `src/models/new_model.py`:
```python
class NewModel(nn.Module):
    def __init__(self, input_dim, **kwargs)
    def forward(self, x)
```

2. 在 `src/models/factory.py` 添加配置和创建逻辑

3. 更新 `model_configs.yaml`

### 10.2 添加新缺失模式

1. 创建模拟函数 `src/missing_data/new_mode.py`:
```python
def simulate_new_mode(X, y, missing_rate, seed) -> Tuple[X_imputed, mask, mim_input]
```

2. 在 `src/core/registry.py` 注册生成器

### 10.3 添加新插补器

```python
@IMPUTERS.register("new_imputer")
class NewImputer(BaseImputer):
    def fit(self, X)
    def transform(self, X_missing)
    @property
    def name(self) -> str
```

---

## 十一、数据流图

```
XJTU原始数据
    ↓
XJTUDataLoader.load_data() → DataFrame
    ↓
features.build_features() → X[16维], y[SOH]
    ↓
splits.train_val_test_split_by_battery() → Train/Val/Test划分
    ↓
[训练阶段]
├── non-MIM: 完整数据直接训练
└── MIM: 
    ├── generate_mcar_missing() → 多MR缺失数据
    ├── impute_missing_values() → 插补填充
    ├── torch.cat([X_imputed, mask], dim=1) → 32维输入
    └── 多MR混合训练
    ↓
模型保存 (分界线)
    ↓
[测试阶段]
├── 生成缺失数据 (MCAR/MAR/MNAR)
├── 插补填充 (mean/knn/iterative/zero)
├── (MIM only) torch.cat([X_imputed, mask], dim=1)
├── 模型预测
└── 计算指标 (MAE/RMSE/R2)
    ↓
结果保存 (JSON/CSV)
```

---

## 十二、关键文件索引

| 文件路径 | 功能类别 | 重要性 |
|----------|----------|--------|
| `src/core/interfaces.py` | 核心接口 | ⭐⭐⭐ |
| `src/core/registry.py` | 插件系统 | ⭐⭐⭐ |
| `src/config/pydantic_config.py` | 配置管理 | ⭐⭐⭐ |
| `src/data/xjtu_loader.py` | 数据加载 | ⭐⭐⭐ |
| `src/data/splits.py` | 数据划分 | ⭐⭐⭐ |
| `src/models/factory.py` | 模型创建 | ⭐⭐⭐ |
| `src/missing_data/{mcar,mar,mnar}.py` | 缺失模拟 | ⭐⭐⭐ |
| `experiments/run_experiment.py` | 主入口 | ⭐⭐⭐ |
| `meta.md` | 架构定义 | ⭐⭐⭐ |
| `tests/test_critical_path.py` | 关键测试 | ⭐⭐⭐ |

---

*本文档基于代码分支 `refactor-code` 的系统梳理，涵盖了项目的所有核心功能和架构设计。*
