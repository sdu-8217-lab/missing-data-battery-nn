# 缺参自适项目代码架构指南

> 本指南面向初学者，帮助快速理解项目代码结构和各模块功能。

## 项目概述

本项目是基于神经网络的电池健康状态（SOH）鲁棒预测框架，核心创新是在输入数据存在缺失的条件下，实现自适应状态估计，无需数据插补。

**核心概念：**
- **MIM（Missing Indicator Method）**：缺失指示器方法，将原始特征与缺失指示器拼接（16维→32维）
- **Baseline**：基线方法，仅使用完整数据训练，不处理缺失情况
- **MAR/MCAR**：缺失数据生成机制（Missing At Random / Missing Completely At Random）

---

## 目录结构

```
src/
├── main.py                    # 主入口
├── config/                    # 配置层
├── data/                      # 数据层
├── missing_data/              # 缺失数据层
├── models/                    # 模型层
├── trainers/                  # 训练层
├── evaluation/                # 评估层
├── experiments/               # 实验层
├── visualization/             # 可视化层
└── utils/                     # 工具层
```

---

## 一、主入口层

### 1.1 `src/main.py`

| 属性 | 说明 |
|------|------|
| **功能** | 项目主入口，负责配置解析、日志初始化、实验启动 |
| **输入** | 命令行参数（通过 Hydra 解析） |
| **输出** | 运行日志、实验结果 CSV |
| **依赖** | Hydra, PyTorch Lightning, 各下层模块 |

**实现方式：**
- 使用 `@hydra.main` 装饰器管理配置
- 调用 `run_experiment()` 执行实际训练
- 集成 WandB 进行实验追踪

**使用示例：**
```bash
# 默认实验
python src/main.py

# 指定实验配置
python src/main.py experiments=mim_mar_0.3

# 组合配置
python src/main.py experiments=mim_mar_0.6 models=lstm
```

---

## 二、配置层 (config/)

### 2.1 `src/config/experiment_config.py`

| 属性 | 说明 |
|------|------|
| **功能** | 定义实验配置的数据类结构 |
| **输入** | 无（定义数据结构） |
| **输出** | `ModelConfig`, `ExperimentConfig` 数据类 |

**核心类：**

```python
@dataclass
class ModelConfig:
    name: str              # 模型名称（MLP, LSTM, GRU, CNN1D）
    model_type: str        # 模型类型
    use_mim: bool          # 是否使用 MIM
    hidden_layers: List    # MLP 隐藏层维度
    hidden_size: int       # LSTM/GRU 隐藏层大小
    num_layers: int        # LSTM/GRU 层数
    channels: List         # CNN 通道数
    dropout: float         # Dropout 概率

@dataclass  
class ExperimentConfig:
    n_repeats: int         # 重复实验次数（100）
    missing_rates: List    # 测试缺失率 [0.1, ..., 0.9]
    training_missing_rates: List  # 训练缺失率 [0.0, ..., 0.9]
    feature_cols: List     # 16个特征列名
    target_col: str        # 目标列（capacity）
```

**关键方法：**
- `get_model_configs()`：返回8种模型配置（4种架构 × 2种MIM设置）

---

## 三、数据层 (data/)

### 3.1 `src/data/loader.py` - 数据加载主入口

| 属性 | 说明 |
|------|------|
| **功能** | 根据配置加载不同数据集 |
| **输入** | `cfg: DictConfig`（包含 dataset, data_dir, batch 等） |
| **输出** | `Dict[str, torch.Tensor]`（包含 X_train, y_train, X_val, y_val, X_test, y_test） |
| **实现** | 工厂模式，支持多数据集扩展 |

**核心函数：**

```python
def load_dataset(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """主入口函数"""
    if dataset == "xjtu":
        return _load_xjtu(cfg)
    elif dataset == "hust":
        return _load_hust(cfg)
    # ...
```

**处理流程：**
1. 根据 `batch` 参数查找匹配文件（如 `3C_battery-*.csv`）
2. 对每个 CSV 调用 `clean_dataframe()` 清洗
3. 调用 `build_features()` 构建特征
4. 调用 `train_val_test_split()` 划分数据集

---

### 3.2 `src/data/preprocessing.py` - 数据预处理

| 属性 | 说明 |
|------|------|
| **功能** | 数据清洗、异常值处理、缺失值填充、标准化 |
| **输入** | `df: pd.DataFrame`, `cfg: DictConfig` |
| **输出** | 清洗后的 `pd.DataFrame` |

**核心函数：**

| 函数 | 功能 | 实现方式 |
|------|------|----------|
| `clean_dataframe()` | 主清洗函数 | 替换inf→NaN → 异常值处理 → 填充NaN |
| `_remove_outliers_3sigma()` | 3-sigma去异常 | 保留 (μ-3σ, μ+3σ) 范围内的数据 |
| `_remove_outliers_iqr()` | IQR去异常 | 保留 (Q1-1.5IQR, Q3+1.5IQR) 范围内的数据 |
| `standardize_features()` | Z-score标准化 | `(X - mean) / std` |

---

### 3.3 `src/data/splits.py` - 数据集划分

| 属性 | 说明 |
|------|------|
| **功能** | 划分训练/验证/测试集，可选标准化 |
| **输入** | `X: np.ndarray`, `y: np.ndarray`, `cfg: DictConfig` |
| **输出** | `Dict[str, torch.Tensor]` 包含划分后的数据和标准化统计量 |

**划分策略：**
```
原始数据
    ├── 测试集 (20%)
    └── 剩余数据 (80%)
        ├── 训练集 (60% = 80% × 75%)
        └── 验证集 (20% = 80% × 25%)
```

**注意：** 标准化使用训练集统计量（mean/std）应用于验证集和测试集。

---

### 3.4 `src/data/features.py` - 特征工程

| 属性 | 说明 |
|------|------|
| **功能** | 从原始数据构建特征矩阵X和目标向量y |
| **输入** | `df: pd.DataFrame`, `cfg: DictConfig` |
| **输出** | `X: np.ndarray [N, 16]`, `y: np.ndarray [N]`（SOH值） |

**关键逻辑：**
```python
def build_features(df, cfg):
    # 提取16维特征
    X = df[feature_cols].values  # [N, 16]
    
    # 计算 SOH = capacity / nominal_capacity
    capacity = df[target_col].values
    nominal_capacity = capacity[0]  # 使用首次容量作为标称容量
    y = capacity / nominal_capacity  # [N]
    
    return X, y
```

**16个输入特征：**
1. Voltage统计：mean, std, kurtosis, skewness
2. Current统计：mean, std, kurtosis, skewness
3. 充电特征：CC Q, CC charge time, CV Q, CV charge time
4. 变化率：voltage slope, current slope
5. 熵值：voltage entropy, current entropy

---

## 四、缺失数据层 (missing_data/)

### 4.1 `src/missing_data/mar.py` - MAR缺失模拟

| 属性 | 说明 |
|------|------|
| **功能** | 生成 Missing At Random 缺失模式（缺失概率依赖于SOH） |
| **输入** | `X: [N, D]`, `sohs: [N]`, `missing_rate: float`, `alpha, beta, gamma` |
| **输出** | `X_imputed: [N, D]`, `mask: [N, D]`, `mim_input: [N, 2D]` |

**核心公式：**
```
p_i = alpha * (1 - SOH_i)^beta + gamma

其中：
- alpha: 缺失强度（自动反推以达成目标missing_rate）
- beta: 非线性参数（默认2.0）
- gamma: 基础缺失率（默认0.05）
```

**算法流程：**
1. 计算 `E[(1-SOH)^beta]`
2. 反推 `alpha = (missing_rate - gamma) / E[...]`
3. 计算每个样本的缺失概率 `p_i`
4. 生成掩码 `mask = (rand > p).float()`
5. 均值插补缺失值
6. 构造 MIM 输入：`[X_imputed, 1-mask]`

**物理意义：** SOH 越低（电池越老化），缺失概率越高。

---

### 4.2 `src/missing_data/mcar.py` - MCAR缺失模拟

| 属性 | 说明 |
|------|------|
| **功能** | 生成 Missing Completely At Random 缺失模式（完全随机缺失） |
| **输入** | `X: [N, D]`, `missing_rate: float`, `seed: int` |
| **输出** | `X_imputed: [N, D]`, `mask: [N, D]`, `mim_input: [N, 2D]` |

**实现：** 简单随机掩码 `mask = (rand > missing_rate).float()`

---

## 五、模型层 (models/)

### 5.1 `src/models/model_factory.py` - 模型工厂

| 属性 | 说明 |
|------|------|
| **功能** | 根据配置创建对应模型 |
| **输入** | `cfg: DictConfig` |
| **输出** | `BaseSOHLightningModule` 实例 |

**关键逻辑：**
```python
def create_model(cfg):
    # 根据 use_mim 确定输入维度
    input_dim = 32 if cfg.experiments.experiment.use_mim else 16
    
    # 创建 backbone
    if cfg.models.name == "cnn1d":
        backbone = CNN1D(input_dim=input_dim, ...)
    elif cfg.models.name == "mlp":
        backbone = MLP(input_dim=input_dim, ...)
    # ...
    
    return BaseSOHLightningModule(backbone, cfg)
```

---

### 5.2 `src/models/base_model.py` - Lightning基类

| 属性 | 说明 |
|------|------|
| **功能** | PyTorch Lightning 基类，封装通用训练逻辑 |
| **输入** | `backbone: nn.Module`, `cfg: DictConfig` |
| **输出** | Lightning Module |

**核心方法：**

| 方法 | 功能 |
|------|------|
| `forward(x)` | 前向传播 |
| `training_step(batch, batch_idx)` | 训练步骤，计算 loss |
| `validation_step(batch, batch_idx)` | 验证步骤，计算 val_loss |
| `test_step(batch, batch_idx)` | 测试步骤，收集预测 |
| `on_test_epoch_end()` | 测试结束，计算 MAE/RMSE/R² |
| `configure_optimizers()` | 配置 Adam + ReduceLROnPlateau |

---

### 5.3 `src/models/cnn1d.py` - 1D-CNN模型

| 属性 | 说明 |
|------|------|
| **功能** | 1D卷积神经网络 |
| **输入** | `x: [N, D]`（D=16或32） |
| **输出** | `y: [N, 1]`（SOH预测值） |
| **结构** | Conv1D → BatchNorm → ReLU → Dropout → Flatten → FC |

**前向传播流程：**
```python
def forward(self, x):
    # [N, D] -> [N, 1, D]
    x = x.unsqueeze(1)
    
    # 卷积层 [N, 1, D] -> [N, channels[-1], D]
    x = self.conv_layers(x)
    
    # Flatten [N, channels[-1], D] -> [N, channels[-1] * D]
    x = x.view(x.size(0), -1)
    
    # 全连接层 -> [N, 1]
    x = self.fc_layers(x)
    return x
```

---

### 5.4 其他模型文件

| 文件 | 功能 | 输入维度 | 输出维度 |
|------|------|----------|----------|
| `mlp.py` | 多层感知机 | [N, D] | [N, 1] |
| `lstm.py` | 长短期记忆网络 | [N, seq_len, D] | [N, 1] |
| `gru.py` | 门控循环单元 | [N, seq_len, D] | [N, 1] |

---

## 六、训练层 (trainers/)

### 6.1 `src/trainers/neural_network_trainer.py`

| 属性 | 说明 |
|------|------|
| **功能** | 主训练器，协调数据加载、缺失生成、模型训练、结果保存 |
| **输入** | `cfg: DictConfig` |
| **输出** | 训练好的模型、CSV结果文件 |

**核心函数 `run_experiment(cfg)` 流程：**

```
1. 加载数据 load_dataset(cfg)
   └── X_train, y_train, X_val, y_val, X_test, y_test

2. 对于每个 missing_rate 和 seed:
   a. 应用缺失机制 _apply_missing_mechanism()
      ├── MAR: simulate_mar()  或
      └── MCAR: simulate_mcar()
   
   b. 准备 DataLoader
   
   c. 创建模型 create_model(cfg)
      └── 根据 use_mim 自动调整 input_dim
   
   d. 训练模型 trainer.fit()
   
   e. 测试模型 trainer.test()
   
   f. 保存结果 append_result_row()
```

---

### 6.2 `src/trainers/lightning_trainer.py`

| 属性 | 说明 |
|------|------|
| **功能** | 创建 PyTorch Lightning Trainer |
| **输入** | `cfg: DictConfig` |
| **输出** | `pl.Trainer` 实例 |

**配置项：**
- `max_epochs`: 最大训练轮数
- `accelerator`: 计算设备（auto/cpu/cuda）
- `callbacks`: EarlyStopping, ModelCheckpoint

---

## 七、评估层 (evaluation/)

### 7.1 `src/evaluation/metrics.py`

| 属性 | 说明 |
|------|------|
| **功能** | 计算回归评估指标 |
| **输入** | `y_true: np.ndarray`, `y_pred: np.ndarray` |
| **输出** | `Dict[str, float]` 包含 mae, rmse, r2 |

**核心函数：**

```python
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}
```

---

### 7.2 `src/evaluation/result_writer.py`

| 属性 | 说明 |
|------|------|
| **功能** | 将实验结果追加到 CSV 文件 |
| **输入** | `metrics: Dict`, `cfg: DictConfig` |
| **输出** | CSV 文件路径 |

**输出格式示例：**
```csv
seed,missing_rate,model,missing_mode,use_mim,test_mae,test_rmse,test_r2,timestamp
42,0.3,cnn1d,mar,True,0.0597,0.0632,-2.28,2026-02-23T14:08:31
```

---

## 八、可视化层 (visualization/)

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `missing_rate_curves.py` | 绘制指标随缺失率变化曲线 | 6个CSV文件 | `mae_vs_missing_rate.png/pdf` |
| `heatmaps.py` | 绘制MIM改善热力图 | 6个CSV文件 | `mim_improvement.png/pdf` |

---

## 九、工具层 (utils/)

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `seed_manager.py` | 随机种子管理 | `set_seed(seed)` |
| `logger.py` | 日志配置 | `setup_logger(name, log_file, level)` |
| `wandb_utils.py` | WandB集成 | `init_wandb(cfg)`, `finish_wandb()` |

---

## 十、数据流图

```
配置 (Hydra YAML)
    ↓
main.py
    ↓
load_dataset(cfg) ──→ XJTU CSV 文件
    ↓                    ↓
    ├── clean_dataframe() ──→ 去除异常值、填充NaN
    ↓
    ├── build_features() ──→ X[16维], y[SOH]
    ↓
    └── train_val_test_split() ──→ 划分 + 标准化
             ↓
    X_train, y_train, X_val, y_val, X_test, y_test
             ↓
    simulate_mar() / simulate_mcar()
             ↓
    X_imputed ──→ MIM? ──Yes──→ mim_input[32维]
        │                      No
        └──────────────────────→ X_imputed[16维]
             ↓
    create_model(cfg) ──→ CNN1D/MLP/LSTM/GRU
             ↓
    trainer.fit() ──→ trainer.test()
             ↓
    compute_metrics() ──→ append_result_row()
             ↓
    results/csv/*.csv ──→ visualization/*.py ──→ 图表
```

---

## 十一、关键设计模式

### 11.1 工厂模式
- `model_factory.py`：根据配置字符串创建不同模型
- `loader.py`：根据 dataset 名称加载不同数据集

### 11.2 策略模式
- 缺失机制：MAR vs MCAR 可互换
- 预处理：3-sigma vs IQR 可配置

### 11.3 模板方法模式
- `BaseSOHLightningModule` 定义训练流程，子类只需实现 backbone

---

## 十二、快速开始指南

### 添加新模型

1. 在 `src/models/` 创建 `my_model.py`
2. 继承 `nn.Module`，实现 `__init__` 和 `forward`
3. 在 `model_factory.py` 添加创建逻辑
4. 在 `configs/models/` 创建配置文件

### 添加新数据集

1. 在 `src/data/loader.py` 添加 `_load_mydata()` 函数
2. 在 `configs/data/` 创建配置文件

### 添加新评估指标

1. 在 `src/evaluation/metrics.py` 添加计算函数
2. 在 `base_model.py` 的 `on_test_epoch_end()` 中添加日志记录

---

## 十三、常见问题

### Q1: MIM 和 Baseline 的区别是什么？

| | Baseline | MIM |
|--|----------|-----|
| 输入维度 | 16 | 32 (16特征 + 16掩码) |
| 缺失处理 | 仅均值插补 | 插补 + 缺失指示 |
| 适用场景 | 简单基线 | 缺失数据较多的场景 |

### Q2: 如何修改超参数？

编辑 `configs/experiments/*.yaml` 文件：
```yaml
training:
  epochs: 200
  learning_rate: 0.001
  batch_size: 64
```

### Q3: 如何调试单个文件？

```python
python -c "from src.data.loader import load_dataset; ..."
```

---

*最后更新: 2026-02-23*
