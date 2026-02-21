# 大实验V2 - 使用指南

## 概述

大实验V2版本整合了以下改进：

| 改进项 | 旧版 | V2版本 | 收益 |
|--------|------|--------|------|
| 配置管理 | dataclass | **Pydantic** | 自动验证、类型安全 |
| 训练循环 | 自定义 | **PyTorch Lightning** | 自动设备管理、内置早停 |
| 日志系统 | logging | **Loguru** | UTF-8编码、自动轮转 |
| 模型参数 | 旧架构 | **configs_v3最佳** | 性能最优 |
| 模型数量 | 5 (含XGBoost) | **4 (清理XGBoost)** | 专注深度学习 |

---

## 模型配置 (configs_v3最佳参数)

| 模型 | 架构 | Dropout | configs_v3 MAE |
|------|------|---------|----------------|
| **MLP** | [192,96,48,24] 4层 | 0.15 | 0.0128 |
| **LSTM** | h=48, l=2 | 0.20 | 0.0077 |
| **GRU** | h=64, l=2 | 0.20 | 0.0069 |
| **CNN1D** | [72,32], kernel=4 | 0.10 | **0.0057** |

---

## 使用方法

### 1. 快速测试 (推荐首次运行)

```bash
# 2次重复，5轮训练
python run_big_experiment_v2.py --n_repeats 2 --epochs 5
```

### 2. 完整实验

```bash
# 100次重复，200轮训练（默认）
python run_big_experiment_v2.py

# 或使用批处理文件
run_big_v2.bat 100 200
```

### 3. 其他批次

```bash
# 2C批次
python run_big_experiment_v2.py --batch 2C --n_repeats 100 --epochs 200

# R2.5批次
python run_big_experiment_v2.py --batch R2.5 --n_repeats 100 --epochs 200
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 3C | 数据批次 (2C, 3C, R2.5, R3, RW, Sim_satellite) |
| `--n_repeats` | 100 | 重复次数 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--batch_size` | 32 | 批量大小 |
| `--patience` | 15 | 早停耐心值 |
| `--seed` | 42 | 随机种子起始值 |

---

## 输出结构

```
experiments_v2/
└── {batch}/
    └── {timestamp}/
        ├── config.json          # 实验配置（Pydantic序列化）
        ├── results_all.csv      # 完整结果
        ├── results_partial.csv  # 中间结果（每10次重复保存）
        └── experiment_*.log     # 日志文件
```

---

## 代码架构

### 主要组件

```
run_big_experiment_v2.py         # 主入口
├── src/config/pydantic_config.py    # Pydantic配置（已更新configs_v3参数）
├── src/trainers/lightning_module.py # PyTorch Lightning模块
├── src/trainers/lightning_trainer.py # Lightning训练器
├── src/utils/logger_v2.py           # Loguru日志
└── src/models/model_factory.py      # 模型工厂（已清理XGBoost）
```

### 实验流程

```
for repeat in 1..n_repeats:
    seed = random_seed + repeat
    data = prepare_data(seed)  # 重新划分数据
    
    for model_config in [MLP, LSTM, GRU, CNN1D]:
        # Baseline版本
        train_baseline(model_config, data)
        evaluate(model, missing_rates=[0.1-0.9])
        
        # MIM版本
        train_mim(model_config, data)  # 混合10个缺失率训练
        evaluate(model, missing_rates=[0.1-0.9])
```

---

## 与V1版本对比

| 特性 | V1 (旧版) | V2 (改进版) |
|------|-----------|-------------|
| 配置验证 | 手动 | **Pydantic自动验证** |
| 训练循环 | 自定义133行 | **Lightning 30行** |
| 设备管理 | 手动 | **自动(CPU/GPU)** |
| 早停 | 手动实现 | **内置回调** |
| 日志编码 | GBK问题 | **UTF-8自动处理** |
| 模型参数 | 旧架构 | **configs_v3最佳** |
| XGBoost | 包含 | **已清理** |

---

## 故障排除

### 问题1: Pydantic验证错误

```
ValidationError: 1 validation error for ExperimentConfig
missing_rates -> 0
  value is not a valid float (type=type_error.float)
```

**解决**: 检查配置文件中的缺失率格式

### 问题2: 内存不足

```bash
# 减小batch_size
python run_big_experiment_v2.py --batch_size 16
```

### 问题3: 训练失败

查看日志文件: `experiments_v2/{batch}/experiment_*.log`

---

## 技术细节

### Pydantic配置验证示例

```python
from src.config.pydantic_config import ExperimentConfig

# 自动验证类型和范围
config = ExperimentConfig(
    batch="3C",
    n_repeats=100,
    epochs=200,
    lr=0.001  # 必须在 (0, 1] 范围内
)

# 保存配置
config.save_json("config.json")

# 加载配置
config = ExperimentConfig.load_json("config.json")
```

### PyTorch Lightning训练示例

```python
from src.trainers.lightning_module import SOHLightningModule
from src.trainers.lightning_trainer import LightningTrainer

# 包装模型
lightning_model = SOHLightningModule(pytorch_model, learning_rate=0.001)

# 训练（自动处理设备、早停等）
trainer = LightningTrainer(max_epochs=200, patience=15)
history = trainer.fit(train_loader, val_loader)
```

---

## 预期运行时间

| 配置 | 单次时间 | 总时间 |
|------|----------|--------|
| 100 repeats × 8 models × 9 MR | ~15秒/次 | ~33小时 |
| (CPU, configs_v3参数) | | |

建议：使用多窗口并行运行不同批次
