# SOH预测缺失数据处理实验代码

## 项目结构

```
src/
├── config/                 # 配置管理
│   └── experiment_config.py
├── data/                   # 数据加载和处理
│   ├── dataset_loader.py
│   └── datasets.py
├── models/                 # 模型定义
│   ├── base_model.py
│   ├── mlp.py
│   ├── lstm.py
│   ├── gru.py
│   ├── cnn1d.py
│   ├── xgboost_model.py
│   └── model_factory.py
├── trainers/               # 训练器
│   ├── neural_network_trainer.py
│   └── xgboost_trainer.py
├── evaluators/             # 评估器
│   ├── metrics.py
│   └── model_evaluator.py
├── experiments/            # 实验运行
│   └── experiment_runner.py
├── visualization/          # 可视化
│   └── missing_rate_curves.py
├── utils/                  # 工具函数
│   ├── seed_manager.py
│   └── logger.py
└── main.py                 # 主入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 运行完整实验（所有模型，100次重复）
python -m src.main --batch 3C --n_repeats 100

# 运行部分模型
python -m src.main --batch 3C --models mlp lstm --n_repeats 10

# 指定其他批次
python -m src.main --batch 2C --n_repeats 100
```

### 3. 实验结果

实验结果将保存在 `results/{timestamp}_{batch}/` 目录下：

```
results/
└── 20240201_143052_3C/
    ├── config.json           # 实验配置
    ├── logs/                 # 日志文件
    ├── models/               # 训练好的模型
    ├── results/              # CSV结果
    └── figures/              # 图表
```

## 模型配置

所有模型参数统一配置，无MIM版本参数量控制在~10,000：

| 模型 | 配置 | 无MIM参数量 |
|------|------|------------|
| MLP | [100, 64, 32] | ~10,145 |
| LSTM | hidden=42, 1层 | ~9,955 |
| GRU | hidden=48, 1层 | ~9,409 |
| CNN1D | Conv[48→32]+FC[48,24] | ~9,817 |
| XGBoost | 60树, depth=5 | ~7,680 |

## 核心特性

1. **MIM训练策略**：多缺失率联合训练（{0.0, 0.1, ..., 0.9}）
2. **统一隐藏层**：有/无MIM版本保持相同隐藏层结构
3. **100次重复实验**：确保统计显著性
4. **早停机制**：patience=15，防止过拟合
5. **完整日志**：记录训练过程和实验结果

## 开发指南

### 添加新模型

1. 在 `models/` 下创建新模型文件
2. 继承 `BaseModel` 类
3. 实现 `fit`, `predict`, `save`, `load` 方法
4. 在 `ModelFactory.create_model` 中注册

### 修改配置

编辑 `config/experiment_config.py` 中的 `ExperimentConfig` 类。
