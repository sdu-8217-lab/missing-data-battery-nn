# 代码架构

## 核心设计原则

1. **单一职责**：每个模块负责一个明确功能
2. **配置驱动**：Hydra 管理所有实验参数
3. **接口统一**：模型实现统一接口，便于扩展
4. **可复用性**：核心组件可在不同场景复用

## 项目结构

```
src/
├── main.py              # 唯一入口，Hydra配置驱动
├── data/                # 数据层
│   ├── loader.py        # 数据加载、电池划分
│   ├── loader_dataloaders.py  # DataLoader创建
│   └── preprocessing.py # 数据预处理
├── models/              # 模型层
│   ├── mlp.py, lstm.py, gru.py, cnn1d.py  # 模型定义
│   └── model_factory.py # 模型工厂
├── trainers/            # 训练层
│   ├── lightning_module.py  # Lightning模块（统一）
│   └── lightning_trainer.py # 训练器
├── missing_data/        # 缺失数据处理
│   ├── mcar.py, mar.py      # 缺失模拟
│   └── imputation.py        # 插补方法
├── evaluation/          # 评估层
│   ├── metrics.py           # 评估指标
│   └── model_evaluator.py   # 模型评估
└── experiments/         # 实验调度
    ├── runner.py            # 单实验运行
    ├── scheduler.py         # 批量调度
    └── database.py          # 实验数据库
```

## 关键设计

### MIM (Missing Indicator Method)

```python
# 传统方法 (16维)
输入 = [特征值]

# MIM方法 (32维)  
输入 = [特征值] + [缺失指示器]
```

### 混合缺失率训练

```python
# 训练时：混合多种缺失率
for mr in [0.0, 0.1, ..., 0.9]:
    X_missing = apply_missing(X_train, mr)
    train(X_missing, y_train)

# 测试时：指定缺失率
X_test_missing = apply_missing(X_test, mr=0.5)
predict = model(X_test_missing)
```

### 按电池划分

```
❌ 随机划分样本
✅ 整个电池只属于 train/val/test 之一

Train: [电池3,4,5,7]
Val:   [电池1,8]
Test:  [电池2,6]
```

## 配置系统

使用 Hydra 配置，支持命令行覆盖：

```bash
# 基础运行
python src/main.py data=xjtu model=mlp method=mim

# 参数覆盖
python src/main.py data=xjtu model=mlp method=mim training.epochs=200

# 多种子运行
python src/main.py data=xjtu model=mlp method=mim experiment.seeds=[42,43,44]
```

## 扩展指南

### 添加新模型

1. 在 `src/models/` 创建模型类
2. 在 `src/models/model_factory.py` 注册

### 添加新方法

1. 在 `configs/method/` 创建配置
2. 如有需要，在 `src/missing_data/` 实现

## 相关文档

- [EXPERIMENTS.md](EXPERIMENTS.md) - 实验设计
- [DATASETS.md](DATASETS.md) - 数据集说明
