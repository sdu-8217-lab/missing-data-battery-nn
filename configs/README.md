# 配置目录

<!-- 
  confidence: B
  reviewed: 2026-03-19
  reviewer: chen
  ai-assisted: true
-->

## 重要说明

⚠️ **当前实验系统（run_batch_experiments.py）使用命令行参数（argparse），
不直接读取本目录下的YAML配置文件。**

## 目录结构

```
configs/
├── README.md                    # 本文档
├── legacy/                      # 旧Hydra配置（已废弃，保留参考）
│   ├── data/                    # 数据集配置
│   ├── method/                  # 方法配置（插补/MIM）
│   ├── missing/                 # 缺失模式配置
│   ├── model/                   # 模型配置
│   ├── paper/                   # 论文实验配置
│   ├── experiment-single/       # 单实验配置（原experiment/）
│   ├── experiments-batch/       # 批量实验配置（原experiments/）
│   ├── base.yaml                # 根级基础配置
│   ├── config.yaml              # 主配置
│   └── model_architectures.yaml # 模型架构定义
├── reference/                   # 参考配置（当前实验使用的参数参考）
│   └── experiment-params.md     # 实验参数说明
└── schemas/                     # 配置schema（预留，未来使用）
    └── experiment-v2.yaml       # 新实验配置schema
```

## 当前实验运行方式

```bash
# 批量实验（推荐）
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds $(seq 0 99) \
    --epochs 50

# 单次实验
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 50
```

## 实验参数（代码中定义）

```python
# experiments/run_batch_experiments.py

SEEDS = [0, 1, 2, ..., 99]          # L1: 随机种子
BATCHES = ["2C", "3C", "R2.5",      # L3: 电池批次
           "R3", "RW", "Sim_satellite"]
MODELS = ["mlp", "lstm", "cnn"]     # L4: 模型架构
USE_MIMS = ["false", "true"]        # L5: 是否使用MIM
MODES = ["MCAR", "MAR", "MNAR"]     # L7: 测试缺失模式
TEST_MRS = [0.0, 0.1, ..., 0.9]     # L8: 测试缺失率
IMPUTATIONS = ["mean", "knn",       # L9: 插补方法
               "iterative", "zero"]
```

## 旧Hydra配置说明

`legacy/` 目录下的YAML文件是旧版Hydra架构的配置，
当前已不再使用，但保留作为参考。

如需查看旧配置：
```bash
ls configs/legacy/
cat configs/legacy/model/mlp.yaml
```

## 未来计划

- `schemas/`：定义配置schema，支持配置验证
- `reference/`：提取当前代码中的参数作为参考文档
