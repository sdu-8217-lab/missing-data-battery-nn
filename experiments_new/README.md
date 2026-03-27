# SOH预测实验框架

基于PyTorch + YAML配置的电池SOH预测实验框架，支持MIM方法和完整的测试流程。

## 特性

- **12种模型架构**：3种模型（MLP/LSTM/CNN）× 4种参数量级别
- **参数化验证**：支持`val_mr_subset`、`base_metric`、`aggregation`配置
- **插补方法**：支持zero/mean/knn/iterative四种插补
- **缺失模式**：支持MCAR/MAR/MNAR三种缺失模式
- **GPU优化**：自动检测CUDA，支持miniforge环境
- **实验流程分离**：train → eval → plot 三个阶段独立执行

## 模型架构配置

| 模型 | Level 1 (4K-8K) | Level 2 (8K-16K) | Level 3 (16K-32K) | Level 4 (32K-64K) |
|------|-----------------|------------------|-------------------|-------------------|
| **MLP** | [80, 56] | [120, 80] | [168, 104] | [256, 128, 56] |
| **LSTM** | h=28, l=1 | h=44, l=1 | h=44, l=2 | h=64, l=2 |
| **CNN** | [40, 28] | [72, 40] | [96, 56] | [136, 72] |

## 快速开始

### 环境要求

- **Conda环境**: `battery-nn` (已配置PyTorch + CUDA)
- **GPU**: NVIDIA RTX 4060 Ti (16GB显存)
- **CUDA**: 12.9

### 完整实验流水线

```bash
cd experiments_new

# 运行完整实验（训练 → 评估 → 绘图）
./run_pipeline.sh --config configs/experiments/phase1_mlp_mim.yaml

# 使用特定模型架构
./run_pipeline.sh \
    --config configs/experiments/phase1_mlp_mim.yaml \
    --model-config configs/models/lstm_level_2.yaml

# 指定多个种子
./run_pipeline.sh \
    --config configs/experiments/phase1_mlp_mim.yaml \
    --seeds "42 123 456"
```

### 分阶段执行

```bash
# 阶段1: 训练模型（分界线以上）
python scripts/train_models.py \
    --config configs/experiments/phase1_mlp_mim.yaml

# 阶段2: 评估模型（分界线以下）
python scripts/evaluate_models.py \
    --models-dir results/phase1/3C/*/models \
    --missing-modes MCAR MAR MNAR \
    --imputation-methods zero mean knn iterative

# 阶段3: 生成图表
python scripts/generate_plots.py \
    --results results/phase1/3C/*/test_results.csv
```

## 实验配置

### YAML配置示例

```yaml
name: "phase1_mlp_mim"
seed: 42
n_repeats: 1

data:
  data_dir: "./data/raw/XJTU"
  batch: "3C"
  test_size: 0.25
  val_size: 0.25

training:
  epochs: 200
  batch_size: 32
  lr: 0.001
  weight_decay: 1e-5  # META §6.3
  early_stopping_patience: 30  # META §6.2
  
  use_mim: true
  imputation_method: "zero"  # zero/mean/knn/iterative
  training_missing_rates:
    - 0.0
    - 0.05
    # ... 20个缺失率
  
  validation:
    val_mr_subset: [-1]        # -1=所有, [0.5]=单点
    base_metric: "mae"         # mae/mse/rmse
    aggregation: "mean"        # mean/max/min/median/worst

evaluation:
  missing_rates:
    - 0.0
    - 0.05
    # ... 20个缺失率
  missing_modes:
    - "MCAR"
    - "MAR"
    - "MNAR"
  imputation_methods:
    - "zero"
    - "mean"
    - "knn"
    - "iterative"

model_config_path: "configs/models/mlp_level_3.yaml"
output_dir: "./results"
device: "auto"
```

### 验证指标配置模式

```yaml
# 模式1: 单点优化（仅用MR=0.5验证）
validation:
  val_mr_subset: [0.5]
  base_metric: "mae"
  aggregation: "mean"

# 模式2: 高缺失率关注
validation:
  val_mr_subset: [0.7, 0.8, 0.9]
  aggregation: "mean"

# 模式3: 最差情况保守模式
validation:
  val_mr_subset: [-1]
  aggregation: "worst"

# 模式4: 均衡模式（默认）
validation:
  val_mr_subset: [-1]
  base_metric: "mae"
  aggregation: "mean"
```

## 测试流程

每个训练好的模型将经过240种组合的测试：

| 维度 | 选项 | 数量 |
|------|------|------|
| 缺失模式 | MCAR, MAR, MNAR | 3 |
| 缺失率 | 0.0, 0.05, ..., 0.95 | 20 |
| 插补方法 | zero, mean, knn, iterative | 4 |
| **总计** | | **240组合** |

## 输出结构

```
results/
└── {experiment_name}/
    └── {batch}/
        └── {timestamp}/
            ├── config.yaml              # 实验配置
            ├── train.log                # 训练日志
            ├── models/                  # 模型文件
            │   ├── model_3C_mlp_mimFalse_seed42.pt
            │   └── model_3C_mlp_mimTrue_seed42.pt
            ├── train_results.csv        # 训练结果汇总
            ├── eval.log                 # 评估日志
            ├── test_results.csv         # 完整测试结果 (240组合/模型)
            ├── plot.log                 # 绘图日志
            └── figures/                 # 图表
                ├── mae_by_missing_rate.png
                ├── imputation_comparison.png
                ├── heatmap.png
                ├── summary_table.csv
                └── report.md
```

## 批量运行所有12种模型配置

```bash
for model in mlp lstm cnn; do
    for level in 1 2 3 4; do
        echo "Running ${model}_level_${level}..."
        ./run_pipeline.sh \
            --config configs/experiments/phase1_mlp_mim.yaml \
            --model-config configs/models/${model}_level_${level}.yaml \
            --seeds "42 123"
    done
done
```

## 关键实现细节

### META文档遵循

- **§2.5**: 验证指标参数化配置 (`val_mr_subset`, `aggregation`)
- **§1.3**: MIM训练使用插补 + 缺失指示器
- **§4**: MCAR/MAR/MNAR缺失模式生成
- **§7.6**: 实验流程分离 (train/eval/plot)

### 分界线原则

```
分界线以上（训练阶段）:
  L1: Seed, L3: Batch, L4: Model, L5: use_mim, L6: 训练MR
  → 训练144个模型

分界线以下（测试阶段）:
  L7: Mode, L8: 测试MR, L9: Imputation
  → 每个模型240种测试组合
```

## 故障排除

```bash
# 验证环境
python3 test_installation.py

# 检查GPU
./run.sh --help

# 只运行训练阶段（跳过评估和绘图）
./run_pipeline.sh --skip-eval --skip-plot

# 只运行评估阶段（使用已训练模型）
./run_pipeline.sh --skip-train --skip-plot
```

## 依赖

- Python 3.8+
- PyTorch 2.10+ (CUDA 12.9)
- pandas, numpy, scikit-learn
- pyyaml, loguru
- matplotlib, seaborn

安装：
```bash
conda activate battery-nn
conda install pytorch pandas numpy scikit-learn matplotlib seaborn pyyaml -c pytorch
pip install loguru
```
