# 实验参数参考

<!-- 
  confidence: B
  reviewed: 2026-03-19
  reviewer: chen
  ai-assisted: true
-->

本文档从代码中提取当前实验使用的参数，作为参考。

## 实验配置（来自 run_batch_experiments.py）

```python
# 9层实验架构参数

# L1: 随机种子
SEEDS = [42]  # 默认单种子
# 完整实验: list(range(0, 100))  # 100个种子

# L2: 数据集（当前固定为XJTU）
# DATASET = "xjtu"

# L3: 电池批次
BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]

# L4: 模型架构
MODELS = ["mlp", "lstm", "cnn"]

# L5: 是否使用MIM
USE_MIMS = ["false", "true"]

# L6: 训练缺失率（由L5决定）
# use_mim=false: Train MR = 0.0
# use_mim=true:  Train MR = 0.0-0.9（混合）

# L7: 测试缺失模式
MODES = ["MCAR", "MAR", "MNAR"]

# L8: 测试缺失率
TEST_MRS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# L9: 插补方法
IMPUTATIONS = ["mean", "knn", "iterative", "zero"]
```

## 实验规模计算

### 分界线以上（训练）

```
训练模型数 = len(SEEDS) × len(BATCHES) × len(MODELS) × len(USE_MIMS)
          = 100 × 6 × 3 × 2
          = 3,600 个模型
```

### 分界线以下（测试）

```
每模型测试数 = len(MODES) × len(TEST_MRS) × len(IMPUTATIONS)
            = 3 × 10 × 4
            = 120 个测试
```

### 总计

```
总结果数 = 3,600 个模型 × 120 个测试
        = 432,000 行结果
```

## 运行参数

```python
# 默认参数
DEFAULT_EPOCHS = 50
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_DIR = "results"

# 超时设置
TRAIN_TIMEOUT = 600  # 10分钟
TEST_TIMEOUT = 60    # 1分钟
```

## 文件命名规范

### 模型文件

```
{model_dir}/seed{seed}_batch{batch}_model{model}_{mim_suffix}.pt

示例:
- seed0_batch2C_modelmlp_mim.pt      (use_mim=true)
- seed0_batch2C_modelmlp_no_mim.pt   (use_mim=false)
```

### 结果文件

```
{results_dir}/seed{seed}_batch{batch}_model{model}_{mim}_{mode}_{mr}_{imp}.json

示例:
- seed42_batch2C_modelmlp_mim_mcar_0.3_mean.json
```
