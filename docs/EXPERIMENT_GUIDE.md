# 完整实验操作指南

## 实验概述

**实验设计**: 多维度控制变量的完整网格扫描

| 维度 | 选项 | 数量 |
|------|------|------|
| 数据集 | XJTU, TJU, HUST, MIT | 4 |
| 缺失机制 | MCAR, MAR | 2 |
| 方法 | MIM, Mean, Median, KNN, Zero | 5 |
| 模型 | MLP, LSTM, GRU, CNN1D | 4 |
| 缺失率评估 | 0.1, 0.2, ..., 0.9 | 9 |
| 随机种子 | 42, 43, ..., 141 | 100 |

**总评估数**: 4 × 2 × 5 × 4 × 9 × 100 = **144,000**

---

## 环境准备

### 1. 激活环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate battery
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
```

---

## 快速测试 (开发阶段)

### 单配置测试

```bash
# 快速测试: 1 dataset × 1 model × 1 method × 1 seed × 3 epochs
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42] training.epochs=3
```

**预期输出**:
```
Loaded 8 batteries from data/XJTU data
Battery split: 4 train, 2 val, 2 test
MR=0.1: MAE=0.0123, RMSE=0.0156, R²=0.9985
...
MR=0.9: MAE=0.0289, RMSE=0.0354, R²=0.9932
Results saved to: results/battery_soh_experiment_mlp_mim.csv
```

### 小网格测试

```bash
# 方法对比: 1 dataset × 1 model × 3 methods × 3 seeds × 10 epochs
for method in mim mean knn; do
    python src/main.py data=xjtu model=mlp method=$method \
        experiment.seeds=[42,43,44] training.epochs=10
done

# 可视化
python plot_results.py
```

---

## 标准实验 (验证阶段)

### 单配置完整实验

```bash
# 标准: 1 dataset × 1 model × 1 method × 10 seeds × 100 epochs
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42,43,44,45,46,47,48,49,50,51] \
    training.epochs=100
```

**运行时间**: ~30-60 分钟 (CPU)

### 模型对比实验

```bash
# 比较 4 个模型: 1 dataset × 4 models × 1 method × 10 seeds × 100 epochs
for model in mlp lstm gru cnn1d; do
    python src/main.py data=xjtu model=$model method=mim \
        experiment.seeds=[42..51] training.epochs=100
done
```

**运行时间**: ~2-4 小时 (CPU)

### 方法对比实验

```bash
# 比较 5 种方法: 1 dataset × 1 model × 5 methods × 10 seeds × 100 epochs
for method in mim mean median knn zero; do
    python src/main.py data=xjtu model=mlp method=$method \
        experiment.seeds=[42..51] training.epochs=100
done
```

**运行时间**: ~2.5-5 小时 (CPU)

---

## 完整论文实验

### 使用脚本自动化

创建 `scripts/run_full_experiment.sh`:

```bash
#!/bin/bash

DATASETS=(xjtu tju hust mit)
MODELS=(mlp lstm gru cnn1d)
METHODS=(mim mean median knn zero)
SEEDS_START=42
SEEDS_END=141
EPOCHS=100

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for method in "${METHODS[@]}"; do
            echo "Running: $dataset / $model / $method"
            python src/main.py \
                data=$dataset \
                model=$model \
                method=$method \
                experiment.seeds="[$(seq -s, $SEEDS_START $SEEDS_END)]" \
                training.epochs=$EPOCHS \
                wandb.enabled=false
        done
    done
done
```

**执行**:
```bash
chmod +x scripts/run_full_experiment.sh
./scripts/run_full_experiment.sh
```

**预期总时间**: ~600-1200 小时 (CPU)  
**建议**: 使用多 GPU 并行或分布式集群

---

## 实验监控

### 实时查看进度

```bash
# 查看结果文件大小 (估算进度)
watch -n 30 'ls -lh results/*.csv | wc -l'

# 查看最新结果
tail -f results/battery_soh_experiment_mlp_mim.csv
```

### 检查已完成配置

```bash
# 统计已完成的实验配置
python -c "
import os
csvs = [f for f in os.listdir('results') if f.endswith('.csv')]
print(f'Completed: {len(csvs)} configurations')
for f in sorted(csvs):
    print(f'  - {f}')
"
```

---

## 结果分析

### 加载所有结果

```python
import pandas as pd
import glob

# 加载所有 CSV
results = []
for csv_file in glob.glob('results/*.csv'):
    df = pd.read_csv(csv_file)
    results.append(df)

all_results = pd.concat(results, ignore_index=True)
print(f"Total evaluations: {len(all_results)}")
```

### 生成对比表

```python
# 按 dataset, model, method, missing_rate 分组统计
summary = all_results.groupby(
    ['dataset', 'model', 'method', 'missing_rate']
)['test_mae'].agg(['mean', 'std', 'count'])

print(summary)
```

### 统计显著性检验

```python
from scipy.stats import ttest_ind

# 比较 MIM vs Mean 在每个缺失率下的差异
for mr in [0.1, 0.3, 0.5, 0.7, 0.9]:
    mim = all_results[
        (all_results.method=='mim') & 
        (all_results.missing_rate==mr)
    ].test_mae
    
    mean = all_results[
        (all_results.method=='mean') & 
        (all_results.missing_rate==mr)
    ].test_mae
    
    t, p = ttest_ind(mim, mean)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    
    print(f"MR={mr:.1f}: MIM={mim.mean():.4f}, Mean={mean.mean():.4f}, p={p:.4f} {sig}")
```

### 可视化

```bash
# 生成对比图
python plot_results.py

# 结果保存在
ls results/comparison_plot.png
```

---

## 中断恢复

### 检查已完成

```bash
# 查看已有结果文件
ls results/*.csv

# 统计行数 (每个种子应有 9 行，对应 9 个缺失率)
wc -l results/*.csv
```

### 续跑未完成

由于每个配置独立保存到 CSV，直接重新运行即可：

```bash
# 重新运行会覆盖原有 CSV，或追加新种子
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42..141] training.epochs=100
```

### 仅跑未完成配置

```bash
# 检查缺失的配置并补跑
python << 'EOF'
import os

datasets = ['xjtu', 'tju', 'hust', 'mit']
models = ['mlp', 'lstm', 'gru', 'cnn1d']
methods = ['mim', 'mean', 'median', 'knn', 'zero']

completed = set()
for f in os.listdir('results'):
    if f.endswith('.csv'):
        # Parse: battery_soh_experiment_{dataset}_{model}_{method}.csv
        parts = f.replace('.csv', '').split('_')
        if len(parts) >= 5:
            dataset = parts[-3]
            model = parts[-2]
            method = parts[-1]
            completed.add((dataset, model, method))

for dataset in datasets:
    for model in models:
        for method in methods:
            if (dataset, model, method) not in completed:
                print(f"Missing: {dataset} / {model} / {method}")
EOF
```

---

## 硬件要求

| 配置 | CPU | 内存 | 磁盘 | 预估时间 |
|------|-----|------|------|----------|
| 开发测试 | 4核 | 8GB | 1GB | 10分钟 |
| 标准实验 | 8核 | 16GB | 5GB | 2-4小时 |
| 完整论文 | 32核+ | 32GB+ | 20GB | 600-1200小时 |

**优化建议**:
- 使用 GPU: `training.accelerator=gpu`
- 多进程并行: 每个配置独立运行
- 分布式: 使用 SLURM/Kubernetes 调度

---

## 故障排查

### OOM (内存不足)

```bash
# 减小 batch size
python src/main.py ... training.batch_size=32  # 默认64
python src/main.py ... training.batch_size=16  # 更小
```

### 特征列错误

```bash
# 检查数据集配置
head configs/data/xjtu.yaml

# 检查数据文件列名
head -1 "data/XJTU data/2C_battery-1.csv"
```

### 电池数量不足

```bash
# 检查数据目录
ls "data/XJTU data/" | grep battery

# 至少需要 3 个电池 (train/val/test 各1)
```

### 结果不一致

```bash
# 固定种子重跑验证
python src/main.py ... experiment.seeds=[42] training.epochs=10

# 比较两次结果
diff results/battery_soh_experiment_mlp_mim.csv results/battery_soh_experiment_mlp_mim.csv.backup
```

---

## 输出文件规范

### CSV 格式

```csv
seed,missing_rate,model,method,dataset,missing_mode,test_mae,test_rmse,test_r2
42,0.1,mlp,mim,xjtu,mcar,0.0104,0.0132,0.9989
42,0.2,mlp,mim,xjtu,mcar,0.0107,0.0135,0.9988
...
```

### 图表格式

```
results/comparison_plot.png
```

---

*最后更新: 2026-02-26*  
*对应代码: dev branch (Hydra + Lightning)*
