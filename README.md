# Battery SOH Prediction with Missing Data

基于 PyTorch Lightning + Hydra 的电池 SOH 预测框架，对比 **MIM (Missing Indicator Method)** 与 **传统插补方法**。

## 核心对比

| 方法 | 训练数据 | 测试时处理 | 输入维度 | 特点 |
|------|---------|-----------|---------|------|
| **MIM (Proposed)** | 混合缺失率 0.0-0.9 | 直接 MIM 格式 | 32 (16+16) | 端到端，不插补 |
| **Mean Imputation** | 完整数据 | 缺失→均值填充 | 16 | 简单 baseline |
| **Median Imputation** | 完整数据 | 缺失→中位数填充 | 16 | 鲁棒 baseline |
| **KNN Imputation** | 完整数据 | 缺失→KNN填充 | 16 | 局部相关性 baseline |
| **Zero Imputation** | 完整数据 | 缺失→0填充 | 16 | 极端 baseline |

## 技术栈

| 组件 | 用途 |
|------|------|
| **Hydra** | 配置管理 |
| **PyTorch Lightning** | 训练框架 |
| **Weights & Biases** | 实验追踪 (可选) |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单次实验

```bash
# MIM 方法 (Proposed)
python src/main.py data=xjtu model=mlp method=mim

# 插补 Baselines
python src/main.py data=xjtu model=mlp method=mean
python src/main.py data=xjtu model=mlp method=median
python src/main.py data=xjtu model=mlp method=knn
python src/main.py data=xjtu model=mlp method=zero
```

### 更换模型

```bash
python src/main.py model=lstm    # LSTM
python src/main.py model=gru     # GRU
python src/main.py model=cnn1d   # 1D-CNN
```

### 更换数据集

```bash
python src/main.py data=xjtu   # 西安交通大学
python src/main.py data=tju    # 天津大学
python src/main.py data=hust   # 华中科技大学
python src/main.py data=mit    # 麻省理工大学
```

### 更换缺失机制

```bash
python src/main.py missing=mcar   # 完全随机缺失
python src/main.py missing=mar    # 依赖 SOH 的缺失
```

### 覆盖参数

```bash
python src/main.py \
    data=xjtu \
    model=mlp \
    method=mim \
    training.epochs=200 \
    training.batch_size=32 \
    experiment.seeds=[42,43,44]
```

## 项目结构

```
.
├── configs/              # Hydra 配置
│   ├── config.yaml       # 主配置 (实验名、种子)
│   ├── data/             # 4个数据集配置
│   ├── model/            # 4个模型配置
│   └── missing/          # 2种缺失机制
├── src/
│   ├── main.py           # 唯一入口
│   ├── models/           # 神经网络定义
│   ├── data/             # 数据加载 (电池级划分)
│   ├── missing_data/     # 缺失模拟 + 插补
│   └── utils/            # 工具函数
├── data/                 # 数据集 (4 universities)
├── results/              # 实验结果 (CSV)
└── plot_results.py       # 可视化脚本
```

## 模型

| 模型 | 输入 dim=16 | 输入 dim=32 | 参数量 |
|------|------------|-------------|--------|
| MLP | [100,64,32] | [84,56,28] | ~10K |
| LSTM | hidden=48×2 | hidden=48×2 | ~10K |
| GRU | hidden=64×2 | hidden=64×2 | ~9K |
| CNN1D | [72,32] | [64,32] | ~10K |

## 实验设计 (控制变量)

```
维度1: 数据集     {XJTU, TJU, HUST, MIT}
维度2: 缺失机制   {MCAR, MAR}  
维度3: 方法       {MIM, Mean, Median, KNN, Zero}
维度4: 模型       {MLP, LSTM, GRU, CNN1D}
维度5: 缺失率     {0.1, 0.2, ..., 0.9}
维度6: 随机种子   {42, 43, ..., 141} (100次)
```

**总实验数**: 4 × 2 × 5 × 4 × 9 × 100 = 144,000 次

**分层架构**:
- **单次实验**: 固定 (dataset, mode, method, model, seed)
- **批量实验**: 跨种子/方法/模型的组合

## 关键特性

### 1. 按电池划分 (Battery-wise Split)

同一电池的所有循环只属于 train/val/test 中的一个集合，避免数据泄漏。

```
Train: [电池3,4,5,7]
Val:   [电池1,8]
Test:  [电池2,6]
```

### 2. 滑动窗口序列 (Sliding Window)

LSTM/GRU/CNN 使用真实时间序列，不是简单重复。

```
输入: [c0,c1,c2,c3,c4] -> 预测 y4
      [c1,c2,c3,c4,c5] -> 预测 y5
```

### 3. MIM 训练策略

训练时混合 10 种缺失率 (0.0-0.9)，测试时指定缺失率。

```python
# 训练: 10× 数据量
for mr in [0.0, 0.1, ..., 0.9]:
    X_mim = simulate_missing(X, mr)  # [N, 32]

# 测试: 指定缺失率
X_test_mim = simulate_missing(X_test, mr=0.5)  # [N, 32]
```

### 4. 插补 Baseline

训练时用完整数据，测试时先缺失再插补。

```python
# 训练 (完整)
X_train  # [N, 16]

# 测试 (缺失+插补)
X_test_missing = simulate_missing(X_test, mr=0.5)  # [N, 16]
X_test_input = mean_imputation(X_test_missing)      # [N, 16]
```

## 结果分析

### 1. 运行实验

```bash
# 生成结果文件
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42,43,44] training.epochs=100

# 结果保存到: results/battery_soh_experiment_mlp_mim.csv
```

### 2. 可视化对比

```bash
python plot_results.py
# 生成: results/comparison_plot.png
```

### 3. 统计检验

```python
import pandas as pd
from scipy import stats

mim = pd.read_csv('results/battery_soh_experiment_mlp_mim.csv')
mean = pd.read_csv('results/battery_soh_experiment_mlp_mean.csv')

# MR=0.5 的 t检验
mim_mae = mim[mim.missing_rate==0.5].test_mae
mean_mae = mean[mean.missing_rate==0.5].test_mae
t_stat, p_value = stats.ttest_ind(mim_mae, mean_mae)
print(f"p-value: {p_value:.4f}")
```

## 实验结果示例

**配置**: XJTU 2C, MLP, 3 seeds, 10 epochs

| MR | MIM | Mean | KNN | MIM vs Best |
|----|-----|------|-----|-------------|
| 0.1 | **0.010** | 0.073 | 0.067 | **85% better** |
| 0.3 | **0.011** | 0.086 | 0.066 | **83% better** |
| 0.5 | **0.013** | 0.108 | 0.067 | **81% better** |
| 0.7 | **0.016** | 0.144 | 0.079 | **80% better** |
| 0.9 | **0.025** | 0.202 | 0.116 | **78% better** |

## 故障排查

### 问题: "至少需要3个电池"
**原因**: 按电池划分需要 train/val/test 各至少1个电池
**解决**: 检查数据目录中有 ≥3 个电池文件

### 问题: "找不到特征列"
**原因**: CSV 列名与配置不匹配
**解决**: 检查 `configs/data/{dataset}.yaml` 中的 `features`

### 问题: CUDA OOM
**原因**: MIM 训练集扩大10倍
**解决**: 减小 `training.batch_size` (64 → 32 或 16)

### 问题: 结果不一致
**原因**: 随机种子未正确设置
**解决**: 确认 `experiment.seeds` 和 `random_state` 一致

## 扩展

### 添加新数据集
1. 创建 `data/{dataset} data/` 目录
2. 添加 `configs/data/{dataset}.yaml`
3. 更新 `src/data/loader.py`

### 添加新模型
1. 在 `src/models/__init__.py` 定义模型
2. 添加 `configs/model/{model}.yaml`
3. 更新 `src/main.py` 的 `get_model_config()`

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{your_paper_2026,
  title={Robust SOH Prediction with Missing Data using Missing Indicator Method},
  author={Your Name},
  journal={Journal of Energy Storage},
  year={2026}
}
```

---

*Last updated: 2026-02-26*
