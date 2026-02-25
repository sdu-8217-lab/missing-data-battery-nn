# AGENTS.md - 电池 SOH 预测实验指南

**项目**: 基于 MIM 的电池 SOH 鲁棒预测  
**架构**: Hydra + PyTorch Lightning  
**核心对比**: MIM vs 传统插补方法

---

## 1. 核心概念

### 1.1 研究目标
在电池数据存在缺失的条件下，比较两种范式：
- **MIM (Missing Indicator Method)**: 端到端学习，不插补
- **传统插补**: 先插补（Mean/KNN/...），再预测

### 1.2 关键创新
| 方面 | 传统方法 | MIM (本文) |
|------|---------|-----------|
| 缺失处理 | 先插补再输入模型 | 直接输入缺失指示器 |
| 输入维度 | 16 | 32 (16特征+16掩码) |
| 训练策略 | 完整数据训练 | 混合缺失率(0.0-0.9)训练 |
| 模型适配 | 需调整输入层 | 需调整输入层 |

### 1.3 实验维度（控制变量）

实验设计为**多维度控制变量**的网格搜索：

```
维度1: 数据集       {XJTU, TJU, HUST, MIT}
维度2: 缺失机制     {MCAR, MAR}
维度3: 方法         {MIM, Mean, Median, KNN, Zero}
维度4: 模型         {MLP, LSTM, GRU, CNN1D}
维度5: 缺失率       {0.1, 0.2, ..., 0.9}
维度6: 随机种子     {42, 43, ..., 141} (100次重复)

总实验数 = 4 × 2 × 5 × 4 × 9 × 100 = 144,000 次
```

---

## 2. 分层实验架构

### 2.1 实验层级

```
大型实验 (Full Experiment)
├── 数据集循环 (4 datasets)
│   ├── 缺失机制循环 (2 modes)
│   │   ├── 方法循环 (5 methods)
│   │   │   ├── 模型循环 (4 models)
│   │   │   │   └── 种子循环 (100 seeds)
│   │   │   │       └── 缺失率评估 (9 rates)
```

### 2.2 单次实验 (Single Experiment)

**定义**: 固定 (dataset, mode, method, model, seed) 的一次完整训练和评估

**流程**:
```
1. 加载数据 (按电池划分)
   └── 训练: [电池3,4,5,7], 验证: [电池1,8], 测试: [电池2,6]
   
2. 训练模型
   ├── MIM: 混合缺失率 0.0-0.9 训练
   └── Baseline: 完整数据训练
   
3. 评估 (9个缺失率)
   ├── MR=0.1: MAE, RMSE, R²
   ├── MR=0.2: MAE, RMSE, R²
   └── ... MR=0.9
   
4. 保存结果
   └── results/battery_soh_{dataset}_{model}_{method}.csv
```

**运行命令**:
```bash
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42] training.epochs=100
```

### 2.3 批量实验 (Batch Experiment)

**定义**: 多个单次实验的组合，通常跨种子或跨方法

**场景1: 种子重复实验** (100 seeds)
```bash
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42,43,44,...,141] training.epochs=100
```

**场景2: 方法对比实验** (MIM vs Mean vs KNN)
```bash
# 分别运行
for method in mim mean knn; do
    python src/main.py data=xjtu model=mlp method=$method \
        experiment.seeds=[42,43,44] training.epochs=100
done
```

**场景3: 全网格实验** (完整论文实验)
```bash
# 使用脚本自动化
./scripts/run_full_experiment.sh
```

---

## 3. 项目结构

```
missing-data-battery-nn/
├── configs/                    # Hydra 配置
│   ├── config.yaml            # 主配置 (实验名称、种子列表)
│   ├── data/                  # 数据集配置
│   │   ├── xjtu.yaml         # XJTU: {batch_id}_battery-*.csv
│   │   ├── tju.yaml          # TJU: Dataset_1_NCA_battery/*.csv
│   │   ├── hust.yaml         # HUST: *.csv
│   │   └── mit.yaml          # MIT: 2017-05-12/*battery*.csv
│   ├── model/                 # 模型配置
│   │   ├── mlp.yaml          # [100,64,32] or [84,56,28]
│   │   ├── lstm.yaml         # hidden=48, num_layers=2
│   │   ├── gru.yaml          # hidden=64, num_layers=2
│   │   └── cnn1d.yaml        # channels=[72,32] or [64,32]
│   └── missing/               # 缺失机制配置
│       ├── mcar.yaml         # 完全随机缺失
│       └── mar.yaml          # 依赖SOH的缺失
│
├── src/
│   ├── main.py                # 唯一入口
│   ├── data/
│   │   ├── loader.py         # 电池级数据加载 + 按电池划分
│   │   ├── loader_dataloaders.py  # DataLoader创建 (滑动窗口)
│   │   ├── splits.py         # 电池划分逻辑
│   │   ├── features.py       # 特征构造 (16维)
│   │   └── preprocessing.py  # 数据清洗 + 标准化
│   ├── models/
│   │   ├── __init__.py       # MLP, LSTM, GRU, CNN1D 定义
│   │   └── lightning_module.py  # PyTorch Lightning 包装
│   ├── missing_data/
│   │   ├── mcar.py           # MCAR缺失模拟
│   │   ├── mar.py            # MAR缺失模拟 (修正公式)
│   │   └── imputation.py     # Mean/Median/KNN/Zero插补
│   └── utils/
│       └── seed_manager.py   # 随机种子管理
│
├── data/                      # 数据集 (4 universities)
│   ├── XJTU data/
│   ├── TJU data/
│   ├── HUST data/
│   └── MIT data/
│
├── results/                   # 实验结果 (gitignored)
│   └── battery_soh_{dataset}_{model}_{method}.csv
│
├── scripts/                   # 批量实验脚本
│   └── run_full_experiment.sh
│
└── plot_results.py           # 结果可视化
```

---

## 4. 关键实现细节

### 4.1 数据划分 (按电池)

**原则**: 同一电池的所有循环只属于一个集合 (train/val/test)

**代码** (`src/data/splits.py`):
```python
# 电池ID列表: ['2C_battery-1', '2C_battery-2', ..., '2C_battery-8']
battery_ids = list(battery_data.keys())

# 划分电池ID (不是划分样本!)
train_ids, val_ids, test_ids = split_batteries(
    battery_ids, test_size=0.25, val_size=0.25
)
# 结果: train=[3,4,5,7], val=[1,8], test=[2,6]

# 收集各组样本
X_train = np.vstack([battery_data[bid][0] for bid in train_ids])
```

### 4.2 序列构造 (滑动窗口)

**原则**: LSTM/GRU/CNN 使用真实时间序列，不是简单重复

**代码** (`src/data/loader_dataloaders.py`):
```python
def create_sliding_windows(X, y, seq_len=5):
    # X: [N, D], y: [N]
    for i in range(N - seq_len + 1):
        window = X[i:i+seq_len]      # [seq_len, D]
        target = y[i+seq_len-1]      # 窗口最后一个时间步
    # 返回: [N-seq_len+1, seq_len, D]
```

**示例** (seq_len=5):
```
循环序列: [c0, c1, c2, c3, c4, c5, c6, ...]
窗口0: [c0,c1,c2,c3,c4] -> target=y4
窗口1: [c1,c2,c3,c4,c5] -> target=y5
...
```

### 4.3 MIM 训练策略

**原则**: 训练时混合 0.0-0.9 缺失率，测试时指定缺失率

**代码**:
```python
# 训练
mim_rates = [0.0, 0.1, 0.2, ..., 0.9]  # 10种
for mr in mim_rates:
    X_mr, mask, mim_input = simulate_mcar(X_train, mr)
    # mim_input: [N, 32] (16 features + 16 mask)
# 合并所有: [10*N, 32]

# 测试 (MR=0.5)
X_test_mr, mask, mim_input = simulate_mcar(X_test, missing_rate=0.5)
```

### 4.4 插补 Baseline 策略

**原则**: 训练用完整数据，测试时先缺失再插补

**代码**:
```python
# 训练 (无缺失)
X_train_input = X_train  # [N, 16]

# 测试 (MR=0.5)
X_test_missing, mask, _ = simulate_mcar(X_test, 0.5)
X_test_input = mean_imputation(X_test_missing, mask)  # [N, 16]
```

---

## 5. 运行实验

### 5.1 快速测试 (1 seed, 3 epochs)
```bash
python src/main.py data=xjtu model=mlp method=mim \
    training.epochs=3 experiment.seeds=[42]
```

### 5.2 标准实验 (3 seeds, 100 epochs)
```bash
python src/main.py data=xjtu model=mlp method=mim \
    training.epochs=100 experiment.seeds=[42,43,44]
```

### 5.3 方法对比 (MIM vs Baselines)
```bash
for method in mim mean median knn zero; do
    python src/main.py data=xjtu model=mlp method=$method \
        training.epochs=100 experiment.seeds=[42,43,44]
done
```

### 5.4 全网格实验 (完整论文)
```bash
# 使用提供的脚本
./scripts/run_full_experiment.sh

# 或手动循环
for dataset in xjtu tju hust mit; do
    for model in mlp lstm gru cnn1d; do
        for method in mim mean knn; do
            python src/main.py data=$dataset model=$model method=$method \
                training.epochs=100 experiment.seeds=[42..141]
        done
    done
done
```

---

## 6. 结果分析

### 6.1 结果文件格式

`results/battery_soh_xjtu_mlp_mim.csv`:
```csv
seed,missing_rate,model,method,test_mae,test_rmse,test_r2
42,0.1,mlp,mim,0.0104,0.0132,0.9989
42,0.3,mlp,mim,0.0111,0.0141,0.9987
...
44,0.9,mlp,mim,0.0253,0.0318,0.9941
```

### 6.2 可视化

```bash
python plot_results.py
# 生成: results/comparison_plot.png
```

### 6.3 统计检验

```python
import pandas as pd
from scipy import stats

# 加载结果
mim = pd.read_csv('results/battery_soh_xjtu_mlp_mim.csv')
mean = pd.read_csv('results/battery_soh_xjtu_mlp_mean.csv')

# t检验 (MR=0.5)
mim_mae = mim[mim.missing_rate==0.5].test_mae
mean_mae = mean[mean.missing_rate==0.5].test_mae
t_stat, p_value = stats.ttest_ind(mim_mae, mean_mae)
print(f"p-value: {p_value:.4f}")  # p<0.05 表示显著差异
```

---

## 7. 故障排查

### 7.1 "至少需要3个电池"
- **原因**: 按电池划分需要 train/val/test 至少各1个电池
- **解决**: 确保数据目录中有 ≥3 个电池文件

### 7.2 "找不到特征列"
- **原因**: CSV 列名与配置不匹配
- **解决**: 检查 `configs/data/{dataset}.yaml` 中的 `features` 列表

### 7.3 CUDA out of memory
- **原因**: MIM 训练集扩大10倍
- **解决**: 减小 `training.batch_size` (默认64 → 32 或 16)

### 7.4 结果复现不一致
- **原因**: 随机种子未正确设置
- **检查**: 确认 `experiment.seeds` 一致，且 `random_state` 相同

---

## 8. 扩展指南

### 8.1 添加新数据集
1. 在 `data/` 下创建 `{dataset} data/` 目录
2. 添加 `configs/data/{dataset}.yaml`
3. 在 `src/data/loader.py` 的 `load_dataset()` 中添加加载逻辑

### 8.2 添加新模型
1. 在 `src/models/__init__.py` 中定义模型类
2. 添加 `configs/model/{model}.yaml`
3. 更新 `src/main.py` 的 `get_model_config()`

### 8.3 添加新的缺失机制
1. 在 `src/missing_data/` 创建 `{mechanism}.py`
2. 实现 `simulate_{mechanism}(X, missing_rate, ...)` 函数
3. 添加 `configs/missing/{mechanism}.yaml`

---

*最后更新: 2026-02-26*  
*对应代码版本: dev branch (post Hydra+Lightning refactor)*
