# 大实验 V3 - 使用Configs_v3最佳参数

## 更新内容

本次更新将大实验的模型配置参数替换为 `configs_v3` 架构搜索的最佳结果：

### 模型配置对比

| 模型 | 旧配置 | 新配置 (configs_v3最佳) | MAE |
|------|--------|------------------------|-----|
| **MLP** | [100,64,32] 3层 | [192,96,48,24] 4层 | 0.0128 |
| **LSTM** | h=42, l=1 | h=48, l=2 | 0.0077 |
| **GRU** | h=48, l=1 | h=64, l=2 | 0.0069 |
| **CNN1D** | [48,32], k=3 | [72,32], k=4 | **0.0057** |

### 移除内容
- ❌ XGBoost 模型（已完全移除）

### 新增内容
- ✅ MLP 支持 dropout（configs_v3: 0.15）
- ✅ 所有模型使用2层架构（比1层更深）
- ✅ CNN1D 使用 kernel=4（configs_v3发现优于k=3）

---

## 实验规模

| 项目 | 数量 |
|------|------|
| 模型 | 4 (MLP, LSTM, GRU, CNN1D) |
| 策略 | 2 (Baseline, MIM) |
| 缺失率 | 9 (0.1 - 0.9) |
| 重复次数 | 100 (种子 42-141) |
| **单次实验数** | **7200 次评估** |

---

## 使用方法

### 运行完整大实验

```bash
python run_batch.py --n_repeats 100 --epochs 200
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 3C | 数据批次 |
| `--n_repeats` | 100 | 重复次数 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--batch_size` | 32 | 批次大小 |
| `--patience` | 15 | 早停耐心值 |
| `--device` | cpu | 计算设备 |
| `--resume` | - | 从检查点恢复 |

### 中断恢复

```bash
# 如果实验中断，使用 --resume 继续
python run_batch.py --n_repeats 100 --epochs 200 --resume
```

### 小测试（快速验证）

```bash
# 2次重复，5轮训练，快速验证
python run_batch.py --n_repeats 2 --epochs 5
```

---

## 实验流程

1. **大实验运行器** (`run_batch.py` → `BatchExperimentRunner`)
   - 管理100次重复实验
   - 支持中断恢复
   - 定期保存聚合结果

2. **小实验运行器** (`SingleExperimentRunner`)
   - 每次使用不同随机种子
   - 运行8个模型配置 (4模型 × 2策略)
   - 每个模型在9个缺失率下评估

3. **训练策略**
   - **Baseline**: 仅用完整数据训练
   - **MIM**: 混合10个缺失率 (0.0-0.9) 的数据训练

---

## 输出结构

```
experiments/3C/{timestamp}/
├── checkpoint.json              # 检查点（用于恢复）
├── logs/
│   └── batch.log               # 主日志
├── seed_42/                    # 第1次重复
│   ├── seed_42.log
│   ├── results.csv            # 8模型 × 9缺失率 = 72行
│   ├── models/                # 8个模型权重
│   └── figures/               # 可视化图表
├── seed_43/                    # 第2次重复
├── ...
└── aggregate/                  # 聚合结果（100次重复）
    ├── results_all.csv        # 合并数据
    └── figures/               # 统计图表
```

---

## 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `src/config/experiment_config.py` | 更新模型配置为configs_v3最佳参数，移除XGBoost |
| `src/experiments/single_experiment.py` | 移除XGBoost配置和特殊处理代码 |
| `src/models/mlp.py` | 添加dropout支持 |
| `src/models/model_factory.py` | MLP支持dropout参数 |

---

## 验证配置

```bash
python -c "
from src.config.experiment_config import ExperimentConfig
config = ExperimentConfig()
for mc in config.get_model_configs():
    print(f'{mc.name}: {mc.model_type}, MIM={mc.use_mim}')
"
```

预期输出8个配置：MLP, MLP-MIM, LSTM, LSTM-MIM, GRU, GRU-MIM, CNN1D, CNN1D-MIM
