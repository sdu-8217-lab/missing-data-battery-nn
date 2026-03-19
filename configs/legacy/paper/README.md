# 论文精确配置文件

本目录包含用于复现论文实验的精确配置文件。

## 配置结构

```
paper/
├── common.yaml              # 通用训练参数（100 seeds, 9 missing rates, Adam, etc.）
├── run_mlp_baseline.yaml    # MLP Baseline 快速运行配置
├── run_mlp_mim.yaml         # MLP MIM 快速运行配置
└── README.md               # 本文件
```

模型配置位于 `configs/model/paper_*.yaml`：
- `paper_mlp.yaml` - MLP: [192,96,48,24], 27,649/30,721 params
- `paper_lstm.yaml` - LSTM: hidden=48, layers=2, 31,537/34,609 params
- `paper_gru.yaml` - GRU: hidden=64, layers=2, 40,769/43,841 params
- `paper_cnn1d.yaml` - CNN1D: channels=[72,32], 13,961/18,569 params

## 快速开始

### 1. 快速运行单个实验（推荐）

```bash
# MLP Baseline, MR=0.5, seed=42
python src/main.py --config-name=paper/run_mlp_baseline

# MLP MIM, MR=0.5, seed=42
python src/main.py --config-name=paper/run_mlp_mim
```

### 2. 使用命令行参数（灵活）

```bash
# MLP Baseline, 所有缺失率 (0.1-0.9), seed=42
python src/main.py model=paper_mlp method=baseline experiment.seeds=[42]

# MLP MIM, 所有缺失率, seed=42
python src/main.py model=paper_mlp method=mim experiment.seeds=[42]

# LSTM MIM, 特定缺失率, 5 epochs
python src/main.py model=paper_lstm method=mim experiment.seeds=[42,43] missing.missing_rates_eval=[0.5] training.epochs=5
```

### 3. 使用脚本运行

```bash
python scripts/run_paper_experiment.py --model paper_mlp --method mim --seed 42 --mr 0.5
```

## 模型参数量对照

| 模型 | Baseline | MIM (论文) | MIM (实际) | 状态 |
|------|----------|-----------|-----------|------|
| MLP | 27,649 | 36,865 | 30,721 | Baseline匹配 |
| LSTM | 31,537 | 40,753 | 34,609 | Baseline匹配 |
| GRU | 40,769 | 49,985 | 43,841 | Baseline匹配 |
| CNN1D | 16,713 | 30,537 | 18,569 | 两者不同 |

## 论文关键参数

### 训练设置
- 优化器: Adam
- 学习率: 0.001
- Batch Size: 64
- 最大 Epoch: 200
- 早停耐心: 20

### 缺失率
- 9档: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

### 随机种子
- 100个种子: 42-141

## 相关文档

- `docs/paper_model_specs.md` - 完整模型参数规格
- `docs/WANDB_SETUP.md` - WandB 配置指南
- `scripts/verify_model_params.py` - 参数量验证脚本
