# 快速开始

> 5分钟运行你的首个实验

---

## 前提条件

- 已完成 [环境配置](ENVIRONMENT_SETUP.md)
- 已激活环境：`conda activate battery-nn`

---

## 1分钟验证

```bash
# 验证环境
python --version  # Python 3.13+
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## 5分钟运行实验

### 单配置测试

```bash
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42] training.epochs=5
```

**预期输出**：
```
Loaded 8 batteries from data/XJTU data
...
MR=0.1: MAE=0.0288
MR=0.5: MAE=0.0232
MR=0.9: MAE=0.0295
✓ Results saved to: results/battery_soh_experiment_mlp_mim.csv
```

### 查看结果

```bash
cat results/battery_soh_experiment_mlp_mim.csv
```

---

## 常用命令

### 切换模型

```bash
python src/main.py model=lstm    # LSTM
python src/main.py model=gru     # GRU
python src/main.py model=cnn1d   # 1D-CNN
```

### 切换方法

```bash
python src/main.py method=mean   # 均值插补
python src/main.py method=mim    # MIM (推荐)
```

### 切换数据集

```bash
python src/main.py data=xjtu     # 西安交通大学
python src/main.py data=tju      # 天津大学
```

---

## 目录结构

```
.
├── configs/           # 配置文件
├── data/              # 数据集
├── docs/              # 文档
├── results/           # 实验结果
├── src/               # 源代码
│   ├── main.py       # 主入口
│   ├── models/       # 模型定义
│   ├── data/         # 数据加载
│   └── missing_data/ # 缺失处理
└── environment.yml   # 环境配置
```

---

## 下一步

- [完整项目指南](PROJECT_GUIDE.md) - 了解项目架构和设计
- [实验操作指南](EXPERIMENT_GUIDE.md) - 运行完整实验
- [代码架构指南](ARCHITECTURE.md) - 深入了解代码结构

---

*最后更新: 2026-03-07*
