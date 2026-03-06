# 项目完全指南 - 8217接手版

> **目标**: 让8217通过阅读本文档,达到与当前OpenClaw相同的项目理解深度
> 
> **阅读时间**: 约30分钟
> 
> **必读**: ⭐ | **参考**: 📖 | **进阶**: 🔬

---

## 📚 文档地图

### 核心必读 (先读这些)

| 文档 | 用途 | 阅读顺序 |
|------|------|---------|
| **本文档** | 项目总览和入口 | 1️⃣ |
| [00-README-8217.md](./00-README-8217.md) | 快速上手和首次运行 | 2️⃣ |
| [01-项目核心概念.md](./01-项目核心概念.md) | 科研背景和技术原理 | 3️⃣ |
| [02-代码架构详解.md](./02-代码架构详解.md) | 代码结构和关键实现 | 4️⃣ |

### 实验执行

| 文档 | 用途 | 阅读时机 |
|------|------|---------|
| [03-实验操作手册.md](./03-实验操作手册.md) | 详细实验步骤 | 准备跑实验时 |
| [04-常见问题排查.md](./04-常见问题排查.md) | 故障解决 | 遇到问题时 |
| [GPU_SETUP_8217.md](../GPU_SETUP_8217.md) | 4060Ti专用配置 | 在8217上配置环境 |

### 科研产出

| 文档 | 用途 | 阅读时机 |
|------|------|---------|
| [09-论文大纲.md](./09-论文大纲.md) | 论文结构 | 开始写论文时 |
| [12-绘图功能指南.md](./12-绘图功能指南.md) | 结果可视化 | 生成图表时 |

### 历史参考 (了解背景)

| 文档 | 用途 | 说明 |
|------|------|------|
| [16-新架构经验教训.md](./16-新架构经验教训.md) | 避免踩坑 | 已发生的问题和解决 |
| [11-文档一致性检查.md](./11-文档一致性检查.md) | 参数规范 | 模型配置标准 |

---

## 🎯 项目速览

### 核心问题
**如何在电池数据存在缺失的情况下,准确预测电池健康状态(SOH)?**

### 核心创新
| 传统方法 | 我们的方法(MIM) |
|---------|----------------|
| 先插补缺失值,再预测 | 直接告诉网络哪些数据缺失 |
| 16维输入(仅特征) | 32维输入(特征+缺失指示器) |
| 单缺失率训练 | 混合缺失率(0.0-0.9)训练 |

### 实验规模
```
4数据集 × 2缺失机制 × 5方法 × 4模型 × 100种子 × 9缺失率
= 144,000 次评估
= 160 个实验配置
```

### 关键成果
- **MIM vs 最佳插补方法**: MAE降低 78-85%
- **高缺失率(MR≥0.7)**: MIM优势更明显
- **论文目标**: SCI期刊 (目标: Journal of Energy Storage 或类似)

---

## 🏗️ 架构理解

### 技术栈
```
Hydra (配置管理)
    ↓
PyTorch Lightning (训练框架)
    ↓
WandB (实验跟踪)
    ↓
GitHub (版本控制)
```

### 代码结构
```
missing-data-battery-nn/
├── configs/              # Hydra配置
│   ├── config.yaml      # 主配置 ⭐
│   ├── data/            # 4数据集
│   ├── model/           # 4模型
│   └── missing/         # 2缺失机制
├── src/
│   ├── main.py          # 唯一入口 ⭐
│   ├── data/            # 数据加载+划分
│   ├── models/          # 神经网络定义
│   ├── missing_data/    # 缺失模拟+插补
│   └── utils/           # 工具函数
├── results/             # 实验结果(CSV)
├── docs/                # 本文档目录
└── run_full_experiment.py  # 批量实验脚本
```

### 关键文件 (必须理解)

| 文件 | 作用 | 关键理解点 |
|------|------|-----------|
| `configs/config.yaml` | 实验配置 | 控制所有参数 |
| `src/main.py` | 主入口 | 实验流程 |
| `src/data/loader_dataloaders.py` | 数据加载 | MIM训练策略 ⭐⭐⭐ |
| `run_full_experiment.py` | 批量运行 | 160配置自动化 |

---

## 🔬 科研核心概念

### 1. MIM (Missing Indicator Method)
**核心思想**: 不掩盖缺失,而是明确告诉网络"哪些数据缺失"

```python
# 传统方法 (16维)
输入 = [voltage_mean, voltage_std, ..., current_entropy]  # 16个特征

# MIM方法 (32维)
输入 = [特征值(16维)] + [缺失指示器(16维)]
      # 例如: [3.7, 0.1, ..., 缺失值设为0] + [0, 0, ..., 1(表示缺失)]
```

### 2. 混合缺失率训练
**为什么?** 让网络学会在各种缺失率下都能预测

```python
# 训练时: 10个缺失率版本
for mr in [0.0, 0.1, 0.2, ..., 0.9]:
    X_missing = 施加缺失率(mr, X_train)
    X_mim = 拼接(X_missing, 缺失指示器)
    训练网络(X_mim, y_train)

# 测试时: 指定缺失率
X_test_mim = 施加缺失率(0.5, X_test)
预测 = 网络(X_test_mim)
```

### 3. 按电池划分
**为什么?** 避免数据泄漏,模拟真实场景

```
❌ 错误: 随机划分样本 (同电池的循环分散在train/test中)
✅ 正确: 按电池划分 (整个电池只属于一个集合)

Train: [电池3,4,5,7]
Val:   [电池1,8]
Test:  [电池2,6]
```

---

## ⚡ 快速开始 (8217首次运行)

### 1. 克隆代码
```bash
cd ~/
git clone git@github.com:sdu-8217-lab/missing-data-battery-nn.git
cd missing-data-battery-nn
git checkout dev
```

### 2. 配置环境
```bash
# 安装conda环境
conda create -n battery python=3.12 -y
conda activate battery

# 安装PyTorch GPU版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install pytorch-lightning hydra-core omegaconf wandb pandas numpy scikit-learn matplotlib

# 验证GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### 3. 配置WandB
```bash
wandb login
# 输入API key
```

### 4. 运行测试
```bash
# 单配置测试 (5分钟验证)
python src/main.py data=xjtu model=mlp method=mim \
    experiment.seeds=[42] training.epochs=10

# 如果正常输出MAE指标 → 环境配置成功
```

### 5. 启动全量实验
```bash
# 160配置 × 100种子 × 100 epochs
python run_full_experiment.py

# 预计时间: 80-160小时 (CPU) 或 7-10天 (GPU 4060Ti)
```

---

## 🤝 协作流程 (主电脑 ←→ 8217)

### 分工
| 任务 | 负责方 | 说明 |
|------|--------|------|
| 代码开发 | 主电脑 | 调试、优化、文档更新 |
| 大规模实验 | 8217 | GPU加速训练 |
| 结果分析 | 双方 | WandB共享 |
| 论文撰写 | 主电脑 | 初稿在主力机 |

### 同步方式
```bash
# 1. 主电脑开发完成 → 推送
git add .
git commit -m "描述修改"
git push origin dev

# 2. 8217拉取 → 运行
git pull origin dev
python run_full_experiment.py

# 3. 结果自动同步到WandB
# 双方查看: https://wandb.ai/chenqingyang-shandong-university/battery-soh-missing-data
```

---

## 📊 监控实验

### WandB (在线)
- URL: https://wandb.ai/chenqingyang-shandong-university/battery-soh-missing-data
- 实时查看训练曲线、MAE指标

### 本地监控
```bash
# 查看日志
tail -f results/full_experiment*.log

# 查看GPU利用率 (8217)
watch -n 1 nvidia-smi

# 查看结果文件
ls -lh results/*.csv
```

---

## ⚠️ 关键注意事项

### 1. MIM训练策略 (最容易出错!)
```python
# ❌ 错误: 单一缺失率训练
X_train_missing = 施加缺失率(0.5, X_train)

# ✅ 正确: 混合缺失率训练
for mr in [0.0, 0.1, ..., 0.9]:
    X_mr = 施加缺失率(mr, X_train)
    合并训练(X_mr)
```

### 2. 输入维度
| 方法 | 输入维度 | 说明 |
|------|---------|------|
| MIM | 32 | 16特征 + 16指示器 |
| Mean/Median/KNN/Zero | 16 | 仅插补后特征 |

### 3. 按电池划分
- 绝对不能随机划分样本
- 必须整个电池只属于train/val/test之一

---

## 🎓 成为专家的阅读路径

### 第1天 (理解项目)
1. 读完本文档
2. 阅读 [01-项目核心概念.md](./01-项目核心概念.md)
3. 运行单配置测试,观察输出

### 第2-3天 (理解代码)
1. 阅读 [02-代码架构详解.md](./02-代码架构详解.md)
2. 精读 `src/data/loader_dataloaders.py` (MIM核心)
3. 修改配置,观察不同参数的效果

### 第4-7天 (实验执行)
1. 阅读 [03-实验操作手册.md](./03-实验操作手册.md)
2. 启动全量实验
3. 监控进度,排查问题

### 第2-4周 (科研产出)
1. 阅读 [09-论文大纲.md](./09-论文大纲.md)
2. 生成图表
3. 撰写论文

---

## 📞 遇到问题?

1. **先查文档**: [04-常见问题排查.md](./04-常见问题排查.md)
2. **再查历史**: [16-新架构经验教训.md](./16-新架构经验教训.md)
3. **最后问**: 在主电脑的OpenClaw对话中提问

---

## ✅ 接手检查清单

完成以下检查,表示你已成功接手项目:

- [ ] 能独立运行单配置测试
- [ ] 理解MIM vs 插补方法的区别
- [ ] 能解释`loader_dataloaders.py`中的MIM训练策略
- [ ] 能查看WandB实验结果
- [ ] 能排查常见错误

---

**完成这些,你就是项目的主人了! 🎉**
