# Battery SOH Prediction with Missing Data

基于 PyTorch Lightning + Hydra 的电池 SOH 预测框架，支持缺失数据机制研究。

## 技术栈

| 组件 | 用途 |
|------|------|
| **Hydra** | 配置管理 |
| **PyTorch Lightning** | 训练框架 |
| **Weights & Biases** | 实验追踪 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行实验 (XJTU + MLP + MCAR)
python src/main.py data=xjtu model=mlp missing=mcar

# 运行 MAR 实验
python src/main.py data=xjtu model=mlp missing=mar

# 使用 MIM
python src/main.py data=xjtu model=mlp missing=mcar model.use_mim=true

# 更换模型
python src/main.py model=lstm
python src/main.py model=gru
python src/main.py model=cnn1d

# 更换数据集
python src/main.py data=tju
python src/main.py data=hust
python src/main.py data=mit

# 覆盖参数
python src/main.py training.epochs=200 training.batch_size=32
```

## 项目结构

```
.
├── configs/              # Hydra 配置
│   ├── config.yaml       # 主配置
│   ├── data/             # 数据集配置 (xjtu/tju/hust/mit)
│   ├── model/            # 模型配置 (mlp/lstm/gru/cnn1d)
│   └── missing/          # 缺失机制 (mcar/mar)
├── src/
│   ├── main.py           # 主入口
│   ├── models/           # 神经网络模型
│   ├── data/             # 数据加载
│   ├── missing_data/     # 缺失数据模拟
│   └── utils/            # 工具函数
└── data/                 # 数据集
    ├── XJTU data/
    ├── TJU data/
    ├── HUST data/
    └── MIT data/
```

## 模型

| 模型 | 输入维度 | 参数量 (~) |
|------|----------|-----------|
| MLP | 16/32 | 10K |
| LSTM | 16/32 | 10K |
| GRU | 16/32 | 9K |
| CNN1D | 16/32 | 10K |

## 缺失机制

- **MCAR**: 完全随机缺失
- **MAR**: 依赖 SOH 值的缺失
- **MIM**: 缺失指示器方法（输入维度翻倍）

## 数据集

支持四个大学数据集：
- **XJTU**: 西安交通大学
- **TJU**: 天津大学  
- **HUST**: 华中科技大学
- **MIT**: 麻省理工大学

## 实验配置

实验配置通过 Hydra 组合：

```yaml
# 例如：configs/experiment/mim_mar_0.3.yaml
defaults:
  - /data: xjtu
  - /model: mlp
  - /missing: mar
  - _self_

experiment:
  name: "mim_mar_0.3"

missing:
  missing_rate_train: 0.3
  
model:
  use_mim: true
```

运行：`python src/main.py experiment=mim_mar_0.3`
