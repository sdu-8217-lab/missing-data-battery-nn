# AGENTS.md - AI助手上下文

## 项目核心信息

**项目名称**: battery-soh-missing-data (电池SOH缺失数据预测)

**技术栈**: Python 3.12+, PyTorch 2.0+, PyTorch Lightning

**核心架构**: 9层实验设计（见meta.md）
- L1-L6: 训练变量（seed, dataset, batch, model, use_mim, train MR）
- L7-L9: 测试变量（missing mode, test MR, imputation）

**包结构** (`battery_soh/`):
```
core/      # 类型定义、常量
data/      # XJTU数据加载、电池划分
models/    # MLP/LSTM/CNN + 工厂
missing/   # MCAR/MAR/MNAR生成器 + 插补方法
training/  # Lightning训练器
evaluation/# 评估指标
utils/     # 工具函数
```

**关键约束**:
- 所有模型参数量: 16K-32K (2^14 ~ 2^15)
- MIM vs Imputation: 正交设计，不是对立
- 电池级别划分: 防止数据泄漏

**快速测试**:
```bash
pytest tests/ -v
```
