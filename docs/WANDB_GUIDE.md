# WandB 成熟配置指南

## 一、为什么要使用 WandB？

### 1. 科研价值

**实验可追溯性**
- 160配置 × 100种子 = 16,000次训练
- CSV文件只能记录最终结果，WandB记录完整过程
- 可回溯每个配置的超参数、随机种子、训练曲线

**论文可复现性**
- 论文提交时提供 WandB 链接 = 可验证的实验记录
- 审稿人可追溯每个图表的来源
- 满足 Nature/Science 等顶刊的数据可用性要求

### 2. 工程价值

**超参数关联分析**
- 可视化：缺失率 vs MAE 曲线
- 自动对比：MIM vs Mean 在不同模型上的表现
- 并行坐标图：探索最佳参数组合

**团队协作**
- 导师/合作者可远程查看实验进展
- 避免"在我机器上能跑"的问题
- 统一实验记录标准

**异常监控**
- 训练发散时自动告警
- 对比历史最佳模型
- 早期停止决策支持

---

## 二、长期主义方案

### 架构：在线模式 + 团队项目 + 自动化

#### Step 1: 注册与配置

```bash
# 1. 注册账号 (使用学校/机构邮箱)
# https://wandb.ai/site

# 2. 登录 (只需一次)
wandb login

# 3. 创建团队项目
# 在 Web UI 中创建团队 "sdu-8217-lab"
# 邀请合作者/导师加入
```

#### Step 2: 环境隔离配置

创建 `.env` 文件（不提交到git）：

```bash
# .env
WANDB_ENTITY=sdu-8217-lab
WANDB_PROJECT=battery-soh-missing-data
WANDB_API_KEY=your_key_here
```

修改 `.gitignore`：
```
.env
wandb/
```

#### Step 3: 代码集成

修改 `configs/config.yaml`：

```yaml
wandb:
  enabled: true
  entity: ${oc.env:WANDB_ENTITY,sdu-8217-lab}
  project: ${oc.env:WANDB_PROJECT,battery-soh}
  name: ${experiment.name}
  tags:
    - ${data.dataset}
    - ${model.type}
    - ${method}
  notes: "MIM vs traditional imputation methods"
```

#### Step 4: 批量实验优化

修改 `run_full_experiment.py`，添加实验组：

```python
import os
import wandb

# 每个配置作为一个 WandB Run
for config in experiments:
    run_name = f"{config['dataset']}_{config['model']}_{config['method']}"
    
    os.environ["WANDB_RUN_GROUP"] = f"batch_{timestamp}"
    os.environ["WANDB_RUN_TAGS"] = f"{config['dataset']},{config['model']},{config['method']}"
    
    # 运行实验
    subprocess.run([...])
```

---

## 三、团队工作流

### 目录结构

```
battery-soh-missing-data/
├── configs/
│   └── config.yaml          # WandB基础配置
├── src/
│   └── main.py
├── results/                 # CSV结果（本地备份）
├── wandb/                   # WandB本地缓存（gitignore）
├── .env                     # 环境变量（gitignore）
└── README.md                # 包含WandB项目链接
```

### 实验命名规范

```
{x_dataset}_{x_model}_{x_method}_s{x_seed}

示例:
- xjtu_mlp_mim_s42
- hust_lstm_mean_s100
- tju_cnn1d_knn_s88
```

### WandB 项目面板设置

创建自定义面板：

1. **Overview Panel**
   - MAE vs Missing Rate (折线图)
   - RMSE vs Missing Rate (折线图)
   - 不同模型的对比柱状图

2. **Filter Panel**
   - 按 dataset 筛选
   - 按 method 筛选
   - 按 model 筛选

3. **Group by Panel**
   - 按 batch_id 分组
   - 自动计算组内均值和标准差

---

## 四、论文集成

### 图表引用

```latex
% 在论文中引用WandB实验
All experiments are tracked and visualized on Weights \& Biases 
\href{https://wandb.ai/sdu-8217-lab/battery-soh}{https://wandb.ai/sdu-8217-lab/battery-soh}.
```

### 可复现性声明

```
Code and experimental data are available at:
- GitHub: https://github.com/sdu-8217-lab/missing-data-battery-nn
- WandB: https://wandb.ai/sdu-8217-lab/battery-soh
```

---

## 五、与 CSV 的对比

| 特性 | CSV | WandB |
|------|-----|-------|
| 记录粒度 | 最终结果 | 完整训练过程 |
| 可视化 | 手动绘图 | 自动交互式图表 |
| 对比分析 | 需写脚本 | 内置并行坐标图 |
| 团队协作 | 文件传输 | 在线共享 |
| 论文引用 | 静态数据 | 可验证链接 |
| 长期保存 | 本地风险 | 云端备份 |

---

## 六、针对当前项目的建议

### 短期（本周）
- 启用 WandB 在线模式
- 完成10种子测试，验证数据上传

### 中期（本月）
- 配置团队项目
- 邀请导师加入
- 设置自动化面板

### 长期（论文阶段）
- 所有实验结果上传 WandB
- 生成可交互的论文补充材料
- 提供公开可访问的实验记录

---

*参考：
- WandB for Research: https://docs.wandb.ai/guides/track/research
- ML Experiment Tracking Best Practices*
