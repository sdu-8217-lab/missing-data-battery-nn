# 大实验（Big Experiment）完整配置

## 实验设计概览

根据论文设计要求，大实验是完整的MIM（Missing Indicator Mechanism）研究：

### 实验矩阵
| 因子 | 水平数 | 说明 |
|------|--------|------|
| 模型 | 4 | MLP, LSTM, GRU, CNN1D（XGBoost已移除） |
| 策略 | 2 | baseline（仅插补）, mim（插补+缺失指示器） |
| 缺失率 | 9 | 0.1, 0.2, ..., 0.9 |
| 重复次数 | 100 | 每次不同随机种子 |

**总计**: 4 × 2 × 9 × 100 = **7200次实验**

---

## 模型配置（来自configs_v3最佳参数）

| 模型 | 架构 | Dropout | Baseline参数量 | MIM参数量 |
|------|------|---------|----------------|-----------|
| **MLP** | [192,96,48,24] 4层 | 0.15 | 27,649 | 27,713 |
| **LSTM** | h=48, l=2 | 0.20 | 31,537 | 33,025 |
| **GRU** | h=64, l=2 | 0.20 | 40,769 | 42,241 |
| **CNN1D** | [72,32] 2层, kernel=4 | 0.10 | 16,713 | 17,577 |

---

## 文件结构

```
configs_big/                    # 实验配置文件
├── mlp_big_experiment.csv      # 1800行
├── lstm_big_experiment.csv     # 1800行
├── gru_big_experiment.csv      # 1800行
└── cnn1d_big_experiment.csv    # 1800行

run_big_experiment.py           # 主运行脚本
analyze_big_experiment.py       # 结果分析脚本

运行脚本:
├── run_big_mlp.bat
├── run_big_lstm.bat
├── run_big_gru.bat
└── run_big_cnn1d.bat

输出目录:
experiments_big/                # 结果输出
├── fig2_mae_vs_missing_rate.png
├── fig5_improvement_heatmap.png
├── fig6_high_missing_comparison.png
├── tab3_main_results.csv
└── tab4_improvement_significance.csv
```

---

## 使用方法

### 1. 单独运行某个模型（推荐，可并行4个窗口）

```bash
# 窗口1: MLP (1800次实验)
run_big_mlp.bat

# 窗口2: LSTM (1800次实验)  
run_big_lstm.bat

# 窗口3: GRU (1800次实验)
run_big_gru.bat

# 窗口4: CNN1D (1800次实验)
run_big_cnn1d.bat
```

### 2. 从指定重复位置开始（断点续传）

```bash
# 从第50次重复开始运行
run_big_cnn1d.bat 50
```

### 3. Python直接运行

```bash
# 运行单个模型
python run_big_experiment.py --model cnn1d --device cpu --start 0

# 限制运行数量（测试用）
python run_big_experiment.py --model cnn1d --limit 10
```

---

## 预计运行时间

基于configs_v3的观察（CPU环境）：
- CNN1D: ~15-20秒/次 → 1800次 ≈ **7-10小时**
- GRU: ~10-15秒/次 → 1800次 ≈ **5-7.5小时**
- LSTM: ~10-15秒/次 → 1800次 ≈ **5-7.5小时**
- MLP: ~8-12秒/次 → 1800次 ≈ **4-6小时**

**4模型并行总计**: 约 **10-12小时**

---

## 实验流程说明

### 阶段1: 训练阶段
对于每个实验配置：
1. 设置唯一随机种子（seed = 10000 + exp_counter）
2. 加载3C批次数据
3. 根据策略准备输入：
   - **Baseline**: 输入维度16（仅插补值，缺失处填0）
   - **MIM**: 输入维度32（插补值 + 缺失指示器）
4. 训练200 epochs
5. 保存模型（可选）

### 阶段2: 评估阶段
1. 在测试集上应用指定缺失率（MCAR）
2. 按策略格式准备测试数据
3. 模型推理
4. 计算MAE/RMSE/R²
5. 记录结果到CSV

---

## 结果分析

实验完成后运行分析脚本：

```bash
python analyze_big_experiment.py
```

### 输出内容

1. **fig2_mae_vs_missing_rate.png**: MAE随缺失率变化曲线（10条线：4模型×2策略）
2. **fig5_improvement_heatmap.png**: MIM改进百分比热力图
3. **fig6_high_missing_comparison.png**: 高缺失率(0.6-0.9)详细对比
4. **tab3_main_results.csv**: 各缺失率下MAE/RMSE/R²汇总
5. **tab4_improvement_significance.csv**: MIM相对改进百分比

---

## 关键研究问题验证

### RQ1: MIM相对Baseline的实际收益
- 查看fig5热力图：绿色表示MIM优于Baseline
- 查看tab4：各缺失率下的改进百分比

### RQ2: 不同模型架构从MIM获益是否一致
- 比较4个模型的fig2曲线
- 查看各模型在tab4中的平均改进百分比

### RQ3: 缺失率0.1→0.9时哪种组合最鲁棒
- 查看fig6高缺失率对比
- 观察fig2中各模型曲线的斜率（斜率小=更鲁棒）

---

## 注意事项

1. **数据准备**: 实验使用3C批次，确保 `data/XJTU data/` 中有3C数据
2. **存储空间**: 7200次实验结果约需100-200MB存储
3. **内存**: 每次实验独立运行，内存占用约500MB-1GB
4. **随机种子**: 每次实验使用唯一种子，确保100次重复的可复现性

---

## 实验状态检查

运行过程中可以检查进度：

```python
import pandas as pd

# 检查CNN1D进度
df = pd.read_csv('configs_big/cnn1d_big_experiment.csv')
completed = len(df[df['status'] == 'completed'])
pending = len(df[df['status'] == 'pending'])
print(f"CNN1D: {completed}/1800 completed, {pending} pending")
```
