# AGENTS.md - 给下一个智能体的交接文档

## 项目状态

**分支**: `refactor/long-term`  
**状态**: 已完成 A1/A2 实验框架，待运行完整实验

## 核心任务

运行 **A1/A2 实验** 验证两个关键问题：

### A1: MIM 增益来源
- **G1**: z = [x̂ | m] - 标准MIM
- **G2**: z = [x̂ | r], r~Bernoulli(0.4) - 随机指示器（测试正则化假说）
- **G3**: z = [x̂ | m_shuffled] - 打乱指示器（测试信息假说）
- **G4**: z = [x̂ | x̂] - 复制特征（测试维度假说）

### A2: 块状缺失鲁棒性
- **B1**: MCAR - 随机均匀缺失
- **B2**: Block(5) - 连续5-cycle缺失（模拟传感器故障）

## 如何运行实验

### 快速测试（5 epochs）
```bash
# 单组测试
python run_A1_A2_experiments.py --group G1 --seed 42 --epochs 5

# 所有组（演示用）
python run_A1_A2_experiments.py --run-all --seeds 42 --epochs 5
```

### 完整实验（200 epochs，3 seeds）
```bash
# 推荐在服务器后台运行，约需 18 小时
nohup python run_A1_A2_experiments.py \
    --run-all \
    --seeds 42 123 456 \
    --epochs 200 \
    > a1_a2_experiments.log 2>&1 &
```

### 分析结果
```bash
# 生成报告
python analyze_A1_A2.py --report

# 查看结果
cat results/A1_A2/A1_A2_results.csv
cat results/A1_A2/A1_A2_findings_report.md
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `run_A1_A2_experiments.py` | 实验主脚本 |
| `analyze_A1_A2.py` | 结果分析脚本 |
| `battery_soh/evaluation/evaluator.py` | Evaluator 类，支持 mim_variant 参数 |
| `battery_soh/missing/generators.py` | BlockMissingGenerator 实现 |
| `results/A1_A2/` | 实验结果输出目录 |

## 代码结构

```
battery_soh/
├── core/           # 类型定义、常量
├── data/           # XJTU数据加载
├── models/         # MLP/LSTM/CNN + 工厂
├── missing/        # MCAR/MAR/MNAR/Block生成器
├── training/       # Lightning训练器
├── evaluation/     # Evaluator（支持A1/A2）
└── experiments/    # 实验运行器
```

## 重要实现细节

### 1. MIM 变体实现位置
`battery_soh/evaluation/evaluator.py` - `_construct_mim_input()` 方法：
- `standard`: [x̂ | m] - 真实缺失指示
- `random`: [x̂ | r] - 随机 Bernoulli(0.4)
- `shuffled`: [x̂ | m_shuffled] - 打乱后的指示
- `copy`: [x̂ | x̂] - 复制插补值

### 2. 块状缺失实现
`battery_soh/missing/generators.py` - `BlockMissingGenerator`：
- 连续 `block_size` 个样本同时缺失
- 默认 block_size=5，模拟5-cycle传感器故障

### 3. 训练数据维度
A1/A2 实验训练时已添加零 mask（因为训练数据完整）：
```python
train_X = [X | zeros]  # 32维
```

## 注意事项

1. **数据路径**: 确保 `data/XJTU data/` 存在（94MB，已加入 .gitignore）

2. **GPU**: 代码支持 GPU 自动检测，但 CPU 也能运行

3. **结果文件**: 
   - CSV: `results/A1_A2/A1_A2_results.csv`
   - 报告: `results/A1_A2/A1_A2_findings_report.md`

4. **断点续跑**: 实验脚本支持追加写入 CSV，可以中断后重新运行

5. **维度检查**: 如果出现 `mat1 and mat2 shapes cannot be multiplied` 错误，检查：
   - 模型是否 `use_mim=True`（需要32维输入）
   - 训练/测试数据是否正确添加 mask

## 预期结论

| 如果... | 结论 |
|---------|------|
| G2 ≈ G1 | 正则化假说：MIM增益来自维度扩展 |
| G2 < G1 | 信息假说：真实缺失指示有价值 |
| G4 ≈ G1 | 维度假说：单纯翻倍即足够 |
| B2 ≈ B1 | MIM对传感器故障鲁棒 |
| B2 < B1 | 论文贡献限于随机缺失场景 |

## 演示结果（1 seed, 5 epochs）

```
G1 (标准MIM):     MAE = 0.1653
G2 (随机指示器):  MAE = 0.1652  (差 -0.0%)
```

**启示**: G2 ≈ G1，正则化假说可能成立！

---

**最后更新**: 2026-04-10  
**作者**: chen  
**提交**: c42713c
