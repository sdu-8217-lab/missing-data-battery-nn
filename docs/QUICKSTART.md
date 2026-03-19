# 快速入门指南

5分钟上手电池SOH预测实验框架。

---

## 第一步：环境配置（1分钟）

```bash
# 激活环境
conda activate battery-nn

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## 第二步：快速测试（2分钟）

运行单次实验验证环境：

```bash
# 训练一个简单模型
python experiments/run_experiment.py \
    --phase train \
    --seed 42 \
    --batch 2C \
    --model mlp \
    --use-mim true \
    --epochs 10
```

预期输出：
```
Epoch 10/10: 100%|██████████| 10/10 [00:15<00:00, 1.5s/it]
✓ Model saved to models/...
```

---

## 第三步：批量实验（可选）

### 小规模测试（推荐先跑这个）

```bash
# 2种子 × 2批次 × 3模型 × 2 MIM = 24个训练
python experiments/run_batch_experiments.py \
    --phase full \
    --seeds 42 43 \
    --batches 2C 3C \
    --epochs 50
```

预计时间：~1-2小时

### 完整实验（100种子）

```bash
# 后台运行
nohup python experiments/run_batch_experiments.py \
    --phase full \
    --seeds $(seq 0 99) \
    --epochs 50 \
    > exp.log 2>&1 &

# 查看进度
tail -f exp.log
```

预计时间：10-15小时

---

## 第四步：查看结果

```bash
# 查看单次结果
cat results/seed42_batch2C_modelmlp_mimtrue_modeMCAR_mr0.3_impmean.json

# 查看聚合结果
head results/aggregated_results.csv
```

---

## 常用命令速查

| 任务 | 命令 |
|------|------|
| 单次训练 | `python experiments/run_experiment.py --phase train --seed 42 --batch 2C --model mlp --use-mim true --epochs 50` |
| 单次测试 | `python experiments/run_experiment.py --phase test --seed 42 --batch 2C --model mlp --use-mim true --mode MCAR --test-mr 0.3 --imputation mean` |
| 批量训练 | `python experiments/run_batch_experiments.py --phase train --epochs 50` |
| 批量测试 | `python experiments/run_batch_experiments.py --phase test` |
| 监控进度 | `tail -f logs/experiments/*.log` |

---

## 下一步

- 深入了解：[meta.md](../meta.md) - 9层架构定义
- 完整指南：[EXPERIMENTS.md](EXPERIMENTS.md)
- 代码架构：[ARCHITECTURE.md](ARCHITECTURE.md)
