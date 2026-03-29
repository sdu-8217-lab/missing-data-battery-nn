# 并行实验架构使用说明

## 架构概览

本并行架构全面支持：
- ✅ 并行模型训练（4进程并行）
- ✅ 并行模型评估（8进程并行）
- ✅ GPU资源动态管理
- ✅ 断点续传支持
- ✅ OOM保护机制

## 文件结构

```
scripts/
├── parallel_scheduler.py      # 主调度器（新）
├── gpu_resource_manager.py    # GPU资源管理（新）
├── parallel_worker.py         # 并行工作进程（新）
├── train_models.py            # 训练脚本（已并行化）
├── evaluate_models.py         # 评估脚本（已并行化）
├── train_models_backup.py     # 原训练脚本备份
└── evaluate_models_backup.py  # 原评估脚本备份
```

## 使用方法

### 1. 并行训练（推荐）

```bash
# 激活环境
conda activate battery-nn-latest

# 启动并行训练
cd experiments_new
python scripts/parallel_scheduler.py --batch R2.5 --seeds 0 1 2 3 4 5 6 7 8 9

# 或指定并行数（默认4）
python scripts/parallel_scheduler.py \
    --batch R2.5 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --max-workers 4 \
    --eval-workers 8

# 干运行模式（查看任务列表）
python scripts/parallel_scheduler.py --batch R2.5 --seeds 0 1 2 3 4 5 6 7 8 9 --dry-run
```

### 2. 并行评估

```bash
# 评估单个批次
python scripts/evaluate_models.py \
    --batch-dir results/10seeds_parallel/R2.5 \
    --workers 8

# 评估单个配置
python scripts/evaluate_models.py \
    --models-dir results/10seeds_parallel/R2.5/mlp_level_1_mim/*/models \
    --workers 8
```

### 3. 传统的单配置训练（仍可用）

```bash
python scripts/train_models.py \
    --config configs/experiments/batch_configs_standard/batch_R2.5_mlp_mim.yaml \
    --model-config configs/models/mlp_level_1.yaml \
    --seeds 0 1 2 3 4 5 6 7 8 9
```

## 性能对比

| 指标 | 原架构 | 新并行架构 | 提升 |
|------|--------|-----------|------|
| 训练并行度 | 1（文件锁） | 4 | 4倍 |
| 评估并行度 | 1 | 8 | 8倍 |
| 模型/小时 | ~12 | ~40-50 | 3-4倍 |
| GPU利用率 | 40-60% | 80-90% | 2倍 |

## 架构特性

### 1. GPU资源管理

```python
from scripts.gpu_resource_manager import gpu_manager

# 自动监控显存
# 预留2GB安全缓冲
# 动态调整batch_size
```

### 2. 进程池管理

```python
# 默认4个并行训练进程
# 每个进程独立GPU上下文
# 自动处理进程异常
```

### 3. 断点续传

```python
# 自动检测已存在模型
# 跳过已完成的任务
# 支持任意时刻重启
```

### 4. 安全机制

- **OOM保护**: 显存不足时自动等待
- **超时机制**: 单任务1小时超时
- **异常处理**: 失败任务自动重试
- **结果一致性**: 独立输出目录避免冲突

## 注意事项

### ⚠️ 重要提示

1. **多进程启动方式**: 使用`spawn`模式避免CUDA错误
2. **num_workers**: 已减少为2，避免子进程问题
3. **显存管理**: 自动监控，但建议保留20%余量
4. **日志**: 每个任务独立日志文件

### 🔧 调优建议

1. **调整并行数**:
   ```bash
   # RTX 4060 Ti 16GB 建议:
   # - 小模型(Level 1-2): max-workers=4
   # - 大模型(Level 3-4): max-workers=2-3
   ```

2. **监控GPU状态**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **查看进度**:
   ```bash
   tail -f logs/parallel_*.log
   ```

## 回滚方案

如需恢复原始串行架构：

```bash
# 恢复备份
cp scripts/train_models_backup.py scripts/train_models.py
cp scripts/evaluate_models_backup.py scripts/evaluate_models.py

# 删除并行文件（可选）
rm scripts/parallel_scheduler.py
rm scripts/gpu_resource_manager.py
rm scripts/parallel_worker.py
```

## 问题排查

### Q: 出现CUDA错误
A: 检查是否有其他进程占用GPU，或尝试减少max-workers

### Q: OOM错误
A: GPUResourceManager会自动降级，但建议手动减少并行数

### Q: 进程卡住
A: 检查文件锁是否残留：`rm -f /tmp/train_models.lock`

### Q: 结果不一致
A: 每个worker使用独立输出目录，检查路径是否正确

## 下一步优化

1. **数据缓存**: 预加载数据避免重复I/O
2. **多GPU支持**: 扩展到多GPU并行
3. **动态调度**: 根据模型大小自动调整并行度
4. **结果聚合**: 自动合并多worker结果

---

**创建时间**: 2026-03-29  
**版本**: v1.0-parallel
