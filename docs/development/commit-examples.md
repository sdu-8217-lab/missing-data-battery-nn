# 提交消息示例库

按类型分类的提交消息范例，供参考使用。

## feat - 新功能

```bash
# 基础功能
feat: 实现MIM多缺失率联合训练

# 带作用域
feat(scheduler): 添加自动重试机制

# 带详细说明
feat(database): 支持新实验数据结构

将实验记录字段从(model,method,mr,seed)改为(seed,model,method)，
支持一个run评估10个缺失率的新范式。

BREAKING CHANGE: 旧版CSV数据库不兼容，需重新初始化

# 关联issue
feat(models): 添加1D-CNN支持

实现参考文献[3]中的1D-CNN架构，
包含3个卷积层（32→64→128通道）和全局平均池化。

Closes #23
```

## fix - Bug修复

```bash
# 简单修复
fix: 修正学习率调度器在epoch=0时的边界错误

# 带问题描述
fix(data): 修正电池分割时的数据泄漏

原始实现按cycle分割后可能导致同一电池的cycle被分配到
训练集和验证集，修正为按battery_id严格分割。

Fixes #15

# 带解决方案对比
fix: 解决CUDA多进程崩溃

问题：
multiprocessing默认使用fork，导致CUDA上下文被继承，
引发RuntimeError: Cannot re-initialize CUDA in forked subprocess。

解决：
使用multiprocessing.set_start_method('spawn', force=True)

参考：
https://pytorch.org/docs/stable/notes/multiprocessing.html
```

## refactor - 代码重构

```bash
# 简单重构
refactor: 删除config_utils冗余代码

# 带影响说明
refactor(data): 重构DataLoader配置

将DataLoader参数提取到配置类中，统一使用：
- pin_memory=True（GPU训练加速）
- persistent_workers=True（减少进程创建开销）
- num_workers=min(12, cpu_count//2)

影响范围：
- src/data/loader_dataloaders.py
- configs/base.yaml

# 架构级重构
refactor: 提取实验调度为独立模块

将原main.py中的实验循环逻辑提取到src/experiments/scheduler.py，
实现职责分离：
- main.py: 配置解析与入口
- scheduler.py: 实验调度与资源管理
- runner.py: 单个实验执行

删除文件：
- src/utils/runner_legacy.py（功能已合并）
```

## docs - 文档更新

```bash
# 简单更新
docs: 补充API使用示例

# 大量文档
docs: 添加完整实验文档体系

新增文档：
- EXPERIMENT_DESIGN.md: 800轮实验设计原理（100×4×2）
- DATASET_INVENTORY.md: 数据集统计与分组规则
- FILE_STRUCTURE_SPEC.md: 项目文件组织规范

更新文档：
- README.md: 添加实验执行说明

# 注释更新
docs(models): 补充MIM模块算法注释

添加参考文献引用和公式说明，
便于后续维护者理解缺失模式模拟原理。
```

## test - 测试相关

```bash
# 添加测试
test: 添加DataLoader单元测试

测试覆盖：
- 正常加载（无缺失）
- 随机缺失模式（MCAR）
- 电池分割正确性

# 修复测试
test(experiments): 修正scheduler并发测试

修复由于test fixture清理不及时导致的端口冲突。
```

## chore - 构建/工具

```bash
# 依赖更新
chore: 更新requirements依赖

升级：
- torch 2.9.0 → 2.10.0
- pytorch-lightning 2.4.0 → 2.5.0

# 配置更新
chore(gitignore): 允许提交实验结果CSV文件

添加例外规则：
- !results/xjtu_2c/*_summary.csv
- !results/xjtu_2c/*_raw_data.csv

# CI配置
chore(ci): 添加GitHub Actions自动测试

触发条件：
- PR到code/main分支
- 每日定时运行（cron: 0 0 * * *）
```

## experiments - 实验结果

```bash
# 基础示例
experiments: XJTU-2C 800轮实验结果

# 完整信息
experiments(xjtu-2c): 添加20260308批次完整结果

实验配置：
- 数据集：XJTU-2C（8 batteries）
- 轮次：800轮（100 seeds × 4 models × 2 methods）
- 执行时长：约14.5小时
- 成功率：99.5%（796/800首次成功，4轮自动重试后100%）

关键发现：
- 1D-CNN-MIM: 86.3%改进（0.0146→0.0020 MAE）
- MLP-MIM: 55.7%改进（0.0183→0.0081 MAE）
- LSTM-MIM: 44.0%改进
- GRU-MIM: 44.0%改进
- 统计显著性：所有结果p<0.001（配对t检验）

交付物：
- results/xjtu_2c/20260308_023704_summary.csv（80行汇总）
- results/xjtu_2c/20260308_023704_raw_data.csv（800行原始）
- paper/figures/20260308_023704/*.png（10张论文图表）

# 补充实验
experiments(nasa): 添加NASA数据集验证

验证MIM方法在不同数据集上的泛化能力，
结果与XJTU趋势一致（CNN改进最大，RNN改进约40%）。
```

## paper - 论文相关

```bash
# 图表更新
paper: 添加消融实验对比图表

新增：
- fig8_boxplot_mr05.png: MR=0.5时的方法对比箱线图
- fig9_avg_improvement.png: 平均改进率柱状图

# 稿件修订
paper(draft): 提交第三版修改稿

修订内容：
- R1意见：补充缺失模式可视化（Fig.3）
- R2意见：修正表格2的RMSE数值
- R3意见：缩短引言至1.5页

# 格式调整
paper(figures): 更新所有图表为高分辨率版本

- 分辨率：300 DPI → 600 DPI
- 格式：PDF矢量图
- 字体：统一为Times New Roman 8pt
```

## 多行完整示例

### 复杂功能提交

```bash
git commit -m "feat(experiments): 实现实验调度框架v2

核心改进：
1. 时间戳隔离
   - 每次实验生成独立目录 experiments/runs/YYYYMMDD_HHMMSS/
   - 自动创建latest软链接指向最新实验
   - 避免历史结果被覆盖

2. MIM多MR合并训练
   - 一个run顺序训练10个缺失率配置
   - 减少进程创建开销约60%
   - 统一模型初始化，确保可比性

3. 并行策略优化
   - GPU worker: 1（执行训练）
   - CPU worker: 3（数据预处理）
   - 自动检测CUDA可用性降级为CPU模式

技术细节：
- 使用multiprocessing.spawn修复CUDA fork问题
- 添加自动重试机制（max_retries=3）
- 实验状态持久化到CSV数据库

Closes #28
Related to #31, #33"
```

### 破坏性变更

```bash
git commit -m "refactor(configs): 重命名配置字段

变更：
batch → batch_size
epochs → max_epochs
lr → learning_rate

原因：
与PyTorch Lightning官方配置命名保持一致，
便于新用户理解和IDE自动补全。

迁移指南：
旧配置文件需手动更新字段名，
或使用提供的迁移脚本：
  python scripts/migrate_configs.py --from v0.3 --to v0.4

BREAKING CHANGE: 旧版配置文件不再兼容，
请参考docs/migration/v0.3-to-v0.4.md升级。"
```

### 回滚提交

```bash
git commit -m "revert: 回滚'feat: 添加动态学习率'

回滚原因：
动态学习率在缺失率>0.5时导致训练不稳定，
验证损失出现NaN。

回滚内容：
- src/training/scheduler.py 中的DynamicLR类
- configs/experiments/dynamic_lr.yaml

后续计划：
待修复数值稳定性问题后重新提交，
见issue #45。"
```

---

## 参考速查

```bash
# 查看提交历史
git log --oneline --graph -20

# 按类型过滤
git log --oneline --grep="^feat"  # 所有feat提交

# 按作者过滤
git log --oneline --author="zhangsan"

# 生成统计
git shortlog -sne  # 提交者统计
git log --pretty=format:"%s" | grep "^feat" | wc -l  # feat数量
```
