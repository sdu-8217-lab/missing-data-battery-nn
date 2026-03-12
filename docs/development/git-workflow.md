# Git 工作流详解

本文档详细说明本项目的Git分支策略、提交规范和协作流程。

## 目录

- [分支模型](#分支模型)
- [提交规范](#提交规范)
- [工作流示例](#工作流示例)
- [常见场景](#常见场景)
- [工具配置](#工具配置)

---

## 分支模型

采用 **GitHub Flow + 实验分支** 的混合模型：

```
main (保护分支)
  ↑
code (保护分支，主开发线)
  ↑
feature/experiment-name (功能分支)
```

### 分支说明

| 分支名 | 类型 | 生命周期 | 命名规范 |
|:---|:---|:---|:---|
| `main` | 长期 | 永久 | 固定 |
| `code` | 长期 | 永久 | 固定 |
| `feature/*` | 短期 | 合并后删除 | `feature/功能简述`，如 `feature/mim-batch-training` |
| `fix/*` | 短期 | 合并后删除 | `fix/问题简述`，如 `fix/cuda-memory-leak` |
| `experiments/*` | 短期 | 可选保留 | `experiments/数据集-时间戳`，如 `experiments/xjtu-2c-20260308` |
| `paper/*` | 短期 | 合并后删除 | `paper/修改内容`，如 `paper/revision-section3` |

---

## 提交规范

### 格式

```
<type>[optional scope]: <简短描述（中文）>
│      │                │
│      │                └─⫸ 描述：做了什么（祈使句，中文）
│      │
│      └─⫸ 作用域：可选，标识改动模块（见下表）
│
└─⫸ 类型：feat|fix|refactor|docs|test|chore|experiments|paper
```

### 类型详解

#### `feat` - 新功能

用于新增功能或特性。

```bash
# ✅ 基础示例
feat: 实现MIM多缺失率联合训练

# ✅ 带作用域
feat(scheduler): 添加自动重试机制

# ✅ 带body说明
feat(database): 支持新实验数据结构

将实验记录字段从(model,method,mr,seed)改为(seed,model,method)，
支持一个run评估10个缺失率的新范式。

BREAKING CHANGE: 旧版CSV数据库不兼容，需重新初始化
```

#### `fix` - Bug修复

用于修复代码缺陷。

```bash
# ✅ 关联issue
fix: 修正CUDA多进程崩溃

使用multiprocessing.spawn替代fork，避免CUDA上下文复制问题。

Fixes #42

# ✅ 带作用域
fix(data): 修正batch size不匹配导致的OOM
```

#### `refactor` - 代码重构

用于代码结构调整，不新增功能也不修复bug。

```bash
# ✅ 说明重构原因
refactor: 删除config_utils冗余代码

内联save_config_snapshot函数，该功能已被Hydra原生替代。
删除后减少代码行数46行，降低维护负担。
```

#### `docs` - 文档更新

用于文档、注释、README等的更新。

```bash
# ✅ 简单更新
docs: 补充API使用示例

# ✅ 大量文档
docs: 添加完整实验文档体系

- EXPERIMENT_DESIGN.md: 800轮实验设计原理
- DATASET_INVENTORY.md: 数据集统计与分组规则
- FILE_STRUCTURE_SPEC.md: 项目文件组织规范
```

#### `test` - 测试相关

用于添加或修改测试代码。

```bash
test: 添加DataLoader单元测试

test(experiments): 补充scheduler并发测试用例
```

#### `chore` - 构建/工具

用于构建流程、依赖更新、配置修改等。

```bash
chore: 更新requirements依赖
chore(gitignore): 允许提交实验结果CSV文件
chore(ci): 添加GitHub Actions自动测试
```

#### `experiments` - 实验结果

**本项目专用类型**，用于提交实验执行结果。

```bash
# ✅ 基础示例
experiments: XJTU-2C 800轮实验结果

# ✅ 带作用域和时间戳
experiments(xjtu-2c): 添加20260308批次完整结果

- 完成800轮实验（100 seeds × 4 models × 2 methods）
- 关键发现：1D-CNN-MIM改进86.3%，p<0.001
- 交付物：summary.csv, raw_data.csv, 10张论文图表
```

#### `paper` - 论文相关

**本项目专用类型**，用于论文稿件、图表、修订。

```bash
paper: 添加消融实验对比图表

paper(draft): 提交第三版修改稿

包含审稿人意见回复：
- R1: 补充了缺失模式可视化
- R2: 修正了表格2的数值
paper(figures): 更新所有图表为高分辨率版本
```

### 作用域（Scope）对照表

| Scope | 对应路径 | 使用示例 |
|:---|:---|:---|
| `data` | `src/data/` | `feat(data): 添加电池数据缓存` |
| `models` | `src/models/` | `fix(models): 修正LSTM隐藏层维度` |
| `experiments` | `src/experiments/` | `refactor(experiments): 重写调度器` |
| `configs` | `configs/` | `chore(configs): 更新默认学习率` |
| `scripts` | `scripts/` | `feat(scripts): 添加批量运行工具` |
| `docs` | `docs/` | `docs(docs): 更新开发指南` |
| `paper` | `paper/` | `paper(paper): 修订方法论章节` |
| `root` | 根目录 | `chore(root): 更新.gitignore` |

---

## 工作流示例

### 场景1：开发新功能

```bash
# 1. 同步上游
git checkout code
git pull origin code

# 2. 创建功能分支
git checkout -b feature/mim-batch-training

# 3. 开发...
# 编辑代码

# 4. 提交（遵循规范）
git add src/experiments/scheduler.py
git commit -m "feat(scheduler): 实现MIM批量训练调度

支持在一个worker中顺序训练多个缺失率配置，
减少GPU显存碎片化和进程切换开销。"

# 5. 继续开发...
git add src/experiments/database.py
git commit -m "feat(database): 支持多MR评估结果存储

新增eval_missing_rates字段，存储10个评估MR的结果。"

# 6. 推送并创建PR
git push origin feature/mim-batch-training
# 在GitHub创建PR，选择code为target分支
```

### 场景2：执行实验批次

```bash
# 1. 从code创建实验分支
git checkout code
git checkout -b experiments/xjtu-2c-march

# 2. 执行实验（不提交中间状态）
python scripts/run_experiments.py init --dataset xjtu_2c
python scripts/run_experiments.py run

# 3. 实验完成，整理结果
python scripts/analyze_results.py --timestamp 20260308_023704

# 4. 提交最终结果
git add results/ paper/figures/
git commit -m "experiments(xjtu-2c): 添加20260308批次完整结果

- 完成800轮实验（100 seeds × 4 models × 2 methods）
- 成功率99.5%（796/800），4轮自动重试后成功
- 关键发现：
  * 1D-CNN-MIM: 86.3%改进（0.0146→0.0020 MAE）
  * MLP-MIM: 55.7%改进（0.0183→0.0081 MAE）
  * LSTM/GRU-MIM: 44%改进
- 统计显著性：所有结果p<0.001（配对t检验）
- 交付物：summary.csv, raw_data.csv, 10张论文图表"

# 5. 合并回code（通过PR或fast-forward）
git checkout code
git merge experiments/xjtu-2c-march

# 6. 可选：保留或删除实验分支
git branch -d experiments/xjtu-2c-march  # 删除
git push origin experiments/xjtu-2c-march  # 或保留推送
```

### 场景3：修复生产Bug

```bash
# 1. 从code创建修复分支
git checkout code
git checkout -b fix/cuda-memory-leak

# 2. 修复代码...

# 3. 提交
git add src/main.py
git commit -m "fix: 修正验证阶段GPU内存泄漏

在evaluate()中添加torch.cuda.empty_cache()，
解决长时间训练导致的OOM问题。

Fixes #56"

# 4. 快速合并回code
git checkout code
git merge fix/cuda-memory-leak
git push origin code

# 5. 清理
git branch -d fix/cuda-memory-leak
```

---

## 常见场景

### 提交太大需要拆分

```bash
# ❌ 错误：一个提交做太多事
git commit -m "更新代码和文档，修复几个bug"

# ✅ 正确：拆分为多个提交
git add src/models/lstm.py
git commit -m "feat(models): 添加LSTM变体架构"

git add docs/API.md
git commit -m "docs: 补充LSTM模块API文档"

git add src/utils/metrics.py
git commit -m "fix: 修正MAE计算中的除零错误"
```

### 提交后发现错误

```bash
# 刚提交，未推送
git commit --amend -m "feat: 正确的提交消息"

# 已推送（谨慎使用，会改写历史）
git commit --amend -m "feat: 修正后的消息"
git push origin feature/xxx --force-with-lease

# 更旧的提交（使用rebase）
git rebase -i HEAD~3
# 在编辑器中将pick改为reword
```

### 临时保存工作进度

```bash
# 使用stash（不推荐用于长期保存）
git stash push -m "WIP: 实验中途配置"

# 更好的做法：创建WIP提交
git add .
git commit -m "WIP: 调试MIM训练

临时提交，请勿合并到code分支"
# 后续用git rebase -i清理或git commit --amend修改
```

---

## 工具配置

### commitizen（交互式提交）

```bash
# 安装
pip install commitizen

# 使用cz代替git commit，引导式填写
cz commit

# 检查历史提交是否符合规范
cz check --rev-range HEAD~10

# 生成CHANGELOG
cz changelog
```

### pre-commit（提交前检查）

已配置在 `.pre-commit-config.yaml`，安装后自动运行：

```bash
pre-commit install  # 仅需执行一次

# 手动检查
git add .
pre-commit run

# 跳过检查（紧急情况下使用）
git commit -m "fix: 紧急修复" --no-verify
```

---

**相关文档**:
- [提交示例库](./commit-examples.md)
- [Conventional Commits 官方规范](https://www.conventionalcommits.org/zh-hans/v1.0.0/)
- [Angular Commit 规范](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
