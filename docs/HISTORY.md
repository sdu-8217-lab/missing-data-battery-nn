# 历史文档索引

<!-- 
  confidence: B
  reviewed: 2026-03-19
  reviewer: chen
  ai-assisted: true
-->

> **说明**
> 
> 本文档记录已删除的历史文档信息。
> 完整内容仍可通过Git历史访问。
> 
> 删除原则：Git已保留完整历史，无需在仓库中维护归档副本。

---

## 已删除文档清单

### 2026-03-19 删除

| 原路径 | 删除日期 | 最后提交 | 内容摘要 | 删除原因 |
|--------|---------|---------|---------|---------|
| `docs/archived/00-README-8217.md` | 2026-03-19 | b763cc3 | 旧README备份 | 内容已整合到README.md |
| `docs/archived/01-分层实验架构设计.md` | 2026-03-19 | b763cc3 | 早期架构设计 | 内容已整合到meta.md |
| `docs/archived/02-新架构实现总结.md` | 2026-03-19 | b763cc3 | 架构迁移总结 | 内容已过时 |
| `docs/archived/05-实验进度报告.md` | 2026-03-19 | b763cc3 | 历史进度报告 | 临时性文档 |
| `docs/archived/06-详细使用指南.md` | 2026-03-19 | b763cc3 | 早期使用指南 | 被QUICKSTART.md替代 |
| `docs/archived/07-新架构README.md` | 2026-03-19 | b763cc3 | 架构README | 被README.md替代 |
| `docs/archived/08-src目录说明.md` | 2026-03-19 | b5328e4 | 代码结构说明 | 内容已整合到ARCHITECTURE.md |
| `docs/archived/09-论文大纲.md` | 2026-03-19 | b5328e4 | 论文结构草稿 | 移至paper/目录 |
| `docs/archived/10-实验设计与计划.md` | 2026-03-19 | b5328e4 | 实验计划 | 被EXPERIMENTS.md替代 |
| `docs/archived/11-文档一致性检查.md` | 2026-03-19 | b763cc3 | 一致性检查记录 | 临时性文档 |
| `docs/archived/12-绘图功能指南.md` | 2026-03-19 | b5328e4 | 绘图指南 | 内容过时 |
| `docs/archived/13-论文图片指南.md` | 2026-03-19 | b5328e4 | 图片制作指南 | 移至paper/目录 |
| `docs/archived/14-快速参考卡片.md` | 2026-03-19 | b763cc3 | 速查卡片 | 内容整合到QUICKSTART.md |
| `docs/archived/15-模型参数调优指南.md` | 2026-03-19 | b5328e4 | 调优指南 | 内容需重写，暂归档 |
| `docs/archived/16-新架构经验教训.md` | 2026-03-19 | b5328e4 | 经验总结 | 内容已整合 |
| `docs/archived/17-模型架构搜索实验设计.md` | 2026-03-19 | b5328e4 | 架构搜索设计 | 实验已完成 |
| `docs/archived/18-细粒度搜索范围定义.md` | 2026-03-19 | b763cc3 | 搜索范围 | 实验已完成 |
| `docs/archived/19-统一搜索配置规则.md` | 2026-03-19 | b763cc3 | 配置规则 | 实验已完成 |
| `docs/archived/EXPERIMENT_FRAMEWORK.md` | 2026-03-19 | b5328e4 | 实验框架 | 被run_batch_experiments.py替代 |
| `docs/archived/FILE_STRUCTURE_SPEC.md` | 2026-03-19 | b5328e4 | 文件结构规范 | 内容已过时 |
| `docs/archived/MIM_CORRECTION_SUMMARY.md` | 2026-03-19 | b5328e4 | MIM修正总结 | 内容已整合到meta.md |
| `docs/archived/PROJECT_GUIDE_MASTER.md` | 2026-03-19 | b763cc3 | 项目总览 | 被README.md替代 |
| `docs/archived/README_old_hydra_20260318.md` | 2026-03-19 | - | 旧README备份 | 刚创建即归档，直接删除 |
| `docs/archived/RESEARCH_LESSONS_LEARNED.md` | 2026-03-19 | b5328e4 | 研究经验 | 内容已整合 |
| `docs/archived/code_architecture_guide.md` | 2026-03-19 | b763cc3 | 代码架构 | 内容过时 |

**总计删除**: 25个文件
**释放空间**: ~150KB

---

## 如何访问历史文档

### 方法1：查看特定文件历史

```bash
# 查看文件历史记录
git log --follow -- docs/archived/01-分层实验架构设计.md

# 查看特定版本内容
git show b763cc3:docs/archived/01-分层实验架构设计.md
```

### 方法2：恢复已删除文件

```bash
# 从Git历史恢复文件到当前目录
git show b763cc3:docs/archived/01-分层实验架构设计.md > restored.md

# 或直接检出历史版本
git checkout b763cc3 -- docs/archived/01-分层实验架构设计.md
```

### 方法3：浏览历史快照

```bash
# 查看删除前的完整目录状态
git show b763cc3:docs/archived/ | head -30

# 或使用GitHub/GitLab界面浏览历史
```

---

## 归档策略说明

### 为什么删除而非保留？

1. **Git已保留完整历史**
   - 所有文件内容永久存储在Git对象库中
   - 可通过commit hash随时访问任何历史版本

2. **减少认知负担**
   - 新成员不会被26个历史文件迷惑
   - 当前文档目录只包含有效文档

3. **避免维护负担**
   - 历史文件无需持续更新
   - 减少重复内容搜索时的干扰

### 什么情况下恢复？

- 发现当前文档缺失关键历史信息
- 需要对比新旧架构差异
- 法律/审计要求提供历史文档

---

## 相关文档

- [CONFIDENCE.md](CONFIDENCE.md) - 文档置信度体系
- Git历史: `git log --all --full-history -- docs/archived/`

---

*最后更新: 2026-03-19*
*删除操作: 文档重构阶段1*
