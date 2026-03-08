# 贡献指南 (Contributing Guide)

感谢您对 missing-data-battery-nn 项目的关注！本文档帮助您快速上手开发工作流。

> **注意**: 详细Git工作流示例请查阅 [docs/development/git-workflow.md](./docs/development/git-workflow.md)

## 开发环境设置

```bash
# 1. Fork并克隆仓库
git clone https://github.com/YOUR_USERNAME/missing-data-battery-nn.git
cd missing-data-battery-nn

# 2. 创建虚拟环境
conda env create -f environment.yml
conda activate battery-nn

# 3. 安装预提交钩子（强制检查提交规范）
pip install pre-commit commitizen
pre-commit install
```

## 快速参考

### 提交消息格式

```
<type>[optional scope]: <简短描述（中文）>

[optional body: 详细说明]

[optional footer: 关联issue等]
```

### 常用类型

| 类型 | 说明 | 示例 |
|:---|:---|:---|
| `feat` | 新功能 | `feat: 实现MIM多缺失率联合训练` |
| `fix` | Bug修复 | `fix: 修正batch size不匹配导致的OOM` |
| `refactor` | 代码重构 | `refactor(data): 重构DataLoader配置` |
| `docs` | 文档更新 | `docs: 补充API使用示例` |
| `test` | 测试相关 | `test: 添加DataLoader单元测试` |
| `chore` | 构建/工具 | `chore: 更新requirements依赖` |
| `experiments` | 实验结果 | `experiments(xjtu-2c): 添加800轮实验结果` |
| `paper` | 论文相关 | `paper(figures): 添加消融实验图表` |

### Scope（作用域）

```
data        →  src/data/
models      →  src/models/
experiments →  src/experiments/
configs     →  configs/
scripts     →  scripts/
docs        →  docs/
paper       →  paper/
root        →  根目录配置文件
```

## 分支策略

| 分支 | 用途 | 保护状态 |
|:---|:---|:---:|
| `main` | 稳定版本，只接受PR合并 | 🔒 保护 |
| `code` | 主开发分支，功能集成 | 🔒 保护 |
| `feature/*` | 新功能开发 | 自由 |
| `fix/*` | Bug修复 | 自由 |
| `experiments/*` | 实验执行 | 自由 |

## Pull Request 流程

1. **从最新 `code` 分支创建功能分支**
2. **开发并提交**（遵循上述提交规范）
3. **推送并创建PR**（自动加载PR模板）
4. **代码审查**（至少1人批准 + CI通过）

## 提问与反馈

- 🐛 Bug报告 → 使用 GitHub Issues
- 💡 功能建议 → 使用 GitHub Discussions

---

**参考阅读**:
- [详细Git工作流](./docs/development/git-workflow.md)
- [提交示例库](./docs/development/commit-examples.md)
- [Conventional Commits 中文规范](https://www.conventionalcommits.org/zh-hans/v1.0.0/)
