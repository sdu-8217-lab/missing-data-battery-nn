# TEMPLATE：文献复现项目模板

> 请将本文件复制到新的复现子文件夹后，替换所有占位符内容。

## 论文信息

- **标题**: （论文完整标题）
- **作者**: （第一作者 et al.）
- **期刊/会议**: （期刊名 / 会议名，年份）
- **DOI / arXiv**: （链接）
- **原文代码**: （如有，附链接；注明是否公开）
- **项目路径**: `reproductions/<shortname_authorYYYY>/`
- **状态**: ⏳ 计划中 / 🚧 实现中 / ✅ 已完成

## 核心贡献

（用 3–5 条 bullet 概括论文核心方法与创新点）

## 复现范围

### 计划复现的实验

- [ ] 实验 1：...
- [ ] 实验 2：...
- [ ] 实验 3：...

### 与原文的已知差异

| 方面 | 原文 | 本复现 | 原因 |
|------|------|--------|------|
| 数据集 | ... | ... | ... |
| 输入特征 | ... | ... | ... |
| 模型细节 | ... | ... | ... |
| 超参数 | ... | ... | ... |
| 评估指标 | ... | ... | ... |

## 快速开始

```bash
# 1. 激活环境
source ~/research/battery-research/.venv/bin/activate

# 2. 安装本复现额外依赖（如有）
pip install -r requirements.txt  # 若存在

# 3. 训练
python scripts/train.py --config configs/default.yaml

# 4. 评估
python scripts/evaluate.py --config configs/default.yaml --checkpoint results/best_model.pth

# 5. 运行测试
python tests/test_placeholder.py
```

## 批判性审视

（在深度阅读论文和实际复现后，从以下角度客观评价原文：
- 核心 idea 是否合理、是否有新意；
- 方法/算法是否存在隐藏假设或简化；
- 实验设计是否公平、可复现；
- 结果声明是否过度；
- 本复现中遇到了哪些与论文描述不一致的地方。
建议整理为 `CRITICAL_REVIEW.md`。）

## 文件说明

```
.
├── src/               # 模型、数据、训练器实现
├── configs/           # 实验配置
├── scripts/           # 可运行脚本
├── tests/             # 单元测试与 smoke test
├── data/              # 数据目录（gitignored）
├── results/           # 结果目录（gitignored）
├── CRITICAL_REVIEW.md # 批判性审视（推荐）
├── notes.md           # 实现笔记与偏差记录
└── AGENTS.md          # 本地约定与决策记录
```
