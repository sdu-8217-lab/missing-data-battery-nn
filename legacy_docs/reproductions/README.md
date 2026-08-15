# 文献复现专区（Reproductions）

本目录用于集中管理项目引用文献的独立复现工作。每个子文件夹对应一篇（或一组紧密相关）论文，包含从数据预处理、模型实现、训练评估到结果汇总的完整链路。

> **复现的目的不仅是“把代码跑通”，更是检验论文结论、发现隐藏假设、明确方法边界的过程。** 因此，每个复现子项目都应包含对原文的批判性审视。

## 设计原则

1. **与主项目解耦**：每个复现子项目拥有独立的 `src/`、`configs/`、`scripts/`、`tests/`，避免污染项目根目录和 `src/` 核心代码。
2. **可复用基础设施**：鼓励通过相对导入或 `sys.path` 复用项目级工具（`src/utils/seed_manager.py`、`src/evaluators/metrics.py` 等），但模型实现应自包含。
3. **可追踪**：每个子项目必须包含 `README.md`、`notes.md`、`AGENTS.md`，记录论文信息、复现偏差、运行方式与关键决策。
4. **批判性审视**：每个子项目鼓励撰写 `CRITICAL_REVIEW.md`，从 idea、方法、算法、实验、结果等角度客观评价原文，记录仅通过阅读难以发现的问题。
5. **结果隔离**：`data/` 与 `results/` 默认加入 `.gitignore`，每个子项目本地管理，通过 `README.md` 说明如何获取或生成。

## 目录结构

```
reproductions/
├── README.md                 # 本文件：复现专区总览
├── TEMPLATE/                 # 新增复现项目模板
│   ├── AGENTS.md
│   ├── README.md
│   ├── notes.md
│   ├── src/
│   ├── configs/
│   ├── scripts/
│   ├── tests/
│   ├── data/                 # 数据目录（gitignored）
│   └── results/              # 结果目录（gitignored）
└── batteryCDE_wang2025/      # BatteryCDE 论文复现
    ├── AGENTS.md
    ├── README.md
    ├── CRITICAL_REVIEW.md    # 批判性审视
    ├── notes.md
    ├── src/
    ├── configs/
    ├── scripts/
    ├── tests/
    ├── data/
    └── results/
```

## 新增复现项目的标准流程

1. 复制 `TEMPLATE/` 为新的子文件夹，建议命名格式：`<短标题>_<第一作者姓氏><发表年份>/`。
2. 填写 `README.md`、`AGENTS.md`、`notes.md`。
3. 在 `src/` 中实现论文核心模型与数据流。
4. 在 `configs/` 中配置数据集、超参与训练策略。
5. 在 `scripts/` 中实现训练、评估、对比基线脚本。
6. 在 `tests/` 中添加至少一个前向传播测试和一个数据加载测试。
7. **深度阅读论文**，记录实现细节、偏差与批判性思考，形成 `CRITICAL_REVIEW.md`。
8. 运行 `tests/` 与 smoke test 验证代码通路。

## 当前复现项目

| 子项目 | 论文 | 状态 | 批判性审视 |
|--------|------|------|-----------|
| `batteryCDE_wang2025/` | BatteryCDE: A Transferable Future Capacity Estimation Method for Battery Degradation With Irregular Sampling and Missing Data (Wang et al., IEEE TTE 2025) | 核心模型与基线已完成；horizon/缺失/迁移实验进行中 | 已完成，见 `CRITICAL_REVIEW.md` |
