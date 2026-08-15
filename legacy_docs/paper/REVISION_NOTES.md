# 论文重写修订笔记

本文档记录针对审稿意见对 `main_zh.tex` 进行的框架级修改，以及后续还需要补充的实验、图表与数据。

## 已完成的修改

1. **标题更新**：体现跨数据集验证与电池感知缺失模式。
2. **摘要重写**：
   - 明确四个数据集（XJTU/HUST/MIT/TJU）。
   - 引入三种训练策略：Baseline、Multi-MR w/o Indicator、MIM。
   - 引入电池感知缺失模式（block/channel/state-dependent）。
3. **引言更新**：
   - 强调 MIM 针对电池 SOH 预测的领域适配（循环级特征、相对容量标签、老化相关缺失）。
   - 在研究不足中补充：跨数据集验证、严格基线、真实缺失模式。
   - 将研究问题扩展为 RQ1–RQ4。
   - 将贡献扩展为四个方面。
4. **方法章节**：
   - 数据集部分改为多数据集描述。
   - 缺失模拟部分增加 block/channel/state-dependent 三种模式。
   - 输入表示部分明确三种训练/输入策略。
5. **NASA 数据集接入与 SOH 归一化修复**（2026-06-16）：
   - 从 `phm-datasets.s3.amazonaws.com` 下载真实 NASA 电池数据集并解压到 `data/NASA data/`。
   - 新增 `NASADatasetLoader`：读取 `.mat` 文件，提取 discharge cycle 的标量特征，并用历史窗口构造 16 维循环级特征。
   - 修复 SOH 归一化 bug：
     - 初始容量由“首个循环容量”改为“前 20 个 discharge cycles 的最大容量”，避免首个循环未充满导致 SOH 被过度放大。
     - 过滤容量 ≤ 0 的无效循环。
     - 过滤 SOH > 1.3 或 SOH < 0.05 的异常样本。
     - 过滤估计初始容量 < 0.5 Ah 的电池（共 34 个电池中加载 25 个）。
   - Smoke test 验证：NASA `MAE` 从原来的 ≈5.1 降至 ≈0.06–0.16（SOH 尺度），归一化问题已解决。

## 待补充内容

### 实验数据

需要运行完整实验并填充以下结果：

1. **跨数据集结果表/图（5 个数据集）**
   - 每个数据集（XJTU/HUST/MIT/TJU/NASA）上，Baseline、Multi-MR、MIM 三种策略在 MR=0.1–0.9 的 MAE/RMSE/R²。
   - 建议：每个数据集取一个代表性 batch，缺失模式 `bernoulli`，`n_repeats=100`，`epochs=100`。

2. **消融实验：Multi-MR w/o Indicator vs. MIM**
   - 关键图/表：在 MR=0.5 时，对比 MLP/LSTM/GRU/CNN1D 三种策略的 MAE。
   - 用于回答 RQ2。

3. **缺失模式对比**
   - 在 `bernoulli`/`block`/`channel`/`state_dependent`/`mixed` 五种模式下，MIM 与 Baseline 的 MAE 随 MR 变化曲线。
   - 优先在 XJTU 3C 上完成；若算力允许，补充 NASA all 以验证跨数据集模式鲁棒性。
   - 用于回答 RQ3。

4. **统计显著性**
   - 对 100 个随机种子的结果计算均值±标准差，并进行配对 t 检验或多重比较校正（如 Bonferroni）。
   - 替换原稿中单一数值为 mean±std，并标注显著性符号（* p<0.05, ** p<0.01）。

### 图表清单

- `fig:overview`: 方法流程图（三种策略 + 五种缺失模式）。
- `fig:dataset_comparison`: 五个数据集上 MIM 改进率对比。
- `fig:ablation`: Multi-MR vs. MIM 消融。
- `fig:missing_patterns`: 五种缺失模式下 MAE 曲线。
- `tab:multi_dataset_results`: 跨数据集 MAE/RMSE/R² 汇总表。
- `tab:missing_pattern_results`: 缺失模式对比表。

### 文字补充

- 实验设置小节：说明统一的数据划分（按电池划分 train/val/test，`test_size=0.25`, `val_size=0.25`）、训练参数（`epochs=100`, `batch=32`, `lr=1e-3`, `patience=15`）。
- 数据集小节：补充 NASA 数据集描述、特征构造方式、筛选规则（25/34 电池）。
- 结果分析：根据实际数据撰写跨数据集泛化、消融、缺失模式、高缺失鲁棒性分析。
- 结论：回应 RQ1–RQ4，强调工程指导意义。

## 建议运行命令

已提供一键脚本 `scripts/run_paper_experiments.sh`：

```bash
# Stage 1：主数据集 × bernoulli（约 5 个任务并行，每任务约 2–3 小时）
bash scripts/run_paper_experiments.sh 1

# Stage 2：缺失模式对比（XJTU 3C + NASA all，4 种非随机模式）
bash scripts/run_paper_experiments.sh 2

# Stage 3：结果汇总（需后续补充 aggregate_results.py）
bash scripts/run_paper_experiments.sh 3
```

单独运行示例：

```bash
# 跨数据集消融
python src/main.py --dataset XJTU --batch 3C --models all --n_repeats 100 --epochs 100 --results_dir ./results/stage1_bernoulli/xjtu_3c
python src/main.py --dataset HUST --batch 1 --data_dir './data/HUST data' --models all --n_repeats 100 --epochs 100 --results_dir ./results/stage1_bernoulli/hust_1
python src/main.py --dataset MIT --batch 2017-05-12 --data_dir './data/MIT data' --models all --n_repeats 100 --epochs 100 --results_dir ./results/stage1_bernoulli/mit_2017-05-12
python src/main.py --dataset TJU --batch Dataset_1_NCA_battery --data_dir './data/TJU data' --models all --n_repeats 100 --epochs 100 --results_dir ./results/stage1_bernoulli/tju_nca
python src/main.py --dataset NASA --batch all --data_dir './data/NASA data' --models all --n_repeats 100 --epochs 100 --results_dir ./results/stage1_bernoulli/nasa_all

# 缺失模式对比
python src/main.py --dataset XJTU --batch 3C --models all --missing_pattern block --n_repeats 50 --epochs 100 --results_dir ./results/stage2_patterns/xjtu_3c_block
python src/main.py --dataset NASA --batch all --data_dir './data/NASA data' --models all --missing_pattern block --n_repeats 50 --epochs 100 --results_dir ./results/stage2_patterns/nasa_all_block
```

## 已知问题

- `src/experiments/architecture_search.py` 存在缩进错误，当前不在本阶段处理范围。
- `src/visualization/single_plots.py` 在绘制单实验图时偶发 `Invalid color argument: []`，不影响 CSV 结果产出，可在 Stage 3 统一修复可视化脚本。

## 引用检查

- 运行 `xelatex -> bibtex -> xelatex -> xelatex` 以解决交叉引用和引用问题。
- 注意：原稿中存在部分引用可能缺失（如 `geng2025uqsoh`），需根据实际文献补充或删除。
