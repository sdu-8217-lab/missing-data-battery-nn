# 项目交接文档：电池 SOH 缺失数据深度学习

> 面向后续接手本项目的 AI Agent / 协作者。当前版本反映 **GNN / GroupMIM / Battle Royale** 阶段，不再以旧版 MIM-vs-Baseline 为主。
> 
> 本文件基于项目实际文件内容与代码结构编写；若你修改了构建流程、依赖、入口或目录结构，请同步更新本文件。

---

## 1. 项目概述

本项目研究**电池健康状态（SOH）估计在结构化特征缺失场景下的深度学习方法**。

- **输入**：每个放电循环提取的 16 维 handcrafted 特征（电压 / 电流 / 温度 / 容量相关统计量）。
- **输出**：该循环的 SOH（容量 / 初始容量）。
- **核心问题**：当 16 维特征以不同结构缺失时（随机、块、通道、组、混合、road_course），神经网络能否保持精度？哪种缺失处理策略最有效？
- **当前方法矩阵**：
  - **MIM-Uniform**：拼接缺失指示器，在 10%–90% 均匀混合数据上训练。
  - **MultiMR-Uniform**：多缺失率混合训练，但不拼接指示器。
  - **FMG-Uniform**：Feature-wise Missing Gate，门控机制。
  - **GroupMIM-Uniform**：基于特征语义分组（电压 / 电流）的组嵌入 + MIM。
  - **GNN-Uniform**：把特征当作图节点，用图注意力做缺失值消息传递。
  - **GraphMIM-Uniform**：GNN + 组级池化的融合尝试。
  - **SAITS / GRIN / Neural CDE**：文献主流插补 / 连续模型基线。

---

## 2. 技术栈与运行环境

### 2.1 主要技术

| 类别 | 技术 / 版本 |
|------|-------------|
| 语言 | Python 3.12.12 |
| 深度学习框架 | PyTorch >= 2.5.1 |
| 梯度提升树 | XGBoost >= 2.0.0 |
| 数值 / 数据处理 | NumPy >= 1.24、Pandas >= 2.0、SciPy >= 1.10 |
| 机器学习工具 | scikit-learn >= 1.3（划分、标准化、指标） |
| 可视化 | Matplotlib >= 3.7、Seaborn >= 0.12 |
| 配置校验 | Pydantic v2（`src/config/pydantic_config.py`） |
| 训练封装 | 自研 `NeuralNetworkTrainer`，另含 PyTorch Lightning 封装 |
| 物理约束 | `src/trainers/physics_loss.py`（单调性、平滑性） |
| 日志 | 标准 `logging`（`src/utils/logger.py`），另有 Loguru 版本（`logger_v2.py`） |
| 其他 | `tqdm`、`pyyaml`、`scipy.io`（读取 NASA `.mat`） |
| 论文排版 | LaTeX（`paper/main_zh.tex`、`paper/references.bib`） |

### 2.2 依赖安装

项目唯一的依赖清单是根目录 `requirements.txt`：

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
torch>=2.5.1
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
```

> ⚠️ **注意**：`requirements.txt` 未列出 `pydantic`、`loguru`、`pytorch_lightning`，但代码已使用（`src/config/pydantic_config.py`、`src/trainers/lightning_*.py`）。运行相关脚本前请确认已安装：
> ```bash
> pip install pydantic loguru pytorch_lightning
> ```

### 2.3 虚拟环境

仓库本身不含 `.venv/`。当前推荐的激活命令为：

```bash
source ~/research/battery-research/.venv/bin/activate
```

> 注意：`docs/03-代码架构与环境配置.md` 仍引用 Windows Miniforge 路径；`scripts/run_paper_experiments.sh` 硬编码使用 `/home/chen/research/battery-research/.venv/bin/python3`。Linux 环境下请使用 `AGENTS.md` 与 `run_paper_experiments.sh` 中的外部 venv。

### 2.4 硬件资源

- **GPU**：RTX 4060 Ti 16GB。
- 不要同时跑多个大规模消融，避免显存 / CPU 争用。`scripts/run_paper_experiments.sh` 已限制 `MAX_JOBS=2`。

---

## 3. 代码组织

项目根目录：`~/research/missing-data-battery-nn/`

```
├── src/                                  # 核心源码
│   ├── main.py                           # 当前推荐主入口
│   ├── config/                           # 实验与模型配置
│   │   ├── experiment_config.py          # ExperimentConfig / ModelConfig（主配置）
│   │   └── pydantic_config.py            # Pydantic v2 版本配置
│   ├── data/                             # 数据加载、Dataset、缺失模式
│   │   ├── dataset_registry.py           # 数据集注册表（XJTU/HUST/MIT/TJU/NASA）
│   │   ├── dataset_loader.py             # 统一加载器与按电池划分
│   │   ├── datasets.py                   # BatteryDataset / SequenceDataset / MIMDataset
│   │   ├── missing_patterns.py           # 缺失模式工厂
│   │   └── nasa_feature_extractor.py     # NASA .mat → 16 维特征 CSV
│   ├── models/                           # 模型定义与工厂
│   │   ├── model_factory.py              # 统一模型工厂
│   │   ├── base_model.py                 # PyTorch / XGBoost 统一包装器
│   │   ├── neural_network_models.py      # MLP / LSTM / GRU / CNN1D / Transformer / iTransformer
│   │   ├── gnn_missing.py                # 特征级 GNN 缺失处理
│   │   ├── group_aware.py                # GroupMIM 分组嵌入
│   │   ├── missingness_aware.py          # MIM / MultiMR / MaskOnly 变体
│   │   ├── missing_gate.py               # FMG 特征级缺失门控
│   │   ├── graphmim.py                   # GraphMIM（GNN + 组级池化）
│   │   ├── saits_model.py                # SAITS 自注意力插补
│   │   ├── grin_model.py                 # GRIN 图循环插补网络
│   │   └── neural_cde_model.py           # Neural CDE
│   ├── trainers/                         # 训练器与损失函数
│   │   ├── neural_network_trainer.py     # PyTorch 训练 / 早停 / 评估
│   │   ├── xgboost_trainer.py            # XGBoost 训练封装
│   │   ├── physics_loss.py               # 单调性、平滑性物理约束
│   │   ├── lightning_module.py           # PyTorch Lightning 封装
│   │   └── lightning_trainer.py          # PyTorch Lightning 训练封装
│   ├── experiments/                      # 实验运行器
│   │   ├── experiment_runner.py          # 当前主实验运行器
│   │   ├── single_experiment.py          # 旧“小实验”运行器
│   │   ├── batch_experiment.py           # 旧“大实验”运行器（支持中断恢复）
│   │   ├── checkpoint.py                 # 检查点管理
│   │   ├── architecture_search.py        # 架构搜索
│   │   ├── fine_grained_search.py        # 细粒度搜索
│   │   └── batch_experiment_runner_v2.py # v2 批量实验
│   ├── evaluators/                       # 评估指标
│   │   ├── metrics.py                    # MAE / RMSE / R²
│   │   └── model_evaluator.py            # 模型评估入口
│   ├── visualization/                    # 可视化
│   │   ├── single_plots.py               # 单种子绘图
│   │   ├── batch_plots.py                # 多种子统计图
│   │   ├── missing_rate_curves.py        # 缺失率曲线
│   │   └── architecture_analysis.py      # 架构分析图
│   └── utils/                            # 工具
│       ├── seed_manager.py               # 随机种子管理
│       ├── logger.py                     # 标准 logging 日志
│       └── logger_v2.py                  # Loguru 版本日志
│
├── scripts/                              # 可直接运行的脚本
│   ├── run_battle_royale.py              # 当前核心实验：8 方法 × 6 缺失模式大乱斗
│   ├── aggregate_results.py              # 汇总结果、统计检验、画图
│   ├── compare_all_methods.py            # 多方法对比
│   ├── preprocess_nasa.py                # NASA 数据预处理
│   ├── linear_baseline.py                # 线性基线
│   ├── linear_imputer_baseline.py        # 线性插补基线
│   ├── cross_dataset_linear.py           # 跨数据集线性基线
│   ├── run_two_stage_baseline.py         # 两阶段插补 + 估计基线
│   ├── run_mask_only_baseline.py         # MaskOnly 基线
│   ├── run_gate_ablation.py              # FMG 门控消融
│   ├── run_gnn_smoke.py                  # GNN 冒烟测试
│   ├── run_group_aware_smoke.py          # GroupMIM 冒烟测试
│   ├── diagnose_nasa_cnn1d.py            # NASA CNN1D 诊断
│   ├── design_nasa_features.py           # NASA 特征设计
│   └── run_paper_experiments.sh          # 论文阶段三阶段批量脚本
│
├── data/                                 # 数据集（已加入 .gitignore）
│   ├── XJTU data/
│   ├── HUST data/
│   ├── MIT data/
│   ├── TJU data/
│   └── NASA data/                        # 含 raw/*.mat 与 processed/*.csv
│
├── results/                              # 实验结果（已加入 .gitignore）
│   ├── battle_royale_*/                  # 大乱斗结果
│   ├── linear_baseline/
│   ├── nasa_smoke/
│   ├── diagnose_nasa_cnn1d/
│   ├── ablation_nasa_bernoulli_n100/
│   └── _archive/                         # 旧结果归档
│
├── experiments/                          # 旧分层实验输出目录（已不再推荐使用）
├── tests/                                # 测试
│   └── test_missing_patterns.py          # 缺失模式单元测试
├── docs/                                 # 项目文档
├── paper/                                # LaTeX 论文源文件
├── literature/                           # 文献整理
├── configs_v3/                           # 模型配置 CSV
├── requirements.txt                      # 唯一依赖清单
└── reproductions/                        # 文献复现专区
    ├── README.md                         # 复现专区总览与模板说明
    ├── TEMPLATE/                         # 新增复现项目模板
    └── batteryCDE_wang2025/              # BatteryCDE 论文复现
```

### 3.1 新旧代码并存说明

- 当前推荐入口为 `src/main.py` 与 `scripts/run_battle_royale.py`。
- 根目录保留大量旧版 / 分析脚本（如 `run_batch.py`、`run_single.py`、`analyze_*.py`、`plot_*.py`），它们依赖 `src/experiments/batch_experiment.py` 等旧运行器。
- 新增功能请优先在 `src/` 下按当前模块划分实现，避免继续扩展根目录脚本。

### 3.2 文献复现专区（`reproductions/`）

- 新增于项目根目录，用于集中管理引用论文的独立复现工作。
- 每个子文件夹对应一篇论文，包含独立的 `src/`、`configs/`、`scripts/`、`tests/`、`data/`、`results/`。
- `reproductions/TEMPLATE/` 是新增复现项目的标准模板，复制后按 `<短标题>_<第一作者姓氏><发表年份>/` 命名。
- 当前已有：
  - `reproductions/batteryCDE_wang2025/`：Wang et al., IEEE TTE 2025，Neural CDE + 双重注意力做未来容量预测。
- 复现子项目应自包含模型实现，但可复用项目级工具（`src/utils/seed_manager.py`、`src/evaluators/metrics.py` 等）。
- 复现产生的 `data/` 与 `results/` 已加入根 `.gitignore`，不进入 git。

---

## 4. 构建与运行

### 4.1 推荐入口

```bash
# 激活环境
source ~/research/battery-research/.venv/bin/activate

# 大乱斗（当前核心实验）
python scripts/run_battle_royale.py --n_repeats 5 --epochs 100

# 单个缺失模式快速消融
python -m src.main \
  --dataset XJTU --batch 3C --data_dir "./data/XJTU data" \
  --missing_pattern road_course --models all \
  --n_repeats 5 --epochs 100 --batch_size 32 --lr 0.001 --patience 15 \
  --results_dir ./results/battle_royale_road_course

# 汇总结果
python scripts/aggregate_results.py \
  --input results/battle_royale_bernoulli \
  --output results/battle_royale_bernoulli/aggregated

# NASA 数据预处理
python scripts/preprocess_nasa.py --data_dir "./data/NASA data"
```

### 4.2 主流程说明

`src/main.py` → `ExperimentRunner.run_all_experiments()` 的执行流程：

1. **参数解析**：数据集、批次、缺失模式、模型过滤、重复次数、训练超参、物理损失权重等。
2. **注册表自动补全**：`src/data/dataset_registry.py` 提供默认 `feature_cols`、`target_col`、`data_dir` 并校验列名。
3. **数据加载**：
   - `DatasetLoaderFactory.create_loader` → `UnifiedBatteryDatasetLoader` 或 `NASADatasetLoader`。
   - 按电池划分 train / val / test（`test_size=0.25`、`val_size=0.25`）。
   - `StandardScaler` 拟合训练集后转换三集合。
   - 记录电池边界，避免序列窗口跨电池。
4. **重复实验**：
   - 使用 `MODEL_SEED_OFFSETS`（`mlp:0, lstm:100000, gru:200000, cnn1d:300000`）。
   - 保证同一 repeat 下 Baseline / MIM / MultiMR 共享同一数据划分。
   - 每个模型 / 种子组合独立加载数据、训练、评估、增量保存结果。
5. **训练策略**：
   - `baseline`：用完整数据训练。
   - `mim` / `multi_missing_rate` / `uniform_multi_rate`：使用 `MIMDataset` / `SequenceMIMDataset`，在 `training_missing_rates=[0.0, 0.1, …, 0.9]` 上混合训练。
6. **训练器**：`NeuralNetworkTrainer`：Adam + MSE + 早停（默认 `patience=15`），可选物理损失。
7. **评估**：对每个 `missing_rate` 生成对应缺失掩码的测试集，计算 MAE、RMSE、R²、推理时间。
8. **保存**：`results/{timestamp}_{batch}/` 下含 `config.json`、`logs/`、`models/`、`results/experiment_results_{timestamp}.csv`、`figures/`。

### 4.3 输出文件命名

- 当前 `ExperimentRunner` 输出目录：`results/{timestamp}_{batch}/`
- 旧分层架构输出目录：`experiments/{batch_name}/{timestamp}/seed_{seed}/`
- 模型文件：`best_model_{model_name}_seed{seed}_{timestamp}.pth`
- 划分文件：`results/splits_seed{seed}.json`
- 结果 CSV：`results/experiment_results_{timestamp}.csv`，列包括：
  ```text
  timestamp, model, model_type, use_mim, use_fmg, use_curriculum, strategy,
  missing_rate, seed, mae, rmse, r2, inference_time, training_time
  ```

### 4.4 论文阶段批量脚本

```bash
# Stage 1/2/3 或全部
bash scripts/run_paper_experiments.sh [1|2|3|all]
```

该脚本使用 `MAX_JOBS=2` 限制并发，并在后台运行。注意它硬编码了 Python 解释器路径 `/home/chen/research/battery-research/.venv/bin/python3`。

---

## 5. 测试策略

### 5.1 测试文件

- **唯一测试文件**：`tests/test_missing_patterns.py`
- **无 pytest 配置**：没有 `pytest.ini`、`tox.ini`、CI/CD；测试通过直接运行 Python 脚本执行。

### 5.2 测试内容

`tests/test_missing_patterns.py` 包含 6 个测试函数：

1. `test_missing_rate_accuracy`：各模式在 0.1–0.9 缺失率下实际缺失率与目标偏差 < 0.02。
2. `test_reproducibility`：同 seed 生成的掩码完全一致。
3. `test_state_dependent_correlation`：低 SOH 样本缺失率高于高 SOH。
4. `test_channel_pattern_structure`：存在未缺失列与高缺失列。
5. `test_block_pattern_temporal_structure`：块缺失转换次数低于伯努利。
6. `test_mixed_pattern_composition`：混合模式形状与缺失率正确。

### 5.3 运行测试

```bash
# 缺失模式单元测试
python tests/test_missing_patterns.py

# 快速冒烟测试（验证代码通路）
python -m src.main \
  --dataset XJTU --batch 3C --data_dir "./data/XJTU data" \
  --missing_pattern bernoulli --models mlp \
  --n_repeats 1 --epochs 1 --patience 1 \
  --results_dir ./results/smoke_$(date +%s)
```

### 5.4 其他验证

`CLEANUP_REPORT.md` 记录：
- 111 个 Python 文件通过 `py_compile` 语法检查。
- `tests/test_missing_patterns.py` 6 项测试通过。
- 5 个数据集均可正常加载。
- 单模型 smoke test 可端到端跑通。

---

## 6. 代码风格与开发约定

### 6.1 命名规范

- **文件 / 函数**：snake_case，例如 `experiment_runner.py`、`create_missing_pattern`。
- **类**：CamelCase，例如 `ExperimentRunner`、`ModelFactory`。
- **常量**：全大写 + snake_case，例如 `MODEL_SEED_OFFSETS`。
- **注释 / 文档字符串**：以**中文**为主；代码标识符以英文为主。

### 6.2 配置约定

- 主配置是 `ExperimentConfig` dataclass（`src/config/experiment_config.py`）。
- `model_filter` 支持：
  - `['all']`：返回所有默认配置。
  - `['mlp']`：匹配所有 MLP 变体。
  - `['mlp-mim']`：精确匹配名称。
- `include_fmg=True` 才会在 `all` 中加入 FMG 相关配置。
- `ModelConfig.strategy` 取值：
  - `'baseline'`：完整数据训练。
  - `'multi_missing_rate'`：多缺失率混合，**不**拼接指示器。
  - `'mim'`：多缺失率混合，拼接指示器。
  - `'uniform_multi_rate'`：每个 epoch 都混合所有缺失率（当前大乱斗默认）。

### 6.3 日志约定

- 使用 `src/utils/logger.py` 中基于标准 `logging` 的 UTF-8 文件 / 控制台日志。
- 日志中避免使用 `R²` 等特殊字符，改用 `R2`，防止 Windows 终端 / 文件编码问题。

### 6.4 随机种子与可复现性

- 所有随机种子由 `src/utils/seed_manager.py` 统一管理。
- 数据划分按电池（battery）而非按样本，序列窗口不跨电池边界。
- 每次实验保存 `config.json` 与 `splits_seed{seed}.json`。
- PyTorch 模型保存 `state_dict`，XGBoost 使用 `joblib`。

### 6.5 Git 与文件管理

- `.gitignore` 排除：`data/`、`results/`、`logs/`、`figures/`、`*.pth`、LaTeX 辅助文件、`__pycache__`、虚拟环境、CSV、JSON 等。
- 大文件不进入 git；NASA 原始 `.mat` 与实验结果均本地保存。
- `CLEANUP_REPORT.md` 记录：旧结果已移至 `results/_archive/`（约 266 MB）。

---

## 7. 数据集与预处理

### 7.1 支持的数据集

| 数据集 | 批次 / 子集示例 | 说明 |
|--------|-----------------|------|
| XJTU | `2C`, `3C`, `R2.5`, `R3`, `RW`, `Sim_satellite` | 已预处理为 per-battery CSV |
| HUST | `1`–`10` | 已预处理为 per-battery CSV |
| MIT | `2017-05-12`, `2017-06-30`, `2018-04-12` | 已预处理为 per-battery CSV |
| TJU | `Dataset_1_NCA_battery`, `Dataset_2_NCM_battery`, `Dataset_3_NCM_NCA_battery` | 已预处理为 per-battery CSV |
| NASA | `all` | 需先运行 `scripts/preprocess_nasa.py` 从 `.mat` 提取 16 维特征 |

### 7.2 特征与目标

- **输入特征**：16 维 handcrafted 特征（电压 / 电流 / 温度 / 容量相关统计量），由 `dataset_registry.py` 定义。
- **目标列**：`capacity`。
- **SOH 计算**：`capacity / 初始容量`。

### 7.3 NASA 预处理

```bash
python scripts/preprocess_nasa.py --data_dir "./data/NASA data"
```

该脚本从 `data/NASA data/*.mat` 提取 16 维特征，生成 `data/NASA data/processed/B*.csv`。`NASADatasetLoader` 在 `processed/` 不存在时也会自动触发提取。

---

## 8. 安全与部署

### 8.1 部署现状

- **无 Docker**：未提供 `Dockerfile` 或 `docker-compose.yml`。
- **无 CI/CD**：未找到 `.github/`、`.gitlab-ci.yml` 等持续集成配置。
- 自动化仅依赖本地 Bash 脚本 `scripts/run_paper_experiments.sh`。

### 8.2 安全注意事项

- 项目运行在本地 Linux 环境，无网络服务暴露。
- 不处理用户输入或敏感数据；输入数据均为公开电池数据集。
- 不要在代码中硬编码绝对路径或凭据；实验输出目录通过参数指定。
- 运行大规模实验前，请确认 `results/` 与 `data/` 磁盘空间充足（NASA 数据约 391 MB，实验结果可能占用数 GB）。

### 8.3 可复现性建议

- 始终通过 `--seed` 固定随机种子。
- 使用 `requirements.txt` + 缺失依赖手动安装的方式重建环境。
- 实验完成后保留 `config.json` 与 `splits_seed*.json`，以便追溯划分。

---

## 9. 当前主要问题与限制

1. **单一数据集**：所有大乱斗默认仅在 XJTU 3C 上完成，外部效度不足。
2. **统计显著性不足**：默认 `n_repeats=5` 重复太少，配对检验功率不够；建议关键模式使用 `n=30`。
3. **缺少文献主流两阶段基线**：领域主流“插补 → SOH 估计”基线仍在完善中。
4. **物理约束处于可选阶段**：`physics_loss.py` 已实现，但在主流程中为可选参数，尚未系统评估。
5. **GNN 与 GroupMIM 未深度融合**：`graphmim.py` 是初步尝试，尚未成为主流程默认方法。
6. **依赖文件不完整**：`pydantic`、`loguru`、`pytorch_lightning` 未写入 `requirements.txt`。
7. **文档部分过时**：`docs/08-src目录说明.md` 等旧文档中的模块文件名与当前 `src/models/` 不完全一致。

---

## 10. 下一步计划（按优先级）

### 10.1 提升统计稳健性

把大乱斗从 `n=5` 扩展到 **n=30 seeds**，至少覆盖关键模式：

```bash
python scripts/run_battle_royale.py \
  --patterns bernoulli channel road_course --n_repeats 30 --epochs 100
```

### 10.2 完善基线矩阵

- 两阶段插补 + 估计（KNN / Iterative Imputer / Self-attention / GPR + MLP/LSTM）。
- 简单 mean / median 填充 + Linear Regression 强非神经网络基线。

### 10.3 跨数据集验证

在 XJTU 关键模式 n=30 后，以相同方法矩阵跑 NASA、HUST batch 1、MIT 2017-05-12、TJU Dataset_1_NCA_battery。

### 10.4 统一架构

把 GroupMIM 的组嵌入和 GNN 的图注意力融合成单一模型，输入序列模型后估计 SOH。

### 10.5 系统化物理约束

将单调性约束与 Lipschitz 平滑损失作为常规超参纳入公平对比。

### 10.6 文档与论文

- 保持 `AGENTS.md` 与代码结构同步。
- 将 `paper/main_zh.tex` 从旧 MIM 叙事更新为“结构化缺失公平 benchmark + 结构感知端到端方法”。

---

## 11. 常用命令速查

```bash
# 激活环境
source ~/research/battery-research/.venv/bin/activate

# 缺失模式单元测试
python tests/test_missing_patterns.py

# 快速冒烟测试
python -m src.main \
  --dataset XJTU --batch 3C --data_dir "./data/XJTU data" \
  --missing_pattern bernoulli --models mlp \
  --n_repeats 1 --epochs 1 --patience 1 \
  --results_dir ./results/smoke_$(date +%s)

# 大乱斗
python scripts/run_battle_royale.py --n_repeats 5 --epochs 100

# 汇总结果
python scripts/aggregate_results.py \
  --input results/battle_royale_bernoulli \
  --output results/battle_royale_bernoulli/aggregated

# NASA 预处理
python scripts/preprocess_nasa.py --data_dir "./data/NASA data"

# BatteryCDE 复现（额外依赖 torchcde）
cd reproductions/batteryCDE_wang2025
pip install -r requirements.txt
python tests/test_battery_cde_forward.py
python scripts/train.py --config configs/nasa.yaml

# 查看结果目录大小
sudo du -sh results/*
```

---

## 12. 决策建议

- **继续路线**：当前方向（结构感知端到端缺失处理 + 公平 benchmark）是正确的，但需要从“自证 MIM 有效”升级为“系统对比多种缺失结构、多数据集、强基线”。
- **最快产出**：先跑 XJTU 3C 上 `bernoulli` / `channel` / `road_course` 的 n=30 大乱斗，获得统计显著结果，然后补一个两阶段插补基线，即可支撑一篇方法论文。
- **最高天花板**：融合 GNN + GroupMIM + 物理约束 + 跨数据集验证，但工作量显著更大。
