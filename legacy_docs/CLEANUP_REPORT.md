# 工作区整理报告

整理时间：2026-06-17

## 1. 已完成的清理操作

### 1.1 Python 缓存与临时文件
- 删除所有 `__pycache__/` 目录（14 处）
- 删除所有 `*.pyc` 文件
- 删除 paper 目录下的 LaTeX 辅助文件：`main_zh.aux`, `main_zh.log`, `main_zh.out`

### 1.2 实验结果归档
将旧的、已废弃或被后续实验覆盖的结果目录统一移至 `results/_archive/`（共 266 MB）：

- `aggregated_test/`
- `aggregated_smoke/`
- `analysis_bernoulli/`
- `stage1_bernoulli/`
- `ablation_xjtu_3c_bernoulli_n30/`
- `ablation_xjtu_3c_bernoulli_n30_v2/`
- 所有 `*_smoke_*` 中间测试目录：
  - `gnn_smoke_*`
  - `group_aware_smoke_*`
  - `innovation_smoke_*`
  - `mask_only_smoke_bernoulli/`
  - `missingness_aware_smoke_*`
  - `uniform_smoke_*`
- `sanity/`
- `smoke_seed_fix/`

保留的当前/参考结果目录：
- `battle_royale_bernoulli/`, `block/`, `channel/`, `group/`, `mixed/`, `road_course/`（当前最终实验）
- `linear_baseline/`（线性基线参考）
- `nasa_smoke/`（NASA 小规模参考）
- `diagnose_nasa_cnn1d/`（诊断记录）
- `ablation_nasa_bernoulli_n100/`（NASA 消融）

### 1.3 数据目录整理
- 将原始下载包 `data/NASA data/nasa_battery.zip`（200 MB）移至 `data/NASA data/raw/nasa_battery.zip`
- 保留已解压的 `.mat` 文件和 `processed/*.csv` 文件

### 1.4 代码小修复
- `src/main.py`：在 `--missing_pattern` 选项中补回 `road_course`，与 `missing_patterns.py` 注册表保持一致。

### 1.5 文档更新
- 重写 `AGENTS.md`：
  - 反映当前 GNN / GroupMIM / Battle Royale 方向
  - 记录 6 种缺失模式与 8 种对比方法
  - 列出当前主要限制与下一步优先级
  - 更新常用命令速查

## 2. 验证结果

- ✅ 全部 111 个 Python 文件通过 `py_compile` 语法检查
- ✅ `tests/test_missing_patterns.py` 全部 6 项测试通过
- ✅ XJTU / HUST / MIT / TJU / NASA 五个数据集均可正常加载
- ✅ 单模型 smoke test（1 seed × 1 epoch）可端到端跑通
- ✅ `scripts/aggregate_results.py` 可正常汇总 `battle_royale_bernoulli`

## 3. 当前磁盘占用

```
项目总计：835 MB
├── data/      484 MB
├── results/   311 MB（其中 _archive/ 占 266 MB，可删除）
└── paper/      16 MB
```

如需要进一步释放空间，可安全删除 `results/_archive/`（约 266 MB）以及 `paper/main_zh.pdf`（编译产物）。

## 4. 待用户决策事项

1. 是否删除 `results/_archive/` 以释放 266 MB？
2. 是否现在启动 XJTU 3C 关键模式（bernoulli / channel / road_course）的 n=30 大乱斗？
3. 是否优先实现两阶段插补基线（KNN/GPR/自注意力插补 + LSTM）？
