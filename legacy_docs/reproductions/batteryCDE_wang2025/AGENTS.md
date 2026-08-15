# BatteryCDE 复现本地约定

## 1. 数据约定

- **输入**：充电过程中的电压、电流、温度轨迹，形状 `[batch, seq_len, 3]`。
- **目标**：未来某循环的容量（capacity）或 SOH。
- **索引**：每个样本还需携带当前循环索引与未来目标循环索引，用于构造预测任务。
- **缺失模拟**：在原始轨迹上按论文 Scenario 1/2 模拟随机缺失和连续缺失，再使用 `torchcde` 自然三次样条插值。

## 2. 模型约定

- 核心模型文件：`src/models/battery_cde.py`。
- 模型由三个 CDE 子网络组成：
  1. `feature_cde`：输出 feature attention，维度 `[batch, seq_len, input_dim]`。
  2. `cycle_cde`：输出 cycle attention，维度 `[batch, seq_len, 1]`。
  3. `learning_cde`：对加权后的路径积分，输出最终隐状态。
- 注意力使用 sigmoid 门控，与论文一致。
- 初始隐状态由序列平均观测投影得到。

## 3. 训练约定

- 损失函数：MSE（容量预测）。
- 优化器：Adam，默认 lr=1e-3。
- 早停：patience=15，监控 validation MAE。
- 每个训练/验证/测试划分按电池（battery）进行，不跨电池。
- 预测 horizon 作为配置项，训练时一次性预测 `horizon` 步后的容量。

## 4. 评估约定

- 主指标：MAE（与论文对齐），同时记录 RMSE、R²。
- 对多个 horizon（20, 50, 80, 100, 200, 400, 600, 800, 1000）分别评估并绘制曲线。
- 缺失数据实验报告不同缺失率（30%、50%）和缺失类型（随机、连续）下的 MAE。

## 5. 与主项目的关系

- 可复用：
  - `src/utils/seed_manager.py` 固定种子。
  - `src/evaluators/metrics.py` 计算指标。
  - `src/data/dataset_registry.py` 获取数据集元信息。
- 不复用：
  - 主项目的 16 维 handcrafted 特征与本复现的原始轨迹输入不同，因此数据加载独立实现。
  - 主项目的缺失模式（bernoulli / block / channel 等）定义在“循环维度”，本复现缺失定义在“原始时间序列维度”，两者不可直接混用。

## 6. 新增依赖

- `torchcde` 必须单独安装：
  ```bash
  pip install torchcde
  ```
- 若后续引入 `torchdiffeq` 或其他微分方程库，需在 `requirements.txt` 中补充。

## 7. 测试要求

- `tests/test_battery_cde_forward.py`：验证 BatteryCDE 输出形状正确。
- `tests/test_spline_interpolation.py`：验证 `torchcde.natural_cubic_coeffs` 能处理含 NaN 的输入（缺失位置）。
- 每次修改模型后必须跑通这两个测试。
