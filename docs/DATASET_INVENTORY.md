# 数据集清单 (Dataset Inventory)

> 项目: missing-data-battery-nn  
> 更新日期: 2026-03-08

---

## 📊 数据集概览

| 数据集 | 类型 | Batches/子集 | 电池总数 | 推荐优先级 |
|--------|------|-------------|---------|-----------|
| **XJTU** | 充放电循环 | 6 batches | 55个 | ⭐⭐⭐ 最高 |
| **HUST** | 充放电循环 | 10个电池组 | 77个 | ⭐⭐⭐ 高 |
| **TJU** | 材料类型 | 3个子集 | 130个 | ⭐⭐ 中 |
| **MIT** | 时间批次 | 3个日期 | 125个 | ⭐⭐ 中 |

**总计**: 387个电池文件

---

## 1. XJTU 数据集 (西安交大)

### 1.1 数据集特点
- **数据类型**: 恒流充放电循环数据
- **实验条件**: 不同放电倍率(C-rate)和工况
- **采样方式**: 连续循环测试

### 1.2 可用 Batches

| Batch | 电池数 | 说明 | 推荐 |
|-------|--------|------|------|
| **2C** | 8 | 2C倍率放电 | ✅ 推荐 |
| **3C** | 15 | 3C倍率放电 | ✅ 推荐 |
| **R2.5** | 8 | 2.5C倍率 | ⚠️ 可选 |
| **R3** | 8 | 3C倍率(重复) | ⚠️ 可选 |
| **RW** | 8 | 随机游走工况 | ⚠️ 可选 |
| **Sim_satellite** | 8 | 卫星仿真工况 | ⚠️ 可选 |

### 1.3 文件命名规范
```
{batch}_battery-{id}.csv

示例:
- 2C_battery-1.csv
- 3C_battery-10.csv
- R2.5_battery-3.csv
- Sim_satellite_battery-5.csv
```

### 1.4 推荐实验配置
```python
# 主实验推荐配置
dataset = "xjtu"
batches = ["2C", "3C"]  # 23个电池

# 扩展实验配置
batches = ["2C", "3C", "R2.5", "R3", "RW"]  # 47个电池

# 全量实验配置
batches = ["2C", "3C", "R2.5", "R3", "RW", "Sim_satellite"]  # 55个电池
```

---

## 2. HUST 数据集 (华中科技大学)

### 2.1 数据集特点
- **数据类型**: 恒流充放电循环
- **电池类型**: 商用18650锂离子电池
- **实验条件**: 不同温度和使用条件

### 2.2 电池分组

| 电池组 | 电池数 | 说明 |
|--------|--------|------|
| **1** | 8 | Batch 1 |
| **2** | 7 | Batch 2 |
| **3** | 8 | Batch 3 |
| **4** | 8 | Batch 4 |
| **5** | 7 | Batch 5 |
| **6** | 7 | Batch 6 |
| **7** | 8 | Batch 7 |
| **8** | 8 | Batch 8 |
| **9** | 8 | Batch 9 |
| **10** | 8 | Batch 10 |

**总计**: 77个电池文件

### 2.3 文件命名规范
```
{batch}-{battery}.csv

示例:
- 1-1.csv  (Batch 1, Battery 1)
- 1-8.csv  (Batch 1, Battery 8)
- 10-3.csv (Batch 10, Battery 3)
```

### 2.4 推荐实验配置
```python
# 推荐使用前5个batch
dataset = "hust"
batches = ["1", "2", "3", "4", "5"]  # 38个电池

# 全量实验
batches = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
```

---

## 3. TJU 数据集 (天津大学)

### 3.1 数据集特点
- **数据类型**: 不同正极材料的电池数据
- **材料类型**: NCA、NCM、NCM+NCA混合

### 3.2 子集划分

| 子集 | 电池数 | 材料类型 |
|------|--------|----------|
| **Dataset_1_NCA** | 66 | NCA (镍钴铝) |
| **Dataset_2_NCM** | 55 | NCM (镍钴锰) |
| **Dataset_3_NCM_NCA** | 9 | NCM+NCA混合 |

**总计**: 130个电池文件

### 3.3 文件命名规范
```
Dataset_{N}_{material}/{test_id}_{battery_id}-#{channel}.csv

示例:
- CY25-025_1-#1.csv
- CY25-025_1-#2.csv
```

### 3.4 推荐实验配置
```python
# 按材料类型实验
dataset = "tju"
batches = ["Dataset_1_NCA_battery"]      # 66个电池
batches = ["Dataset_2_NCM_battery"]      # 55个电池

# 对比实验 (NCA vs NCM)
batches = ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery"]
```

---

## 4. MIT 数据集 (麻省理工)

### 4.1 数据集特点
- **数据类型**: 快速充放电测试
- **时间跨度**: 2017-2018年多批次测试
- **文件组织**: 按测试日期分文件夹

### 4.2 时间批次

| 日期批次 | 电池数 | 测试时间 |
|----------|--------|----------|
| **2017-05-12** | 46 | 2017年5月12日 |
| **2017-06-30** | 43 | 2017年6月30日 |
| **2018-04-12** | 36 | 2018年4月12日 |

**总计**: 125个电池文件

### 4.3 文件命名规范
```
{date}/{date}_battery-{id}.csv

示例:
- 2017-05-12/2017-05-12_battery-1.csv
- 2018-04-12/2018-04-12_battery-30.csv
```

### 4.4 推荐实验配置
```python
# 跨时间泛化实验
dataset = "mit"
batches = ["2017-05-12", "2017-06-30", "2018-04-12"]

# 单批次实验
batches = ["2017-05-12"]  # 46个电池
```

---

## 🎯 推荐实验策略

### 策略1: 单一数据集深入 (推荐起步)

**XJTU 2C + 3C batch**
```python
# 实验规模
dataset = "xjtu"
batches = ["2C", "3C"]
# 23个电池 × 100 seeds × 4 models × 2 methods = 18,400 runs
```

**优势**:
- 数据同质性好（相同实验条件）
- 23个电池提供足够的训练/测试分割
- 2C和3C可以对比放电倍率的影响

### 策略2: 跨数据集验证 (论文级别)

**主实验: XJTU 2C + 3C**
```python
runs = 23 × 100 × 4 × 2 = 18,400
```

**验证集: HUST + TJU + MIT**
```python
# HUST
runs_hust = 38 × 50 × 4 × 2 = 15,200

# TJU
runs_tju = 55 × 50 × 4 × 2 = 22,000

# MIT
runs_mit = 46 × 50 × 4 × 2 = 18,400
```

### 策略3: 全面泛化验证

**所有数据集参与**
```
XJTU (2C,3C):      18,400 runs
HUST (1-10):       30,800 runs  
TJU (NCA,NCM):     48,400 runs
MIT (all dates):   55,200 runs
────────────────────────────────
总计: ~152,800 runs
```

⚠️ **注意**: 此规模需要大量计算资源，建议仅在验证方法普适性时使用

---

## 📋 配置示例

### 实验配置 - XJTU 2C
```bash
python scripts/run_experiments.py init \
    --dataset xjtu \
    --batch 2C \
    --seeds 42-141
```

### 实验配置 - XJTU 多batch
```bash
python scripts/run_experiments.py init \
    --dataset xjtu \
    --batch 2C 3C R2.5 R3 \
    --seeds 42-141
```

### 实验配置 - HUST
```bash
python scripts/run_experiments.py init \
    --dataset hust \
    --batch 1 2 3 4 5 \
    --seeds 42-141
```

### 实验配置 - TJU (NCA)
```bash
python scripts/run_experiments.py init \
    --dataset tju \
    --batch Dataset_1_NCA_battery \
    --seeds 42-141
```

### 实验配置 - MIT
```bash
python scripts/run_experiments.py init \
    --dataset mit \
    --batch 2017-05-12 \
    --seeds 42-141
```

---

## 🔍 数据质量评估

| 数据集 | 数据完整性 | 特征丰富度 | 循环数 | 推荐用途 |
|--------|-----------|-----------|--------|----------|
| XJTU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 多 | 主实验/基准测试 |
| HUST | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中等 | 跨数据集验证 |
| TJU | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 多 | 材料类型泛化 |
| MIT | ⭐⭐⭐ | ⭐⭐⭐ | 少 | 快速验证 |

---

## 📝 备注

1. **XJTU数据**命名中的"C"表示放电倍率（C-rate）
   - 2C = 2倍率放电（30分钟放完）
   - 3C = 3倍率放电（20分钟放完）

2. **TJU数据**按正极材料分类
   - NCA: 镍钴铝酸锂（高能量密度）
   - NCM: 镍钴锰酸锂（综合性能好）

3. **MIT数据**按测试日期组织
   - 适合验证方法的跨时间稳定性

4. **文件大小参考**
   - XJTU: ~100-200 KB/文件
   - HUST: ~400-800 KB/文件
   - TJU: ~50-200 KB/文件
   - MIT: ~100-500 KB/文件
