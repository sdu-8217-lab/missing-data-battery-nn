# 实验文件结构规范

> 版本: v1.0  
> 日期: 2026-03-08  
> 目的: 规范多次大量实验的文件存放，确保可管理、可追溯、不冲突

---

## 📁 顶层目录结构

```
missing-data-battery-nn/
├── configs/                    # Hydra配置（版本控制）
├── data/                       # 原始数据集（只读）
├── docs/                       # 文档（版本控制）
├── experiments/                # 实验运行时数据（大文件，不版本控制）
│   ├── runs/                   # 时间戳组织的实验运行
│   ├── aggregated/             # 汇总分析结果
│   └── archive/                # 归档的历史实验
├── outputs/                    # Hydra默认输出（临时）
├── paper/                      # 论文相关文件
├── results/                    # 结果汇总（旧结构，逐步迁移）
├── src/                        # 源代码（版本控制）
└── tests/                      # 测试代码
```

---

## 📂 核心目录详解

### 1. experiments/ - 实验运行时数据

```
experiments/
├── .gitignore                  # 忽略所有子目录内容
├── README.md                   # 目录说明
│
├── runs/                       # 实验运行目录（核心）
│   ├── {YYYYMMDD_HHMMSS}/      # 时间戳命名的单次实验
│   │   ├── experiment_db.csv   # 实验状态数据库
│   │   ├── config_snapshot.yaml # 配置快照
│   │   ├── logs/               # 训练日志
│   │   │   ├── {exp_id}.log
│   │   │   └── ...
│   │   ├── results/            # 实验结果CSV
│   │   │   ├── {exp_id}.csv
│   │   │   └── ...
│   │   ├── checkpoints/        # 模型检查点（可选）
│   │   │   └── {exp_id}/
│   │   └── figures/            # 中间图表（可选）
│   │       └── ...
│   └── ...
│
├── latest -> runs/{timestamp}/ # 软链接指向最新实验
│
├── aggregated/                 # 汇总分析结果
│   ├── cross_dataset/          # 跨数据集汇总
│   ├── cross_batch/            # 跨批次汇总
│   ├── robustness_curves/      # 鲁棒性曲线
│   └── statistics/             # 统计分析
│
└── archive/                    # 归档目录
    └── {YYYY-MM}/              # 按月归档
```

#### 命名规范

| 元素 | 格式 | 示例 |
|------|------|------|
| 时间戳目录 | `YYYYMMDD_HHMMSS` | `20260308_143052` |
| 实验ID | `{dataset}_{batch}_seed{seed}_{model}_{method}` | `xjtu_2C_seed42_mlp_baseline` |
| 日志文件 | `{exp_id}.log` | `xjtu_2C_seed42_mlp_baseline.log` |
| 结果文件 | `{exp_id}.csv` | `xjtu_2C_seed42_mlp_baseline.csv` |

---

### 2. outputs/ - Hydra默认输出

```
outputs/
├── {YYYY-MM-DD}/               # 日期目录
│   ├── {HH-MM-SS}/             # 时间目录
│   │   ├── .hydra/             # Hydra配置
│   │   │   ├── config.yaml
│   │   │   ├── hydra.yaml
│   │   │   └── overrides.yaml
│   │   └── ...                 # 其他输出
│   └── ...
└── ...
```

**注意**: 这是Hydra的默认输出目录，用于单次运行。批量实验使用 `experiments/runs/`。

---

### 3. results/ - 结果汇总（旧结构）

```
results/
├── battery_soh_experiment_{model}_{method}.csv  # 旧格式结果
└── archive/                                     # 归档
```

**迁移计划**: 逐步迁移到 `experiments/runs/{timestamp}/results/`

---

## 📊 文件类型与大小预估

### 单实验文件大小

| 文件类型 | 大小 | 说明 |
|----------|------|------|
| 日志文件 (.log) | 500KB - 2MB | 包含训练和评估输出 |
| 结果CSV (.csv) | 5KB - 20KB | 10档MR结果 |
| 模型检查点 (.ckpt) | 1MB - 5MB | 可选保存 |
| 配置文件 (.yaml) | 1KB - 5KB | 配置快照 |

### 大规模实验存储预估

| 实验规模 | 日志总量 | 结果总量 | 总计 |
|----------|---------|---------|------|
| 800 runs | ~800MB | ~10MB | ~810MB |
| 18,400 runs (XJTU) | ~18GB | ~200MB | ~18.2GB |
| 152,800 runs (全量) | ~150GB | ~1.5GB | ~151.5GB |

**存储建议**:
- 定期清理旧日志（保留汇总结果）
- 模型检查点按需保存
- 大实验前确认磁盘空间 > 50GB

---

## 🔧 自动清理策略

### 1. 日志自动归档

```python
# scripts/archive_experiments.py

def archive_old_experiments(days=30):
    """归档30天前的实验"""
    for run_dir in Path("experiments/runs").glob("*"):
        timestamp = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
        if (datetime.now() - timestamp).days > days:
            # 只保留数据库和结果，删除日志
            archive_logs(run_dir)
```

### 2. 磁盘空间监控

```python
# 在scheduler中集成
import shutil

def check_disk_space(min_gb=10):
    """检查磁盘空间"""
    stat = shutil.disk_usage("experiments/")
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        warnings.warn(f"磁盘空间不足: {free_gb:.1f}GB < {min_gb}GB")
```

### 3. 自动删除策略

```bash
# 保留最近30天的完整数据
# 保留最近90天的结果+数据库
# 删除所有日志（保留汇总）

# crontab设置
0 2 * * * cd /path/to/project && python scripts/cleanup.py --days=30
```

---

## 📋 文件创建规范

### 1. 初始化时创建

```python
def create_experiment_structure(timestamp: str) -> Path:
    """创建实验目录结构"""
    run_dir = Path(f"experiments/runs/{timestamp}")
    
    # 创建子目录
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    
    # 创建.gitkeep
    for subdir in ["logs", "results", "checkpoints", "figures"]:
        (run_dir / subdir / ".gitkeep").touch()
    
    return run_dir
```

### 2. 运行时创建

```python
# 日志文件
log_file = run_dir / f"logs/{exp_id}.log"

# 结果文件
results_file = run_dir / f"results/{exp_id}.csv"

# 检查点（可选）
checkpoint_dir = run_dir / f"checkpoints/{exp_id}"
```

### 3. 汇总时创建

```python
# 跨实验汇总
aggregated_dir = Path("experiments/aggregated")
(aggregated_dir / "cross_dataset").mkdir(parents=True, exist_ok=True)
(aggregated_dir / "robustness_curves").mkdir(exist_ok=True)
```

---

## 🔗 文件关联关系

```
experiments/runs/{timestamp}/
│
├── experiment_db.csv          # 主索引文件
│   └── 记录所有实验的状态和路径
│
├── logs/{exp_id}.log          # 详细日志
│   └── 与数据库中的log_file字段对应
│
├── results/{exp_id}.csv       # 结果数据
│   └── 10档MR的详细结果
│
└── config_snapshot.yaml       # 配置快照
    └── 记录本次实验的完整配置
```

---

## 🛡️ 冲突避免机制

### 1. 时间戳隔离

- 每次 `init` 自动生成唯一时间戳
- 不同实验完全隔离
- 防止结果覆盖

### 2. 数据库唯一性

```python
exp_id = f"{dataset}_{batch}_seed{seed}_{model}_{method}"
# 确保: dataset + batch + seed + model + method 组合唯一
```

### 3. 文件锁机制

```python
# 防止同时写入
from filelock import FileLock

with FileLock("experiments/runs/{timestamp}/experiment_db.csv.lock"):
    update_database()
```

---

## 📊 查询和汇总

### 1. 列出所有实验

```bash
# 列出所有时间戳
ls -la experiments/runs/

# 查看最新实验
ls -la experiments/latest/

# 查看特定实验
ls -la experiments/runs/20260308_143052/
```

### 2. 汇总结果

```python
# 汇总所有已完成实验
import pandas as pd
from pathlib import Path

all_results = []
for run_dir in Path("experiments/runs").glob("*"):
    db_file = run_dir / "experiment_db.csv"
    if db_file.exists():
        df = pd.read_csv(db_file)
        all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("experiments/aggregated/all_experiments.csv", index=False)
```

### 3. 生成报告

```python
# 生成实验统计报告
def generate_report():
    stats = {
        "total_runs": count_runs(),
        "completed": count_completed(),
        "failed": count_failed(),
        "disk_usage": get_disk_usage(),
    }
    with open("experiments/REPORT.json", "w") as f:
        json.dump(stats, f, indent=2)
```

---

## 🧹 维护工具

### 1. 清理脚本

```bash
#!/bin/bash
# scripts/cleanup.sh

# 删除30天前的日志
find experiments/runs/*/logs -name "*.log" -mtime +30 -delete

# 删除失败实验的检查点
find experiments/runs/*/checkpoints -type d -empty -delete

# 压缩旧结果
find experiments/runs/*/results -name "*.csv" -mtime +90 -exec gzip {} \;
```

### 2. 备份脚本

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/battery-nn/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# 备份数据库
rsync -av experiments/runs/*/experiment_db.csv "$BACKUP_DIR/"

# 备份结果
rsync -av experiments/runs/*/results/ "$BACKUP_DIR/results/"

# 备份汇总
rsync -av experiments/aggregated/ "$BACKUP_DIR/aggregated/"
```

### 3. 监控脚本

```bash
#!/bin/bash
# scripts/monitor.sh

# 磁盘空间监控
DISK_USAGE=$(df -h experiments/ | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "WARNING: Disk usage ${DISK_USAGE}% > 90%" | mail -s "Battery-NN Disk Alert" admin@example.com
fi

# 实验进度监控
python scripts/report_progress.py
```

---

## 📚 相关文档

- [实验设计规范](EXPERIMENT_DESIGN.md)
- [数据集清单](DATASET_INVENTORY.md)
- [优化总结报告](OPTIMIZATION_SUMMARY.md)

---

## 📝 更新记录

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-03-08 | 初始版本，规范实验文件结构 |
