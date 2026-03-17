# P1 Architecture Refactoring

## Overview

P1重构将实验执行与可视化解耦，支持按批次组织结果。

## Directory Structure

```
results/
├── {timestamp}_{batch_id}/          # 批次特定结果目录
│   ├── results.csv                  # 该批次的所有实验结果
│   └── figures/                     # 该批次的图表
│       ├── fig_mr_mae_curves.png
│       ├── fig_mim_improvement.png
│       └── fig_summary_table.png
├── aggregated/                      # 聚合结果（可选）
│   └── {timestamp}/
│       ├── raw_results.csv
│       └── statistics.csv
```

## New Components

### 1. Path Management (`src/utils/paths.py`)

Centralized path management for batch-specific experiments.

```python
from src.utils.paths import get_results_dir, ensure_results_structure

# Get results directory for a batch
results_dir = get_results_dir("2C", "20260318_120000")
# Returns: Path("results/20260318_120000_2C")

# Ensure directory structure exists
paths = ensure_results_structure(results_dir)
# Returns: {"results_dir": ..., "figures_dir": ..., "csv_path": ...}
```

### 2. Standalone Plotting (`scripts/plot_csv.py`)

Decoupled visualization - can re-run on existing CSV files.

```bash
# Plot from specific CSV
python scripts/plot_csv.py results/20260318_120000_2C/results.csv

# Plot from latest results for a batch
python scripts/plot_csv.py --batch 2C

# Generate specific figure
python scripts/plot_csv.py results/xxx/results.csv --figure mr_mae
```

### 3. Batch Experiment Runner (`experiments/run_batch_experiments.py`)

Run experiments across all 6 XJTU batches with consistent timestamp.

```bash
# Run all batches with MAR mode and MIM method
python experiments/run_batch_experiments.py --mode mar --method mim

# Run specific batches
python experiments/run_batch_experiments.py --batches 2C 3C --mode mcar

# Dry run (show what would be executed)
python experiments/run_batch_experiments.py --dry-run
```

### 4. Results Aggregation (`scripts/aggregate_results.py`)

Combine results from multiple batch experiments.

```bash
# Aggregate all batches with same timestamp
python scripts/aggregate_results.py --timestamp 20260318_120000

# Aggregate specific batches
python scripts/aggregate_results.py --timestamp 20260318_120000 --batches 2C 3C

# Generate statistics
python scripts/aggregate_results.py --timestamp 20260318_120000 --stats
```

## Usage Workflow

### Single Batch Experiment

```bash
# Run experiment for single batch
python experiments/run_experiment.py \
    experiment=youth_mar_mim \
    data.batch_id=2C \
    model.type=cnn \
    missing.rate_eval=0.3

# Results saved to: results/{timestamp}_2C/results.csv
# Generate plots from results
python scripts/plot_csv.py --batch 2C
```

### Full Matrix Experiment (20 seeds × 6 batches × 3 modes × 6 methods × 4 models)

```bash
# Step 1: Run experiments for all batches with consistent timestamp
python experiments/run_batch_experiments.py --mode mar --method mim --missing-rates 0.1 0.3 0.5 0.7 0.9

# Step 2: Aggregate results
python scripts/aggregate_results.py --stats

# Step 3: Generate plots from aggregated results
python scripts/plot_csv.py results/aggregated/{timestamp}/raw_results.csv
```

### Re-generate Plots from Existing Results

```bash
# Find and plot latest results for a batch
python scripts/plot_csv.py --batch 2C --figure all

# Plot specific figure from CSV
python scripts/plot_csv.py results/20260318_120000_2C/results.csv --figure mr_mae
```

## Migration from Old Structure

### Old (P0)
```
results/
├── csv/
│   ├── youth_mar_mim.csv
│   └── youth_mar_baseline.csv
└── figures/
```

### New (P1)
```
results/
├── {timestamp}_2C/
│   ├── results.csv
│   └── figures/
├── {timestamp}_3C/
│   ├── results.csv
│   └── figures/
...
```

## XJTU Batch Identifiers

- `2C` - 2C charge/discharge
- `3C` - 3C charge/discharge
- `R2.5` - Random 2.5 (normalized)
- `R3` - Random 3 (normalized)
- `RW` - Random walk
- `Sim_satellite` - Simulated satellite

## Configuration

The batch experiment runner uses existing Hydra configs:
- `configs/experiments/youth_mar_mim.yaml`
- `configs/experiments/youth_mar_baseline.yaml`
- `configs/experiments/youth_mcar_mim.yaml`
- `configs/experiments/youth_mcar_baseline.yaml`

## Backward Compatibility

- `run_experiment.py` maintains same CLI interface
- Existing configs work without modification
- Old results in `results/csv/` remain accessible
