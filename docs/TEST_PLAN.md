# P1 Architecture Test Plan

## Environment Requirements

```bash
# Required dependencies
pip install numpy pandas matplotlib torch pytorch-lightning hydra-core scikit-learn
```

## Test 1: Small-Scale Experiment Run

### Objective
Verify `run_experiment.py` works with P1 directory structure.

### Configuration
```yaml
# Test matrix (8 experiments total)
Batches: [2C, 3C]
Seeds: [42, 123]
Missing Rates: [0.3, 0.7]
Models: [mlp]
Methods: [baseline, mim]
Mode: mar
```

### Steps

```bash
# 1. Run baseline experiment
python experiments/run_experiment.py \
    experiment=youth_mar_baseline \
    data.batch_id=2C \
    model.type=mlp \
    missing.rate_eval=0.3 \
    training.seeds=[42,123]

# 2. Run MIM experiment
python experiments/run_experiment.py \
    experiment=youth_mar_mim \
    data.batch_id=2C \
    model.type=mlp \
    missing.rate_eval=0.3 \
    training.seeds=[42,123]

# 3. Verify results directory created
ls -la results/2026*_2C/
# Expected:
#   results.csv
#   figures/

# 4. Check CSV content
cat results/2026*_2C/results.csv
```

### Expected Results
- Directory: `results/{timestamp}_2C/`
- CSV contains columns: seed, model, method, missing_rate, test_mae, test_rmse, test_r2
- 4 rows (2 seeds × 2 methods)

## Test 2: Batch Experiment Runner

### Objective
Verify `run_batch_experiments.py` orchestrates multiple batches.

### Steps

```bash
# Dry run first
python experiments/run_batch_experiments.py \
    --mode mar \
    --method baseline \
    --batches 2C 3C \
    --missing-rates 0.3 \
    --seeds 42 \
    --dry-run

# Actual run (if dry run looks correct)
python experiments/run_batch_experiments.py \
    --mode mar \
    --method baseline \
    --batches 2C 3C \
    --missing-rates 0.3 0.7 \
    --seeds 42 123
```

### Expected Results
- Directories created:
  - `results/{timestamp}_2C/`
  - `results/{timestamp}_3C/`
- Both have `results.csv` with 4 rows each
- Same timestamp prefix for both directories

## Test 3: Results Aggregation

### Objective
Verify `aggregate_results.py` combines multiple batches.

### Prerequisites
Complete Test 2 to have data to aggregate.

### Steps

```bash
# Aggregate with statistics
python scripts/aggregate_results.py --stats

# Or with specific timestamp
python scripts/aggregate_results.py \
    --timestamp 20260318 \
    --batches 2C 3C \
    --stats
```

### Expected Results
- Output directory: `results/aggregated/{timestamp}/`
- Files created:
  - `raw_results.csv` - all rows from all batches
  - `statistics.csv` - mean/std per (batch, model, method, mr) group

## Test 4: Standalone Plotting

### Objective
Verify `plot_csv.py` generates figures from CSV files.

### Prerequisites
Have results CSV from Test 1 or Test 2.

### Steps

```bash
# Plot from specific CSV
python scripts/plot_csv.py results/{timestamp}_2C/results.csv

# Plot from latest results for batch
python scripts/plot_csv.py --batch 2C

# Generate specific figure
python scripts/plot_csv.py \
    results/{timestamp}_2C/results.csv \
    --figure mr_mae \
    --output-dir my_figures/
```

### Expected Results
- Figures directory created (default: `results/{timestamp}_2C/figures/`)
- PNG files generated:
  - `fig_mr_mae_curves.png` - MR-MAE curves per model
  - `fig_mim_improvement.png` - MIM improvement bar chart
  - `fig_summary_table.png` - statistics table

## Test 5: MNAR Mode

### Objective
Verify MNAR missing pattern works end-to-end.

### Configuration
Create `configs/experiments/youth_mnar_baseline.yaml`:

```yaml
# @package _global_
experiment: youth_mnar_baseline
method: baseline

data:
  batch_id: 2C
  data_dir: data/XJTU

model:
  type: mlp

missing:
  mode: mnar
  rate_eval: 0.3
  alpha: null  # auto-calculate
  beta: 0.05
  feature_index: 0

training:
  seeds: [42]
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

hydra:
  run:
    dir: outputs/${experiment}/${now:%Y%m%d_%H%M%S}
```

### Steps

```bash
python experiments/run_experiment.py \
    experiment=youth_mnar_baseline \
    data.batch_id=2C \
    model.type=mlp \
    missing.rate_eval=0.5
```

### Expected Results
- Experiment completes without error
- CSV shows `missing_mode=mnar`

## Automated Smoke Test

Create a quick smoke test script:

```bash
#!/bin/bash
# tests/smoke_test_p1.sh

set -e

echo "=== P1 Architecture Smoke Test ==="

# Test 1: Path module
python -c "from src.utils.paths import *; print('paths OK')"

# Test 2: Single experiment (1 seed, 1 epoch for speed)
python experiments/run_experiment.py \
    experiment=youth_mar_baseline \
    data.batch_id=2C \
    model.type=mlp \
    missing.rate_eval=0.3 \
    training.seeds=[42] \
    training.epochs=1

# Test 3: Check results created
RESULTS_DIR=$(ls -d results/*_2C | tail -1)
echo "Results dir: $RESULTS_DIR"
test -f "$RESULTS_DIR/results.csv" || exit 1

# Test 4: Plot results
python scripts/plot_csv.py "$RESULTS_DIR/results.csv" --figure summary

echo "=== All tests passed ==="
```

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Install missing dependencies
pip install -r requirements.txt
```

### Issue: No data files
```bash
# Check data directory structure
ls -la data/XJTU/
```

### Issue: CUDA out of memory
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Hydra config not found
```bash
# Verify config structure
ls configs/experiments/
cat configs/experiments/youth_mar_baseline.yaml
```
