# Batch Experiment Framework

## Overview

This framework manages large-scale experiments (e.g., 7200 experiments for paper reproduction), supporting:
- **Experiment status tracking**: pending/running/completed/failed
- **Parallel execution**: GPU + CPU mixed scheduling
- **Resume capability**: Support for interruption and recovery
- **Auto-retry**: Automatic retry of failed experiments
- **Result collection**: Automatic statistical charts generation

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
conda activate battery-nn

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### 2. Initialize Experiments

```bash
# Mini test (4 experiments, quick verification)
python scripts/run_experiments.py init --mini

# Small test (24 experiments: 2 models x 2 methods x 2 MR x 3 seeds)
python scripts/run_experiments.py init --test

# Full paper reproduction (7200 experiments)
python scripts/run_experiments.py init --all
```

### 3. Run Experiments

```bash
# Using GPU (recommended: 2 GPU workers + 2 CPU workers)
python scripts/run_experiments.py run --gpu-workers 2 --cpu-workers 2

# CPU only (slower, for testing)
python scripts/run_experiments.py run --no-gpu --cpu-workers 4

# Single batch only
python scripts/run_experiments.py run --once
```

### 4. Monitor Progress

```bash
# Check status
python scripts/run_experiments.py status

# Real-time monitoring (in another terminal)
watch -n 30 "python scripts/run_experiments.py status"
```

### 5. Handle Failures

```bash
# Manual retry of failed experiments
python scripts/run_experiments.py retry

# Recover from crash (reset running to pending)
python scripts/run_experiments.py recover
```

### 6. Collect Results

```bash
# Generate summary
python scripts/collect_results.py --output results/summary.csv

# Generate charts
python scripts/plot_results.py --output-dir results/figures/
```

## Configuration

### Scheduler Configuration

Located in `src/experiments/scheduler.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| gpu_workers | 2 | Number of GPU parallel processes |
| cpu_workers | 2 | Number of CPU parallel processes |
| max_retries | 3 | Max retry attempts for failed experiments |
| retry_delay_seconds | 5 | Delay between retry attempts |
| report_interval_seconds | 60 | Status report interval |

### Run Configuration

Located in `configs/paper/run_*.yaml`:
- `run_baseline.yaml`: Baseline method configuration
- `run_mim.yaml`: MIM method configuration

## File Structure

```
experiments/
├── experiment_db.csv     # Experiment database (CSV format)
└── logs/                 # Experiment logs
    ├── mlp_baseline_mr0_5_seed42.log
    └── ...

results/
├── *.csv                 # Experiment results
├── summary.csv           # Summary results
└── figures/              # Charts
    ├── fig_mr_mae_curves.png
    ├── fig_mim_improvement.png
    └── fig_model_comparison.png
```

## Time Estimation

Based on typical hardware (8GB GPU):

| Scale | Experiments | Estimated Time (GPU) | Estimated Time (CPU) |
|-------|-------------|---------------------|---------------------|
| Mini | 4 | ~2 min | ~5 min |
| Test | 24 | ~15 min | ~40 min |
| Full | 7200 | ~20 hours | ~5 days |

## Troubleshooting

### Issue: GPU Out of Memory

Reduce GPU workers:
```bash
python scripts/run_experiments.py run --gpu-workers 1 --cpu-workers 3
```

### Issue: Process Stuck

Terminate and recover:
```bash
# Ctrl+C to terminate
python scripts/run_experiments.py recover
python scripts/run_experiments.py run
```

### Issue: Many Experiments Fail

Check logs and retry:
```bash
# View failed logs
ls experiments/logs/
cat experiments/logs/failed_exp.log

# Manual retry
python scripts/run_experiments.py retry
```

## Architecture Notes

### Database Safety

The scheduler uses **CSV format** for easy analysis in Excel/R/Python. Database operations are centralized in the main process to avoid multi-process race conditions. Workers only run experiments and return results.

### Auto-Retry Mechanism

Failed experiments are automatically retried up to `max_retries` times. If an experiment consistently fails, it likely indicates a bug in the code rather than a transient error.

### Process Model

```
Main Process:
  - Read/Write CSV database
  - Dispatch experiments to workers
  - Update experiment status

Worker Processes (via ProcessPoolExecutor):
  - Run single experiment
  - Return results to main process
  - No direct database access
```

## Next Steps

After completing small-scale tests (24 experiments):
1. **Full-scale experiment**: Initialize and run all 7200 experiments
2. **Result analysis**: Compare with paper figures, assess reproduction quality
3. **Architecture upgrade**: Proceed to Phase 4 modular refactoring
