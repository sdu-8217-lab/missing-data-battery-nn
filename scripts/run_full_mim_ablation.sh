#!/bin/bash

# ============================================================================
# MIM Ablation Study - Full Experiment Batch Runner (Linux/macOS)
# ============================================================================
# 运行矩阵: 3方法 × 3MR × 20seeds = 180次实验
# 方法: Baseline / MIM-Single / MIM-Multi
# MR: 0.3 / 0.6 / 0.9
# Seeds: 42, 101-119
# ============================================================================

PYTHON="python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/batch_run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/results/csv"

echo "============================================================================"
echo "MIM Ablation Study - Full Experiment Batch"
echo "Start: $(date)"
echo "Log: $LOG_DIR"
echo "============================================================================"
echo ""

# 定义seeds数组
SEEDS=(42 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119)
MRS=(0.3 0.6 0.9)
TOTAL=180
COUNT=0

# 失败记录文件
FAILURES_LOG="$LOG_DIR/failures.log"

# ============================================================================
# 函数: 运行单次实验
# 参数: $1=方法名 $2=配置名 $3=MR $4=seed $5=日志前缀
# ============================================================================
run_experiment() {
    local method=$1
    local config=$2
    local mr=$3
    local seed=$4
    local prefix=$5
    
    ((COUNT++))
    echo "[$COUNT/$TOTAL] $method MR=$mr Seed=$seed"
    
    # 修正: experiments.training.seeds (不是 training.seeds)
    $PYTHON "$PROJECT_ROOT/src/main.py" experiments=$config experiments.training.seeds=[$seed] \
        > "$LOG_DIR/${prefix}_mr${mr}_seed${seed}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  [OK]"
        return 0
    else
        echo "  [FAIL] See $LOG_DIR/${prefix}_mr${mr}_seed${seed}.log"
        echo "[$COUNT/$TOTAL] [FAIL] $method MR=$mr Seed=$seed" >> "$FAILURES_LOG"
        return 1
    fi
}

# ============================================================================
# 1. Baseline Experiments
# ============================================================================
echo "[Phase 1/3] Baseline Experiments (MR=0.3, 0.6, 0.9)"
echo "----------------------------------------------------------------------------"

for mr in "${MRS[@]}"; do
    echo ""
    echo "[Baseline] MR=$mr"
    for seed in "${SEEDS[@]}"; do
        run_experiment "Baseline" "mar_$mr" "$mr" "$seed" "baseline"
    done
done

# ============================================================================
# 2. MIM-Single Experiments
# ============================================================================
echo ""
echo "[Phase 2/3] MIM-Single Experiments (MR=0.3, 0.6, 0.9)"
echo "----------------------------------------------------------------------------"

for mr in "${MRS[@]}"; do
    echo ""
    echo "[MIM-Single] MR=$mr"
    for seed in "${SEEDS[@]}"; do
        run_experiment "MIM-Single" "mim_mar_$mr" "$mr" "$seed" "mim_single"
    done
done

# ============================================================================
# 3. MIM-Multi Experiments
# ============================================================================
echo ""
echo "[Phase 3/3] MIM-Multi Experiments (MR=0.3, 0.6, 0.9)"
echo "----------------------------------------------------------------------------"

for mr in "${MRS[@]}"; do
    echo ""
    echo "[MIM-Multi] MR=$mr"
    for seed in "${SEEDS[@]}"; do
        run_experiment "MIM-Multi" "mim_mar_${mr}_corrected" "$mr" "$seed" "mim_multi"
    done
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "Batch Run Completed"
echo "End: $(date)"
echo "Log Directory: $LOG_DIR"
echo ""

if [ -f "$FAILURES_LOG" ]; then
    echo "[WARNING] Some experiments failed. See $FAILURES_LOG"
    cat "$FAILURES_LOG"
else
    echo "[SUCCESS] All experiments completed successfully!"
fi

echo ""
echo "Results directory: $PROJECT_ROOT/results/csv/"
echo "============================================================================"
