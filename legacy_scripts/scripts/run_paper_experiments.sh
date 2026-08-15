#!/usr/bin/env bash
# 论文修订阶段完整实验脚本
# 建议：在 tmux/screen 中运行，或通过 nohup 放到后台
# 用法：bash scripts/run_paper_experiments.sh [STAGE]
#   STAGE=1 : 主数据集 × bernoulli（最高优先级）
#   STAGE=2 : 缺失模式对比（XJTU 3C + NASA all）
#   STAGE=3 : 统计后处理与图表生成（需补充 aggregate_results.py）

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="/home/chen/research/battery-research/.venv/bin/python3"
RESULTS_ROOT="${PROJECT_ROOT}/results"
mkdir -p "$RESULTS_ROOT"

# 限制并行任务数，避免显存占满
# 根据当前 GPU 显存与任务耗时，建议最多 2-3 个任务并行
MAX_JOBS=${MAX_JOBS:-2}

run_experiment() {
    local dataset=$1
    local batch=$2
    local data_dir=$3
    local pattern=$4
    local repeats=$5
    local out_dir=$6

    local log_file="${out_dir}.log"
    local cmd="$PYTHON src/main.py \
        --dataset $dataset \
        --batch $batch \
        --data_dir \"$data_dir\" \
        --missing_pattern $pattern \
        --models all \
        --n_repeats $repeats \
        --epochs 100 \
        --batch_size 32 \
        --lr 0.001 \
        --patience 15 \
        --results_dir \"$out_dir\""

    echo "[START] $dataset/$batch/$pattern (repeats=$repeats) -> $out_dir"
    echo "$cmd" > "$log_file"
    eval "$cmd" >> "$log_file" 2>&1
    echo "[DONE]  $dataset/$batch/$pattern"
}

stage1_main_datasets() {
    echo "===== Stage 1: 跨数据集主实验 (bernoulli, n_repeats=100) ====="
    mkdir -p "$RESULTS_ROOT/stage1_bernoulli"

    local jobs=()
    run_experiment XJTU 3C          "./data/XJTU data"       bernoulli 100 "$RESULTS_ROOT/stage1_bernoulli/xjtu_3c" &
    jobs+=($!)
    run_experiment HUST 1           "./data/HUST data"       bernoulli 100 "$RESULTS_ROOT/stage1_bernoulli/hust_1" &
    jobs+=($!)
    run_experiment MIT 2017-05-12   "./data/MIT data"        bernoulli 100 "$RESULTS_ROOT/stage1_bernoulli/mit_2017-05-12" &
    jobs+=($!)
    run_experiment TJU Dataset_1_NCA_battery "./data/TJU data" bernoulli 100 "$RESULTS_ROOT/stage1_bernoulli/tju_nca" &
    jobs+=($!)
    run_experiment NASA all         "./data/NASA data"       bernoulli 100 "$RESULTS_ROOT/stage1_bernoulli/nasa_all" &
    jobs+=($!)

    for pid in "${jobs[@]}"; do
        wait "$pid" || true
    done
}

stage2_missing_patterns() {
    echo "===== Stage 2: 缺失模式对比 (n_repeats=50，时间允许可提到100) ====="
    mkdir -p "$RESULTS_ROOT/stage2_patterns"

    local patterns=(block channel state_dependent mixed)
    local jobs=()
    for pattern in "${patterns[@]}"; do
        while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
            sleep 10
        done
        run_experiment XJTU 3C  "./data/XJTU data" "$pattern" 50 "$RESULTS_ROOT/stage2_patterns/xjtu_3c_$pattern" &
        jobs+=($!)
    done
    for pattern in "${patterns[@]}"; do
        while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
            sleep 10
        done
        run_experiment NASA all "./data/NASA data" "$pattern" 50 "$RESULTS_ROOT/stage2_patterns/nasa_all_$pattern" &
        jobs+=($!)
    done

    for pid in "${jobs[@]}"; do
        wait "$pid" || true
    done
}

stage3_postprocess() {
    echo "===== Stage 3: 结果汇总、统计检验与图表生成 ====="
    if [[ -f scripts/aggregate_results.py ]]; then
        "$PYTHON" scripts/aggregate_results.py \
            --input "$RESULTS_ROOT" \
            --output "$RESULTS_ROOT/aggregated"
        echo "汇总结果已保存到 $RESULTS_ROOT/aggregated"
    else
        echo "scripts/aggregate_results.py 尚未创建，请手动汇总结果。"
    fi
}

STAGE=${1:-all}
case "$STAGE" in
    1) stage1_main_datasets ;;
    2) stage2_missing_patterns ;;
    3) stage3_postprocess ;;
    all)
        stage1_main_datasets
        stage2_missing_patterns
        stage3_postprocess
        ;;
    *)
        echo "未知 stage: $STAGE"
        echo "用法: $0 [1|2|3|all]"
        exit 1
        ;;
esac
