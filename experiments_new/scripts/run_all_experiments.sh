#!/bin/bash
# 全量实验运行脚本 - 支持多种子
# 6 batches × 3 architectures × 4 levels × 2 modes × n seeds = 144×n 实验
#
# 用法:
#   ./scripts/run_all_experiments.sh                    # 运行全部（单种子）
#   ./scripts/run_all_experiments.sh 3C                 # 只运行3C batch
#   ./scripts/run_all_experiments.sh 3C mlp             # 只运行3C的MLP
#   SEEDS="0 1 2" ./scripts/run_all_experiments.sh 3C   # 指定种子

set -e
source ~/miniforge3/bin/activate battery-nn
cd /home/chen/github/missing-data-battery-nn/experiments_new

# 参数
BATCH_FILTER="${1:-all}"
ARCH_FILTER="${2:-all}"

# 种子配置（可通过环境变量覆盖）
SEEDS=(${SEEDS:-0})  # 默认单种子0，可设置 SEEDS="0 1 2 3 4 5 6 7 8 9"

# 实验矩阵
BATCHES=("2C" "3C" "R2.5" "R3" "RW" "Sim_satellite")
ARCHS=("mlp" "lstm" "cnn")
LEVELS=("level_1" "level_2" "level_3" "level_4")
CONFIG_DIR="configs/experiments/batch_configs_standard"
OUTPUT_DIR="./results/full_scale"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# 统计
total=0
success=0
fail=0

for batch in "${BATCHES[@]}"; do
    [[ "$BATCH_FILTER" == "all" || "$BATCH_FILTER" == "$batch" ]] || continue
    for arch in "${ARCHS[@]}"; do
        [[ "$ARCH_FILTER" == "all" || "$ARCH_FILTER" == "$arch" ]] || continue
        for level in "${LEVELS[@]}"; do
            for mode in "mim" "baseline"; do
                ((total++))
            done
        done
    done
done

total_models=$((total * ${#SEEDS[@]}))

echo "========================================" | tee -a "$LOG_FILE"
echo "全量实验运行 - 支持多种子" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "种子: ${SEEDS[@]} (${#SEEDS[@]} 个)" | tee -a "$LOG_FILE"
echo "Filter: batch=$BATCH_FILTER, arch=$ARCH_FILTER" | tee -a "$LOG_FILE"
echo "配置数: $total (batch × arch × level × mode)" | tee -a "$LOG_FILE"
echo "总模型数: $total_models (configs × ${#SEEDS[@]} seeds)" | tee -a "$LOG_FILE"
echo "预计: $((total_models*2))-$((total_models*4)) 分钟" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

current=0
start=$(date +%s)

for batch in "${BATCHES[@]}"; do
    [[ "$BATCH_FILTER" == "all" || "$BATCH_FILTER" == "$batch" ]] || continue
    
    for arch in "${ARCHS[@]}"; do
        [[ "$ARCH_FILTER" == "all" || "$ARCH_FILTER" == "$arch" ]] || continue
        
        for level in "${LEVELS[@]}"; do
            for mode in "mim" "baseline"; do
                ((current++))
                exp_start=$(date +%s)
                
                config="${CONFIG_DIR}/batch_${batch}_${arch}_${mode}.yaml"
                model_config="configs/models/${arch}_${level}.yaml"
                exp_out="$OUTPUT_DIR/${batch}_${arch}_${level}_${mode}"
                
                printf "\n[%3d/%3d] %s/%s/%s/%s " "$current" "$total" "$batch" "$arch" "$level" "$mode" | tee -a "$LOG_FILE"
                
                if [ ! -f "$config" ] || [ ! -f "$model_config" ]; then
                    echo "SKIP(文件缺失)" | tee -a "$LOG_FILE"
                    ((fail++))
                    continue
                fi
                
                if python scripts/run_full_experiment.py \
                    --config "$config" \
                    --model-config "$model_config" \
                    --output-dir "$exp_out" \
                    --seeds ${SEEDS[@]} \
                    >> "$LOG_FILE" 2>&1; then
                    
                    exp_end=$(date +%s)
                    duration=$((exp_end - exp_start))
                    echo "OK(${duration}s)" | tee -a "$LOG_FILE"
                    ((success++))
                else
                    exp_end=$(date +%s)
                    duration=$((exp_end - exp_start))
                    echo "FAIL(${duration}s)" | tee -a "$LOG_FILE"
                    ((fail++))
                fi
            done
        done
    done
done

end=$(date +%s)
total_time=$((end - start))

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "完成: 成功=$success 失败=$fail 总计=$total 配置" | tee -a "$LOG_FILE"
echo "总模型数: $((success * ${#SEEDS[@]}))" | tee -a "$LOG_FILE"
echo "耗时: $((total_time/60))分$((total_time%60))秒" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 生成图表
if [ $success -gt 0 ]; then
    echo "生成图表..." | tee -a "$LOG_FILE"
    python scripts/plot_batch_results.py --results-dir "$OUTPUT_DIR" --output-dir "./outputs/full_scale" 2>&1 | tee -a "$LOG_FILE"
fi
