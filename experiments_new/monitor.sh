#!/bin/bash
# 实验监控脚本

LOG="results/10seeds_smart_3C.log"
RESULT_DIR="results/10seeds_smart/3C"

echo "========================================"
echo "3C Batch 实验监控"
echo "时间: $(date)"
echo "========================================"

# 检查进程
echo -e "\n【运行进程】"
ps aux | grep -E "(run_10seeds_smart|train_models|evaluate)" | grep -v grep | grep -v monitor

# GPU状态
echo -e "\n【GPU状态】"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

# 最新日志
echo -e "\n【最新日志】"
find $RESULT_DIR -name "*.log" -mmin -10 -exec tail -5 {} \; 2>/dev/null | head -20

# 统计
echo -e "\n【进度统计】"
MODEL_COUNT=$(find $RESULT_DIR -name "model_*.pt" 2>/dev/null | wc -l)
echo "已生成模型: $MODEL_COUNT"

# 最近修改的目录
echo -e "\n【最近活动】"
find $RESULT_DIR -type d -mmin -30 2>/dev/null | tail -5

echo -e "\n========================================"
