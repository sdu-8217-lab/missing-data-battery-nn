#!/usr/bin/env bash
# 补齐两阶段基线到 n=30 × 6 模式 × 3 插补器
# 目的：验证 GraphMIM (端到端) vs 两阶段 imputer+MLP 的对比是否可靠
#
# 用法：
#   nohup bash scripts/run_two_stage_n30.sh > results/two_stage_n30.log 2>&1 &
#
# 预计耗时：18 组合 × 30 seed × ~40s/seed ≈ 6 小时（串行 CPU 跑，避免与 GPU 大任务冲突）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="/home/chen/research/battery-research/.venv/bin/python"
RESULTS_DIR="./results/two_stage_n30"
mkdir -p "$RESULTS_DIR"

PATTERNS=(bernoulli block channel group mixed road_course)
IMPUTERS=(mean knn iterative)
N_REPEATS=30
EPOCHS=100
# 与 GPU 大任务共存：CPU 上跑 MLP，18 组合 × 30 seed × ~40s ≈ 6h
DEVICE="cpu"

echo "===== Two-Stage n=30 Battle-Royale (device=$DEVICE) ====="
echo "start: $(date -Iseconds)"

for pattern in "${PATTERNS[@]}"; do
  for imputer in "${IMPUTERS[@]}"; do
    tag="${pattern}_${imputer}"
    echo ""
    echo "----- [$tag] $(date -Iseconds) -----"
    if [ -f "$RESULTS_DIR/${pattern}_${imputer}_n${N_REPEATS}/results.csv" ]; then
      echo "  [skip] already done"
      continue
    fi
    $PY scripts/run_two_stage_baseline.py \
      --imputer "$imputer" \
      --pattern "$pattern" \
      --n_repeats "$N_REPEATS" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --results_dir "$RESULTS_DIR"
  done
done

echo ""
echo "===== finished: $(date -Iseconds) ====="
