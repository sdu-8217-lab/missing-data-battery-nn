#!/bin/bash
# 运行 MIM vs Baseline 对比实验
# Usage: ./run_comparison.sh [dataset] [model]

DATASET=${1:-xjtu}
MODEL=${2:-mlp}

echo "========================================"
echo "MIM vs Traditional Imputation Baseline"
echo "Dataset: $DATASET, Model: $MODEL"
echo "========================================"

# 快速测试参数
EPOCHS=2
SEEDS="[42]"

echo ""
echo "1. MIM Method (Proposed)"
python src/main.py data=$DATASET model=$MODEL method=mim \
    training.epochs=$EPOCHS experiment.seeds=$SEEDS wandb.enabled=false

echo ""
echo "2. Mean Imputation (Baseline)"
python src/main.py data=$DATASET model=$MODEL method=mean \
    training.epochs=$EPOCHS experiment.seeds=$SEEDS wandb.enabled=false

echo ""
echo "3. Median Imputation (Baseline)"
python src/main.py data=$DATASET model=$MODEL method=median \
    training.epochs=$EPOCHS experiment.seeds=$SEEDS wandb.enabled=false

echo ""
echo "4. KNN Imputation (Baseline, k=5)"
python src/main.py data=$DATASET model=$MODEL method=knn \
    training.epochs=$EPOCHS experiment.seeds=$SEEDS wandb.enabled=false

echo ""
echo "5. Zero Imputation (Baseline)"
python src/main.py data=$DATASET model=$MODEL method=zero \
    training.epochs=$EPOCHS experiment.seeds=$SEEDS wandb.enabled=false

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Check results/ directory for CSV outputs"
echo "========================================"
