#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate battery
cd ~/missing-data-battery-nn
python run_full_experiment.py > results/full_experiment_$(date +%m%d_%H%M).log 2>&1
echo "实验完成，退出码: $?" >> results/full_experiment_status.log
