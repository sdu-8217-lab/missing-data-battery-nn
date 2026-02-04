@echo off
chcp 65001 >nul
set PYTORCH_LIGHTNING_LOG_LEVEL=ERROR
set PYTHONIOENCODING=utf-8

D:\App\miniforge3\envs\pytorch-cpu\python.exe run_full_experiment_v2.py --batch-name %1 --seed %2 --epochs %3
