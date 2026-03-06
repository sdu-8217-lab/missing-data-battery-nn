@echo off
chcp 65001 >nul
setlocal

:: 设置环境变量
set PYTORCH_LIGHTNING_LOG_LEVEL=ERROR
set PYTHONIOENCODING=utf-8

echo ========================================
echo 2C 批次 10 次重复实验 (V2)
echo ========================================
echo.

:: 运行批量实验
cd /d "%~dp0"
D:\App\miniforge3\envs\pytorch-cpu\python.exe run_batch_experiment_v2.py --batch 2C --n-repeats 10 --start-seed 42 --epochs 250

echo.
echo ========================================
echo 按任意键退出...
echo ========================================
pause >nul
