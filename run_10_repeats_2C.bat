@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set BATCH_NAME=2C
set EPOCHS=50
set PYTORCH_LIGHTNING_LOG_LEVEL=ERROR
set PYTHONIOENCODING=utf-8

echo ========================================
echo 2C 批次 10 次重复实验
echo ========================================
echo.

for /L %%i in (42,1,51) do (
    echo.
    echo ========================================
    echo 运行第 %%i-41/10 次实验 - 种子: %%i
    echo ========================================
    D:\App\miniforge3\envs\pytorch-cpu\python.exe run_full_experiment_v2.py --batch-name %BATCH_NAME% --seed %%i --epochs %EPOCHS%
    if !errorlevel! neq 0 (
        echo [警告] 种子 %%i 实验失败，继续下一个...
    )
)

echo.
echo ========================================
echo 所有实验完成！
echo 结果位置: ./experiments_v2/%BATCH_NAME%/
echo ========================================
