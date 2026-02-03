@echo off
chcp 65001 >nul
echo =========================================
echo SOH预测缺失数据处理完整实验 (100次重复)
echo =========================================
echo.
echo 实验配置:
echo   - 重复次数: 100
echo   - 模型数量: 10 (5模型 × 2配置)
echo   - 预估时间: 6-12小时
echo   - 输出目录: experiments/
echo.
echo 按任意键开始实验，或关闭窗口取消...
pause >nul

cd /d "%~dp0"
D:\App\miniforge3\envs\pytorch-cpu\python.exe run_full_experiment.py --batch 3C --n_repeats 100 --epochs 50 > experiments\full_experiment_100repeats.log 2>&1

echo.
echo 实验完成！
echo 结果保存在 experiments/ 目录
echo 日志保存在 experiments\full_experiment_100repeats.log
pause
