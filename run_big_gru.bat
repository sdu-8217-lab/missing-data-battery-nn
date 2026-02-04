@echo off
echo ==========================================
echo 大实验 - GRU (1800次实验)
echo ==========================================
python run_big_experiment.py --model gru --device cpu --start %1
echo.
echo GRU大实验完成!
pause
