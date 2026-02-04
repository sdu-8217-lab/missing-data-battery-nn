@echo off
echo ==========================================
echo 大实验 - MLP (1800次实验)
echo ==========================================
python run_big_experiment.py --model mlp --device cpu --start %1
echo.
echo MLP大实验完成!
pause
