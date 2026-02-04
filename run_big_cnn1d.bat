@echo off
echo ==========================================
echo 大实验 - CNN1D (1800次实验)
echo ==========================================
python run_big_experiment.py --model cnn1d --device cpu --start %1
echo.
echo CNN1D大实验完成!
pause
