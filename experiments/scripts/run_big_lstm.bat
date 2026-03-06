@echo off
echo ==========================================
echo 大实验 - LSTM (1800次实验)
echo ==========================================
python run_big_experiment.py --model lstm --device cpu --start %1
echo.
echo LSTM大实验完成!
pause
