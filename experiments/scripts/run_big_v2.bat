@echo off
echo ==========================================
echo Big Experiment V2
echo PyTorch Lightning + Pydantic + Loguru
echo ==========================================

if "%~1"=="" (
    set REPEATS=100
) else (
    set REPEATS=%~1
)

if "%~2"=="" (
    set EPOCHS=200
) else (
    set EPOCHS=%~2
)

echo Configuration:
echo   Repeats: %REPEATS%
echo   Epochs: %EPOCHS%
echo   Batch: 3C (default)
echo ==========================================

python run_big_experiment_v2.py --batch 3C --n_repeats %REPEATS% --epochs %EPOCHS%

echo.
echo Experiment completed!
pause
