@echo off
chcp 65001 >nul
echo ==========================================
echo MAR + CNN1D Batch Experiments
echo Start: %date% %time%
echo ==========================================

rem Use Python from PATH or specify full path
set "PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"

rem Create log directory with safe name
set "LOGDIR=logs\batch_run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "LOGDIR=%LOGDIR: =_%"
mkdir "%LOGDIR%" 2>nul

echo.
echo Python: %PYTHON%
echo LogDir: %LOGDIR%
echo.

echo [1/6] Running: mim_mar_0.3 (MIM + MAR MR=0.3)...
"%PYTHON%" -u src/main.py experiments=mim_mar_0.3 2>&1 | tee "%LOGDIR%\mim_mar_0.3.log"
echo [1/6] Done
echo.

echo [2/6] Running: mim_mar_0.6 (MIM + MAR MR=0.6)...
"%PYTHON%" -u src/main.py experiments=mim_mar_0.6 2>&1 | tee "%LOGDIR%\mim_mar_0.6.log"
echo [2/6] Done
echo.

echo [3/6] Running: mim_mar_0.9 (MIM + MAR MR=0.9)...
"%PYTHON%" -u src/main.py experiments=mim_mar_0.9 2>&1 | tee "%LOGDIR%\mim_mar_0.9.log"
echo [3/6] Done
echo.

echo [4/6] Running: mar_0.3 (Baseline + MAR MR=0.3)...
"%PYTHON%" -u src/main.py experiments=mar_0.3 2>&1 | tee "%LOGDIR%\mar_0.3.log"
echo [4/6] Done
echo.

echo [5/6] Running: mar_0.6 (Baseline + MAR MR=0.6)...
"%PYTHON%" -u src/main.py experiments=mar_0.6 2>&1 | tee "%LOGDIR%\mar_0.6.log"
echo [5/6] Done
echo.

echo [6/6] Running: mar_0.9 (Baseline + MAR MR=0.9)...
"%PYTHON%" -u src/main.py experiments=mar_0.9 2>&1 | tee "%LOGDIR%\mar_0.9.log"
echo [6/6] Done
echo.

echo ==========================================
echo All experiments completed!
echo End: %date% %time%
echo LogDir: %LOGDIR%
echo ==========================================
pause
