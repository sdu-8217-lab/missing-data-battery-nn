@echo off
chcp 65001 >nul
echo ==========================================
echo MAR + CNN1D 批量实验脚本
echo 开始时间: %date% %time%
echo ==========================================

set PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe
set LOGDIR=logs\batch_run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
mkdir %LOGDIR% 2>nul

echo.
echo [1/6] Running: mim_mar_0.3 (MIM + MAR MR=0.3)...
%PYTHON% src/main.py experiments=mim_mar_0.3 > %LOGDIR%\mim_mar_0.3.log 2>&1
echo   ✓ Completed

echo.
echo [2/6] Running: mim_mar_0.6 (MIM + MAR MR=0.6)...
%PYTHON% src/main.py experiments=mim_mar_0.6 > %LOGDIR%\mim_mar_0.6.log 2>&1
echo   ✓ Completed

echo.
echo [3/6] Running: mim_mar_0.9 (MIM + MAR MR=0.9)...
%PYTHON% src/main.py experiments=mim_mar_0.9 > %LOGDIR%\mim_mar_0.9.log 2>&1
echo   ✓ Completed

echo.
echo [4/6] Running: mar_0.3 (Baseline + MAR MR=0.3)...
%PYTHON% src/main.py experiments=mar_0.3 > %LOGDIR%\mar_0.3.log 2>&1
echo   ✓ Completed

echo.
echo [5/6] Running: mar_0.6 (Baseline + MAR MR=0.6)...
%PYTHON% src/main.py experiments=mar_0.6 > %LOGDIR%\mar_0.6.log 2>&1
echo   ✓ Completed

echo.
echo [6/6] Running: mar_0.9 (Baseline + MAR MR=0.9)...
%PYTHON% src/main.py experiments=mar_0.9 > %LOGDIR%\mar_0.9.log 2>&1
echo   ✓ Completed

echo.
echo ==========================================
echo 所有实验完成！
echo 结束时间: %date% %time%
echo 日志目录: %LOGDIR%
echo 结果目录: results\csv\
echo ==========================================
pause
