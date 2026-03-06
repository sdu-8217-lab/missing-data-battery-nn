@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: Youth Version Large Experiment - MAR + Baseline/MIM (FIXED VERSION)
:: ============================================================================
:: Full Matrix: 2 batches x 2 methods x 3 models x 9 MRs x 50 seeds = 5,400 runs
::
:: Methods: baseline, mim
:: Models: mlp, lstm, cnn
:: MRs: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
:: Seeds: 50 seeds
:: ============================================================================

set "PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python314\python.exe"
set "PROJECT_ROOT=%~dp0.."
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =_%"
set "LOG_DIR=%PROJECT_ROOT%\logs\youth_fixed_%TIMESTAMP%"

mkdir "%LOG_DIR%" 2>nul
mkdir "%PROJECT_ROOT%\results\csv" 2>nul

echo ============================================================================
echo Youth Version Large Experiment - MAR (FIXED)
echo Start: %date% %time%
echo Log: %LOG_DIR%
echo ============================================================================
echo.

:: 实验参数
set "BATCHES=2c 3c"
set "METHODS=baseline mim"
set "MODELS=mlp lstm cnn"
set "MRS=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

:: Seeds (50个)
set "SEEDS=42 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 201 202 203 204 205 206 207 208 209 210 301 302 303 304 305 306 307 308 309 310"

:: 计算总数
set /a TOTAL=2*2*3*9
set /a COUNT=0

for %%B in (%BATCHES%) do (
    for %%M in (%METHODS%) do (
        for %%O in (%MODELS%) do (
            for %%R in (%MRS%) do (
                set /a COUNT+=1
                
                echo [!COUNT!/%TOTAL%] Batch=%%B Method=%%M Model=%%O MR=%%R
                
                %PYTHON% %PROJECT_ROOT%\src\experiments\run_youth_exp.py ^
                    --method %%M ^
                    --model %%O ^
                    --batch %%B ^
                    --mr %%R ^
                    --seeds %SEEDS% ^
                    > "%LOG_DIR%\%%B_%%M_%%O_mr%%R.log" 2>&1
                
                if !errorlevel! equ 0 (
                    echo   [OK]
                ) else (
                    echo   [FAIL]
                    echo [!COUNT!/%TOTAL%] FAIL: %%B_%%M_%%O_mr%%R >> "%LOG_DIR%\failures.log"
                )
            )
        )
    )
)

echo.
echo ============================================================================
echo Youth Experiment Batch Completed
echo End: %date% %time%
echo ============================================================================

if exist "%LOG_DIR%\failures.log" (
    echo [WARNING] Some experiments failed
    type "%LOG_DIR%\failures.log"
) else (
    echo [SUCCESS] All experiments completed!
)

pause
