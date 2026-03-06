@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: Youth Version Large Experiment - MAR + Baseline/MIM
:: ============================================================================
:: Full Matrix: 2 batches × 2 methods × 3 models × 9 MRs × 50 seeds = 5,400 runs
:: Youth Subset: 2 batches × 2 methods × 3 models × 9 MRs × 50 seeds = 5,400 runs
::
:: Methods: baseline, mim
:: Models: mlp, lstm, cnn
:: MRs: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
:: Seeds: 50 seeds (configured in YAML)
:: ============================================================================

set "PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"
set "PROJECT_ROOT=%~dp0.."
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =_%"
set "LOG_DIR=%PROJECT_ROOT%\logs\youth_run_%TIMESTAMP%"

mkdir "%LOG_DIR%" 2>nul
mkdir "%PROJECT_ROOT%\results\csv" 2>nul

echo ============================================================================
echo Youth Version Large Experiment - MAR
echo Start: %date% %time%
echo Log: %LOG_DIR%
echo ============================================================================
echo.

:: 实验参数
set "BATCHES=2c 3c"
set "METHODS=baseline mim"
set "MODELS=mlp lstm cnn"
set "MRS=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

:: 计算总数
set /a TOTAL=2*2*3*9
set /a COUNT=0

for %%B in (%BATCHES%) do (
    for %%M in (%METHODS%) do (
        for %%O in (%MODELS%) do (
            for %%R in (%MRS%) do (
                set /a COUNT+=1
                
                echo [!COUNT!/%TOTAL%] Batch=%%B Method=%%M Model=%%O MR=%%R
                
                if "%%M"=="baseline" (
                    set "EXPERIMENT=youth_mar_baseline"
                ) else (
                    set "EXPERIMENT=youth_mar_mim"
                )
                
                %PYTHON% %PROJECT_ROOT%\src\experiments\run_experiment.py ^
                    --config-name=!EXPERIMENT! ^
                    data.batch_id=%%B ^
                    model.type=%%O ^
                    missing.rate_eval=%%R ^
                    method=%%M ^
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
