@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: Youth Experiment - Final Version (Using Verified src/main.py)
:: ============================================================================

set "PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"
set "PROJECT_ROOT=%~dp0.."
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =_%"
set "LOG_DIR=%PROJECT_ROOT%\logs\youth_final_%TIMESTAMP%"

mkdir "%LOG_DIR%" 2>nul

echo ============================================================================
echo Youth Experiment - Final Run
echo Using verified src/main.py path
echo Start: %date% %time%
echo Log: %LOG_DIR%
echo ============================================================================
echo.

:: 实验参数
set "BATCHES=2C 3C"
set "MRS=0.3 0.6 0.9"
set "SEEDS=42 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119"
set /a TOTAL=2*3*20*2
set /a COUNT=0

for %%B in (%BATCHES%) do (
  for %%M in (%MRS%) do (
    for %%S in (%SEEDS%) do (
      set /a COUNT+=1
      
      echo [!COUNT!/%TOTAL%] Batch=%%B MR=%%M Seed=%%S
      
      :: Baseline
      %PYTHON% %PROJECT_ROOT%\src\main.py ^
        experiments=mar_%%M ^
        data.batch=%%B ^
        experiments.training.seeds=[%%S] ^
        > "%LOG_DIR%\%%B_baseline_mr%%M_seed%%S.log" 2>&1
      
      if !errorlevel! equ 0 (
        echo   [Baseline OK]
      ) else (
        echo   [Baseline FAIL]
      )
      
      :: MIM
      %PYTHON% %PROJECT_ROOT%\src\main.py ^
        experiments=mim_mar_%%M_corrected ^
        data.batch=%%B ^
        experiments.training.seeds=[%%S] ^
        > "%LOG_DIR%\%%B_mim_mr%%M_seed%%S.log" 2>&1
      
      if !errorlevel! equ 0 (
        echo   [MIM OK]
      ) else (
        echo   [MIM FAIL]
      )
    )
  )
)

echo.
echo ============================================================================
echo Completed
echo End: %date% %time%
echo ============================================================================

pause
