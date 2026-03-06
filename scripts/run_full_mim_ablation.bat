@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: MIM Ablation Study - Full Experiment Batch Runner (Windows)
:: ============================================================================
:: 运行矩阵: 3方法 × 3MR × 20seeds = 180次实验
:: 方法: Baseline / MIM-Single / MIM-Multi
:: MR: 0.3 / 0.6 / 0.9
:: Seeds: 42, 101-119
:: ============================================================================

set "PYTHON=C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"
set "PROJECT_ROOT=%~dp0.."
set "LOG_DIR=%PROJECT_ROOT%\logs\batch_run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "LOG_DIR=%LOG_DIR: =_%"

mkdir "%LOG_DIR%" 2>nul
mkdir "%PROJECT_ROOT%\results\csv" 2>nul

echo ============================================================================
echo MIM Ablation Study - Full Experiment Batch
echo Start: %date% %time%
echo Log: %LOG_DIR%
echo ============================================================================
echo.

:: 定义seeds数组
set "SEEDS=42 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119"
set /a TOTAL=180
set /a COUNT=0

:: ============================================================================
:: 1. Baseline Experiments
:: ============================================================================
echo [Phase 1/3] Baseline Experiments (MR=0.3, 0.6, 0.9)
echo ----------------------------------------------------------------------------

for %%M in (0.3 0.6 0.9) do (
    echo.
    echo [Baseline] MR=%%M
    for %%S in (%SEEDS%) do (
        set /a COUNT+=1
        echo [!COUNT!/%TOTAL%] Baseline MR=%%M Seed=%%S
        
        :: 修正: experiments.training.seeds (不是 training.seeds)
        %PYTHON% %PROJECT_ROOT%\src\main.py experiments=mar_%%M experiments.training.seeds=[%%S] ^
            > "%LOG_DIR%\baseline_mr%%M_seed%%S.log" 2>&1
        
        if !errorlevel! equ 0 (
            echo   [OK]
        ) else (
            echo   [FAIL] See %LOG_DIR%\baseline_mr%%M_seed%%S.log
            echo [!COUNT!/%TOTAL%] [FAIL] Baseline MR=%%M Seed=%%S >> "%LOG_DIR%\failures.log"
        )
    )
)

:: ============================================================================
:: 2. MIM-Single Experiments
:: ============================================================================
echo.
echo [Phase 2/3] MIM-Single Experiments (MR=0.3, 0.6, 0.9)
echo ----------------------------------------------------------------------------

for %%M in (0.3 0.6 0.9) do (
    echo.
    echo [MIM-Single] MR=%%M
    for %%S in (%SEEDS%) do (
        set /a COUNT+=1
        echo [!COUNT!/%TOTAL%] MIM-Single MR=%%M Seed=%%S
        
        :: 修正: experiments.training.seeds
        %PYTHON% %PROJECT_ROOT%\src\main.py experiments=mim_mar_%%M experiments.training.seeds=[%%S] ^
            > "%LOG_DIR%\mim_single_mr%%M_seed%%S.log" 2>&1
        
        if !errorlevel! equ 0 (
            echo   [OK]
        ) else (
            echo   [FAIL] See %LOG_DIR%\mim_single_mr%%M_seed%%S.log
            echo [!COUNT!/%TOTAL%] [FAIL] MIM-Single MR=%%M Seed=%%S >> "%LOG_DIR%\failures.log"
        )
    )
)

:: ============================================================================
:: 3. MIM-Multi Experiments
:: ============================================================================
echo.
echo [Phase 3/3] MIM-Multi Experiments (MR=0.3, 0.6, 0.9)
echo ----------------------------------------------------------------------------

for %%M in (0.3 0.6 0.9) do (
    echo.
    echo [MIM-Multi] MR=%%M
    for %%S in (%SEEDS%) do (
        set /a COUNT+=1
        echo [!COUNT!/%TOTAL%] MIM-Multi MR=%%M Seed=%%S
        
        :: 修正: experiments.training.seeds
        %PYTHON% %PROJECT_ROOT%\src\main.py experiments=mim_mar_%%M_corrected experiments.training.seeds=[%%S] ^
            > "%LOG_DIR%\mim_multi_mr%%M_seed%%S.log" 2>&1
        
        if !errorlevel! equ 0 (
            echo   [OK]
        ) else (
            echo   [FAIL] See %LOG_DIR%\mim_multi_mr%%M_seed%%S.log
            echo [!COUNT!/%TOTAL%] [FAIL] MIM-Multi MR=%%M Seed=%%S >> "%LOG_DIR%\failures.log"
        )
    )
)

:: ============================================================================
:: Summary
:: ============================================================================
echo.
echo ============================================================================
echo Batch Run Completed
echo End: %date% %time%
echo Log Directory: %LOG_DIR%
echo.

if exist "%LOG_DIR%\failures.log" (
    echo [WARNING] Some experiments failed. See %LOG_DIR%\failures.log
    type "%LOG_DIR%\failures.log"
) else (
    echo [SUCCESS] All experiments completed successfully!
)

echo.
echo Results directory: %PROJECT_ROOT%\results\csv\
echo ============================================================================

pause
