# MAR + CNN1D 批量实验脚本
# 运行 6 个实验配置：3 个 MIM + 3 个 Baseline

$python = "C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs/batch_run_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "==========================================" -ForegroundColor Green
Write-Host "MAR + CNN1D 批量实验开始" -ForegroundColor Green
Write-Host "开始时间: $(Get-Date)" -ForegroundColor Green
Write-Host "日志目录: $logDir" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# 实验配置列表
$experiments = @(
    # MIM 实验
    @{ Name = "mim_mar_0.3"; Desc = "MIM + MAR MR=0.3"; Output = "results/csv/mar_0.3_cnn1d_mim.csv" },
    @{ Name = "mim_mar_0.6"; Desc = "MIM + MAR MR=0.6"; Output = "results/csv/mar_0.6_cnn1d_mim.csv" },
    @{ Name = "mim_mar_0.9"; Desc = "MIM + MAR MR=0.9"; Output = "results/csv/mar_0.9_cnn1d_mim.csv" },
    # Baseline 实验
    @{ Name = "mar_0.3"; Desc = "Baseline + MAR MR=0.3"; Output = "results/csv/mar_0.3_cnn1d_baseline.csv" },
    @{ Name = "mar_0.6"; Desc = "Baseline + MAR MR=0.6"; Output = "results/csv/mar_0.6_cnn1d_baseline.csv" },
    @{ Name = "mar_0.9"; Desc = "Baseline + MAR MR=0.9"; Output = "results/csv/mar_0.9_cnn1d_baseline.csv" }
)

$total = $experiments.Count
$current = 0

foreach ($exp in $experiments) {
    $current++
    $name = $exp.Name
    $desc = $exp.Desc
    $output = $exp.Output
    $logFile = "$logDir/${name}.log"
    
    Write-Host ""
    Write-Host "[$current/$total] 运行: $desc" -ForegroundColor Cyan
    Write-Host "    配置: +experiments=$name" -ForegroundColor Gray
    Write-Host "    输出: $output" -ForegroundColor Gray
    Write-Host "    日志: $logFile" -ForegroundColor Gray
    Write-Host "    开始时间: $(Get-Date)" -ForegroundColor Gray
    
    $startTime = Get-Date
    
    # 运行实验
    & $python src/main.py +experiments=$name data=xjtu models=cnn1d missing=mar 2>&1 | Tee-Object -FilePath $logFile
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    # 检查结果文件
    if (Test-Path $output) {
        $rows = (Import-Csv $output | Measure-Object).Count
        Write-Host "    ✓ 完成: $rows 行结果已保存" -ForegroundColor Green
        Write-Host "    耗时: $($duration.TotalMinutes.ToString('F1')) 分钟" -ForegroundColor Green
    } else {
        Write-Host "    ✗ 警告: 输出文件未找到" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "所有实验完成！" -ForegroundColor Green
Write-Host "结束时间: $(Get-Date)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
