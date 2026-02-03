# PowerShell脚本 - 运行完整实验
# 使用方式: .\run_complete_experiment.ps1

$batch = "3C"
$nRepeats = 100
$epochs = 100
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "==============================================" -ForegroundColor Green
Write-Host "SOH预测缺失数据处理 - 完整实验" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "批次: $batch"
Write-Host "重复次数: $nRepeats"
Write-Host "训练轮数: $epochs"
Write-Host "开始时间: $(Get-Date)"
Write-Host "预计耗时: 6-12小时"
Write-Host "==============================================" -ForegroundColor Green

# 激活环境
& D:\App\miniforge3\Scripts\conda.exe activate pytorch-cpu

# 运行实验
python run_full_experiment.py `
    --batch $batch `
    --n_repeats $nRepeats `
    --epochs $epochs `
    --results_dir "./experiments" `
    > "experiments/log_$timestamp.txt" 2>&1

Write-Host ""
Write-Host "实验完成！" -ForegroundColor Green
Write-Host "完成时间: $(Get-Date)"
Write-Host "日志文件: experiments/log_$timestamp.txt"
