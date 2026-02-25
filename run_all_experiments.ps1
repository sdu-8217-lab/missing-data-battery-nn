# MAR + CNN1D Batch Experiments Script
# Usage: .\run_all_experiments.ps1

$python = "C:\Users\chen\AppData\Local\Programs\Python\Python313\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\batch_run_$timestamp"

# Create log directory
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "==========================================" -ForegroundColor Green
Write-Host "MAR + CNN1D Batch Experiments" -ForegroundColor Green
Write-Host "Start: $(Get-Date)" -ForegroundColor Green
Write-Host "LogDir: $logDir" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

$experiments = @(
    @{ Name = "mim_mar_0.3"; Desc = "MIM + MAR MR=0.3" },
    @{ Name = "mim_mar_0.6"; Desc = "MIM + MAR MR=0.6" },
    @{ Name = "mim_mar_0.9"; Desc = "MIM + MAR MR=0.9" },
    @{ Name = "mar_0.3"; Desc = "Baseline + MAR MR=0.3" },
    @{ Name = "mar_0.6"; Desc = "Baseline + MAR MR=0.6" },
    @{ Name = "mar_0.9"; Desc = "Baseline + MAR MR=0.9" }
)

$total = $experiments.Count
$current = 0

foreach ($exp in $experiments) {
    $current++
    $name = $exp.Name
    $desc = $exp.Desc
    $logFile = "$logDir\$name.log"
    
    Write-Host ""
    Write-Host "[$current/$total] Running: $desc" -ForegroundColor Cyan
    Write-Host "  Config: experiments=$name" -ForegroundColor Gray
    Write-Host "  Log: $logFile" -ForegroundColor Gray
    Write-Host "  Start: $(Get-Date)" -ForegroundColor Gray
    
    $startTime = Get-Date
    
    # Run experiment and capture output
    & $python src/main.py experiments=$name 2>&1 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor DarkGray
        $_ | Out-File -Append -FilePath $logFile
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host "  Completed in $($duration.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "End: $(Get-Date)" -ForegroundColor Green
Write-Host "LogDir: $logDir" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "Results Summary:" -ForegroundColor Yellow
Get-ChildItem "results\csv\*.csv" | ForEach-Object {
    $rows = (Get-Content $_.FullName | Measure-Object).Line - 1
    Write-Host "  $($_.Name): $rows rows" -ForegroundColor White
}

Read-Host "`nPress Enter to exit"
