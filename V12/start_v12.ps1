# V12 多模态时序算法启动脚本 (PowerShell版本)
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("test", "train", "all")]
    [string]$Action = "all",
    
    [Parameter(Mandatory=$false)]
    [string]$Config = "config.yaml"
)

function Write-Banner {
    param([string]$Text)
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    for ($i = 0; $i -lt 58; $i++) { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
    Write-Host "🚀 $Text" -ForegroundColor Yellow
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    for ($i = 0; $i -lt 58; $i++) { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
}

function Run-Command {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Banner $Description
    
    try {
        $result = Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
            Write-Host "✅ 执行成功" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ 执行失败 (退出码: $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ 执行出错: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 主程序
Write-Host "🎯 V12多模态时序算法启动器 (PowerShell版)" -ForegroundColor Magenta
Write-Host "=" -ForegroundColor Cyan -NoNewline
for ($i = 0; $i -lt 58; $i++) { Write-Host "=" -ForegroundColor Cyan -NoNewline }
Write-Host ""
Write-Host "📁 当前目录: $(Get-Location)" -ForegroundColor Cyan
Write-Host "⚙️  配置文件: $Config" -ForegroundColor Cyan
Write-Host "🎬 执行动作: $Action" -ForegroundColor Cyan

# 检查必要文件
$requiredFiles = @(
    "config.yaml",
    "train.py", 
    "simple_multimodal_integration.py",
    "enhanced_validation_integration.py"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ 缺少必要文件: $($missingFiles -join ', ')" -ForegroundColor Red
    Write-Host "请确保在V12目录中运行此脚本" -ForegroundColor Yellow
    exit 1
}

$success = $true

# 运行测试
if ($Action -eq "test" -or $Action -eq "all") {
    Write-Host ""
    Write-Host "🧪 开始运行测试套件" -ForegroundColor Yellow
    
    $tests = @(
        @("python test_v12_integration.py", "V12集成测试"),
        @("python test_multimodal_modifications.py", "多模态损失测试"),
        @("python test_enhanced_validation_integration.py", "增强验证测试")
    )
    
    foreach ($test in $tests) {
        if (-not (Run-Command $test[0] $test[1])) {
            $success = $false
            break
        }
    }
}

# 运行训练
if (($Action -eq "train" -or $Action -eq "all") -and $success) {
    $trainCmd = "python train.py --config $Config"
    $success = Run-Command $trainCmd "模型训练"
}

# 结果总结
Write-Host ""
if ($success) {
    Write-Host "🎉 V12启动完成！" -ForegroundColor Green
    Write-Host "=" -ForegroundColor Cyan -NoNewline
    for ($i = 0; $i -lt 58; $i++) { Write-Host "=" -ForegroundColor Cyan -NoNewline }
    Write-Host ""
    Write-Host "📋 后续步骤:" -ForegroundColor Cyan
    Write-Host "1. 检查训练日志和TensorBoard" -ForegroundColor White
    Write-Host "2. 监控验证指标变化" -ForegroundColor White
    Write-Host "3. 调整配置参数优化性能" -ForegroundColor White
    Write-Host "4. 使用可视化工具分析结果" -ForegroundColor White
} else {
    Write-Host "❌ 执行过程中出现错误" -ForegroundColor Red
    exit 1
}

exit 0
