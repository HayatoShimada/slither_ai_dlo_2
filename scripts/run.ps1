# 仮想環境を利用して main.py を実行する。venv が無い／壊れていれば setup を実行してから起動。（Windows）
# 使い方:
#   .\scripts\run.ps1          # 骨格可視化
#   .\scripts\run.ps1 debug    # HSV デバッグ
#   .\scripts\run.ps1 bot      # 自動運転 + 強化学習

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$needSetup = $true
if (Test-Path $VenvPython) {
    try {
        & $VenvPython -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $needSetup = $false }
    } catch {}
}

if ($needSetup) {
    Write-Host "Virtual environment missing or broken. Running setup..."
    & (Join-Path $ScriptDir "setup.ps1")
}

$mainPy = Join-Path $ProjectRoot "main.py"
& $VenvPython $mainPy $args
