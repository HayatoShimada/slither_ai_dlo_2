# どの環境でも同じ手順で仮想環境を用意するスクリプト（Windows）
# 使い方: .\scripts\setup.ps1  または  pwsh -File scripts\setup.ps1
# - Python は py -3, python3, python の順で検出
# - .venv が無い／壊れている場合は作り直す
# - requirements.txt をインストール

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$pythonExe = $null
$pyArg = $null
foreach ($cand in @("py", "python3", "python")) {
    $args = if ($cand -eq "py") { @("-3", "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)") } else { @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)") }
    try {
        & $cand $args 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $cand; $pyArg = if ($cand -eq "py") { "-3" } else { "" }; break }
    } catch {}
}

if (-not $pythonExe) {
    Write-Error "Python 3.9+ not found. Install Python or py launcher and retry."
    exit 1
}

Write-Host "Using: $pythonExe $pyArg"

$venvPath = Join-Path $ProjectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

$recreate = $false
if (Test-Path $venvPath) {
    if (-not (Test-Path $venvPython)) { $recreate = $true }
    else {
        try {
            & $venvPython -c "import sys" 2>$null
            if ($LASTEXITCODE -ne 0) { $recreate = $true }
        } catch { $recreate = $true }
    }
    if ($recreate) {
        Write-Host "Removing broken or foreign .venv and recreating..."
        Remove-Item -Recurse -Force $venvPath
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating .venv..."
    $venvArgs = if ($pythonExe -eq "py") { @("-3", "-m", "venv", $venvPath) } else { @("-m", "venv", $venvPath) }
    & $pythonExe $venvArgs
}

Write-Host "Installing dependencies..."
& $venvPython -m pip install -q --upgrade pip
& $venvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Setup done. Run: .\scripts\run.ps1 bot   or   .\scripts\run.ps1 debug"
