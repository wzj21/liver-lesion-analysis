param(
    [string]$Python = "python",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"

if (-not $SkipInstall) {
    Write-Host "Installing runtime dependencies..."
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
    & $Python -m pip install pyinstaller
}

Write-Host "Building Windows desktop app..."
& $Python -m PyInstaller packaging\windows_desktop.spec --clean --noconfirm

$ExePath = Join-Path $ProjectRoot "dist\LiverLesionAI\LiverLesionAI.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build finished but exe was not found: $ExePath"
}

Write-Host ""
Write-Host "Build complete:"
Write-Host $ExePath
Write-Host ""
Write-Host "Copy trained checkpoints into dist\LiverLesionAI\checkpoints or edit configs\software_inference.yaml before clinical use."
