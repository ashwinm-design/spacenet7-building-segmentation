# scripts/start_frontend.ps1
# Starts the SIH26012 WebGIS Frontend on http://localhost:5173

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting SIH26012 WebGIS Frontend (Vite + React)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path (Split-Path -Parent $scriptDir) "frontend"

if (-not (Test-Path $frontendDir)) {
    Write-Host "[ERROR] Frontend directory not found at $frontendDir" -ForegroundColor Red
    exit 1
}

Set-Location $frontendDir

$nodeVersion = node -v 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Node.js is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Using Node.js: $nodeVersion" -ForegroundColor Green
Write-Host "[INFO] Starting WebGIS dev server on http://localhost:5173" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan

npm run dev
