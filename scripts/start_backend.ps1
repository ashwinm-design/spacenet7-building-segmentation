# scripts/start_backend.ps1
# Starts the SIH26012 FastAPI backend server on http://127.0.0.1:8000

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting SIH26012 AI Mapping Backend Service" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

# Verify python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Using Python: $pythonVersion" -ForegroundColor Green
Write-Host "[INFO] Starting FastAPI application on http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "[INFO] Health check available at: http://127.0.0.1:8000/api/health" -ForegroundColor Gray
Write-Host "[INFO] Swagger API Docs at:       http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
