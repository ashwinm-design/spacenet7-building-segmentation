@echo off
REM scripts/start_backend.bat
REM Starts the SIH26012 FastAPI backend server on http://127.0.0.1:8000

echo ============================================================
echo   Starting SIH26012 AI Mapping Backend Service
echo ============================================================

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
