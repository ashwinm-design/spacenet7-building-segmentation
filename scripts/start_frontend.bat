@echo off
REM scripts/start_frontend.bat
REM Starts the SIH26012 WebGIS Frontend on http://localhost:5173

echo ============================================================
echo   Starting SIH26012 WebGIS Frontend (Vite + React)
echo ============================================================

cd "%~dp0..\frontend"
npm run dev
