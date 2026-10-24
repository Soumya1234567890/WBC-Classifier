@echo off
title White Blood Cell Differential Classifier
echo =====================================================================
echo       White Blood Cell Differential Classifier (Explainable AI)
echo =====================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Error: Isolated Python environment .venv was not found in:
    echo %~dp0.venv
    echo.
    echo Please make sure the environment is installed.
    pause
    exit /b
)

echo [1/3] Ensuring port 8000 is free...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Terminating stale process on port 8000 (PID: %%a)...
    taskkill /F /T /PID %%a >nul 2>&1
)

echo [2/3] Setting up smart browser launcher...
start "" /b powershell -NoProfile -Command "$url = 'http://localhost:8000'; for ($i=0; $i -lt 45; $i++) { Start-Sleep -Seconds 1; try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; break } } catch {} }"

echo [3/3] Starting Deep Learning Server on http://localhost:8000 ...
echo (Loading Keras model and Grad-CAM layer... Browser will open automatically once ready)
echo.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause


