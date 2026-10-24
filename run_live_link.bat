@echo off
title White Blood Cell Differential Classifier - Live Web Link
echo =====================================================================
echo    White Blood Cell Classifier (Live Public Web Link Generator)
echo =====================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Error: Isolated Python environment .venv was not found in:
    echo %~dp0.venv
    pause
    exit /b
)

REM Check for Cloudflared
set "CLOUDFLARED_BIN="
if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" set "CLOUDFLARED_BIN=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if exist "C:\Program Files\cloudflared\cloudflared.exe" set "CLOUDFLARED_BIN=C:\Program Files\cloudflared\cloudflared.exe"

if "%CLOUDFLARED_BIN%"=="" (
    echo Error: cloudflared was not found.
    echo Please install cloudflared or use run_app.bat for local access.
    pause
    exit /b
)

echo [1/4] Ensuring port 8000 is clean...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%a >nul 2>&1
)

echo [2/4] Starting Deep Learning Inference Server on port 8000...
start "WBC-Backend-Server" /min .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo [3/4] Initializing AI model (Keras Sequential + Grad-CAM)...
powershell -NoProfile -Command "for ($i=0; $i -lt 45; $i++) { Start-Sleep -Seconds 1; try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { break } } catch {} }"

echo.
echo [4/4] Establishing secure public live web tunnel via Cloudflare...
echo =====================================================================
echo Your live public link will appear below (look for https://*.trycloudflare.com):
echo This link can be opened on ANY device (phone, laptop, tablet) anywhere!
echo Keep this window OPEN while sharing the live link.
echo =====================================================================
echo.

"%CLOUDFLARED_BIN%" tunnel --url http://127.0.0.1:8000

echo.
echo Live session terminated.
pause
