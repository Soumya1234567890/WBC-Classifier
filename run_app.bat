@echo off
title White Blood Cell Differential Classifier
echo =====================================================================
echo       White Blood Cell Differential Classifier (Explainable AI)
echo =====================================================================
echo.
echo Starting web application server...
echo Access URL: http://localhost:8000
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment .venv not found.
    pause
    exit /b
)

REM Open default web browser after short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

REM Run server with Python in .venv
.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000

pause
