@echo off
title Deploy White Blood Cell Classifier to Vercel
echo =====================================================================
echo       Deploy White Blood Cell Classifier to Vercel
echo =====================================================================
echo.

cd /d "%~dp0"

REM Ensure Node.js and npx are found
set "PATH=C:\Program Files\nodejs;C:\Program Files (x86)\nodejs;%PATH%"

where npx >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Node.js and npx were not detected on your system.
    echo Please install Node.js from https://nodejs.org or run the local app via run_app.bat.
    pause
    exit /b
)

echo Make sure you have a free account at https://vercel.com
echo Deploying static client-side bundle in ./public directory...
echo.

call npx vercel

echo.
echo =====================================================================
echo Deployment completed!
echo To deploy directly to production with a custom permanent URL, run:
echo npx vercel --prod
echo =====================================================================
echo.
pause

