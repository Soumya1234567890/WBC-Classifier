@echo off
title Deploy White Blood Cell Classifier to Vercel
echo =====================================================================
echo       Deploy White Blood Cell Classifier to Vercel
echo =====================================================================
echo.
echo Make sure you have a free account at https://vercel.com
echo.
cd /d "%~dp0"

echo Running npx vercel...
npx vercel

echo.
echo If you wish to promote to production, run:
echo npx vercel --prod
echo.
pause
