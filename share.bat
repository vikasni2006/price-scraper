@echo off
rem Starts Price Finder and gives you a public link your friends can open.
cd /d "%~dp0"
title Price Finder - public link

if not exist "cloudflared.exe" (
  echo Downloading Cloudflare Tunnel from Cloudflare's official GitHub page.
  echo About 55 MB, only the first time...
  echo.
  curl -L --fail -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
  if errorlevel 1 (
    del cloudflared.exe 2>nul
    echo.
    echo The download failed. Check your internet connection and run share.bat again.
    pause
    exit /b 1
  )
)

netstat -ano | findstr ":5000 " | findstr "LISTENING" >nul
if errorlevel 1 (
  echo Starting the app in a new window. Keep that window open.
  start "Price Finder app - keep this open" cmd /k python app.py
  timeout /t 5 /nobreak >nul
) else (
  echo The app is already running.
)

echo.
echo ==============================================================
echo  In a few seconds a link ending in  trycloudflare.com  appears.
echo  Send that link to your friends. Keep this window open.
echo  To stop sharing, close this window.
echo ==============================================================
echo.
cloudflared.exe tunnel --url http://localhost:5000
pause
