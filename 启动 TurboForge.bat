@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0scripts\start_web.ps1" (
  echo Missing startup script: scripts\start_web.ps1
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_web.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. See the message above and the log under .turboforge\logs.
  pause
)
