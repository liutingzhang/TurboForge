@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo TurboForge - keep this window open while using the application.
echo First launch needs internet access. The application UI is currently Chinese.
echo Read INSTALL_EN.md for requirements, supported routes and troubleshooting.
if not exist "%~dp0scripts\start_web.ps1" (
  echo Missing startup script. Extract ALL release ZIP parts to the same folder.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_web.ps1" %*
if errorlevel 1 (
  echo Startup failed. See the message above and .turboforge\logs.
  pause
)
