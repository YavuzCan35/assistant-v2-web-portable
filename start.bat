@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\start.ps1" %*
set "EXITCODE=%errorlevel%"
if not "%EXITCODE%"=="0" (
  echo.
  echo [start.bat] start.ps1 exited with code %EXITCODE%.
  echo [start.bat] Press any key to close this window.
  pause >nul
)
exit /b %EXITCODE%
