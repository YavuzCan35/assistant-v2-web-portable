@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\start.ps1" %*
set "EXITCODE=%errorlevel%"
echo.
if "%EXITCODE%"=="0" (
  echo [start.bat] start.ps1 ended with code 0.
) else (
  echo.
  echo [start.bat] start.ps1 exited with code %EXITCODE%.
)
echo [start.bat] Press any key to close this window.
pause >nul
exit /b %EXITCODE%
