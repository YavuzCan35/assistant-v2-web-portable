@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\uninstall.ps1" %*
exit /b %errorlevel%
