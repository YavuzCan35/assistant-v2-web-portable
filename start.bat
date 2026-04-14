@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\start.ps1" %*
exit /b %errorlevel%
