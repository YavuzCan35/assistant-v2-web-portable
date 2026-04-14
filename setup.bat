@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\setup.ps1" %*
exit /b %errorlevel%
