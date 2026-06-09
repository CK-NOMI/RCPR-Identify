@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cosod_web.ps1"
pause
