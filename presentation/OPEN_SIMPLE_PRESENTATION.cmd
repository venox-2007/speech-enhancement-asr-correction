@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_simple_presentation.ps1"
if errorlevel 1 pause
