@echo off
setlocal
cd /d "%~dp0"
rem Load the supplied fonts only for this presentation session.
rem They are removed again after the presentation closes.
powershell.exe -NoProfile -WindowStyle Hidden -File "%~dp0open_presentation.ps1"
if errorlevel 1 (
 echo The PowerPoint opener could not finish. Open Speech-Enhancement-ASR-FINAL.pdf instead.
 pause
)
