@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing. Follow the README setup first.
  pause
  exit /b 1
)
if not exist "results\enhancement_train\checkpoint_best.pt" (
  echo The trained U-Net checkpoint is missing.
  pause
  exit /b 1
)
rem Use the cached models so venue Wi-Fi is not required.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
rem Reuse a running demo instead of starting a second server on the same port.
".venv\Scripts\python.exe" -c "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read().strip() == b'ok'" >nul 2>&1
if not errorlevel 1 (
  echo The demo is already running. Opening it in your browser.
  start "" "http://127.0.0.1:8501"
  echo You can close this launcher window. The existing demo will keep running.
  pause
  exit /b 0
)
echo Open http://127.0.0.1:8501 in Chrome or Edge.
echo Press Ctrl+C in this window when you finish the demo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
set "DEMO_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%DEMO_EXIT_CODE%"=="0" (
  echo The demo could not start. The error is shown above.
  echo If port 8501 is in use, open http://127.0.0.1:8501 in your browser.
) else (
  echo The demo has stopped.
)
pause
exit /b %DEMO_EXIT_CODE%
