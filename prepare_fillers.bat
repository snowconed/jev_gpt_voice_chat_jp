@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  exit /b 1
)
echo Preparing cached fillers. GPT speech uses a small paid API request when the cache is missing.
".venv\Scripts\python.exe" -B prepare_fillers.py %*
exit /b %errorlevel%
