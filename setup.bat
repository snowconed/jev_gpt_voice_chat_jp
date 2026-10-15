@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto python_fallback
py -3 -m venv "%~dp0.venv"
if errorlevel 1 goto failed
goto install
:python_fallback
python -m venv "%~dp0.venv"
if errorlevel 1 goto failed
:install
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.lock.txt"
if errorlevel 1 goto failed
echo Setup completed. Run start_jev.bat for GPT speech.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.11 or newer and try again.
pause
exit /b 1
