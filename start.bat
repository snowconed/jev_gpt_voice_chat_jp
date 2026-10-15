@echo off
setlocal
cd /d "%~dp0"
set "VOICE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VOICE_PYTHON%" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
if not defined OPENAI_API_KEY (
  echo OPENAI_API_KEY is missing. Set it and reopen this window.
  pause
  exit /b 1
)
echo Open the address above in your browser. Press Ctrl+C to stop.
"%VOICE_PYTHON%" -B "%~dp0server.py" %*
set "VOICE_EXIT=%ERRORLEVEL%"
if not "%VOICE_EXIT%"=="0" pause
exit /b %VOICE_EXIT%
