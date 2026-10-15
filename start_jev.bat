@echo off
setlocal
echo Jev voice chat: http://127.0.0.1:8783
call "%~dp0start.bat" --turn-mode jev --port 8783 --data-dir "%~dp0data_jev" %*
exit /b %ERRORLEVEL%
