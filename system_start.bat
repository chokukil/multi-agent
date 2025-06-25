@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "LOG_DIR=%PROJECT_ROOT%logs"
set "PID_FILE=%LOG_DIR%\pandas_server.pid"

echo ================================================
echo Starting CherryAI System for Windows...
echo ================================================

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo ================================================
echo Step 1: Starting Pandas Data Analyst Server in the background...
echo Log file: %LOG_DIR%\pandas_server.log
echo ================================================

REM Start the server in a new minimized window and capture its PID
start "Pandas Data Analyst Server" /B powershell -Command "& { $process = Start-Process 'uv' -ArgumentList 'run', 'python', 'a2a_servers/pandas_server.py' -PassThru; $process.Id | Out-File -FilePath '%PID_FILE%' -Encoding ascii }"

echo ✅ Server starting in the background...

echo.
echo Waiting 5 seconds for the server to initialize...
timeout /t 5 /nobreak > nul

echo.
echo ================================================
echo Step 2: Starting Streamlit Application in this terminal.
echo You will see live logs below. Press Ctrl+C to stop.
echo ================================================

REM Run streamlit in the foreground
uv run streamlit run app.py

echo.
echo Streamlit app stopped. Cleaning up background server...
for /f %%i in ('type "%PID_FILE%"') do (
    taskkill /PID %%i /F
)
del "%PID_FILE%"
echo ✅ System shutdown complete.

endlocal 