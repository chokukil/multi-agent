@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "PID_FILE=%LOG_DIR%\pandas_server.pid"

echo ================================================
echo Stopping CherryAI System for Windows...
echo Project Root: %PROJECT_ROOT%
echo ================================================

:: Function to safely kill a process by PID
:: Usage: call :kill_process_by_pid PID ProcessName
goto :skip_functions

:kill_process_by_pid
set "PID=%~1"
set "NAME=%~2"

:: Check if process exists
tasklist /fi "pid eq %PID%" 2>nul | find /i "%PID%" >nul
if errorlevel 1 (
    echo ⚠️  %NAME% ^(PID: %PID%^) is not running.
    goto :eof
)

echo Stopping %NAME% ^(PID: %PID%^)...
taskkill /pid %PID% /f >nul 2>&1
if errorlevel 1 (
    echo ❌ Failed to terminate %NAME%.
) else (
    echo ✅ %NAME% terminated successfully.
)
goto :eof

:skip_functions

:: 1. Stop background pandas server from PID file
echo.
echo Step 1: Stopping Pandas Data Analyst Server...
if exist "%PID_FILE%" (
    for /f %%i in ('type "%PID_FILE%"') do (
        call :kill_process_by_pid %%i "Pandas Server"
    )
    del "%PID_FILE%" >nul 2>&1
    echo PID file removed.
) else (
    echo - No PID file found at: %PID_FILE%
)

:: 2. Stop any remaining pandas_server processes
echo.
echo Step 2: Searching for remaining pandas_server processes...
set "FOUND_PANDAS=false"
for /f "tokens=2 delims=," %%a in ('tasklist /nh /fi "imagename eq python.exe" /fo csv 2^>nul ^| findstr "pandas_server"') do (
    set "FOUND_PANDAS=true"
    call :kill_process_by_pid %%~a "Pandas Server Process"
)
if "!FOUND_PANDAS!"=="false" (
    echo - No pandas_server processes found.
)

:: 3. Stop Streamlit processes
echo.
echo Step 3: Stopping Streamlit processes...
set "FOUND_STREAMLIT=false"

:: Kill streamlit.exe directly
tasklist /fi "imagename eq streamlit.exe" 2>nul | find /i "streamlit.exe" >nul
if not errorlevel 1 (
    echo Found streamlit.exe processes...
    taskkill /im streamlit.exe /f >nul 2>&1
    set "FOUND_STREAMLIT=true"
)

:: Kill python processes running streamlit
for /f "tokens=1,2 delims=," %%a in ('tasklist /nh /fi "imagename eq python.exe" /fo csv 2^>nul') do (
    set "IMAGE=%%~a"
    set "PID=%%~b"
    
    :: Check if this python process is running streamlit
    wmic process where "ProcessId=!PID!" get CommandLine 2>nul | findstr "streamlit run" >nul
    if not errorlevel 1 (
        set "FOUND_STREAMLIT=true"
        call :kill_process_by_pid !PID! "Streamlit Process"
    )
)

if "!FOUND_STREAMLIT!"=="false" (
    echo - No Streamlit processes found.
)

:: 4. Stop any system_start.bat processes
echo.
echo Step 4: Stopping system_start.bat processes...
set "FOUND_SYSTEM_START=false"
for /f "tokens=1,2 delims=," %%a in ('tasklist /nh /fi "imagename eq cmd.exe" /fo csv 2^>nul') do (
    set "IMAGE=%%~a"
    set "PID=%%~b"
    
    :: Check if this cmd process is running system_start.bat
    wmic process where "ProcessId=!PID!" get CommandLine 2>nul | findstr "system_start.bat" >nul
    if not errorlevel 1 (
        set "FOUND_SYSTEM_START=true"
        call :kill_process_by_pid !PID! "System Start Script"
    )
)

if "!FOUND_SYSTEM_START!"=="false" (
    echo - No system_start.bat processes found.
)

:: 5. Clean up any remaining UV processes related to the project
echo.
echo Step 5: Cleaning up UV processes...
set "FOUND_UV=false"
for /f "tokens=1,2 delims=," %%a in ('tasklist /nh /fi "imagename eq python.exe" /fo csv 2^>nul') do (
    set "IMAGE=%%~a"
    set "PID=%%~b"
    
    :: Check if this python process is related to UV and our project
    wmic process where "ProcessId=!PID!" get CommandLine 2>nul | findstr "CherryAI_0623" >nul
    if not errorlevel 1 (
        wmic process where "ProcessId=!PID!" get CommandLine 2>nul | findstr "uv" >nul
        if not errorlevel 1 (
            set "FOUND_UV=true"
            call :kill_process_by_pid !PID! "UV Process"
        )
    )
)

if "!FOUND_UV!"=="false" (
    echo - No project-related UV processes found.
)

:: 6. Verify all processes are stopped
echo.
echo Step 6: Final verification...
timeout /t 2 /nobreak >nul

set "REMAINING_PROCESSES="
tasklist /fi "imagename eq python.exe" 2>nul | findstr "pandas_server" >nul
if not errorlevel 1 (
    set "REMAINING_PROCESSES=%REMAINING_PROCESSES% pandas_server"
)

tasklist /fi "imagename eq python.exe" 2>nul | findstr "streamlit" >nul
if not errorlevel 1 (
    set "REMAINING_PROCESSES=%REMAINING_PROCESSES% streamlit"
)

if not "!REMAINING_PROCESSES!"=="" (
    echo ⚠️  Warning: Some processes may still be running:!REMAINING_PROCESSES!
    echo You may need to manually terminate them.
) else (
    echo ✅ All CherryAI processes successfully terminated.
)

echo.
echo ================================================
echo ✅ System shutdown complete.
echo ================================================

endlocal
pause 