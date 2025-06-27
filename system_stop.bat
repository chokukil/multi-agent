@echo off
setlocal enabledelayedexpansion

:: --- Configuration ---
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "PID_FILE=%LOG_DIR%\pandas_server.pid"

echo ================================================
echo Stopping CherryAI A2A Data Science System for Windows...
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

:: 1. Stop A2A Data Science Servers
echo.
echo Step 1: Stopping A2A Data Science Servers...

echo Stopping Data Loader Server (Port 8000)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Loader Server" 2>nul
if errorlevel 1 (echo - Data Loader Server not running) else (echo ✅ Data Loader Server stopped)

echo Stopping Pandas Data Analyst Server (Port 8001)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Pandas Data Analyst Server" 2>nul
if errorlevel 1 (echo - Pandas Data Analyst Server not running) else (echo ✅ Pandas Data Analyst Server stopped)

echo Stopping SQL Data Analyst Server (Port 8002)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SQL Data Analyst Server" 2>nul
if errorlevel 1 (echo - SQL Data Analyst Server not running) else (echo ✅ SQL Data Analyst Server stopped)

echo Stopping EDA Tools Server (Port 8003)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq EDA Tools Server" 2>nul
if errorlevel 1 (echo - EDA Tools Server not running) else (echo ✅ EDA Tools Server stopped)

echo Stopping Data Visualization Server (Port 8004)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Visualization Server" 2>nul
if errorlevel 1 (echo - Data Visualization Server not running) else (echo ✅ Data Visualization Server stopped)

echo Stopping Orchestrator Server (Port 8100)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Orchestrator Server" 2>nul
if errorlevel 1 (echo - Orchestrator Server not running) else (echo ✅ Orchestrator Server stopped)

:: 2. Stop any remaining A2A server processes by port
echo.
echo Step 2: Stopping A2A servers by port...
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8000 terminated
    )
)

netstat -ano | findstr ":8001 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8001 terminated
    )
)

netstat -ano | findstr ":8002 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8002 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8002 terminated
    )
)

netstat -ano | findstr ":8003 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8003 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8003 terminated
    )
)

netstat -ano | findstr ":8004 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8004 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8004 terminated
    )
)

netstat -ano | findstr ":8100 " | findstr "LISTENING" >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8100 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
        echo ✅ Process on port 8100 terminated
    )
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