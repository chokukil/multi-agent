@echo off
setlocal

echo 🛑 Stopping A2A Data Science Team System for Windows
echo ================================================

echo 🎨 Stopping Streamlit...
taskkill /F /IM python.exe /FI "COMMANDLINE eq *streamlit*ai.py*" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Streamlit stopped
) else (
    echo ⚠️  Streamlit was not running
)

echo.
echo 🤖 Stopping A2A servers...

echo Stopping Orchestrator Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Orchestrator Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *orchestrator_server.py*" 2>nul

echo Stopping SQL Data Analyst Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SQL Data Analyst Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *sql_data_analyst_server.py*" 2>nul

echo Stopping Data Visualization Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Visualization Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *data_visualization_server.py*" 2>nul

echo Stopping EDA Tools Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq EDA Tools Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *eda_tools_server.py*" 2>nul

echo Stopping Feature Engineering Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Feature Engineering Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *feature_engineering_server.py*" 2>nul

echo Stopping Data Cleaning Server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Cleaning Server" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *data_cleaning_server.py*" 2>nul

echo.
echo 🧹 Final cleanup...

REM Kill any remaining A2A related processes
taskkill /F /IM python.exe /FI "COMMANDLINE eq *a2a_ds_servers*" 2>nul

echo.
echo 🔍 Checking port status...

REM Check if ports are still in use (pandas data analyst port 8200 제거됨)
netstat -an | findstr ":8100 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8100 (Orchestrator): Free
) else (
    echo ⚠️  Port 8100 (Orchestrator): Still in use
)

netstat -an | findstr ":8201 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8201 (SQL Analyst): Free
) else (
    echo ⚠️  Port 8201 (SQL Analyst): Still in use
)

netstat -an | findstr ":8202 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8202 (Data Viz): Free
) else (
    echo ⚠️  Port 8202 (Data Viz): Still in use
)

netstat -an | findstr ":8203 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8203 (EDA Tools): Free
) else (
    echo ⚠️  Port 8203 (EDA Tools): Still in use
)

netstat -an | findstr ":8204 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8204 (Feature Engineering): Free
) else (
    echo ⚠️  Port 8204 (Feature Engineering): Still in use
)

netstat -an | findstr ":8205 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8205 (Data Cleaning): Free
) else (
    echo ⚠️  Port 8205 (Data Cleaning): Still in use
)

netstat -an | findstr ":8501 " >nul 2>&1
if %errorlevel% neq 0 (
    echo ✅ Port 8501 (Streamlit): Free
) else (
    echo ⚠️  Port 8501 (Streamlit): Still in use
)

echo.
echo ================================================
echo ✅ A2A Data Science System shutdown complete!
echo ================================================
echo 🎯 Stopped Services:
echo    📱 Streamlit UI (8501)
echo    🎯 Orchestrator (8100)
echo    🗃️  SQL Data Analyst (8201)
echo    📈 Data Visualization (8202)
echo    🔍 EDA Tools (8203)
echo    🔧 Feature Engineering (8204)
echo    🧹 Data Cleaning (8205)
echo.
echo 💡 If any ports are still in use, you may need to:
echo    - Restart the terminal/command prompt
echo    - Check Task Manager for remaining Python processes
echo    - Use netstat -ano to find specific PIDs and kill them
echo ================================================

endlocal
pause 