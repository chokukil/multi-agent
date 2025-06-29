@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "LOG_DIR=%PROJECT_ROOT%logs"
set "A2A_LOG_DIR=%PROJECT_ROOT%a2a_ds_servers\logs"

echo ================================================
echo Starting CherryAI A2A Data Science System for Windows...
echo ================================================

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%A2A_LOG_DIR%" mkdir "%A2A_LOG_DIR%"

echo.
echo ================================================
echo Step 1: Starting A2A Data Science Servers...
echo ================================================

echo Starting Orchestrator Server (Port 8100)...
start "Orchestrator Server" /B uv run python a2a_ds_servers/a2a_orchestrator_v6.py

echo Starting SQL Data Analyst Server (Port 8201)...
start "SQL Data Analyst Server" /B uv run python a2a_ds_servers/sql_data_analyst_server.py

echo Starting Data Visualization Server (Port 8202)...
start "Data Visualization Server" /B uv run python a2a_ds_servers/data_visualization_server.py

echo Starting EDA Tools Server (Port 8203)...
start "EDA Tools Server" /B uv run python a2a_ds_servers/eda_tools_server.py

echo Starting Feature Engineering Server (Port 8204)...
start "Feature Engineering Server" /B uv run python a2a_ds_servers/feature_engineering_server.py

echo Starting Data Cleaning Server (Port 8205)...
start "Data Cleaning Server" /B uv run python a2a_ds_servers/data_cleaning_server.py

echo ✅ All A2A servers starting in the background...

echo.
echo Waiting 15 seconds for all servers to initialize...
timeout /t 15 /nobreak > nul

echo.
echo ================================================
echo Step 2: Verifying A2A Server Status...
echo ================================================

REM Check if servers are responding
echo Checking Orchestrator (8100)...
curl -s -f "http://localhost:8100/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Orchestrator: Ready
) else (
    echo ❌ Orchestrator: Not responding
)

echo Checking SQL Data Analyst (8201)...
curl -s -f "http://localhost:8201/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ SQL Data Analyst: Ready
) else (
    echo ❌ SQL Data Analyst: Not responding
)

echo Checking Data Visualization (8202)...
curl -s -f "http://localhost:8202/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Data Visualization: Ready
) else (
    echo ❌ Data Visualization: Not responding
)

echo Checking EDA Tools (8203)...
curl -s -f "http://localhost:8203/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ EDA Tools: Ready
) else (
    echo ❌ EDA Tools: Not responding
)

echo Checking Feature Engineering (8204)...
curl -s -f "http://localhost:8204/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Feature Engineering: Ready
) else (
    echo ❌ Feature Engineering: Not responding
)

echo Checking Data Cleaning (8205)...
curl -s -f "http://localhost:8205/.well-known/agent.json" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Data Cleaning: Ready
) else (
    echo ❌ Data Cleaning: Not responding
)

echo.
echo ================================================
echo Step 3: Starting Streamlit Application...
echo ================================================
echo 🎉 A2A Data Science System is operational!
echo ================================================
echo 📊 Available Services:
echo    🎯 Orchestrator:          http://localhost:8100
echo    🗃️  SQL Data Analyst:      http://localhost:8201
echo    📈 Data Visualization:    http://localhost:8202
echo    🔍 EDA Tools:             http://localhost:8203
echo    🔧 Feature Engineering:   http://localhost:8204
echo    🧹 Data Cleaning:         http://localhost:8205
echo.
echo 🌐 Starting Streamlit UI on http://localhost:8501
echo 🛑 Press Ctrl+C to stop the system
echo ================================================

REM Run streamlit in the foreground
uv run streamlit run ai.py

echo.
echo Streamlit app stopped. Cleaning up A2A servers...

REM Kill all A2A server processes - 최신 서버명 기준
echo Stopping A2A Data Science Servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Orchestrator Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SQL Data Analyst Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Visualization Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq EDA Tools Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Feature Engineering Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Cleaning Server" 2>nul

echo.
echo 🧹 Additional cleanup...
REM Force kill any remaining A2A processes
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "orchestrator_server.py"') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "sql_data_analyst_server.py"') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "data_visualization_server.py"') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "eda_tools_server.py"') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "feature_engineering_server.py"') do taskkill /F /PID %%i 2>nul
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "data_cleaning_server.py"') do taskkill /F /PID %%i 2>nul

echo ✅ A2A Data Science System shutdown complete.

endlocal 