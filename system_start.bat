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

echo Starting Data Loader Server (Port 8000)...
start "Data Loader Server" /B uv run python a2a_ds_servers/data_loader_server.py

echo Starting Pandas Data Analyst Server (Port 8001)...
start "Pandas Data Analyst Server" /B uv run python a2a_ds_servers/pandas_data_analyst_server.py

echo Starting SQL Data Analyst Server (Port 8002)...
start "SQL Data Analyst Server" /B uv run python a2a_ds_servers/sql_data_analyst_server.py

echo Starting EDA Tools Server (Port 8003)...
start "EDA Tools Server" /B uv run python a2a_ds_servers/eda_tools_server.py

echo Starting Data Visualization Server (Port 8004)...
start "Data Visualization Server" /B uv run python a2a_ds_servers/data_visualization_server.py

echo Starting Orchestrator Server (Port 8100)...
start "Orchestrator Server" /B uv run python a2a_ds_servers/orchestrator_server.py

echo ✅ All A2A servers starting in the background...

echo.
echo Waiting 10 seconds for all servers to initialize...
timeout /t 10 /nobreak > nul

echo.
echo ================================================
echo Step 2: Starting Streamlit Application in this terminal.
echo You will see live logs below. Press Ctrl+C to stop.
echo ================================================

REM Run streamlit in the foreground
uv run streamlit run app.py

echo.
echo Streamlit app stopped. Cleaning up A2A servers...

REM Kill all A2A server processes
echo Stopping A2A Data Science Servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Loader Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Pandas Data Analyst Server" 2>nul  
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SQL Data Analyst Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq EDA Tools Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Data Visualization Server" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Orchestrator Server" 2>nul

echo ✅ A2A Data Science System shutdown complete.

endlocal 