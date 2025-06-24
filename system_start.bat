@echo off
:: ==============================================================================
:: CherryAI - System Start Script (BAT)
:: ==============================================================================
:: This script starts all the necessary agent servers for the CherryAI system
:: to be fully operational. It starts each agent in a new background window.
::
:: Use system_stop.bat to terminate these agents.
:: ==============================================================================

echo.
echo #############################################################
echo #         Starting all CherryAI Agent Servers
echo #############################################################
echo.

echo [INFO] Starting agents in separate background windows...
echo.

START "DataloaderAgent" /B uv run python -m mcp_agents.mcp_dataloader_agent
echo   - Dataloader Agent (Port 8001) launched.

START "WranglingAgent" /B uv run python -m mcp_agents.mcp_datawrangling_agent
echo   - Data Wrangling Agent (Port 8002) launched.

START "VisualizationAgent" /B uv run python -m mcp_agents.mcp_datavisualization_agent
echo   - Data Visualization Agent (Port 8003) launched.

echo.
echo [SUCCESS] All agents have been launched.
echo           They are running in the background. Use the Streamlit UI
echo           or run 'system_stop.bat' to terminate them.
echo.

timeout /t 3 /nobreak > nul

REM system_start.bat - Complete System Startup: MCP Servers + Streamlit App

echo ================================================
echo      CherryAI System Startup Script
echo ================================================
echo.
echo This script will:
echo 1. Start all MCP servers
echo 2. Wait for initialization
echo 3. Launch the main Streamlit application
echo.
echo ================================================

REM Check if UV is available
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] UV is not installed!
    echo Please install UV first and then run this script.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run install_mcp_dependencies.bat first.
    pause
    exit /b 1
)

echo [OK] Prerequisites check passed
echo.

echo ================================================
echo Step 1: Starting MCP Servers
echo ================================================

REM Start MCP servers
echo Launching MCP server startup script...
call mcp_server_start.bat

REM Additional wait time to ensure all servers are fully loaded
echo.
echo ================================================
echo Step 2: Waiting for MCP servers to stabilize...
echo ================================================
echo Please wait while all MCP servers finish initialization...

timeout /t 45 /nobreak >nul

echo.
echo ================================================
echo Step 3: Starting Streamlit Application
echo ================================================

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found!
    echo Please ensure app.py is in the current directory.
    pause
    exit /b 1
)

echo Starting CherryAI Streamlit application...
echo.
echo [OK] System startup complete!
echo.
echo The application will open in your default web browser.
echo URL: http://localhost:8501
echo.
echo To stop the system:
echo 1. Close this window (Ctrl+C)
echo 2. Close all MCP server windows
echo.
echo ================================================

REM Start Streamlit app
uv run streamlit run app.py

echo.
echo ================================================
echo System shutdown initiated
echo ================================================
echo Please close all MCP server windows manually.
pause 