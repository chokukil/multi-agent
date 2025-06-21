@echo off
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