@echo off
REM quick_start.bat - Quick Start for CherryAI (Background MCP servers + Streamlit)

title CherryAI Quick Start

echo ================================================
echo       CherryAI Quick Start Launcher
echo ================================================

REM Check prerequisites
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Run: install_mcp_dependencies.bat
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ❌ app.py not found!
    pause
    exit /b 1
)

echo ✅ Starting CherryAI system...

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting MCP servers in background...

REM Start MCP servers silently in background
start /min "" cmd /c "uv run python mcp-servers/mcp_file_management.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_data_science_tools.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_data_preprocessing_tools.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_statistical_analysis_tools.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_advanced_ml_tools.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_semiconductor_yield_analysis.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_process_control_charts.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_semiconductor_equipment_analysis.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_defect_pattern_analysis.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_process_optimization.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_semiconductor_process_tools.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_timeseries_analysis.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_anomaly_detection.py"
start /min "" cmd /c "uv run python mcp-servers/mcp_report_writing_tools.py"

echo Waiting for servers to initialize...
timeout /t 15 /nobreak >nul

echo.
echo ✅ Starting Streamlit application...
echo.
echo 🌐 Opening http://localhost:8501
echo.

REM Start Streamlit
uv run streamlit run app.py 