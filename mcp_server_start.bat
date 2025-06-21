@echo off
REM mcp_server_start.bat - Start MCP Servers using UV package manager

echo ================================================
echo      MCP Server Batch Launcher (UV)
echo ================================================

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] UV is not installed!
    echo.
    echo Please install UV first:
    echo   1. Download and install from: https://docs.astral.sh/uv/getting-started/installation/
    echo   2. Or run: winget install --id=astral-sh.uv -e
    echo   3. Or use pip: pip install uv
    echo.
    pause
    exit /b 1
)

echo [OK] UV found. Checking project setup...

REM Create virtual environment with uv if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment with UV...
    uv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies using uv with pyproject.toml
echo Installing dependencies with UV...
echo This will install all required packages for MCP servers!
uv sync
if %errorlevel% neq 0 (
    echo [ERROR] UV sync failed. Trying alternative installation methods...
    echo.
    echo [WARNING] Installing from pyproject.toml...
    uv pip install -e .
    if %errorlevel% neq 0 (
        echo [ERROR] pyproject.toml installation failed
        echo Falling back to requirements.txt...
        uv pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERROR] Requirements.txt installation also failed
            echo Please check your dependencies manually
            pause
            exit /b 1
        )
    )
)

REM Install critical MCP server dependencies individually
echo Installing critical MCP server dependencies...
uv pip install fastmcp mcp uvicorn xgboost scikit-learn scipy pandas numpy matplotlib seaborn plotly

REM Install optional advanced ML dependencies (non-critical)
echo Installing optional ML dependencies...
uv pip install --no-deps catboost lightgbm imbalanced-learn statsmodels pingouin mlxtend umap-learn shap lime optuna 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Some optional ML packages failed to install. This is not critical.
)

REM Install Windows-specific dependencies
echo Installing Windows-specific dependencies...
uv pip install pywin32

echo [OK] Dependencies installed successfully!

REM Create necessary directories
echo Setting up directories...
if not exist "prompt-configs" mkdir prompt-configs
if not exist "mcp-configs" mkdir mcp-configs
if not exist "results" mkdir results
if not exist "reports" mkdir reports
if not exist "logs" mkdir logs
if not exist "generated_code" mkdir generated_code

REM Check for .env file
if not exist ".env" (
    echo ================================================
    echo WARNING: .env file not found!
    echo Creating example .env file...
    echo.
    echo # OpenAI API Key > .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo # Anthropic API Key >> .env
    echo ANTHROPIC_API_KEY=your_anthropic_api_key_here >> .env
    echo # Other API Keys >> .env
    echo PERPLEXITY_API_KEY=your_perplexity_api_key_here >> .env
    echo LANGCHAIN_API_KEY=your_langchain_api_key_here >> .env
    echo.
    echo Please edit .env file with your actual API keys
    echo ================================================
)

echo ================================================
echo Checking MCP server configuration...
echo ================================================

REM Check if MCP configuration files exist
if not exist "mcp_config.py" (
    echo [ERROR] Missing: mcp_config.py
    echo    Please ensure all required files are present
    goto :end
)

REM Show current MCP configuration
echo Displaying MCP server configuration...
uv run python mcp_config.py

echo ================================================
echo Starting MCP servers...
echo ================================================

REM Start each MCP server in background with correct port environment variables
echo Starting File Management Server...
start "MCP File Management" /min cmd /c "uv run python mcp-servers/mcp_file_management.py --port 8006 || pause"

echo Starting Data Science Tools Server...
start "MCP Data Science" /min cmd /c "set SERVER_PORT=8007 && uv run python mcp-servers/mcp_data_science_tools.py || pause"

echo Starting Data Preprocessing Server...
start "MCP Data Preprocessing" /min cmd /c "set SERVER_PORT=8017 && uv run python mcp-servers/mcp_data_preprocessing_tools.py || pause"

echo Starting Statistical Analysis Server...
start "MCP Statistical Analysis" /min cmd /c "set SERVER_PORT=8018 && uv run python mcp-servers/mcp_statistical_analysis_tools.py || pause"

echo Starting Advanced ML Tools Server...
start "MCP Advanced ML" /min cmd /c "set SERVER_PORT=8016 && uv run python mcp-servers/mcp_advanced_ml_tools.py || pause"

echo Starting Semiconductor Yield Analysis Server...
start "MCP Yield Analysis" /min cmd /c "set SERVER_PORT=8008 && uv run python mcp-servers/mcp_semiconductor_yield_analysis.py || pause"

echo Starting Process Control Charts Server...
start "MCP Process Control" /min cmd /c "set SERVER_PORT=8009 && uv run python mcp-servers/mcp_process_control_charts.py || pause"

echo Starting Equipment Analysis Server...
start "MCP Equipment Analysis" /min cmd /c "set SERVER_PORT=8010 && uv run python mcp-servers/mcp_semiconductor_equipment_analysis.py || pause"

echo Starting Defect Pattern Analysis Server...
start "MCP Defect Pattern" /min cmd /c "set SERVER_PORT=8011 && uv run python mcp-servers/mcp_defect_pattern_analysis.py || pause"

echo Starting Process Optimization Server...
start "MCP Process Optimization" /min cmd /c "set SERVER_PORT=8012 && uv run python mcp-servers/mcp_process_optimization.py || pause"

echo Starting Semiconductor Process Tools Server...
start "MCP Semiconductor Process" /min cmd /c "set SERVER_PORT=8020 && uv run python mcp-servers/mcp_semiconductor_process_tools.py || pause"

echo Starting Time Series Analysis Server...
start "MCP Time Series" /min cmd /c "set SERVER_PORT=8013 && uv run python mcp-servers/mcp_timeseries_analysis.py || pause"

echo Starting Anomaly Detection Server...
start "MCP Anomaly Detection" /min cmd /c "set SERVER_PORT=8014 && uv run python mcp-servers/mcp_anomaly_detection.py || pause"

echo Starting Report Writing Tools Server...
start "MCP Report Writing" /min cmd /c "set SERVER_PORT=8019 && uv run python mcp-servers/mcp_report_writing_tools.py || pause"

REM Wait for servers to start
echo ================================================
echo Waiting for MCP servers to initialize...
echo ================================================
timeout /t 30 /nobreak >nul

echo [OK] All MCP servers startup initiated!
echo.
echo Check individual server windows for detailed status.
echo To stop all servers, close their respective windows or use Ctrl+C in each window.
echo.
echo Server URLs (default):
echo   - File Management: http://localhost:8006
echo   - Data Science: http://localhost:8007
echo   - Semiconductor Yield: http://localhost:8008
echo   - Process Control: http://localhost:8009
echo   - Equipment Analysis: http://localhost:8010
echo   - Defect Pattern: http://localhost:8011
echo   - Process Optimization: http://localhost:8012
echo   - Time Series Analysis: http://localhost:8013
echo   - Anomaly Detection: http://localhost:8014
echo   - Advanced ML: http://localhost:8016
echo   - Data Preprocessing: http://localhost:8017
echo   - Statistical Analysis: http://localhost:8018
echo   - Report Writing: http://localhost:8019
echo   - Semiconductor Process: http://localhost:8020

:end
echo ================================================
echo MCP Server startup complete.
echo ================================================
pause 