@echo off
REM install_mcp_dependencies.bat - Install MCP Server Dependencies using UV

echo ================================================
echo  MCP Server Dependencies Installer (UV)
echo ================================================

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ UV is not installed!
    echo Please install UV first and run this script again.
    pause
    exit /b 1
)

echo ✅ UV found. Installing MCP dependencies...

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️ Virtual environment not found. Creating one...
    uv venv
    call .venv\Scripts\activate.bat
)

echo ================================================
echo Step 1: Core Dependencies
echo ================================================
uv pip install pandas numpy matplotlib seaborn plotly scipy
if %errorlevel% neq 0 (
    echo ❌ Core dependencies installation failed
    pause
    exit /b 1
)
echo ✅ Core dependencies installed

echo ================================================
echo Step 2: MCP Framework
echo ================================================
uv pip install fastmcp mcp uvicorn langchain-mcp-adapters
if %errorlevel% neq 0 (
    echo ❌ MCP framework installation failed
    pause
    exit /b 1
)
echo ✅ MCP framework installed

echo ================================================
echo Step 3: Machine Learning Core
echo ================================================
uv pip install scikit-learn xgboost
if %errorlevel% neq 0 (
    echo ❌ ML core installation failed
    pause
    exit /b 1
)
echo ✅ ML core installed

echo ================================================
echo Step 4: Advanced ML (Optional)
echo ================================================
echo Installing CatBoost...
uv pip install catboost
if %errorlevel% neq 0 (
    echo ⚠️ CatBoost installation failed (this is optional)
)

echo Installing LightGBM...
uv pip install lightgbm
if %errorlevel% neq 0 (
    echo ⚠️ LightGBM installation failed (this is optional)
)

echo Installing imbalanced-learn...
uv pip install imbalanced-learn
if %errorlevel% neq 0 (
    echo ⚠️ imbalanced-learn installation failed (this is optional)
)

echo ================================================
echo Step 5: Statistical Analysis
echo ================================================
echo Installing statsmodels...
uv pip install statsmodels
if %errorlevel% neq 0 (
    echo ⚠️ statsmodels installation failed (this is optional)
)

echo Installing pingouin...
uv pip install pingouin
if %errorlevel% neq 0 (
    echo ⚠️ pingouin installation failed (this is optional)
)

echo Installing mlxtend...
uv pip install mlxtend
if %errorlevel% neq 0 (
    echo ⚠️ mlxtend installation failed (this is optional)
)

echo ================================================
echo Step 6: Dimensionality Reduction & Explainability
echo ================================================
echo Installing umap-learn...
uv pip install umap-learn
if %errorlevel% neq 0 (
    echo ⚠️ umap-learn installation failed (this is optional)
)

echo Installing SHAP...
uv pip install shap
if %errorlevel% neq 0 (
    echo ⚠️ SHAP installation failed (this is optional)
)

echo Installing LIME...
uv pip install lime
if %errorlevel% neq 0 (
    echo ⚠️ LIME installation failed (this is optional)
)

echo ================================================
echo Step 7: Hyperparameter Optimization
echo ================================================
echo Installing Optuna...
uv pip install optuna
if %errorlevel% neq 0 (
    echo ⚠️ Optuna installation failed (this is optional)
)

echo ================================================
echo Step 8: Windows-specific
echo ================================================
echo Installing pywin32...
uv pip install pywin32
if %errorlevel% neq 0 (
    echo ⚠️ pywin32 installation failed (this is optional)
)

echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo ✅ Critical dependencies installed successfully!
echo ⚠️ Some optional packages may have failed - this is normal
echo.
echo You can now run the MCP servers using:
echo   mcp_server_start.bat
echo.
pause 