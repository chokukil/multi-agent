@echo off
echo ================================================
echo     Cherry AI - MCP Dependencies Installation
echo ================================================
echo.

echo 🔧 Installing MCP core packages...
pip install mcp>=1.0.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install mcp package
    goto :error
)

echo 🔧 Installing MCP client packages...
pip install fastmcp>=0.3.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install fastmcp package
    goto :error
)

echo 🔧 Installing MCP server dependencies...
pip install uvicorn>=0.30.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install uvicorn package
    goto :error
)

echo 🔧 Installing LangChain MCP adapters...
pip install langchain-mcp-adapters>=0.1.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install langchain-mcp-adapters package
    goto :error
)

echo 🔧 Installing additional MCP dependencies...
pip install aiohttp>=3.9.0 pydantic>=2.0.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install additional dependencies
    goto :error
)

echo.
echo ✅ All MCP packages installed successfully!
echo.
echo 📋 Installed packages:
echo   - mcp (core MCP package)
echo   - fastmcp (fast MCP server)
echo   - uvicorn (ASGI server)
echo   - langchain-mcp-adapters (LangChain integration)
echo   - aiohttp (async HTTP client)
echo   - pydantic (data validation)
echo.
echo 🚀 You can now restart the system to use MCP tools!
echo    Run: .\system_start.bat
echo.
pause
exit /b 0

:error
echo.
echo ❌ Installation failed!
echo.
echo 💡 Try these solutions:
echo   1. Update pip: python -m pip install --upgrade pip
echo   2. Check Python version (3.8+ required)
echo   3. Try with --user flag: pip install --user [package]
echo   4. Use virtual environment
echo.
pause
exit /b 1 