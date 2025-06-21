@echo off
REM system_stop.bat - Stop all CherryAI services (MCP servers + Streamlit)

title CherryAI System Stop

echo ================================================
echo      CherryAI System Shutdown Script
echo ================================================

echo Stopping CherryAI services...

REM Stop Streamlit processes
echo Stopping Streamlit applications...
taskkill /f /im "streamlit.exe" >nul 2>&1
taskkill /f /im "python.exe" /fi "WINDOWTITLE eq Streamlit*" >nul 2>&1

REM Stop Python processes that might be MCP servers
echo Stopping MCP server processes...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| find "python.exe"') do (
    taskkill /f /pid %%i >nul 2>&1
)

REM Stop UV processes
echo Stopping UV processes...
taskkill /f /im "uv.exe" >nul 2>&1

REM Stop any remaining CherryAI related processes
echo Cleaning up remaining processes...
taskkill /f /fi "WINDOWTITLE eq *MCP*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq *CherryAI*" >nul 2>&1

echo.
echo ✅ All CherryAI services have been stopped.
echo.
echo Services that were terminated:
echo - Streamlit web application
echo - All MCP servers
echo - Related Python processes
echo.
echo You can now safely restart the system using:
echo   system_start.bat or quick_start.bat
echo.
pause 