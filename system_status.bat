@echo off
setlocal

echo 🔍 A2A Data Science System Status Check
echo ================================================

set running_count=0
set total_count=8

echo 🤖 Checking A2A Servers:
echo ────────────────────────────────────────────────

REM Check Orchestrator (8100)
echo Checking Orchestrator (Port 8100)...
netstat -an | findstr ":8100 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8100/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Orchestrator: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Orchestrator: Port occupied but not responding properly
    )
) else (
    echo ❌ Orchestrator: Not running
)

REM Check Pandas Data Analyst (8200)
echo Checking Pandas Data Analyst (Port 8200)...
netstat -an | findstr ":8200 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8200/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Pandas Data Analyst: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Pandas Data Analyst: Port occupied but not responding properly
    )
) else (
    echo ❌ Pandas Data Analyst: Not running
)

REM Check SQL Data Analyst (8201)
echo Checking SQL Data Analyst (Port 8201)...
netstat -an | findstr ":8201 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8201/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ SQL Data Analyst: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  SQL Data Analyst: Port occupied but not responding properly
    )
) else (
    echo ❌ SQL Data Analyst: Not running
)

REM Check Data Visualization (8202)
echo Checking Data Visualization (Port 8202)...
netstat -an | findstr ":8202 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8202/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Data Visualization: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Data Visualization: Port occupied but not responding properly
    )
) else (
    echo ❌ Data Visualization: Not running
)

REM Check EDA Tools (8203)
echo Checking EDA Tools (Port 8203)...
netstat -an | findstr ":8203 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8203/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ EDA Tools: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  EDA Tools: Port occupied but not responding properly
    )
) else (
    echo ❌ EDA Tools: Not running
)

REM Check Feature Engineering (8204)
echo Checking Feature Engineering (Port 8204)...
netstat -an | findstr ":8204 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8204/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Feature Engineering: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Feature Engineering: Port occupied but not responding properly
    )
) else (
    echo ❌ Feature Engineering: Not running
)

REM Check Data Cleaning (8205)
echo Checking Data Cleaning (Port 8205)...
netstat -an | findstr ":8205 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8205/.well-known/agent.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Data Cleaning: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Data Cleaning: Port occupied but not responding properly
    )
) else (
    echo ❌ Data Cleaning: Not running
)

echo.
echo 🎨 Checking Streamlit UI:
echo ────────────────────────────────────────────────

REM Check Streamlit (8501)
echo Checking Streamlit UI (Port 8501)...
netstat -an | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    curl -s -f "http://localhost:8501" >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Streamlit UI: Running
        set /a running_count+=1
    ) else (
        echo ⚠️  Streamlit UI: Port occupied but not responding properly
    )
) else (
    echo ❌ Streamlit UI: Not running
)

echo.
echo 📊 System Summary:
echo ────────────────────────────────────────────────
echo Services running: %running_count%/%total_count%

if %running_count% equ %total_count% (
    echo 🎉 All services are operational!
    echo.
    echo 🌐 Access URLs:
    echo    📱 Streamlit UI:        http://localhost:8501
    echo    🎯 Orchestrator:        http://localhost:8100
    echo    🐼 Pandas Data Analyst: http://localhost:8200
    echo    🗃️  SQL Data Analyst:    http://localhost:8201
    echo    📈 Data Visualization:  http://localhost:8202
    echo    🔍 EDA Tools:           http://localhost:8203
    echo    🔧 Feature Engineering: http://localhost:8204
    echo    🧹 Data Cleaning:       http://localhost:8205
    echo.
    echo 🔧 Quick Actions:
    echo    system_stop.bat         - Stop all services
    echo    system_start.bat        - Start all services
    echo    system_status.bat       - Check system status
) else if %running_count% equ 0 (
    echo 🛑 System is not running
    echo.
    echo 💡 To start the system:
    echo    system_start.bat
) else (
    echo ⚠️  Partial system failure ^(%running_count%/%total_count% services running^)
    echo.
    echo 💡 Recommended actions:
    echo    system_stop.bat ^&^& system_start.bat  - Restart all services
    echo    Check logs in logs\a2a_servers\        - Check server logs
)

echo.
echo 🕒 Status checked at: %date% %time%
echo ================================================

endlocal
pause 