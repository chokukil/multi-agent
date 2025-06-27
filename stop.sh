#!/bin/bash

# A2A Data Science Team System Stop Script
# Stops all A2A servers and Streamlit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PID_DIR="$PROJECT_ROOT/logs/pids"

echo "🛑 Stopping A2A Data Science Team System"
echo "================================================"

# Check if PID directory exists
if [ ! -d "$PID_DIR" ]; then
    echo "📁 No PID directory found. System may not be running."
    exit 0
fi

# Server names - 최신 구성
SERVERS=(
    "orchestrator"
    "pandas_analyst"
    "sql_analyst"
    "data_viz"
    "eda_tools"
    "feature_eng"
    "data_cleaning"
)

stopped_count=0
total_count=0

# Stop Streamlit first
echo "🎨 Stopping Streamlit..."
if [ -f "$PID_DIR/streamlit.pid" ]; then
    STREAMLIT_PID=$(cat "$PID_DIR/streamlit.pid")
    if [ -n "$STREAMLIT_PID" ] && ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
        kill "$STREAMLIT_PID"
        sleep 2
        if ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
            kill -9 "$STREAMLIT_PID" 2>/dev/null
        fi
        echo "✅ Streamlit stopped"
        stopped_count=$((stopped_count + 1))
    else
        echo "⚠️  Streamlit was not running"
    fi
    rm -f "$PID_DIR/streamlit.pid"
    total_count=$((total_count + 1))
else
    echo "⚠️  Streamlit PID file not found"
fi

# Stop A2A servers
echo ""
echo "🤖 Stopping A2A servers..."
for name in "${SERVERS[@]}"; do
    PID_FILE="$PID_DIR/${name}.pid"
    total_count=$((total_count + 1))
    
    # Get display name
    case $name in
        "orchestrator") display_name="Orchestrator" ;;
        "pandas_analyst") display_name="Pandas Data Analyst" ;;
        "sql_analyst") display_name="SQL Data Analyst" ;;
        "data_viz") display_name="Data Visualization" ;;
        "eda_tools") display_name="EDA Tools" ;;
        "feature_eng") display_name="Feature Engineering" ;;
        "data_cleaning") display_name="Data Cleaning" ;;
        *) display_name="$name" ;;
    esac
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo "🔄 Stopping $display_name (PID: $PID)..."
            kill "$PID"
            sleep 1
            # Force kill if still running
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null
                echo "   ⚡ Force killed $display_name"
            else
                echo "   ✅ $display_name stopped gracefully"
            fi
            stopped_count=$((stopped_count + 1))
        else
            echo "⚠️  $display_name was not running"
        fi
        rm -f "$PID_FILE"
    else
        echo "⚠️  $display_name PID file not found"
    fi
done

# Also kill any remaining processes by name (backup cleanup)
echo ""
echo "🧹 Cleaning up any remaining processes..."

# Kill any remaining A2A server processes - 최신 서버 이름
SERVER_SCRIPTS=(
    "orchestrator_server.py"
    "pandas_data_analyst_server.py"
    "sql_data_analyst_server.py"
    "data_visualization_server.py"
    "eda_tools_server.py"
    "feature_engineering_server.py"
    "data_cleaning_server.py"
)

for script in "${SERVER_SCRIPTS[@]}"; do
    PIDS=$(pgrep -f "$script" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "🔄 Found remaining $script processes: $PIDS"
        kill $PIDS 2>/dev/null || true
        sleep 1
        PIDS=$(pgrep -f "$script" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            kill -9 $PIDS 2>/dev/null || true
        fi
    fi
done

# Kill any remaining Streamlit processes
STREAMLIT_PIDS=$(pgrep -f "streamlit.*app.py" 2>/dev/null || true)
if [ -n "$STREAMLIT_PIDS" ]; then
    echo "🔄 Found remaining Streamlit processes: $STREAMLIT_PIDS"
    kill $STREAMLIT_PIDS 2>/dev/null || true
    sleep 1
    STREAMLIT_PIDS=$(pgrep -f "streamlit.*app.py" 2>/dev/null || true)
    if [ -n "$STREAMLIT_PIDS" ]; then
        kill -9 $STREAMLIT_PIDS 2>/dev/null || true
    fi
fi

# Check final status
echo ""
echo "🔍 Final status check..."

# Check ports - 최신 포트 번호
PORTS=(8100 8200 8201 8202 8203 8204 8205 8501)
ports_in_use=0
for port in "${PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is still in use"
        ports_in_use=$((ports_in_use + 1))
    fi
done

# Summary
echo ""
echo "================================================"
if [ $ports_in_use -eq 0 ]; then
    echo "✅ A2A Data Science System completely stopped!"
    echo "📊 Stopped $stopped_count out of $total_count services"
    echo "🌐 All ports are now free"
    echo ""
    echo "🎯 Stopped Services:"
    echo "   📱 Streamlit UI (8501)"
    echo "   🎯 Orchestrator (8100)"
    echo "   🐼 Pandas Data Analyst (8200)"
    echo "   🗃️  SQL Data Analyst (8201)"
    echo "   📈 Data Visualization (8202)"
    echo "   🔍 EDA Tools (8203)"
    echo "   🔧 Feature Engineering (8204)"
    echo "   🧹 Data Cleaning (8205)"
else
    echo "⚠️  System mostly stopped, but $ports_in_use ports still in use"
    echo "📊 Stopped $stopped_count out of $total_count services"
    echo "💡 You may need to manually kill remaining processes"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   - Check running processes: ps aux | grep python"
    echo "   - Check port usage: lsof -i :PORT_NUMBER"
    echo "   - Force kill: kill -9 PID"
fi
echo "================================================"

# Clean up empty PID directory if no files remain
if [ -d "$PID_DIR" ] && [ -z "$(ls -A "$PID_DIR" 2>/dev/null)" ]; then
    rmdir "$PID_DIR" 2>/dev/null || true
fi 