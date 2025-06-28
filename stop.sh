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

# Server names
SERVERS=(
    "orchestrator"
    "data_loader"
    "data_cleaning"
    "data_wrangling"
    "eda_tools"
    "data_viz"
    "feature_eng"
    "h2o_ml"
    "mlflow_tools"
    "sql_database"
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
        "data_loader") display_name="Data Loader" ;;
        "data_cleaning") display_name="Data Cleaning" ;;
        "data_wrangling") display_name="Data Wrangling" ;;
        "eda_tools") display_name="EDA Tools" ;;
        "data_viz") display_name="Data Visualization" ;;
        "feature_eng") display_name="Feature Engineering" ;;
        "h2o_ml") display_name="H2O ML" ;;
        "mlflow_tools") display_name="MLflow Tools" ;;
        "sql_database") display_name="SQL Database" ;;
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
    "ai_ds_team_data_loader_server.py"
    "ai_ds_team_data_cleaning_server.py"
    "ai_ds_team_data_wrangling_server.py"
    "ai_ds_team_eda_tools_server.py"
    "ai_ds_team_data_visualization_server.py"
    "ai_ds_team_feature_engineering_server.py"
    "ai_ds_team_h2o_ml_server.py"
    "ai_ds_team_mlflow_tools_server.py"
    "ai_ds_team_sql_database_server.py"
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

# Check ports
PORTS=(8100 8307 8306 8309 8312 8308 8310 8313 8314 8311 8501)
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
    echo "   📂 Data Loader (8307)"
    echo "   🧹 Data Cleaning (8306)"
    echo "   🛠️ Data Wrangling (8309)"
    echo "   🔍 EDA Tools (8312)"
    echo "   🎨 Data Visualization (8308)"
    echo "   🔧 Feature Engineering (8310)"
    echo "   🤖 H2O ML (8313)"
    echo "   📈 MLflow Tools (8314)"
    echo "   🗄️ SQL Database (8311)"
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