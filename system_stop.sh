#!/bin/bash

# A2A Data Science Team System Shutdown Script

# Get the directory of the script and set the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PID_DIR="$PROJECT_ROOT/logs/pids"

echo "================================================"
echo "🛑 Stopping A2A Data Science System..."
echo "================================================"

# Server definitions (must match the start script)
declare -A SERVERS=(
    ["orchestrator"]="a2a_ds_servers/orchestrator_server.py:8100"
    ["data_loader"]="a2a_ds_servers/ai_ds_team_data_loader_server.py:8200"
    ["data_cleaning"]="a2a_ds_servers/ai_ds_team_data_cleaning_server.py:8201"
    ["data_wrangling"]="a2a_ds_servers/ai_ds_team_data_wrangling_server.py:8202"
    ["eda_tools"]="a2a_ds_servers/ai_ds_team_eda_tools_server.py:8203"
    ["data_viz"]="a2a_ds_servers/ai_ds_team_data_visualization_server.py:8204"
    ["feature_eng"]="a2a_ds_servers/ai_ds_team_feature_engineering_server.py:8205"
    ["h2o_ml"]="a2a_ds_servers/ai_ds_team_h2o_ml_server.py:8206"
    ["mlflow_tools"]="a2a_ds_servers/ai_ds_team_mlflow_tools_server.py:8207"
    ["sql_database"]="a2a_ds_servers/ai_ds_team_sql_database_server.py:8208"
)

# Kill all background processes
all_stopped=true
for name in "${!SERVERS[@]}"; do
    PID_FILE="$PID_DIR/${name}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo "🔄 Stopping $name (PID: $PID)..."
            kill "$PID"
            sleep 1
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null
            fi
            
            if ! ps -p "$PID" > /dev/null 2>&1; then
                 echo "   ✅ $name stopped."
            else
                 echo "   ❌ Failed to stop $name."
                 all_stopped=false
            fi
        else
            echo "✅ $name is not running."
        fi
        rm -f "$PID_FILE"
    else
        echo "🤔 No PID file for $name, skipping."
    fi
done

# Stop any remaining Streamlit app
echo "🔄 Stopping Streamlit app..."
# Adding -f to match the full command, making it more specific
STREAMLIT_PIDS=$(pgrep -f "streamlit run app.py")
if [ -n "$STREAMLIT_PIDS" ]; then
    kill -9 $STREAMLIT_PIDS
    echo "✅ Streamlit app stopped."
else
    echo "✅ Streamlit app is not running."
fi


echo ""
if [ "$all_stopped" = true ]; then
    echo "✅ A2A Data Science System shutdown complete."
else
    echo "⚠️ Some servers may not have stopped correctly. Please check manually."
fi
echo "================================================" 