#\!/bin/bash

# CherryAI A2A Agent System Shutdown Script
# This script stops all A2A agents and the orchestrator

echo "🍒 CherryAI A2A Agent System Stopping..."
echo "========================================="

# Function to stop a service by PID file
stop_service() {
    local service_name=$1
    local pid_file="pids/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo -n "Stopping $service_name (PID: $pid)... "
        
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid
            sleep 2
            
            # Check if process is still running
            if ps -p $pid > /dev/null 2>&1; then
                echo -n "Force killing... "
                kill -9 $pid
                sleep 1
            fi
            
            if ps -p $pid > /dev/null 2>&1; then
                echo "❌ Failed to stop"
                return 1
            else
                echo "✅ Stopped"
                rm -f "$pid_file"
                return 0
            fi
        else
            echo "⚠️  Process not running"
            rm -f "$pid_file"
            return 0
        fi
    else
        echo "⚠️  No PID file found for $service_name"
        return 0
    fi
}

# Define agents
AGENTS=(
    "data_cleaning_server"
    "data_visualization_server"
    "data_wrangling_server"
    "feature_engineering_server"
    "sql_database_server"
    "eda_tools_server"
    "h2o_ml_server"
    "mlflow_tools_server"
    "pandas_analyst_server"
    "report_generator_server"
)

echo ""
echo "🛑 Stopping A2A Agents..."
echo "-------------------------"

SUCCESS_COUNT=0
TOTAL_COUNT=0

# Stop all agents
for agent in "${AGENTS[@]}"; do
    ((TOTAL_COUNT++))
    if stop_service "$agent"; then
        ((SUCCESS_COUNT++))
    fi
done

echo ""
echo "🎛️  Stopping Orchestrator..."
echo "----------------------------"

# Stop orchestrator
((TOTAL_COUNT++))
if stop_service "orchestrator"; then
    ((SUCCESS_COUNT++))
fi

echo ""
echo "🎨 Stopping Cherry AI Streamlit Platform..."
echo "-------------------------------------------"

# Stop Cherry AI Streamlit Platform
((TOTAL_COUNT++))
if stop_service "cherry_ai_streamlit"; then
    ((SUCCESS_COUNT++))
fi

echo ""
echo "🧹 Cleanup..."
echo "-------------"

# Kill any remaining processes on the ports (safety measure)
PORTS=(8100 8306 8308 8309 8310 8311 8312 8313 8314 8315 8316 8501)

for port in "${PORTS[@]}"; do
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ \! -z "$pid" ]; then
        echo "Killing remaining process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# Clean up empty PID files
find pids/ -name "*.pid" -empty -delete 2>/dev/null

echo ""
echo "========================================="
echo "🎯 Summary: $SUCCESS_COUNT/$TOTAL_COUNT services stopped"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo "✅ All services stopped successfully\!"
else
    echo "⚠️  Some services may still be running"
    echo "You can check with: ps aux | grep python"
fi

echo ""
echo "🔄 To start all services again, run: ./start.sh"
echo "========================================="
