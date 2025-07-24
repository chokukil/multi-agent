#\!/bin/bash

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
            echo "⚠️ Process not running"
            rm -f "$pid_file"
            return 0
        fi
    else
        echo "⚠️ No PID file found for $service_name"
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

success=0
total=0

# Stop all agents
for agent in "${AGENTS[@]}"; do
    ((total++))
    if stop_service "$agent"; then
        ((success++))
    fi
done

echo ""
echo "🎛️ Stopping Orchestrator..."
echo "----------------------------"

((total++))
if stop_service "orchestrator"; then
    ((success++))
fi

echo ""
echo "🧹 Cleanup..."
echo "-------------"

# Kill any remaining processes on the ports
PORTS=(8100 8306 8308 8309 8310 8311 8312 8313 8314 8315 8316)

for port in "${PORTS[@]}"; do
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ \! -z "$pid" ]; then
        echo "Killing remaining process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

echo ""
echo "========================================="
echo "🎯 Summary: $success/$total services stopped"

if [ $success -eq $total ]; then
    echo "✅ All services stopped successfully\!"
else
    echo "⚠️ Some services may still be running"
fi

echo "🔄 To start all services again, run: ./start.sh"
echo "========================================="
