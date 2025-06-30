#!/bin/bash

# A2A Data Science Team System Startup Script (macOS Compatible)
# Starts orchestrator and all 6 specialized agents

# Get the directory of the script and set the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/a2a_servers"
PID_DIR="$PROJECT_ROOT/logs/pids"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Server configurations (name:script:port)
SERVERS=(
    "orchestrator:a2a_ds_servers/a2a_orchestrator.py:8100"
    "pandas_analyst:a2a_ds_servers/pandas_data_analyst_server.py:8200"
    "sql_analyst:a2a_ds_servers/sql_data_analyst_server.py:8201"
    "data_viz:a2a_ds_servers/data_visualization_server.py:8202"
    "eda_tools:a2a_ds_servers/eda_tools_server.py:8203"
    "feature_eng:a2a_ds_servers/feature_engineering_server.py:8204"
    "data_cleaning:a2a_ds_servers/data_cleaning_server.py:8205"
)

# Store PIDs for cleanup
PIDS=()

# Cleanup function
cleanup() {
    echo -e "\n🛑 Stopping A2A Data Science System..."
    
    # Kill all background processes
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r name script port <<< "$server_config"
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo "🔄 Stopping $name (PID: $PID)..."
                kill "$PID"
                # Wait a bit for graceful shutdown
                sleep 2
                # Force kill if still running
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
            fi
            rm -f "$PID_FILE"
        fi
    done
    
    echo "✅ A2A Data Science System shutdown complete."
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "❌ Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to wait for server to be ready
wait_for_server() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo "⏳ Waiting for $name to be ready on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo "   Attempt $attempt/$max_attempts..."
    done
    
    echo "❌ $name failed to start within timeout"
    return 1
}

echo "================================================"
echo "🚀 Starting A2A Data Science Team System"
echo "================================================"

# Check UV environment
if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
    echo "❌ UV environment not found. Please run 'uv venv' first."
    exit 1
fi

echo "📦 Using UV environment..."

# Check ports availability
echo "🔍 Checking port availability..."
for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    if ! check_port "$port"; then
        echo "❌ Cannot start system: Port $port is already in use"
        exit 1
    fi
done

echo "✅ All ports are available"
echo ""

# Start servers
echo "🚀 Starting A2A servers..."

for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    
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
    
    echo "🤖 Starting $display_name (Port $port)..."
    uv run python "$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PID=$!
    echo $PID > "$PID_DIR/${name}.pid"
    PIDS+=($PID)
    echo "   Started with PID: $PID"
    
    # Wait for server to be ready
    if ! wait_for_server "$display_name" "$port"; then
        echo "❌ Failed to start $display_name"
        exit 1
    fi
    echo ""
done

echo "================================================"
echo "🎉 A2A Data Science System is fully operational!"
echo "================================================"
echo ""
echo "🌐 Available Services:"
echo "   📋 Orchestrator:        http://localhost:8100"
echo "   📊 Pandas Analyst:      http://localhost:8200"
echo "   🗄️  SQL Analyst:        http://localhost:8201"
echo "   🎨 Data Visualization:  http://localhost:8202"
echo "   🔬 EDA Tools:           http://localhost:8203"
echo "   🔧 Feature Engineering: http://localhost:8204"
echo "   🧹 Data Cleaning:       http://localhost:8205"
echo ""
echo "📋 Agent Cards:"
for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    echo "   http://localhost:$port/.well-known/agent.json"
done
echo ""
echo "📁 Log files in: $LOG_DIR"
echo "📁 PID files in: $PID_DIR"
echo ""
echo "🔄 To test the system:"
echo "   python test_simple_a2a_client.py"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Keep the script running and show live status
echo "📊 System Status Monitor (Press Ctrl+C to stop):"
echo "================================================"

while true; do
    all_running=true
    current_time=$(date "+%H:%M:%S")
    status_line="[$current_time] "
    
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r name script port <<< "$server_config"
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                status_line+="✅"
            else
                status_line+="❌"
                all_running=false
            fi
        else
            status_line+="❌"
            all_running=false
        fi
    done
    
    if [ "$all_running" = true ]; then
        status_line+=" All services running"
    else
        status_line+=" Some services failed"
    fi
    
    echo -ne "\r$status_line"
    sleep 5
done 