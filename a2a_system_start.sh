#!/bin/bash

# A2A Data Science Team System Startup Script
# Starts orchestrator and all 6 specialized agents

# Get the directory of the script and set the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/a2a_servers"
PID_DIR="$PROJECT_ROOT/logs/pids"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Server definitions
declare -A SERVERS=(
    ["orchestrator"]="a2a_ds_servers/orchestrator_server.py:8100"
    ["data_loader"]="a2a_ds_servers/ai_ds_team_data_loader_server.py:8307"
    ["data_cleaning"]="a2a_ds_servers/ai_ds_team_data_cleaning_server.py:8306"
    ["data_wrangling"]="a2a_ds_servers/ai_ds_team_data_wrangling_server.py:8309"
    ["eda_tools"]="a2a_ds_servers/ai_ds_team_eda_tools_server.py:8312"
    ["data_viz"]="a2a_ds_servers/ai_ds_team_data_visualization_server.py:8308"
    ["feature_eng"]="a2a_ds_servers/ai_ds_team_feature_engineering_server.py:8310"
    ["h2o_ml"]="a2a_ds_servers/ai_ds_team_h2o_ml_server.py:8313"
    ["mlflow_tools"]="a2a_ds_servers/ai_ds_team_mlflow_tools_server.py:8314"
    ["sql_database"]="a2a_ds_servers/ai_ds_team_sql_database_server.py:8311"
)

# Store PIDs for cleanup
PIDS=()

# Cleanup function
cleanup() {
    echo -e "\n🛑 Stopping A2A Data Science System..."
    
    # Kill all background processes
    for name in "${!SERVERS[@]}"; do
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
for name in "${!SERVERS[@]}"; do
    IFS=':' read -r script port <<< "${SERVERS[$name]}"
    if ! check_port "$port"; then
        echo "❌ Cannot start system: Port $port is already in use"
        exit 1
    fi
done

echo "✅ All ports are available"
echo ""

# Start servers in order
echo "🚀 Starting A2A servers..."

# Start orchestrator first
echo "📋 Starting Orchestrator (Port 8100)..."
IFS=':' read -r script port <<< "${SERVERS["orchestrator"]}"
uv run python "$script" > "$LOG_DIR/orchestrator.log" 2>&1 &
PID=$!
echo $PID > "$PID_DIR/orchestrator.pid"
PIDS+=($PID)
echo "   Started with PID: $PID"

# Wait for orchestrator
if ! wait_for_server "Orchestrator" "8100"; then
    echo "❌ Failed to start Orchestrator"
    exit 1
fi

echo ""

# Start agent servers
agents=("data_loader" "data_cleaning" "data_wrangling" "eda_tools" "data_viz" "feature_eng" "h2o_ml" "mlflow_tools" "sql_database")
agent_names=("Data Loader" "Data Cleaning" "Data Wrangling" "EDA Tools" "Data Visualization" "Feature Engineering" "H2O ML" "MLflow Tools" "SQL Database")

for i in "${!agents[@]}"; do
    name="${agents[$i]}"
    display_name="${agent_names[$i]}"
    IFS=':' read -r script port <<< "${SERVERS[$name]}"
    
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
echo "   📂 Data Loader:          http://localhost:8307"
echo "   🧹 Data Cleaning:       http://localhost:8306"
echo "   🛠️ Data Wrangling:      http://localhost:8309"
echo "   🔬 EDA Tools:           http://localhost:8312"
echo "   🎨 Data Visualization:  http://localhost:8308"
echo "   🔧 Feature Engineering: http://localhost:8310"
echo "   🤖 H2O ML:              http://localhost:8313"
echo "   📈 MLflow Tools:        http://localhost:8314"
echo "   🗄️ SQL Database:        http://localhost:8311"
echo ""
echo "📋 Agent Cards:"
for name in "${!SERVERS[@]}"; do
    IFS=':' read -r script port <<< "${SERVERS[$name]}"
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
    
    for name in "${!SERVERS[@]}"; do
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