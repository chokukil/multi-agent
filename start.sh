#!/bin/bash

# A2A Data Science Team System Startup Script
# Starts orchestrator, all agents, and Streamlit UI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/a2a_servers"
PID_DIR="$PROJECT_ROOT/logs/pids"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Server configurations (name:script:port)
SERVERS=(
    "orchestrator:a2a_ds_servers/orchestrator_server.py:8100"
    "data_loader:a2a_ds_servers/ai_ds_team_data_loader_server.py:8307"
    "data_cleaning:a2a_ds_servers/ai_ds_team_data_cleaning_server.py:8306"
    "data_wrangling:a2a_ds_servers/ai_ds_team_data_wrangling_server.py:8309"
    "eda_tools:a2a_ds_servers/ai_ds_team_eda_tools_server.py:8312"
    "data_viz:a2a_ds_servers/ai_ds_team_data_visualization_server.py:8308"
    "feature_eng:a2a_ds_servers/ai_ds_team_feature_engineering_server.py:8310"
    "h2o_ml:a2a_ds_servers/ai_ds_team_h2o_ml_server.py:8313"
    "mlflow_tools:a2a_ds_servers/ai_ds_team_mlflow_tools_server.py:8314"
    "sql_database:a2a_ds_servers/ai_ds_team_sql_database_server.py:8311"
)

# Cleanup function
cleanup() {
    echo -e "\n🛑 Stopping A2A Data Science System..."
    
    # Stop Streamlit if running
    if [ -f "$PID_DIR/streamlit.pid" ]; then
        STREAMLIT_PID=$(cat "$PID_DIR/streamlit.pid")
        if [ -n "$STREAMLIT_PID" ] && ps -p "$STREAMLIT_PID" > /dev/null 2>&1; then
            echo "🔄 Stopping Streamlit (PID: $STREAMLIT_PID)..."
            kill "$STREAMLIT_PID"
        fi
        rm -f "$PID_DIR/streamlit.pid"
    fi
    
    # Stop all A2A servers
    for server_config in "${SERVERS[@]}"; do
        IFS=':' read -r name script port <<< "$server_config"
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo "🔄 Stopping $name (PID: $PID)..."
                kill "$PID"
                sleep 1
                # Force kill if still running
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
            fi
            rm -f "$PID_FILE"
        fi
    done
    
    echo "✅ System shutdown complete."
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
    
    echo "⏳ Waiting for $name..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        if [ $((attempt % 5)) -eq 0 ]; then
            echo "   Still waiting... ($attempt/$max_attempts)"
        fi
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

# Check ports availability (except 8501 for Streamlit)
echo "🔍 Checking port availability..."
all_ports_available=true
for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    if ! check_port "$port"; then
        all_ports_available=false
    fi
done

# Check Streamlit port
if ! check_port "8501"; then
    echo "❌ Port 8501 (Streamlit) is already in use"
    all_ports_available=false
fi

if [ "$all_ports_available" = false ]; then
    echo "❌ Cannot start system: Some ports are in use"
    echo "💡 Run './stop.sh' to stop existing services"
    exit 1
fi

echo "✅ All ports are available"
echo ""

# Start A2A servers
echo "🚀 Starting A2A servers..."

for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    
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
    
    echo "🤖 Starting $display_name (Port $port)..."
    uv run python "$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PID=$!
    echo $PID > "$PID_DIR/${name}.pid"
    echo "   Started with PID: $PID"
    
    # Give server time to initialize
    sleep 2
done

echo ""
echo "⏱️  Waiting for servers to initialize..."
sleep 5

# Check all servers are ready
echo "🔍 Checking server status..."
all_ready=true
for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    
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
    
    if ! wait_for_server "$display_name" "$port"; then
        all_ready=false
        echo "❌ Failed to start $display_name"
        echo "📄 Check log: $LOG_DIR/${name}.log"
    fi
done

if [ "$all_ready" = false ]; then
    echo "❌ Some servers failed to start. Exiting..."
    exit 1
fi

echo ""
echo "================================================"
echo "🎉 A2A Data Science System is operational!"
echo "================================================"
echo "📊 Available Services:"
echo "   🎯 Orchestrator:          http://localhost:8100"
echo "   📂 Data Loader:          http://localhost:8307"
echo "   🧹 Data Cleaning:         http://localhost:8306"
echo "   🛠️ Data Wrangling:        http://localhost:8309"
echo "   🔍 EDA Tools:             http://localhost:8312"
echo "   🎨 Data Visualization:    http://localhost:8308"
echo "   🔧 Feature Engineering:   http://localhost:8310"
echo "   🤖 H2O ML:              http://localhost:8313"
echo "   📈 MLflow Tools:        http://localhost:8314"
echo "   🗄️ SQL Database:        http://localhost:8311"
echo ""
echo "🌐 Starting Streamlit UI on http://localhost:8501"
echo "================================================"

# Start Streamlit
echo "🎨 Starting Streamlit..."
uv run streamlit run app.py &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > "$PID_DIR/streamlit.pid"

echo ""
echo "🎉 System ready!"
echo "📱 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the system"
echo ""

# Wait for user interrupt
wait $STREAMLIT_PID 