#!/bin/bash

# Universal Engine Startup Script
# Version: 1.0
# Description: Starts the complete Universal Engine system including LLM services and A2A agents

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/pids"
ENV_FILE="${PROJECT_ROOT}/.env"

# Default configuration
LLM_PROVIDER=${LLM_PROVIDER:-"ollama"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"llama2"}
A2A_PORT_START=${A2A_PORT_START:-8306}
A2A_PORT_END=${A2A_PORT_END:-8315}
UNIVERSAL_ENGINE_PORT=${UNIVERSAL_ENGINE_PORT:-8000}

echo -e "${BLUE}🚀 Universal Engine Startup Script${NC}"
echo "=================================="

# Create necessary directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
}

# Function to check if a service is running
is_service_running() {
    local port=$1
    nc -z localhost "$port" 2>/dev/null
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if is_service_running "$port"; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start on port $port"
    return 1
}

# Function to start Ollama service
start_ollama() {
    if [ "$LLM_PROVIDER" != "ollama" ]; then
        print_status "Skipping Ollama (LLM_PROVIDER=$LLM_PROVIDER)"
        return 0
    fi
    
    print_status "Starting Ollama service..."
    
    if is_service_running 11434; then
        print_warning "Ollama is already running"
        return 0
    fi
    
    # Start Ollama in background
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 & 
    echo $! > "$PID_DIR/ollama.pid"
    
    # Wait for Ollama to be ready
    if wait_for_service 11434 "Ollama"; then
        # Check if model exists, download if necessary
        if ! ollama list | grep -q "$OLLAMA_MODEL"; then
            print_status "Downloading model: $OLLAMA_MODEL"
            ollama pull "$OLLAMA_MODEL"
        fi
        return 0
    else
        return 1
    fi
}

# Function to start A2A agents
start_a2a_agents() {
    print_status "Starting A2A agents (ports $A2A_PORT_START-$A2A_PORT_END)..."
    
    local started_agents=0
    
    for port in $(seq "$A2A_PORT_START" "$A2A_PORT_END"); do
        if is_service_running "$port"; then
            print_warning "Port $port already in use"
            continue
        fi
        
        # Start A2A agent for this port
        agent_name="a2a_agent_$port"
        
        # Create agent startup command based on port
        case $port in
            8306) agent_type="data_cleaner" ;;
            8307) agent_type="eda_tools" ;;
            8308) agent_type="statistical_analyzer" ;;
            8309) agent_type="visualization" ;;
            8310) agent_type="ml_modeling" ;;
            8311) agent_type="text_analytics" ;;
            8312) agent_type="time_series" ;;
            8313) agent_type="report_generator" ;;
            8314) agent_type="data_validator" ;;
            8315) agent_type="performance_monitor" ;;
            *) agent_type="generic" ;;
        esac
        
        # Start agent in background
        cd "$PROJECT_ROOT"
        nohup python -m core.universal_engine.a2a_integration.agents."$agent_type" \
            --port "$port" \
            > "$LOG_DIR/${agent_name}.log" 2>&1 &
        
        echo $! > "$PID_DIR/${agent_name}.pid"
        started_agents=$((started_agents + 1))
        
        print_status "Started $agent_type agent on port $port"
    done
    
    if [ $started_agents -gt 0 ]; then
        print_status "Started $started_agents A2A agents"
        # Give agents time to initialize
        sleep 5
    fi
}

# Function to start Redis (if available)
start_redis() {
    if command -v redis-server >/dev/null 2>&1; then
        if is_service_running 6379; then
            print_warning "Redis is already running"
        else
            print_status "Starting Redis server..."
            nohup redis-server > "$LOG_DIR/redis.log" 2>&1 &
            echo $! > "$PID_DIR/redis.pid"
            wait_for_service 6379 "Redis"
        fi
    else
        print_warning "Redis not available (optional)"
    fi
}

# Function to initialize Universal Engine
initialize_universal_engine() {
    print_status "Initializing Universal Engine system..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment if it exists
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        print_status "Activated Python virtual environment"
    fi
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Initialize system components
    python -c "
import asyncio
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer
    
    async def initialize():
        initializer = UniversalEngineInitializer()
        success = await initializer.initialize_system()
        return success
    
    result = asyncio.run(initialize())
    if result:
        print('✅ Universal Engine system initialized successfully')
        sys.exit(0)
    else:
        print('❌ Universal Engine initialization failed')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Initialization error: {e}')
    sys.exit(1)
"
    
    return $?
}

# Function to start Universal Engine main service
start_universal_engine() {
    print_status "Starting Universal Engine main service..."
    
    if is_service_running "$UNIVERSAL_ENGINE_PORT"; then
        print_warning "Universal Engine is already running on port $UNIVERSAL_ENGINE_PORT"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Start the main Universal Engine service
    nohup python -m uvicorn main:app \
        --host 0.0.0.0 \
        --port "$UNIVERSAL_ENGINE_PORT" \
        --workers 1 \
        --access-log \
        > "$LOG_DIR/universal_engine.log" 2>&1 &
    
    echo $! > "$PID_DIR/universal_engine.pid"
    
    # Wait for service to be ready
    if wait_for_service "$UNIVERSAL_ENGINE_PORT" "Universal Engine"; then
        print_status "Universal Engine is ready at http://localhost:$UNIVERSAL_ENGINE_PORT"
        return 0
    else
        return 1
    fi
}

# Function to perform health checks
perform_health_checks() {
    print_status "Performing health checks..."
    
    local health_status=0
    
    # Check Universal Engine
    if curl -s "http://localhost:$UNIVERSAL_ENGINE_PORT/health" | grep -q "healthy"; then
        print_status "✅ Universal Engine: Healthy"
    else
        print_error "❌ Universal Engine: Unhealthy"
        health_status=1
    fi
    
    # Check LLM service
    if [ "$LLM_PROVIDER" = "ollama" ]; then
        if curl -s "http://localhost:11434/api/version" >/dev/null; then
            print_status "✅ Ollama: Healthy"
        else
            print_error "❌ Ollama: Unhealthy"
            health_status=1
        fi
    fi
    
    # Check A2A agents
    local healthy_agents=0
    local total_agents=$((A2A_PORT_END - A2A_PORT_START + 1))
    
    for port in $(seq "$A2A_PORT_START" "$A2A_PORT_END"); do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            healthy_agents=$((healthy_agents + 1))
        fi
    done
    
    print_status "A2A Agents: $healthy_agents/$total_agents healthy"
    if [ $healthy_agents -lt $((total_agents / 2)) ]; then
        print_warning "Less than 50% of A2A agents are healthy"
        health_status=1
    fi
    
    return $health_status
}

# Function to save startup configuration
save_startup_config() {
    cat > "$PROJECT_ROOT/startup_config.json" <<EOF
{
    "startup_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "llm_provider": "$LLM_PROVIDER",
    "ollama_model": "$OLLAMA_MODEL",
    "universal_engine_port": $UNIVERSAL_ENGINE_PORT,
    "a2a_port_range": "$A2A_PORT_START-$A2A_PORT_END",
    "project_root": "$PROJECT_ROOT",
    "log_directory": "$LOG_DIR",
    "pid_directory": "$PID_DIR"
}
EOF
    print_status "Startup configuration saved to startup_config.json"
}

# Main startup sequence
main() {
    print_status "Starting Universal Engine system startup sequence..."
    
    # Load environment variables
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
        print_status "Loaded environment variables from $ENV_FILE"
    fi
    
    # Check prerequisites
    if ! command -v python >/dev/null 2>&1; then
        print_error "Python is not installed"
        exit 1
    fi
    
    if [ "$LLM_PROVIDER" = "ollama" ] && ! command -v ollama >/dev/null 2>&1; then
        print_error "Ollama is not installed but LLM_PROVIDER is set to ollama"
        exit 1
    fi
    
    # Start services in order
    if ! start_redis; then
        print_warning "Redis startup failed, continuing without cache"
    fi
    
    if ! start_ollama; then
        if [ "$LLM_PROVIDER" = "ollama" ]; then
            print_error "Ollama startup failed"
            exit 1
        fi
    fi
    
    if ! start_a2a_agents; then
        print_warning "Some A2A agents failed to start"
    fi
    
    if ! initialize_universal_engine; then
        print_error "Universal Engine initialization failed"
        exit 1
    fi
    
    if ! start_universal_engine; then
        print_error "Universal Engine main service failed to start"
        exit 1
    fi
    
    # Wait a moment for all services to stabilize
    sleep 10
    
    # Perform health checks
    if perform_health_checks; then
        print_status "🎉 All systems healthy!"
    else
        print_warning "Some services are unhealthy, check logs"
    fi
    
    # Save configuration
    save_startup_config
    
    echo ""
    print_status "🚀 Universal Engine is now running!"
    print_status "📊 Dashboard: http://localhost:$UNIVERSAL_ENGINE_PORT"
    print_status "📋 Health: http://localhost:$UNIVERSAL_ENGINE_PORT/health"
    print_status "📁 Logs: $LOG_DIR"
    print_status "🔧 Stop with: ./scripts/stop_universal_engine.sh"
    echo ""
}

# Trap to handle interruption
trap 'print_error "Startup interrupted"; exit 1' INT TERM

# Run main function
main "$@"