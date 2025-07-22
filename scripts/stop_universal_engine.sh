#!/bin/bash

# Universal Engine Stop Script
# Version: 1.0
# Description: Safely stops the Universal Engine system and all related services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PID_DIR="${PROJECT_ROOT}/pids"
LOG_DIR="${PROJECT_ROOT}/logs"

echo -e "${BLUE}🛑 Universal Engine Stop Script${NC}"
echo "================================="

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

# Function to gracefully stop a process
stop_process() {
    local pid_file=$1
    local service_name=$2
    local timeout=${3:-10}
    
    if [ ! -f "$pid_file" ]; then
        print_warning "PID file not found for $service_name: $pid_file"
        return 1
    fi
    
    local pid
    pid=$(cat "$pid_file")
    
    if [ -z "$pid" ]; then
        print_warning "Empty PID file for $service_name"
        rm -f "$pid_file"
        return 1
    fi
    
    # Check if process is running
    if ! kill -0 "$pid" 2>/dev/null; then
        print_status "$service_name (PID $pid) is not running"
        rm -f "$pid_file"
        return 0
    fi
    
    print_status "Stopping $service_name (PID $pid)..."
    
    # Try graceful shutdown first (SIGTERM)
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for process to exit
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""
    
    # If still running, force kill (SIGKILL)
    if kill -0 "$pid" 2>/dev/null; then
        print_warning "$service_name didn't stop gracefully, forcing shutdown..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 2
        
        if kill -0 "$pid" 2>/dev/null; then
            print_error "Failed to stop $service_name (PID $pid)"
            return 1
        fi
    fi
    
    print_status "$service_name stopped successfully"
    rm -f "$pid_file"
    return 0
}

# Function to stop all processes by port
stop_by_port() {
    local port=$1
    local service_name=$2
    
    local pids
    pids=$(lsof -ti :$port 2>/dev/null || true)
    
    if [ -z "$pids" ]; then
        print_status "No process found on port $port ($service_name)"
        return 0
    fi
    
    print_status "Stopping $service_name on port $port..."
    
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            print_status "Stopping process $pid on port $port"
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait a bit for graceful shutdown
            sleep 3
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    # Verify port is free
    if lsof -ti :$port >/dev/null 2>&1; then
        print_error "Port $port is still in use"
        return 1
    else
        print_status "Port $port is now free"
        return 0
    fi
}

# Function to stop Universal Engine main service
stop_universal_engine() {
    print_status "Stopping Universal Engine main service..."
    
    local pid_file="$PID_DIR/universal_engine.pid"
    local port=${UNIVERSAL_ENGINE_PORT:-8000}
    
    # Try to stop using PID file first
    if stop_process "$pid_file" "Universal Engine" 15; then
        return 0
    fi
    
    # Fallback to stopping by port
    stop_by_port "$port" "Universal Engine"
}

# Function to stop A2A agents
stop_a2a_agents() {
    print_status "Stopping A2A agents..."
    
    local port_start=${A2A_PORT_START:-8306}
    local port_end=${A2A_PORT_END:-8315}
    local stopped_count=0
    
    for port in $(seq "$port_start" "$port_end"); do
        local agent_name="a2a_agent_$port"
        local pid_file="$PID_DIR/${agent_name}.pid"
        
        # Try PID file first
        if stop_process "$pid_file" "A2A Agent ($port)" 5; then
            stopped_count=$((stopped_count + 1))
            continue
        fi
        
        # Fallback to port-based stopping
        if stop_by_port "$port" "A2A Agent"; then
            stopped_count=$((stopped_count + 1))
        fi
    done
    
    print_status "Stopped $stopped_count A2A agents"
}

# Function to stop Ollama
stop_ollama() {
    print_status "Stopping Ollama service..."
    
    local pid_file="$PID_DIR/ollama.pid"
    
    # Try PID file first
    if stop_process "$pid_file" "Ollama" 10; then
        return 0
    fi
    
    # Fallback to port-based stopping
    stop_by_port 11434 "Ollama"
}

# Function to stop Redis
stop_redis() {
    print_status "Stopping Redis service..."
    
    local pid_file="$PID_DIR/redis.pid"
    
    # Try PID file first
    if stop_process "$pid_file" "Redis" 5; then
        return 0
    fi
    
    # Fallback to port-based stopping
    stop_by_port 6379 "Redis"
    
    # Also try redis-cli shutdown if available
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli shutdown 2>/dev/null || true
    fi
}

# Function to cleanup resources
cleanup_resources() {
    print_status "Cleaning up resources..."
    
    # Remove all PID files
    if [ -d "$PID_DIR" ]; then
        rm -f "$PID_DIR"/*.pid
        print_status "Removed PID files"
    fi
    
    # Clean up temporary files
    if [ -f "$PROJECT_ROOT/startup_config.json" ]; then
        rm -f "$PROJECT_ROOT/startup_config.json"
        print_status "Removed startup configuration"
    fi
    
    # Clean up any socket files
    find /tmp -name "*universal_engine*" -type s -delete 2>/dev/null || true
    
    # Clean up any lock files
    find "$PROJECT_ROOT" -name "*.lock" -delete 2>/dev/null || true
    
    print_status "Resource cleanup completed"
}

# Function to verify all services stopped
verify_shutdown() {
    print_status "Verifying all services are stopped..."
    
    local issues=0
    local ports_to_check=(8000 11434 6379)
    
    # Add A2A agent ports
    for port in $(seq ${A2A_PORT_START:-8306} ${A2A_PORT_END:-8315}); do
        ports_to_check+=($port)
    done
    
    for port in "${ports_to_check[@]}"; do
        if lsof -ti :$port >/dev/null 2>&1; then
            print_warning "Port $port is still in use"
            issues=$((issues + 1))
        fi
    done
    
    # Check for any remaining Universal Engine processes
    local remaining_processes
    remaining_processes=$(ps aux | grep -E "(universal_engine|ollama|redis)" | grep -v grep | wc -l)
    
    if [ "$remaining_processes" -gt 0 ]; then
        print_warning "$remaining_processes related processes still running"
        ps aux | grep -E "(universal_engine|ollama|redis)" | grep -v grep
        issues=$((issues + 1))
    fi
    
    if [ $issues -eq 0 ]; then
        print_status "✅ All services stopped successfully"
        return 0
    else
        print_warning "⚠️ $issues issues found during shutdown"
        return 1
    fi
}

# Function to force cleanup (emergency stop)
force_cleanup() {
    print_warning "Performing emergency cleanup..."
    
    # Kill all processes containing universal_engine
    pkill -f "universal_engine" 2>/dev/null || true
    
    # Kill processes on known ports
    local ports=(8000 11434 6379)
    for port in $(seq ${A2A_PORT_START:-8306} ${A2A_PORT_END:-8315}); do
        ports+=($port)
    done
    
    for port in "${ports[@]}"; do
        local pids
        pids=$(lsof -ti :$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -KILL 2>/dev/null || true
        fi
    done
    
    # Clean up resources
    cleanup_resources
    
    print_status "Emergency cleanup completed"
}

# Main stop sequence
main() {
    local force_mode=false
    local skip_verify=false
    
    # Parse command line arguments
    while [ $# -gt 0 ]; do
        case $1 in
            --force)
                force_mode=true
                shift
                ;;
            --skip-verify)
                skip_verify=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--force] [--skip-verify]"
                echo "  --force       Force stop all processes (emergency mode)"
                echo "  --skip-verify Skip verification of shutdown"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    if [ "$force_mode" = true ]; then
        force_cleanup
        exit 0
    fi
    
    print_status "Starting Universal Engine shutdown sequence..."
    
    # Load environment variables if available
    if [ -f "$PROJECT_ROOT/.env" ]; then
        source "$PROJECT_ROOT/.env"
    fi
    
    # Stop services in reverse order of startup
    stop_universal_engine || print_warning "Universal Engine stop had issues"
    
    stop_a2a_agents || print_warning "A2A agents stop had issues"
    
    # Stop LLM service if it was started by us
    if [ "${LLM_PROVIDER:-ollama}" = "ollama" ]; then
        stop_ollama || print_warning "Ollama stop had issues"
    fi
    
    stop_redis || print_warning "Redis stop had issues"
    
    # Clean up resources
    cleanup_resources
    
    # Wait a moment for all processes to fully terminate
    sleep 3
    
    # Verify shutdown unless skipped
    if [ "$skip_verify" = false ]; then
        if ! verify_shutdown; then
            print_warning "Some issues detected. Use --force for emergency cleanup"
        fi
    fi
    
    # Save shutdown log
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): Universal Engine stopped" >> "$LOG_DIR/shutdown.log"
    
    print_status "🛑 Universal Engine shutdown completed"
}

# Trap to handle interruption
trap 'print_error "Stop script interrupted"; exit 1' INT TERM

# Ensure directories exist
mkdir -p "$PID_DIR" "$LOG_DIR"

# Run main function
main "$@"