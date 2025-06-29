#!/bin/bash

# Get the directory of the script and set the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/pandas_server.pid"

# Create logs directory and clear old pid file
mkdir -p "$LOG_DIR"
rm -f "$PID_FILE"

# Cleanup function to be called on exit
cleanup() {
    echo -e "\n🛑 Stopping system..."
    if [ -f "$PID_FILE" ]; then
        PID_TO_KILL=$(cat "$PID_FILE")
        if [ -n "$PID_TO_KILL" ] && ps -p "$PID_TO_KILL" > /dev/null; then
            echo "Killing background server with PID: $PID_TO_KILL"
            kill "$PID_TO_KILL"
            wait "$PID_TO_KILL" 2>/dev/null
            echo "✅ Server stopped."
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found. Server may not have started correctly."
    fi
    echo "✅ System shutdown complete."
}

# Register the cleanup function to be called on any script exit.
trap cleanup EXIT

echo "================================================"
echo "Starting CherryAI System..."
echo "Project Root: $PROJECT_ROOT"
echo "================================================"

# Setup logs directory
mkdir -p "$LOG_DIR"

# Check for UV environment
if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
    echo "UV environment not found. Please run 'uv venv' first."
    exit 1
fi
echo "Activating UV environment..."

# Start the Pandas Data Analyst A2A server in the background
echo ""
echo "================================================"
echo "Step 1: Starting Pandas Data Analyst Server in the background."
echo "Log file will be at: $LOG_DIR/pandas_server.log"
echo "================================================"

# Run server in the background, redirecting output to a log file
# and store its PID in the PID_FILE.
uv run python a2a_servers/pandas_server.py > "$LOG_DIR/pandas_server.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "✅ Server started in the background with PID: $SERVER_PID"

echo ""
echo "Waiting 5 seconds for the server to initialize..."
sleep 5

echo ""
echo "================================================"
echo "Step 2: Starting Streamlit Application in this terminal."
echo "You will see live logs below. Press Ctrl+C to stop."
echo "================================================"

# Run streamlit in the foreground of the current terminal
uv run streamlit run ai.py 