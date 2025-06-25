#!/bin/bash

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/pandas_server.pid"

echo "================================================"
echo "Stopping CherryAI System..."
echo "Project Root: $PROJECT_ROOT"
echo "================================================"

# Function to safely kill a process by PID
kill_process_by_pid() {
    local pid=$1
    local name=$2
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping $name (PID: $pid)..."
        kill "$pid" 2>/dev/null
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Force killing $name (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null
        fi
        
        # Verify termination
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ $name terminated successfully."
        else
            echo "❌ Failed to terminate $name."
        fi
    else
        echo "⚠️  $name (PID: $pid) is not running."
    fi
}

# 1. Stop background pandas server from PID file
echo ""
echo "Step 1: Stopping Pandas Data Analyst Server..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill_process_by_pid "$PID" "Pandas Server"
    rm -f "$PID_FILE"
    echo "PID file removed."
else
    echo "- No PID file found at: $PID_FILE"
fi

# 2. Stop any remaining pandas_server processes
echo ""
echo "Step 2: Searching for remaining pandas_server processes..."
PANDAS_PIDS=$(pgrep -f "pandas_server.py")
if [ -n "$PANDAS_PIDS" ]; then
    echo "Found pandas_server processes: $PANDAS_PIDS"
    for pid in $PANDAS_PIDS; do
        kill_process_by_pid "$pid" "Pandas Server Process"
    done
else
    echo "- No pandas_server processes found."
fi

# 3. Stop Streamlit processes
echo ""
echo "Step 3: Stopping Streamlit processes..."
STREAMLIT_PIDS=$(pgrep -f "streamlit run")
if [ -n "$STREAMLIT_PIDS" ]; then
    echo "Found Streamlit processes: $STREAMLIT_PIDS"
    for pid in $STREAMLIT_PIDS; do
        kill_process_by_pid "$pid" "Streamlit Process"
    done
else
    echo "- No Streamlit processes found."
fi

# 4. Stop any system_start.sh processes
echo ""
echo "Step 4: Stopping system_start.sh processes..."
SYSTEM_START_PIDS=$(pgrep -f "system_start.sh")
if [ -n "$SYSTEM_START_PIDS" ]; then
    echo "Found system_start.sh processes: $SYSTEM_START_PIDS"
    for pid in $SYSTEM_START_PIDS; do
        # Don't kill the current script
        if [ "$pid" != "$$" ]; then
            kill_process_by_pid "$pid" "System Start Script"
        fi
    done
else
    echo "- No system_start.sh processes found."
fi

# 5. Clean up any remaining UV processes related to the project
echo ""
echo "Step 5: Cleaning up UV processes..."
UV_PIDS=$(pgrep -f "uv.*CherryAI_0623")
if [ -n "$UV_PIDS" ]; then
    echo "Found UV processes: $UV_PIDS"
    for pid in $UV_PIDS; do
        kill_process_by_pid "$pid" "UV Process"
    done
else
    echo "- No project-related UV processes found."
fi

# 6. Verify all processes are stopped
echo ""
echo "Step 6: Final verification..."
sleep 1

REMAINING_PROCESSES=""
if pgrep -f "pandas_server.py" > /dev/null; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES pandas_server"
fi
if pgrep -f "streamlit run.*app.py" > /dev/null; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES streamlit"
fi

if [ -n "$REMAINING_PROCESSES" ]; then
    echo "⚠️  Warning: Some processes may still be running:$REMAINING_PROCESSES"
    echo "You may need to manually terminate them."
else
    echo "✅ All CherryAI processes successfully terminated."
fi

echo ""
echo "================================================"
echo "✅ System shutdown complete."
echo "================================================" 