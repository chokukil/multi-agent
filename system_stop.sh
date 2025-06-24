#!/bin/bash
# ==============================================================================
# CherryAI - System Stop Script (SH)
# ==============================================================================
# This script finds and terminates all running CherryAI agent servers.
# It uses pkill to find processes matching the agent module paths.
# ==============================================================================

echo "🛑 Stopping all CherryAI agent servers..."

# Use pkill to find and kill processes whose command line matches the pattern.
# The '-f' flag matches against the full command line.
pkill -f "python -m mcp_agents.mcp_dataloader_agent"
pkill -f "python -m mcp_agents.mcp_datawrangling_agent"
pkill -f "python -m mcp_agents.mcp_datavisualization_agent"

echo "✅ All agent processes have been terminated."

echo "================================================"
echo "      CherryAI System Shutdown Script - macOS"
echo "================================================"
echo ""
echo "This script will stop all running services:"
echo "1. MCP servers"
echo "2. Streamlit application"
echo ""
echo "================================================"

# Function to kill processes by pattern
kill_processes() {
    local pattern="$1"
    local description="$2"
    
    echo "Stopping $description..."
    pids=$(pgrep -f "$pattern")
    if [ -n "$pids" ]; then
        echo "Found running processes: $pids"
        pkill -f "$pattern"
        sleep 2
        
        # Check if processes are still running
        remaining_pids=$(pgrep -f "$pattern")
        if [ -n "$remaining_pids" ]; then
            echo "Force killing remaining processes: $remaining_pids"
            pkill -9 -f "$pattern"
            sleep 1
        fi
        echo "✓ $description stopped"
    else
        echo "✓ No $description processes found"
    fi
}

# Stop MCP servers
echo "================================================"
echo "Step 1: Stopping MCP Servers"
echo "================================================"

kill_processes "mcp-servers" "MCP servers"

# Stop Streamlit
echo ""
echo "================================================"
echo "Step 2: Stopping Streamlit Application"
echo "================================================"

kill_processes "streamlit" "Streamlit application"

# Stop UV processes
echo ""
echo "================================================"
echo "Step 3: Stopping UV processes"
echo "================================================"

kill_processes "uv run" "UV processes"

# Clean up any remaining Python processes related to the project
echo ""
echo "================================================"
echo "Step 4: Cleaning up project processes"
echo "================================================"

current_dir=$(basename "$PWD")
kill_processes "$current_dir" "project-related processes"

echo ""
echo "================================================"
echo "System Shutdown Complete"
echo "================================================"
echo ""
echo "All CherryAI services have been stopped."
echo "You can now safely close this terminal."
echo ""
read -p "Press any key to continue..." 