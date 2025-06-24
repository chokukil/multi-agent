#!/bin/bash
# ==============================================================================
# CherryAI - System Start Script (SH)
# ==============================================================================
# This script starts all the necessary agent servers for the CherryAI system
# to be fully operational. It starts each agent in the background.
#
# Use system_stop.sh to terminate these agents.
# ==============================================================================

echo "🚀 Starting all CherryAI agent servers..."

# Activate virtual environment if necessary and start agents
# Using 'uv run' which handles the virtual environment automatically.
uv run python -m mcp_agents.mcp_dataloader_agent &
DATALOADER_PID=$!
echo "   - Dataloader Agent started with PID: $DATALOADER_PID (Port 8001)"

uv run python -m mcp_agents.mcp_datawrangling_agent &
DATAWRANGLING_PID=$!
echo "   - Data Wrangling Agent started with PID: $DATAWRANGLING_PID (Port 8002)"

uv run python -m mcp_agents.mcp_datavisualization_agent &
DATAVISUALIZATION_PID=$!
echo "   - Data Visualization Agent started with PID: $DATAVISUALIZATION_PID (Port 8003)"

echo -e "\n✅ All agents have been launched in the background."
echo "   Use 'system_stop.sh' or the Streamlit UI to stop them."

echo "================================================"
echo "      CherryAI System Startup Script - macOS"
echo "================================================"
echo ""
echo "This script will:"
echo "1. Start all MCP servers"
echo "2. Wait for initialization"
echo "3. Launch the main Streamlit application"
echo ""
echo "================================================"

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "[ERROR] UV is not installed!"
    echo "Please install UV first and then run this script."
    echo ""
    echo "Installation options:"
    echo "  1. Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  2. Or run: brew install uv" 
    echo "  3. Or run: pip install uv"
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run ./mcp_server_start.sh first to set up the environment."
    read -p "Press any key to continue..."
    exit 1
fi

echo "[OK] Prerequisites check passed"
echo ""

echo "================================================"
echo "Step 1: Starting MCP Servers"
echo "================================================"

# Start MCP servers
echo "Launching MCP server startup script..."
chmod +x mcp_server_start.sh
./mcp_server_start.sh

# Additional wait time to ensure all servers are fully loaded
echo ""
echo "================================================"
echo "Step 2: Waiting for MCP servers to stabilize..."
echo "================================================"
echo "Please wait while all MCP servers finish initialization..."

sleep 45

echo ""
echo "================================================"
echo "Step 3: Starting Streamlit Application"
echo "================================================"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "[ERROR] app.py not found!"
    echo "Please ensure app.py is in the current directory."
    read -p "Press any key to continue..."
    exit 1
fi

echo "Starting CherryAI Streamlit application..."
echo ""
echo "[OK] System startup complete!"
echo ""
echo "The application will open in your default web browser."
echo "URL: http://localhost:8501"
echo ""
echo "To stop the system:"
echo "1. Press Ctrl+C in this terminal"
echo "2. Run: pkill -f 'mcp-servers' to stop all MCP servers"
echo ""
echo "================================================"

# Start Streamlit app
uv run streamlit run app.py

echo ""
echo "================================================"
echo "System shutdown initiated"
echo "================================================"
echo "To completely stop all MCP servers, run:"
echo "pkill -f 'mcp-servers'"
echo ""
read -p "Press any key to continue..." 