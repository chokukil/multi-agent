#!/bin/bash

# ==============================================================================
# CherryAI - Pandas Data Analyst Workflow Test Runner (SH)
# ==============================================================================
# This script automates the process of:
# 1. Starting all required agent servers in the background.
# 2. Waiting for the servers to initialize.
# 3. Running the integration test for the Pandas Data Analyst workflow.
# 4. Shutting down all agent servers after the test is complete.
# ==============================================================================

echo "🚀 Starting agent servers for the test..."

# Activate virtual environment and start agents in the background
uv run python -m mcp_agents.mcp_dataloader_agent &
DATALOADER_PID=$!
echo "   - Dataloader Agent started with PID: $DATALOADER_PID"

uv run python -m mcp_agents.mcp_datawrangling_agent &
DATAWRANGLING_PID=$!
echo "   - Data Wrangling Agent started with PID: $DATAWRANGLING_PID"

uv run python -m mcp_agents.mcp_datavisualization_agent &
DATAVISUALIZATION_PID=$!
echo "   - Data Visualization Agent started with PID: $DATAVISUALIZATION_PID"

# Function to clean up all background processes
cleanup() {
    echo -e "\n🛑 Shutting down agent servers..."
    kill $DATALOADER_PID $DATAWRANGLING_PID $DATAVISUALIZATION_PID
    echo "   - All agents stopped."
}

# Trap EXIT signal to ensure cleanup runs automatically
trap cleanup EXIT

# Wait a moment for all servers to be fully ready
echo -e "\n⏳ Waiting for servers to initialize..."
sleep 5

# Run the specific integration test
echo -e "\n🧪 Running the Pandas Data Analyst integration test..."
uv run pytest tests/integration/test_pandas_data_analyst_workflow.py -v

TEST_EXIT_CODE=$?

# The 'trap' will handle the cleanup automatically upon exit.
echo -e "\n✅ Test run finished."

exit $TEST_EXIT_CODE 