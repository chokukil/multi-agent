#!/bin/bash
# mcp_server_start.sh - Start MCP Servers using UV package manager

echo "================================================"
echo "      MCP Server Launcher (UV) - macOS"
echo "================================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[ERROR] UV is not installed!"
    echo ""
    echo "Please install UV first:"
    echo "  1. Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  2. Or run: brew install uv"
    echo "  3. Or run: pip install uv"
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

echo "[OK] UV found. Checking project setup..."

# Create virtual environment with uv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with UV..."
    uv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        read -p "Press any key to continue..."
        exit 1
    fi
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using uv with pyproject.toml
echo "Installing dependencies with UV..."
echo "This will install all required packages for MCP servers!"
uv sync
if [ $? -ne 0 ]; then
    echo "[ERROR] UV sync failed. Trying alternative installation methods..."
    echo ""
    echo "[WARNING] Installing from pyproject.toml..."
    uv pip install -e .
    if [ $? -ne 0 ]; then
        echo "[ERROR] pyproject.toml installation failed"
        echo "Falling back to requirements.txt..."
        uv pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "[ERROR] Requirements.txt installation also failed"
            echo "Please check your dependencies manually"
            read -p "Press any key to continue..."
            exit 1
        fi
    fi
fi

# Install critical MCP server dependencies individually
echo "Installing critical MCP server dependencies..."
uv pip install fastmcp mcp uvicorn xgboost scikit-learn scipy pandas numpy matplotlib seaborn plotly

# Install optional advanced ML dependencies (non-critical)
echo "Installing optional ML dependencies..."
uv pip install --no-deps catboost lightgbm imbalanced-learn statsmodels pingouin mlxtend umap-learn shap lime optuna 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Some optional ML packages failed to install. This is not critical."
fi

echo "[OK] Dependencies installed successfully!"

# Create necessary directories
echo "Setting up directories..."
mkdir -p prompt-configs
mkdir -p mcp-configs
mkdir -p results
mkdir -p reports
mkdir -p logs
mkdir -p generated_code

# Check for .env file
if [ ! -f ".env" ]; then
    echo "================================================"
    echo "WARNING: .env file not found!"
    echo "Creating example .env file..."
    echo ""
    cat > .env << EOF
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Other API Keys
PERPLEXITY_API_KEY=your_perplexity_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
EOF
    echo "Please edit .env file with your actual API keys"
    echo "================================================"
fi

echo "================================================"
echo "Checking MCP server configuration..."
echo "================================================"

# Check if MCP configuration files exist
if [ ! -f "mcp_config.py" ]; then
    echo "[ERROR] Missing: mcp_config.py"
    echo "    Please ensure all required files are present"
    exit 1
fi

# Show current MCP configuration
echo "Displaying MCP server configuration..."
uv run python mcp_config.py

echo "================================================"
echo "Starting MCP servers..."
echo "================================================"

# Function to start a server in background
start_server() {
    local name="$1"
    local script="$2"
    local port="$3"
    
    echo "Starting $name..."
    if [ -n "$port" ]; then
        SERVER_PORT=$port nohup uv run python "$script" > "logs/${name// /_}.log" 2>&1 &
    else
        nohup uv run python "$script" --port 8006 > "logs/${name// /_}.log" 2>&1 &
    fi
    echo "  Started $name (PID: $!)"
    sleep 1
}

# Start each MCP server in background
start_server "File Management" "mcp-servers/mcp_file_management.py" ""
start_server "Data Science" "mcp-servers/mcp_data_science_tools.py" "8007"
start_server "Data Preprocessing" "mcp-servers/mcp_data_preprocessing_tools.py" "8017"
start_server "Statistical Analysis" "mcp-servers/mcp_statistical_analysis_tools.py" "8018"
start_server "Advanced ML" "mcp-servers/mcp_advanced_ml_tools.py" "8016"
start_server "Semiconductor Yield Analysis" "mcp-servers/mcp_semiconductor_yield_analysis.py" "8008"
start_server "Process Control Charts" "mcp-servers/mcp_process_control_charts.py" "8009"
start_server "Equipment Analysis" "mcp-servers/mcp_semiconductor_equipment_analysis.py" "8010"
start_server "Defect Pattern Analysis" "mcp-servers/mcp_defect_pattern_analysis.py" "8011"
start_server "Process Optimization" "mcp-servers/mcp_process_optimization.py" "8012"
start_server "Semiconductor Process Tools" "mcp-servers/mcp_semiconductor_process_tools.py" "8020"
start_server "Time Series Analysis" "mcp-servers/mcp_timeseries_analysis.py" "8013"
start_server "Anomaly Detection" "mcp-servers/mcp_anomaly_detection.py" "8014"
start_server "Report Writing Tools" "mcp-servers/mcp_report_writing_tools.py" "8019"

# Wait for servers to start
echo "================================================"
echo "Waiting for MCP servers to initialize..."
echo "================================================"
sleep 30

echo "[OK] All MCP servers startup initiated!"
echo ""
echo "Check individual server logs in the logs/ directory for detailed status."
echo "To stop all servers, run: pkill -f 'mcp-servers'"
echo ""
echo "Server URLs (default):"
echo "  - File Management: http://localhost:8006"
echo "  - Data Science: http://localhost:8007"
echo "  - Semiconductor Yield: http://localhost:8008"
echo "  - Process Control: http://localhost:8009"
echo "  - Equipment Analysis: http://localhost:8010"
echo "  - Defect Pattern: http://localhost:8011"
echo "  - Process Optimization: http://localhost:8012"
echo "  - Time Series Analysis: http://localhost:8013"
echo "  - Anomaly Detection: http://localhost:8014"
echo "  - Advanced ML: http://localhost:8016"
echo "  - Data Preprocessing: http://localhost:8017"
echo "  - Statistical Analysis: http://localhost:8018"
echo "  - Report Writing: http://localhost:8019"
echo "  - Semiconductor Process: http://localhost:8020"

echo "================================================"
echo "MCP Server startup complete."
echo "================================================" 