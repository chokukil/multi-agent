#!/bin/bash

echo "🍒 CherryAI A2A Agent System Starting..."
echo "==========================================="

mkdir -p pids logs

# Define agents and ports
agents=(
    "data_cleaning:8306"
    "data_visualization:8308"
    "data_wrangling:8309"
    "feature_engineering:8310"
    "sql_database:8311"
    "eda_tools:8312"
    "h2o_ml:8313"
    "mlflow_tools:8314"
    "pandas_analyst:8315"
    "report_generator:8316"
)

check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

start_agent() {
    local name=$1
    local port=$2
    
    echo -n "Starting $name agent on port $port... "
    
    if check_port $port; then
        echo "❌ Port $port already in use"
        return 1
    fi
    
    nohup python a2a_ds_servers/${name}_server.py > logs/${name}_server.log 2>&1 &
    local pid=$!
    echo $pid > pids/${name}_server.pid
    sleep 3
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ Started (PID: $pid)"
        return 0
    else
        echo "❌ Failed to start"
        return 1
    fi
}

echo ""
echo "📡 Starting A2A Agents..."
echo "------------------------"

success=0
total=0

for agent_info in "${agents[@]}"; do
    name=${agent_info%:*}
    port=${agent_info#*:}
    ((total++))
    if start_agent "$name" "$port"; then
        ((success++))
    fi
done

echo ""
echo "🎛️ Starting Orchestrator..."
echo "---------------------------"

echo -n "Starting orchestrator on port 8100... "
((total++))

if check_port 8100; then
    echo "❌ Port 8100 already in use"
else  
    nohup python a2a_ds_servers/a2a_orchestrator.py > logs/orchestrator.log 2>&1 &
    pid=$!
    echo $pid > pids/orchestrator.pid
    sleep 3
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ Started (PID: $pid)"
        ((success++))
    else
        echo "❌ Failed to start"
    fi
fi

echo ""
echo "==========================================="
echo "🎯 Summary: $success/$total services started successfully"

if [ $success -eq $total ]; then
    echo "✅ All services are running!"
    echo ""
    echo "📊 Agent Endpoints:"
    echo "  • Orchestrator:        http://localhost:8100"
    echo "  • Data Cleaning:       http://localhost:8306"
    echo "  • Data Visualization:  http://localhost:8308"
    echo "  • Data Wrangling:      http://localhost:8309"
    echo "  • Feature Engineering: http://localhost:8310"
    echo "  • SQL Database:        http://localhost:8311"
    echo "  • EDA Tools:           http://localhost:8312"
    echo "  • H2O ML:              http://localhost:8313"
    echo "  • MLflow Tools:        http://localhost:8314"
    echo "  • Pandas Analyst:      http://localhost:8315"
    echo "  • Report Generator:    http://localhost:8316"
else
    echo "⚠️ Some services failed to start"
    echo "Check logs/ directory for details"
fi

echo ""
echo "🎨 Starting Cherry AI Streamlit Platform..."
echo "------------------------------------------"

# Check if Streamlit is already running
if check_port 8501; then
    echo "⚠️  Cherry AI Streamlit Platform already running on port 8501"
else
    echo -n "Starting Cherry AI Streamlit Platform on port 8501... "
    
    # Check if start_cherry_ai_streamlit.sh exists
    if [ -f "./start_cherry_ai_streamlit.sh" ]; then
        # Make it executable if not already
        chmod +x ./start_cherry_ai_streamlit.sh
        
        # Start in background
        nohup ./start_cherry_ai_streamlit.sh > logs/cherry_ai_streamlit.log 2>&1 &
        streamlit_pid=$!
        echo $streamlit_pid > pids/cherry_ai_streamlit.pid
        sleep 5
        
        if ps -p $streamlit_pid > /dev/null 2>&1; then
            echo "✅ Started (PID: $streamlit_pid)"
            echo ""
            echo "🌟 Cherry AI Streamlit Platform Features:"
            echo "  • Enhanced ChatGPT/Claude-style interface"
            echo "  • Real-time agent collaboration visualization"
            echo "  • Drag-and-drop multi-format file processing"
            echo "  • Interactive artifacts with smart downloads"
            echo "  • LLM-powered analysis recommendations"
            echo ""
            echo "🔗 Access the platform at: http://localhost:8501"
        else
            echo "❌ Failed to start"
            echo "Check logs/cherry_ai_streamlit.log for details"
        fi
    else
        echo "❌ start_cherry_ai_streamlit.sh not found"
    fi
fi

echo ""
echo "🛑 To stop all services, run: ./stop.sh"
echo "==========================================="
