#\!/bin/bash

echo "🍒 CherryAI A2A Agent System Starting..."
echo "==========================================="

mkdir -p pids logs

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
    
    nohup python /Users/gukil/CherryAI/CherryAI_0717/a2a_ds_servers/${name}_server.py > logs/${name}_server.log 2>&1 &
    local pid=$\!
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
echo "==========================================="
echo "🎯 Summary: $success/$total services started successfully"

if [ $success -eq $total ]; then
    echo "✅ All agents are running\!"
    echo ""
    echo "📊 Agent Endpoints:"
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
    echo ""
    echo "🧪 Test all agents:"
    echo "   python test_all_4_llm_first_agents.py"
else
    echo "⚠️ Some services failed to start"
    echo "Check logs/ directory for details"
fi

echo "🛑 To stop all services, run: ./stop.sh"
echo "==========================================="
