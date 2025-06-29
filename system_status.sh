#!/bin/bash

# A2A Data Science Team System Status Checker
# Checks the status of all A2A servers and Streamlit

echo "🔍 A2A Data Science System Status Check"
echo "================================================"

# Server configurations (name:script:port)
SERVERS=(
    "Orchestrator:a2a_orchestrator_v6.py:8100"
    "Pandas_Analyst:pandas_data_analyst_server.py:8200"
    "SQL_Analyst:sql_data_analyst_server.py:8201"
    "Data_Viz:data_visualization_server.py:8202"
    "EDA_Tools:eda_tools_server.py:8203"
    "Feature_Eng:feature_engineering_server.py:8204"
    "Data_Cleaning:data_cleaning_server.py:8205"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

running_count=0
total_count=0

# Function to check if a service is running
check_service() {
    local name="$1"
    local script="$2"
    local port="$3"
    
    # Check if port is in use
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        # Try to get agent card to verify it's our service
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo -e "✅ ${GREEN}$name${NC} (Port $port): Running"
            return 0
        else
            echo -e "⚠️  ${YELLOW}$name${NC} (Port $port): Port occupied but not responding properly"
            return 1
        fi
    else
        echo -e "❌ ${RED}$name${NC} (Port $port): Not running"
        return 1
    fi
}

# Check A2A servers
echo "🤖 Checking A2A Servers:"
echo "────────────────────────────────────────────────"

for server_config in "${SERVERS[@]}"; do
    IFS=':' read -r name script port <<< "$server_config"
    total_count=$((total_count + 1))
    
    if check_service "$name" "$script" "$port"; then
        running_count=$((running_count + 1))
    fi
done

# Check Streamlit
echo ""
echo "🎨 Checking Streamlit UI:"
echo "────────────────────────────────────────────────"
total_count=$((total_count + 1))

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    if curl -s -f "http://localhost:8501" >/dev/null 2>&1; then
        echo -e "✅ ${GREEN}Streamlit UI${NC} (Port 8501): Running"
        running_count=$((running_count + 1))
    else
        echo -e "⚠️  ${YELLOW}Streamlit UI${NC} (Port 8501): Port occupied but not responding"
    fi
else
    echo -e "❌ ${RED}Streamlit UI${NC} (Port 8501): Not running"
fi

# Summary
echo ""
echo "📊 System Summary:"
echo "────────────────────────────────────────────────"
echo "Services running: $running_count/$total_count"

if [ $running_count -eq $total_count ]; then
    echo -e "🎉 ${GREEN}All services are operational!${NC}"
    echo ""
    echo "🌐 Access URLs:"
    echo "   📱 Streamlit UI:        http://localhost:8501"
    echo "   🎯 Orchestrator:        http://localhost:8100"
    echo "   🐼 Pandas Data Analyst: http://localhost:8200"
    echo "   🗃️  SQL Data Analyst:    http://localhost:8201"
    echo "   📈 Data Visualization:  http://localhost:8202"
    echo "   🔍 EDA Tools:           http://localhost:8203"
    echo "   🔧 Feature Engineering: http://localhost:8204"
    echo "   🧹 Data Cleaning:       http://localhost:8205"
    
    echo ""
    echo "🔧 Quick Actions:"
    echo "   ./stop.sh              - Stop all services"
    echo "   ./start.sh             - Start all services"
    echo "   ./system_status.sh     - Check system status"
elif [ $running_count -eq 0 ]; then
    echo -e "🛑 ${RED}System is not running${NC}"
    echo ""
    echo "💡 To start the system:"
    echo "   ./start.sh"
else
    echo -e "⚠️  ${YELLOW}Partial system failure${NC} ($running_count/$total_count services running)"
    echo ""
    echo "💡 Recommended actions:"
    echo "   ./stop.sh && ./start.sh    - Restart all services"
    echo "   tail -f logs/a2a_servers/*.log - Check server logs"
fi

echo ""
echo "🕒 Status checked at: $(date)"
echo "================================================" 