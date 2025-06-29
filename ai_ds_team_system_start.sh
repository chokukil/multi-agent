#!/bin/bash

# AI_DS_Team A2A System Start Script
# CherryAI 프로젝트 - AI_DS_Team 통합 시스템

echo "🧬 AI_DS_Team A2A System Starting..."

# .env 파일 로드
if [ -f .env ]; then
    echo "🔧 Loading environment variables from .env..."
    export $(cat .env | grep -v "^#" | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️ .env file not found. Please create .env file with required settings."
fi
echo "=================================="

# .env 파일 로드
if [ -f .env ]; then
    echo "🔧 Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️ .env file not found. Please create .env file with required settings."
fi

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리 확인
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found. Please run from project root.${NC}"
    exit 1
fi

# Python 가상환경 활성화 확인
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected. Activating uv environment...${NC}"
    source .venv/bin/activate 2>/dev/null || {
        echo -e "${RED}Error: Could not activate virtual environment${NC}"
        exit 1
    }
fi

# 로그 디렉토리 생성
mkdir -p a2a_ds_servers/logs

# 서버 정의 (name:port:script 형태)
AI_DS_SERVERS=(
    "AI_DS_Team_DataCleaning:8306:a2a_ds_servers/ai_ds_team_data_cleaning_server.py"
    "AI_DS_Team_DataLoader:8307:a2a_ds_servers/ai_ds_team_data_loader_server.py"
    "AI_DS_Team_DataVisualization:8308:a2a_ds_servers/ai_ds_team_data_visualization_server.py"
    "AI_DS_Team_DataWrangling:8309:a2a_ds_servers/ai_ds_team_data_wrangling_server.py"
    "AI_DS_Team_FeatureEngineering:8310:a2a_ds_servers/ai_ds_team_feature_engineering_server.py"
    "AI_DS_Team_SQLDatabase:8311:a2a_ds_servers/ai_ds_team_sql_database_server.py"
    "AI_DS_Team_EDATools:8312:a2a_ds_servers/ai_ds_team_eda_tools_server.py"
    "AI_DS_Team_H2OML:8313:a2a_ds_servers/ai_ds_team_h2o_ml_server.py"
    "AI_DS_Team_MLflowTools:8314:a2a_ds_servers/ai_ds_team_mlflow_tools_server.py"
)

CORE_SERVERS=(
    "A2A_Orchestrator:8100:a2a_ds_servers/a2a_orchestrator_v6.py"
)

# 함수: 포트 사용 중인지 확인
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # 포트 사용 중
    else
        return 1  # 포트 사용 가능
    fi
}

# 함수: 서버 시작
start_server() {
    local name=$1
    local port=$2
    local script=$3
    
    if check_port $port; then
        echo -e "${YELLOW}⚠️  $name (port $port) is already running${NC}"
        return 1
    fi
    
    if [ ! -f "$script" ]; then
        echo -e "${RED}❌ Script not found: $script${NC}"
        return 1
    fi
    
    echo -e "${BLUE}🚀 Starting $name on port $port...${NC}"
    nohup python "$script" > "a2a_ds_servers/logs/${name}.log" 2>&1 &
    local pid=$!
    echo $pid > "a2a_ds_servers/logs/${name}.pid"
    
    # 서버 시작 확인 (최대 10초 대기)
    local count=0
    while [ $count -lt 10 ]; do
        if check_port $port; then
            echo -e "${GREEN}✅ $name started successfully (PID: $pid)${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    echo -e "${RED}❌ Failed to start $name${NC}"
    return 1
}

# 메인 실행
echo -e "${CYAN}📋 Starting Core A2A Servers...${NC}"

# 1. 코어 서버들 시작 (오케스트레이터)
for server_entry in "${CORE_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    start_server "$name" "$port" "$script"
    sleep 2
done

echo ""
echo -e "${PURPLE}🧬 Starting AI_DS_Team Agents...${NC}"

# 2. AI_DS_Team 서버들 시작
started_count=0
total_count=${#AI_DS_SERVERS[@]}

for server_entry in "${AI_DS_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    
    if start_server "$name" "$port" "$script"; then
        started_count=$((started_count + 1))
    fi
    sleep 1
done

echo ""
echo "=================================="

# 최종 상태 확인
echo -e "${CYAN}📊 Final System Status:${NC}"
echo "Core Servers: ${#CORE_SERVERS[@]}"
echo "AI_DS_Team Agents: $started_count/$total_count"

# 포트 상태 확인
echo ""
echo -e "${CYAN}🔍 Port Status Check:${NC}"

# 코어 서버 포트 확인
for server_entry in "${CORE_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    if check_port $port; then
        echo -e "${GREEN}✅ Port $port: $name${NC}"
    else
        echo -e "${RED}❌ Port $port: $name (not responding)${NC}"
    fi
done

# AI_DS_Team 서버 포트 확인
for server_entry in "${AI_DS_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    if check_port $port; then
        echo -e "${GREEN}✅ Port $port: $name${NC}"
    else
        echo -e "${RED}❌ Port $port: $name (not responding)${NC}"
    fi
done

echo ""
echo -e "${CYAN}🌐 Streamlit UI:${NC}"
echo "Main Application: http://localhost:8501"
echo "AI_DS_Team Orchestrator: http://localhost:8501 -> 7_🧬_AI_DS_Team_Orchestrator"

echo ""
echo -e "${GREEN}🎉 AI_DS_Team A2A System startup complete!${NC}"
echo -e "${YELLOW}💡 Use './ai_ds_team_system_stop.sh' to stop all services${NC}"

# 사용법 안내
echo ""
echo -e "${CYAN}📖 Quick Usage Guide:${NC}"
echo "1. Open Streamlit: streamlit run ai.py"
echo "2. Navigate to: 7_🧬_AI_DS_Team_Orchestrator"
echo "3. Upload data and chat with AI_DS_Team agents"
echo "4. Monitor agent status in the dashboard"

# 로그 모니터링 안내
echo ""
echo -e "${CYAN}📝 Log Monitoring:${NC}"
echo "tail -f a2a_ds_servers/logs/*.log  # All logs"
echo "ls a2a_ds_servers/logs/           # List all log files" 