#!/bin/bash

# 🍒 CherryAI - Unified A2A Agent System Startup Script
# 🚀 World's First A2A + MCP Integrated Platform
# 📅 Version: 2025.07.19 - LLM First Architecture

echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
echo "🍒                                                                      🍒"
echo "🍒                        🍒 CherryAI System Starting 🍒                🍒"
echo "🍒                                                                      🍒"
echo "🍒    🌟 World's First A2A + MCP Integrated Platform 🌟                🍒"
echo "🍒    🧠 LLM First Architecture with 11 Specialized Agents 🧠          🍒"
echo "🍒    🚀 Hardcoding-Free Dynamic Data Processing 🚀                    🍒"
echo "🍒                                                                      🍒"
echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
CHERRY='\033[1;35m'  # Cherry 색상
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs/cherryai"
PID_DIR="$PROJECT_ROOT/logs/pids"

# 디렉토리 생성
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"#
 🍒 CherryAI 11개 A2A 에이전트 정의 (검증 완료된 서버들)
declare -A CHERRY_AGENTS=(
    # 🧹 Data Processing Agents
    ["data_cleaning"]="a2a_ds_servers/data_cleaning_server.py:8316"
    ["pandas_analyst"]="a2a_ds_servers/pandas_analyst_server.py:8317"
    ["visualization"]="a2a_ds_servers/visualization_server.py:8318"
    ["wrangling"]="a2a_ds_servers/wrangling_server.py:8319"
    
    # 🔬 Analysis Agents  
    ["eda"]="a2a_ds_servers/eda_server.py:8320"
    ["feature_engineering"]="a2a_ds_servers/feature_engineering_server.py:8321"
    ["data_loader"]="a2a_ds_servers/data_loader_server.py:8322"
    
    # 🤖 ML Agents
    ["h2o_ml"]="a2a_ds_servers/h2o_ml_server.py:8323"
    ["sql_database"]="a2a_ds_servers/sql_database_server.py:8324"
    
    # 🧠 Knowledge Agents
    ["knowledge_bank"]="a2a_ds_servers/knowledge_bank_server.py:8325"
    ["report"]="a2a_ds_servers/report_server.py:8326"
)

# 🍒 CherryAI 오케스트레이터
declare -A CHERRY_ORCHESTRATOR=(
    ["orchestrator"]="a2a_orchestrator.py:8100"
)

# 에이전트 표시명
declare -A AGENT_NAMES=(
    ["data_cleaning"]="🧹 Data Cleaning Agent"
    ["pandas_analyst"]="📊 Pandas Analyst Agent"
    ["visualization"]="🎨 Visualization Agent"
    ["wrangling"]="🛠️ Data Wrangling Agent"
    ["eda"]="🔬 EDA Analysis Agent"
    ["feature_engineering"]="⚙️ Feature Engineering Agent"
    ["data_loader"]="📂 Data Loader Agent"
    ["h2o_ml"]="🤖 H2O ML Agent"
    ["sql_database"]="🗄️ SQL Database Agent"
    ["knowledge_bank"]="🧠 Knowledge Bank Agent"
    ["report"]="📋 Report Generator Agent"
    ["orchestrator"]="🎯 CherryAI Orchestrator"
)

# PID 저장 배열
PIDS=()

# 🍒 CherryAI 정리 함수
cleanup() {
    echo -e "\n${CHERRY}🍒 CherryAI System Shutdown Initiated...${NC}"
    
    # 모든 백그라운드 프로세스 종료
    for name in "${!CHERRY_AGENTS[@]}"; do
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${BLUE}🔄 Stopping ${AGENT_NAMES[$name]} (PID: $PID)...${NC}"
                kill "$PID"
                sleep 2
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
            fi
            rm -f "$PID_FILE"
        fi
    done
    
    # 오케스트레이터 종료
    for name in "${!CHERRY_ORCHESTRATOR[@]}"; do
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${BLUE}🔄 Stopping ${AGENT_NAMES[$name]} (PID: $PID)...${NC}"
                kill "$PID"
                sleep 2
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
            fi
            rm -f "$PID_FILE"
        fi
    done
    
    echo -e "${GREEN}🍒 CherryAI System Shutdown Complete!${NC}"
}

# 정리 함수 등록
trap cleanup EXIT INT TERM# 포트 
사용 가능 확인 함수
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# 서버 준비 상태 확인 함수
wait_for_server() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}⏳ Waiting for ${AGENT_NAMES[$name]} to be ready on port $port...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ ${AGENT_NAMES[$name]} is ready!${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -e "${CYAN}   🔍 Attempt $attempt/$max_attempts...${NC}"
    done
    
    echo -e "${RED}❌ ${AGENT_NAMES[$name]} failed to start within timeout${NC}"
    return 1
}

# 🍒 CherryAI 환경 검증
echo -e "${CHERRY}🍒 CherryAI Environment Verification${NC}"
echo "=================================================="

# UV 환경 확인
if [ ! -f "$PROJECT_ROOT/uv.lock" ]; then
    echo -e "${RED}❌ UV environment not found. Please run 'uv venv' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ UV environment detected${NC}"

# .env 파일 로드
if [ -f .env ]; then
    echo -e "${BLUE}🔧 Loading environment variables from .env...${NC}"
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✅ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠️ .env file not found. Using default settings.${NC}"
fi

# 포트 가용성 확인
echo -e "${CYAN}🔍 Checking port availability...${NC}"

# 오케스트레이터 포트 확인
for name in "${!CHERRY_ORCHESTRATOR[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_ORCHESTRATOR[$name]}"
    if ! check_port "$port"; then
        echo -e "${RED}❌ Cannot start CherryAI: Port $port is already in use${NC}"
        exit 1
    fi
done

# 에이전트 포트 확인
for name in "${!CHERRY_AGENTS[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_AGENTS[$name]}"
    if ! check_port "$port"; then
        echo -e "${RED}❌ Cannot start CherryAI: Port $port is already in use${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All ports are available${NC}"
echo ""#
 🍒 CherryAI 시스템 시작
echo -e "${CHERRY}🍒 Starting CherryAI A2A Agent System${NC}"
echo "=================================================="

# 1. 오케스트레이터 먼저 시작
echo -e "${PURPLE}🎯 Starting CherryAI Orchestrator...${NC}"
for name in "${!CHERRY_ORCHESTRATOR[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_ORCHESTRATOR[$name]}"
    
    echo -e "${BLUE}🚀 Starting ${AGENT_NAMES[$name]} (Port $port)...${NC}"
    uv run python "$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PID=$!
    echo $PID > "$PID_DIR/${name}.pid"
    PIDS+=($PID)
    echo -e "${CYAN}   📝 Started with PID: $PID${NC}"
    
    if ! wait_for_server "$name" "$port"; then
        echo -e "${RED}❌ Failed to start ${AGENT_NAMES[$name]}${NC}"
        exit 1
    fi
done

echo ""

# 2. 11개 CherryAI 에이전트 시작
echo -e "${PURPLE}🤖 Starting 11 CherryAI A2A Agents...${NC}"

# 에이전트 시작 순서 (의존성 고려)
agent_order=("data_loader" "data_cleaning" "pandas_analyst" "wrangling" "eda" "feature_engineering" "visualization" "h2o_ml" "sql_database" "knowledge_bank" "report")

for name in "${agent_order[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_AGENTS[$name]}"
    
    echo -e "${BLUE}🚀 Starting ${AGENT_NAMES[$name]} (Port $port)...${NC}"
    uv run python "$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PID=$!
    echo $PID > "$PID_DIR/${name}.pid"
    PIDS+=($PID)
    echo -e "${CYAN}   📝 Started with PID: $PID${NC}"
    
    if ! wait_for_server "$name" "$port"; then
        echo -e "${RED}❌ Failed to start ${AGENT_NAMES[$name]}${NC}"
        exit 1
    fi
    echo ""
done

# 🍒 CherryAI 시스템 완료
echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
echo "🍒                                                                      🍒"
echo "🍒                🎉 CherryAI System Fully Operational! 🎉             🍒"
echo "🍒                                                                      🍒"
echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
echo ""

# 🍒 서비스 정보 표시
echo -e "${CHERRY}🍒 CherryAI Services Dashboard${NC}"
echo "=================================================="
echo -e "${PURPLE}🎯 Orchestrator:${NC}"
echo "   http://localhost:8100"
echo ""
echo -e "${BLUE}🧹 Data Processing Agents:${NC}"
echo "   🧹 Data Cleaning:        http://localhost:8316"
echo "   📊 Pandas Analyst:       http://localhost:8317"
echo "   🎨 Visualization:        http://localhost:8318"
echo "   🛠️ Data Wrangling:       http://localhost:8319"
echo ""
echo -e "${GREEN}🔬 Analysis Agents:${NC}"
echo "   🔬 EDA Analysis:         http://localhost:8320"
echo "   ⚙️ Feature Engineering:  http://localhost:8321"
echo "   📂 Data Loader:          http://localhost:8322"
echo ""
echo -e "${YELLOW}🤖 ML & Knowledge Agents:${NC}"
echo "   🤖 H2O ML:               http://localhost:8323"
echo "   🗄️ SQL Database:         http://localhost:8324"
echo "   🧠 Knowledge Bank:       http://localhost:8325"
echo "   📋 Report Generator:     http://localhost:8326"
echo ""# 🍒 Age
nt Cards 정보
echo -e "${CYAN}📋 A2A Agent Cards:${NC}"
for name in "${!CHERRY_ORCHESTRATOR[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_ORCHESTRATOR[$name]}"
    echo "   http://localhost:$port/.well-known/agent.json"
done
for name in "${!CHERRY_AGENTS[@]}"; do
    IFS=':' read -r script port <<< "${CHERRY_AGENTS[$name]}"
    echo "   http://localhost:$port/.well-known/agent.json"
done
echo ""

# 🍒 시스템 정보
echo -e "${CHERRY}🍒 CherryAI System Information${NC}"
echo "=================================================="
echo -e "${BLUE}📁 Log Directory:${NC} $LOG_DIR"
echo -e "${BLUE}📁 PID Directory:${NC} $PID_DIR"
echo -e "${BLUE}🏗️ Architecture:${NC} LLM First + A2A Protocol + MCP Integration"
echo -e "${BLUE}🚀 Features:${NC} Hardcoding-Free Dynamic Processing"
echo -e "${BLUE}🌟 Status:${NC} World's First A2A + MCP Integrated Platform"
echo ""

# 🍒 사용법 안내
echo -e "${CHERRY}🍒 CherryAI Quick Start Guide${NC}"
echo "=================================================="
echo -e "${GREEN}🔄 Test the system:${NC}"
echo "   python test_simple_a2a_client.py"
echo ""
echo -e "${GREEN}🌐 Access main application:${NC}"
echo "   streamlit run main.py"
echo ""
echo -e "${GREEN}🛑 Stop all services:${NC}"
echo "   ./stop.sh  (or Press Ctrl+C)"
echo ""

# 🍒 실시간 상태 모니터링
echo -e "${CHERRY}🍒 CherryAI System Monitor (Press Ctrl+C to stop)${NC}"
echo "=================================================="

while true; do
    all_running=true
    current_time=$(date "+%H:%M:%S")
    status_line="🍒 [$current_time] "
    
    # 오케스트레이터 상태
    for name in "${!CHERRY_ORCHESTRATOR[@]}"; do
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                status_line+="🎯"
            else
                status_line+="❌"
                all_running=false
            fi
        else
            status_line+="❌"
            all_running=false
        fi
    done
    
    status_line+=" | "
    
    # 에이전트 상태
    for name in "${agent_order[@]}"; do
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                status_line+="✅"
            else
                status_line+="❌"
                all_running=false
            fi
        else
            status_line+="❌"
            all_running=false
        fi
    done
    
    if [ "$all_running" = true ]; then
        status_line+=" | 🍒 All CherryAI services running perfectly!"
    else
        status_line+=" | ⚠️ Some CherryAI services need attention"
    fi
    
    echo -ne "\r$status_line"
    sleep 5
done