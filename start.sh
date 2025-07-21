#!/bin/bash

# 🍒 CherryAI - Unified A2A Agent System Startup Script
# 🚀 World's First A2A Platform with Universal Engine
# 📅 Version: 2025.07.21 - LLM First Universal Engine Architecture

echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
echo "🍒                                                                      🍒"
echo "🍒                        🍒 CherryAI System Starting 🍒                🍒"
echo "🍒                                                                      🍒"
echo "🍒    🌟 World's First Universal Engine + A2A Platform 🌟              🍒"
echo "🍒    🧠 Universal Engine + 11 Specialized A2A Agents 🧠              🍒"
echo "🍒    🚀 100% Zero-Hardcoding LLM First Architecture 🚀               🍒"
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
mkdir -p "$PID_DIR"

# 🍒 CherryAI 11개 A2A 에이전트 정의 (Universal Engine 호환 포트)
# Universal Engine의 A2AAgentDiscoverySystem은 8306-8315 포트를 스캔합니다
# orchestrator와 정확히 매칭되도록 수정

# Bash 3.2 호환 - 배열 방식 사용 (orchestrator 포트 매핑과 일치)
CHERRY_AGENT_NAMES=(
    "data_cleaning"
    "data_loader" 
    "data_visualization"
    "data_wrangling"
    "feature_engineering"
    "sql_database"
    "eda_tools"
    "h2o_ml"
    "pandas_analyst"
    "knowledge_bank"
    "report"
)

CHERRY_AGENT_SCRIPTS=(
    "a2a_ds_servers/data_cleaning_server.py"
    "a2a_ds_servers/data_loader_server.py"
    "a2a_ds_servers/visualization_server.py"
    "a2a_ds_servers/wrangling_server.py"
    "a2a_ds_servers/feature_engineering_server.py"
    "a2a_ds_servers/sql_database_server.py"
    "a2a_ds_servers/eda_server.py"
    "a2a_ds_servers/h2o_ml_server.py"
    "a2a_ds_servers/pandas_analyst_server.py"
    "a2a_ds_servers/knowledge_bank_server.py"
    "a2a_ds_servers/report_server.py"
)

CHERRY_AGENT_PORTS=(
    8306
    8307
    8308
    8309
    8310
    8311
    8312
    8313
    8314
    8315
    8316
)

CHERRY_AGENT_DISPLAY_NAMES=(
    "🧹 Data Cleaning Agent"
    "📂 Data Loader Agent"
    "🎨 Data Visualization Agent"
    "🛠️ Data Wrangling Agent"
    "⚙️ Feature Engineering Agent"
    "�️ SQ L Database Agent"
    "🔬 EDA Tools Agent"
    "🤖 H2O ML Agent"
    "� Pandasa Analyst Agent"
    "🧠 Knowledge Bank Agent"
    "📋 Report Generator Agent"
)

# 🍒 CherryAI 오케스트레이터
ORCHESTRATOR_SCRIPT="a2a_orchestrator.py"
ORCHESTRATOR_PORT=8100
ORCHESTRATOR_NAME="🎯 CherryAI Orchestrator"

# PID 저장 배열
PIDS=()

# 🍒 CherryAI 정리 함수
cleanup() {
    echo -e "\n${CHERRY}🍒 CherryAI System Shutdown Initiated...${NC}"
    
    # 모든 에이전트 프로세스 종료
    for i in "${!CHERRY_AGENT_NAMES[@]}"; do
        name="${CHERRY_AGENT_NAMES[$i]}"
        display_name="${CHERRY_AGENT_DISPLAY_NAMES[$i]}"
        PID_FILE="$PID_DIR/${name}.pid"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${BLUE}🔄 Stopping ${display_name} (PID: $PID)...${NC}"
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
    PID_FILE="$PID_DIR/orchestrator.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${BLUE}🔄 Stopping ${ORCHESTRATOR_NAME} (PID: $PID)...${NC}"
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    echo -e "${GREEN}🍒 CherryAI System Shutdown Complete!${NC}"
}

# 정리 함수 등록
trap cleanup EXIT INT TERM

# 포트 사용 가능 확인 함수
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
    
    # 배열에서 display name 찾기
    local display_name=""
    for i in "${!CHERRY_AGENT_NAMES[@]}"; do
        if [ "${CHERRY_AGENT_NAMES[$i]}" = "$name" ]; then
            display_name="${CHERRY_AGENT_DISPLAY_NAMES[$i]}"
            break
        fi
    done
    
    echo -e "${YELLOW}⏳ Waiting for ${display_name} to be ready on port $port...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ ${display_name} is ready!${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -e "${CYAN}   🔍 Attempt $attempt/$max_attempts...${NC}"
    done
    
    echo -e "${RED}❌ ${display_name} failed to start within timeout${NC}"
    return 1
}

# 오케스트레이터 준비 상태 확인 함수
wait_for_orchestrator() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}⏳ Waiting for ${ORCHESTRATOR_NAME} to be ready on port $port...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ ${ORCHESTRATOR_NAME} is ready!${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -e "${CYAN}   🔍 Attempt $attempt/$max_attempts...${NC}"
    done
    
    echo -e "${RED}❌ ${ORCHESTRATOR_NAME} failed to start within timeout${NC}"
    return 1
}

# 🍒 Universal Engine 포트 구성 안내
echo -e "${CHERRY}🍒 Universal Engine Configuration${NC}"
echo "=================================================="
echo -e "${GREEN}✅ A2A agents configured for Universal Engine compatibility${NC}"
echo -e "${BLUE}📌 Universal Engine discovery range: 8306-8315${NC}"
echo -e "${BLUE}📌 10 agents in discovery range, 1 agent on port 8316${NC}"
echo ""

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
if ! check_port "$ORCHESTRATOR_PORT"; then
    echo -e "${RED}❌ Cannot start CherryAI: Port $ORCHESTRATOR_PORT is already in use${NC}"
    exit 1
fi

# 에이전트 포트 확인
for i in "${!CHERRY_AGENT_PORTS[@]}"; do
    port="${CHERRY_AGENT_PORTS[$i]}"
    if ! check_port "$port"; then
        echo -e "${RED}❌ Cannot start CherryAI: Port $port is already in use${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All ports are available${NC}"
echo ""

# 🍒 CherryAI 시스템 시작
echo -e "${CHERRY}🍒 Starting CherryAI A2A Agent System${NC}"
echo "=================================================="

# 1. 오케스트레이터 먼저 시작
echo -e "${PURPLE}🎯 Starting CherryAI Orchestrator...${NC}"

echo -e "${BLUE}🚀 Starting ${ORCHESTRATOR_NAME} (Port $ORCHESTRATOR_PORT)...${NC}"
uv run python "$ORCHESTRATOR_SCRIPT" > "$LOG_DIR/orchestrator.log" 2>&1 &
PID=$!
echo $PID > "$PID_DIR/orchestrator.pid"
PIDS+=($PID)
echo -e "${CYAN}   📝 Started with PID: $PID${NC}"

if ! wait_for_orchestrator "orchestrator" "$ORCHESTRATOR_PORT"; then
    echo -e "${RED}❌ Failed to start ${ORCHESTRATOR_NAME}${NC}"
    exit 1
fi

echo ""

# 2. 11개 CherryAI 에이전트 시작
echo -e "${PURPLE}🤖 Starting 11 CherryAI A2A Agents...${NC}"

# 에이전트 시작 (배열 인덱스 사용)
for i in "${!CHERRY_AGENT_NAMES[@]}"; do
    name="${CHERRY_AGENT_NAMES[$i]}"
    script="${CHERRY_AGENT_SCRIPTS[$i]}"
    port="${CHERRY_AGENT_PORTS[$i]}"
    display_name="${CHERRY_AGENT_DISPLAY_NAMES[$i]}"
    
    echo -e "${BLUE}🚀 Starting ${display_name} (Port $port)...${NC}"
    uv run python "$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PID=$!
    echo $PID > "$PID_DIR/${name}.pid"
    PIDS+=($PID)
    echo -e "${CYAN}   📝 Started with PID: $PID${NC}"
    
    if ! wait_for_server "$name" "$port"; then
        echo -e "${RED}❌ Failed to start ${display_name}${NC}"
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
echo "   📂 Data Loader:          http://localhost:8306"
echo "   🧹 Data Cleaning:        http://localhost:8307"
echo "   📊 Pandas Analyst:       http://localhost:8308"
echo "   🎨 Visualization:        http://localhost:8309"
echo "   🛠️ Data Wrangling:       http://localhost:8310"
echo ""
echo -e "${GREEN}🔬 Analysis Agents:${NC}"
echo "   🔬 EDA Analysis:         http://localhost:8311"
echo "   ⚙️ Feature Engineering:  http://localhost:8312"
echo ""
echo -e "${YELLOW}🤖 ML & Knowledge Agents:${NC}"
echo "   🤖 H2O ML:               http://localhost:8313"
echo "   🗄️ SQL Database:         http://localhost:8314"
echo "   🧠 Knowledge Bank:       http://localhost:8315"
echo "   📋 Report Generator:     http://localhost:8316"
echo ""

# 🍒 Agent Cards 정보
echo -e "${CYAN}📋 A2A Agent Cards:${NC}"
echo "   http://localhost:$ORCHESTRATOR_PORT/.well-known/agent.json"
for i in "${!CHERRY_AGENT_PORTS[@]}"; do
    port="${CHERRY_AGENT_PORTS[$i]}"
    echo "   http://localhost:$port/.well-known/agent.json"
done
echo ""

# 🍒 시스템 정보
echo -e "${CHERRY}🍒 CherryAI System Information${NC}"
echo "=================================================="
echo -e "${BLUE}📁 Log Directory:${NC} $LOG_DIR"
echo -e "${BLUE}📁 PID Directory:${NC} $PID_DIR"
echo -e "${BLUE}🏗️ Architecture:${NC} LLM First Universal Engine + A2A Protocol"
echo -e "${BLUE}🚀 Features:${NC} Zero-Hardcoding Dynamic Processing with Universal Engine"
echo -e "${BLUE}🌟 Status:${NC} World's First Universal Engine + A2A Platform"
echo ""

# 🍒 사용법 안내
echo -e "${CHERRY}🍒 CherryAI Quick Start Guide${NC}"
echo "=================================================="
echo -e "${GREEN}🔄 Test the system:${NC}"
echo "   python test_simple_a2a_client.py"
echo ""
echo -e "${GREEN}🌐 Access main application:${NC}"
echo "   streamlit run cherry_ai.py  (Universal Engine powered)"
echo "   streamlit run main.py      (Classic interface)"
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
    PID_FILE="$PID_DIR/orchestrator.pid"
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
    
    status_line+=" | "
    
    # 에이전트 상태
    for i in "${!CHERRY_AGENT_NAMES[@]}"; do
        name="${CHERRY_AGENT_NAMES[$i]}"
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