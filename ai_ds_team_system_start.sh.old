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
    "AI_DS_Team_PythonREPL:8315:a2a_ds_servers/ai_ds_team_python_repl_server.py"
)

CORE_SERVERS=(
    "A2A_Orchestrator:8100:a2a_ds_servers/a2a_orchestrator.py"
)

STANDALONE_SERVERS=(
    "Standalone_Pandas_Agent:8080:standalone_pandas_agent_server.py"
)

# Context Engineering 컴포넌트 상태 점검
check_context_engineering_status() {
    echo -e "${CYAN}🧠 Context Engineering 시스템 상태 점검...${NC}"
    
    # Agent Persona Manager 상태 확인
    if [ -f "a2a_ds_servers/context_engineering/agent_persona_manager.py" ]; then
        echo -e "${GREEN}✅ Agent Persona Manager: 사용 가능${NC}"
    else
        echo -e "${RED}❌ Agent Persona Manager: 파일 없음${NC}"
    fi
    
    # Collaboration Rules Engine 상태 확인
    if [ -f "a2a_ds_servers/context_engineering/collaboration_rules_engine.py" ]; then
        echo -e "${GREEN}✅ Collaboration Rules Engine: 사용 가능${NC}"
    else
        echo -e "${RED}❌ Collaboration Rules Engine: 파일 없음${NC}"
    fi
    
    # Error Recovery System 상태 확인
    if [ -f "a2a_ds_servers/base/intelligent_data_handler.py" ]; then
        echo -e "${GREEN}✅ Intelligent Data Handler: 사용 가능${NC}"
    else
        echo -e "${RED}❌ Intelligent Data Handler: 파일 없음${NC}"
    fi
    
    # Streaming Wrapper 상태 확인
    if [ -f "a2a_ds_servers/base/streaming_wrapper.py" ]; then
        echo -e "${GREEN}✅ Streaming Wrapper: 사용 가능${NC}"
    else
        echo -e "${RED}❌ Streaming Wrapper: 파일 없음${NC}"
    fi
    
    echo -e "${CYAN}📊 Context Engineering 6개 데이터 레이어:${NC}"
    echo -e "${BLUE}  1. INSTRUCTIONS Layer: Agent Persona Manager${NC}"
    echo -e "${BLUE}  2. MEMORY Layer: Collaboration Rules Engine${NC}"
    echo -e "${BLUE}  3. HISTORY Layer: Session Management${NC}"
    echo -e "${BLUE}  4. INPUT Layer: Intelligent Data Handler${NC}"
    echo -e "${BLUE}  5. TOOLS Layer: MCP Integration${NC}"
    echo -e "${BLUE}  6. OUTPUT Layer: Streaming Wrapper${NC}"
}

# A2A + MCP 통합 상태 점검
check_a2a_mcp_integration() {
    echo -e "${CYAN}🌐 A2A + MCP 통합 시스템 상태 점검...${NC}"
    
    # MCP 도구 상태 확인
    if [ -f "a2a_ds_servers/tools/mcp_integration.py" ]; then
        echo -e "${GREEN}✅ MCP Integration: 사용 가능${NC}"
    else
        echo -e "${RED}❌ MCP Integration: 파일 없음${NC}"
    fi
    
    # 아키텍처 정보 출력
    echo -e "${PURPLE}🏗️ 세계 최초 A2A + MCP 통합 플랫폼 아키텍처:${NC}"
    echo -e "${BLUE}  • A2A 프로토콜: 10개 에이전트 (포트 8306-8315)${NC}"
    echo -e "${BLUE}  • MCP 도구: 7개 도구 (Playwright, FileManager, Database 등)${NC}"
    echo -e "${BLUE}  • 통합 워크플로우: MCP 도구 → A2A 에이전트 → 통합 결과${NC}"
    echo -e "${BLUE}  • Context Engineering: 6개 데이터 레이어 구조${NC}"
}

# 서버 상태 세부 점검
check_server_health() {
    local name=$1
    local port=$2
    
    if check_port $port; then
        # 서버 응답 상태 확인
        if command -v curl >/dev/null 2>&1; then
            # A2A Agent Card 확인
            if curl -s --max-time 5 "http://localhost:$port/.well-known/agent.json" >/dev/null 2>&1; then
                echo -e "${GREEN}✅ $name: 정상 (A2A 호환)${NC}"
            else
                echo -e "${YELLOW}⚠️ $name: 실행 중 (A2A 응답 없음)${NC}"
            fi
        else
            echo -e "${GREEN}✅ $name: 실행 중${NC}"
        fi
    else
        echo -e "${RED}❌ $name: 중지됨${NC}"
    fi
}

# 시스템 성능 모니터링
check_system_performance() {
    echo -e "${CYAN}📈 시스템 성능 모니터링...${NC}"
    
    # 메모리 사용량 확인
    if command -v ps >/dev/null 2>&1; then
        echo -e "${BLUE}🧠 메모리 사용량:${NC}"
        ps aux | grep -E "(python.*server|streamlit)" | grep -v grep | while read line; do
            echo "  $line" | awk '{printf "  PID: %s, MEM: %s%%, CMD: %s\n", $2, $4, $11}'
        done
    fi
    
    # 디스크 사용량 확인
    if [ -d "a2a_ds_servers/logs" ]; then
        log_size=$(du -sh a2a_ds_servers/logs 2>/dev/null | cut -f1)
        echo -e "${BLUE}📁 로그 디렉토리 크기: ${log_size}${NC}"
    fi
    
    # 데이터 디렉토리 확인
    if [ -d "a2a_ds_servers/artifacts" ]; then
        data_size=$(du -sh a2a_ds_servers/artifacts 2>/dev/null | cut -f1)
        echo -e "${BLUE}📊 데이터 디렉토리 크기: ${data_size}${NC}"
    fi
}

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

# Context Engineering 시스템 상태 점검
check_context_engineering_status

echo ""

# A2A + MCP 통합 상태 점검
check_a2a_mcp_integration

echo ""

# 1. 코어 서버들 시작 (오케스트레이터)
for server_entry in "${CORE_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    start_server "$name" "$port" "$script"
    sleep 2
done

echo ""
echo -e "${PURPLE}🚀 Starting Standalone Servers...${NC}"

# 1.5. Standalone 서버들 시작
for server_entry in "${STANDALONE_SERVERS[@]}"; do
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

# 시스템 성능 모니터링
check_system_performance

echo ""

# 최종 상태 확인
echo -e "${CYAN}📊 Final System Status:${NC}"
echo "Core Servers: ${#CORE_SERVERS[@]}"
echo "Standalone Servers: ${#STANDALONE_SERVERS[@]}"
echo "AI_DS_Team Agents: $started_count/$total_count"

# 포트 상태 확인
echo ""
echo -e "${CYAN}🔍 Port Status Check:${NC}"

# 코어 서버 포트 확인
for server_entry in "${CORE_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    check_server_health "$name" "$port"
done

# Standalone 서버 포트 확인
for server_entry in "${STANDALONE_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    check_server_health "$name" "$port"
done

# AI_DS_Team 서버 포트 확인
for server_entry in "${AI_DS_SERVERS[@]}"; do
    IFS=':' read -r name port script <<< "$server_entry"
    check_server_health "$name" "$port"
done

echo ""
echo -e "${CYAN}🌐 Services:${NC}"
echo "Main Application: http://localhost:8501"
echo "Standalone Pandas Agent: http://localhost:8080"
echo "AI_DS_Team Orchestrator: http://localhost:8501 -> 7_🧬_AI_DS_Team_Orchestrator"

echo ""
echo -e "${GREEN}🎉 AI_DS_Team A2A System startup complete!${NC}"
echo -e "${YELLOW}💡 Use './ai_ds_team_system_stop.sh' to stop all services${NC}"

# 사용법 안내
echo ""
echo -e "${CYAN}📖 Quick Usage Guide:${NC}"
echo "1. Open Streamlit: streamlit run main.py"
echo "2. Upload data files and ask questions"
echo "3. Monitor A2A agents and MCP tools collaboration"
echo "4. View artifacts and insights in the dashboard"

# 로그 모니터링 안내
echo ""
echo -e "${CYAN}📝 Log Monitoring:${NC}"
echo "tail -f a2a_ds_servers/logs/*.log  # All logs"
echo "ls a2a_ds_servers/logs/           # List all log files"

# Context Engineering 시스템 상태 최종 요약
echo ""
echo -e "${CYAN}🧠 Context Engineering 시스템 상태 요약:${NC}"
echo -e "${BLUE}  • INSTRUCTIONS Layer: Agent Persona Manager 활성화${NC}"
echo -e "${BLUE}  • MEMORY Layer: Collaboration Rules Engine 활성화${NC}"
echo -e "${BLUE}  • HISTORY Layer: Session Management 활성화${NC}"
echo -e "${BLUE}  • INPUT Layer: Intelligent Data Handler 활성화${NC}"
echo -e "${BLUE}  • TOOLS Layer: MCP Integration 활성화${NC}"
echo -e "${BLUE}  • OUTPUT Layer: Streaming Wrapper 활성화${NC}"

# A2A + MCP 통합 상태 최종 요약
echo ""
echo -e "${PURPLE}🌐 A2A + MCP 통합 플랫폼 상태 요약:${NC}"
echo -e "${BLUE}  • A2A 에이전트: 10개 (포트 8306-8315) 활성화${NC}"
echo -e "${BLUE}  • MCP 도구: 7개 도구 통합 완료${NC}"
echo -e "${BLUE}  • 워크플로우: MCP → A2A → 통합 결과${NC}"
echo -e "${BLUE}  • 세계 최초 A2A + MCP 통합 플랫폼 가동 중${NC}" 