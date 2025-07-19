#!/bin/bash

# 🍒 CherryAI - Unified A2A Agent System Stop Script
# 🛑 World's First A2A + MCP Integrated Platform Shutdown
# 📅 Version: 2025.07.19 - LLM First Architecture

echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
echo "🍒                                                                      🍒"
echo "🍒                        🍒 CherryAI System Stopping 🍒               🍒"
echo "🍒                                                                      🍒"
echo "🍒    🛑 Graceful Shutdown of A2A + MCP Integrated Platform 🛑         🍒"
echo "🍒    🧠 LLM First Architecture with 11 Specialized Agents 🧠          🍒"
echo "🍒    🔧 Safe System Cleanup and Resource Management 🔧                🍒"
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

# 🍒 CherryAI 포트 정의
CHERRY_PORTS=(8100 8316 8317 8318 8319 8320 8321 8322 8323 8324 8325 8326)

# 🍒 CherryAI 서비스 이름 매핑
declare -A CHERRY_SERVICES=(
    [8100]="🎯 CherryAI Orchestrator"
    [8316]="🧹 Data Cleaning Agent"
    [8317]="📊 Pandas Analyst Agent"
    [8318]="🎨 Visualization Agent"
    [8319]="🛠️ Data Wrangling Agent"
    [8320]="🔬 EDA Analysis Agent"
    [8321]="⚙️ Feature Engineering Agent"
    [8322]="📂 Data Loader Agent"
    [8323]="🤖 H2O ML Agent"
    [8324]="🗄️ SQL Database Agent"
    [8325]="🧠 Knowledge Bank Agent"
    [8326]="📋 Report Generator Agent"
)

# 🍒 CherryAI PID 파일 매핑
declare -A CHERRY_PID_FILES=(
    [8100]="orchestrator"
    [8316]="data_cleaning"
    [8317]="pandas_analyst"
    [8318]="visualization"
    [8319]="wrangling"
    [8320]="eda"
    [8321]="feature_engineering"
    [8322]="data_loader"
    [8323]="h2o_ml"
    [8324]="sql_database"
    [8325]="knowledge_bank"
    [8326]="report"
)

# 🍒 포트로 프로세스 종료 함수
kill_process_by_port() {
    local port=$1
    local service_name=$2
    
    # lsof로 포트 사용 프로세스 찾기
    local pid=$(lsof -ti :$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo -e "${BLUE}🔍 Port $port에서 실행 중인 ${service_name} 발견 (PID: $pid)${NC}"
        
        # SIGTERM으로 정상 종료 시도
        if kill $pid 2>/dev/null; then
            echo -e "${YELLOW}⏳ ${service_name} 정상 종료 중...${NC}"
            
            # 5초 대기
            local count=0
            while [ $count -lt 5 ]; do
                if ! kill -0 $pid 2>/dev/null; then
                    echo -e "${GREEN}✅ ${service_name} 정상 종료됨${NC}"
                    return 0
                fi
                sleep 1
                count=$((count + 1))
            done
            
            # 강제 종료
            echo -e "${RED}⚡ ${service_name} 강제 종료 중...${NC}"
            if kill -9 $pid 2>/dev/null; then
                echo -e "${GREEN}✅ ${service_name} 강제 종료됨${NC}"
                return 0
            else
                echo -e "${RED}❌ ${service_name} 종료 실패${NC}"
                return 1
            fi
        else
            echo -e "${RED}❌ ${service_name} 종료 신호 전송 실패${NC}"
            return 1
        fi
    else
        echo -e "${CYAN}ℹ️  Port $port: ${service_name} 실행 중이지 않음${NC}"
        return 0
    fi
}

# 🍒 PID 파일로 프로세스 종료 함수
kill_process_by_pidfile() {
    local pidfile=$1
    local service_name=$2
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo -e "${BLUE}🔍 PID 파일에서 ${service_name} 프로세스 발견 (PID: $pid)${NC}"
            
            if kill $pid 2>/dev/null; then
                echo -e "${GREEN}✅ ${service_name} (PID: $pid) 종료됨${NC}"
            else
                echo -e "${RED}❌ ${service_name} (PID: $pid) 종료 실패${NC}"
            fi
        else
            echo -e "${YELLOW}ℹ️  ${service_name}: 유효하지 않은 PID${NC}"
        fi
        
        # PID 파일 제거
        rm -f "$pidfile"
        echo -e "${CYAN}🗑️  PID 파일 제거: $pidfile${NC}"
    fi
}

# 🍒 CherryAI 시스템 정리 시작
echo -e "${CHERRY}🍒 CherryAI System Cleanup Initiated${NC}"
echo "=================================================="

# 1. 🍒 CherryAI 에이전트들 종료 (역순으로)
echo -e "${PURPLE}🤖 Stopping 11 CherryAI A2A Agents...${NC}"

stopped_count=0
total_agents=${#CHERRY_PORTS[@]}

# 에이전트 종료 순서 (역순)
agent_shutdown_order=(8326 8325 8324 8323 8322 8321 8320 8319 8318 8317 8316)

for port in "${agent_shutdown_order[@]}"; do
    if [ $port -ne 8100 ]; then  # 오케스트레이터는 나중에
        service_name="${CHERRY_SERVICES[$port]}"
        if kill_process_by_port $port "$service_name"; then
            stopped_count=$((stopped_count + 1))
        fi
        
        # PID 파일도 확인
        pid_name="${CHERRY_PID_FILES[$port]}"
        pidfile="$PID_DIR/${pid_name}.pid"
        kill_process_by_pidfile "$pidfile" "$service_name"
        
        sleep 0.5
    fi
done

echo ""

# 2. 🍒 CherryAI 오케스트레이터 종료
echo -e "${PURPLE}🎯 Stopping CherryAI Orchestrator...${NC}"

orchestrator_port=8100
service_name="${CHERRY_SERVICES[$orchestrator_port]}"
kill_process_by_port $orchestrator_port "$service_name"

# PID 파일도 확인
pid_name="${CHERRY_PID_FILES[$orchestrator_port]}"
pidfile="$PID_DIR/${pid_name}.pid"
kill_process_by_pidfile "$pidfile" "$service_name"

echo ""

# 3. 🍒 추가 Python 프로세스 정리
echo -e "${CYAN}🧹 Additional CherryAI Process Cleanup...${NC}"

# CherryAI 관련 프로세스들 찾기
cherry_processes=$(ps aux | grep python | grep -E "(a2a_ds_servers|orchestrator|cherry)" | grep -v grep | awk '{print $2}')

if [ -n "$cherry_processes" ]; then
    echo -e "${YELLOW}⚡ CherryAI 관련 Python 프로세스 정리 중...${NC}"
    for pid in $cherry_processes; do
        if kill $pid 2>/dev/null; then
            echo -e "${GREEN}✅ CherryAI Python 프로세스 종료됨 (PID: $pid)${NC}"
        fi
    done
else
    echo -e "${GREEN}✅ CherryAI 관련 Python 프로세스 없음${NC}"
fi

echo ""

# 4. 🍒 로그 및 임시 파일 정리
echo -e "${CYAN}📁 CherryAI Log and Temporary File Cleanup...${NC}"

if [ -d "$LOG_DIR" ]; then
    # PID 파일들 제거
    find "$PID_DIR" -name "*.pid" -delete 2>/dev/null
    echo -e "${GREEN}✅ PID 파일들 정리됨${NC}"
    
    # 오래된 로그 파일들 압축 (7일 이상)
    find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null
    echo -e "${GREEN}✅ 오래된 로그 파일들 압축됨${NC}"
fi

# 임시 파일들 정리
if [ -d "a2a_ds_servers" ]; then
    find a2a_ds_servers -name "*.tmp" -delete 2>/dev/null
    find a2a_ds_servers -name "*.lock" -delete 2>/dev/null
    echo -e "${GREEN}✅ 임시 파일들 정리됨${NC}"
fi

echo ""

# 5. 🍒 최종 포트 상태 확인
echo -e "${CHERRY}🍒 Final CherryAI Port Status Check${NC}"
echo "=================================================="

active_ports=0

for port in "${CHERRY_PORTS[@]}"; do
    service_name="${CHERRY_SERVICES[$port]}"
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port: ${service_name} 여전히 사용 중${NC}"
        active_ports=$((active_ports + 1))
    else
        echo -e "${GREEN}✅ Port $port: ${service_name} 정리됨${NC}"
    fi
done

echo ""

# 6. 🍒 시스템 리소스 상태 확인
echo -e "${CYAN}📊 CherryAI System Resource Status${NC}"
echo "=================================================="

# 메모리 사용량 확인
if command -v ps >/dev/null 2>&1; then
    remaining_processes=$(ps aux | grep -E "(python.*server|streamlit|cherry)" | grep -v grep | wc -l)
    if [ $remaining_processes -gt 0 ]; then
        echo -e "${YELLOW}⚠️ 남은 프로세스: ${remaining_processes}개${NC}"
        echo -e "${BLUE}🔍 남은 프로세스 목록:${NC}"
        ps aux | grep -E "(python.*server|streamlit|cherry)" | grep -v grep | while read line; do
            echo "  $line" | awk '{printf "  PID: %s, MEM: %s%%, CMD: %s\n", $2, $4, $11}'
        done
    else
        echo -e "${GREEN}✅ 모든 CherryAI 프로세스 정리 완료${NC}"
    fi
fi

# 디스크 사용량 확인
if [ -d "$LOG_DIR" ]; then
    log_size=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
    echo -e "${BLUE}📁 로그 디렉토리 크기: ${log_size}${NC}"
fi

echo ""

# 7. 🍒 최종 결과 요약
echo -e "${CHERRY}🍒 CherryAI Shutdown Summary${NC}"
echo "=================================================="
echo -e "${BLUE}🤖 CherryAI 에이전트: $stopped_count/11 종료됨${NC}"
echo -e "${BLUE}🔌 활성 포트: $active_ports개${NC}"
echo -e "${BLUE}📁 로그 디렉토리: $LOG_DIR${NC}"
echo -e "${BLUE}📁 PID 디렉토리: $PID_DIR${NC}"

echo ""

if [ $active_ports -eq 0 ]; then
    echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
    echo "🍒                                                                      🍒"
    echo "🍒                🎉 CherryAI System Shutdown Complete! 🎉             🍒"
    echo "🍒                                                                      🍒"
    echo "🍒    ✅ All 11 A2A Agents Successfully Terminated                     🍒"
    echo "🍒    ✅ LLM First Architecture Safely Shutdown                        🍒"
    echo "🍒    ✅ World's First A2A + MCP Platform Gracefully Stopped          🍒"
    echo "🍒                                                                      🍒"
    echo "🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒"
else
    echo -e "${YELLOW}⚠️  일부 CherryAI 서비스가 여전히 실행 중입니다.${NC}"
    echo -e "${YELLOW}💡 강제 종료가 필요하면 다음 명령을 실행하세요:${NC}"
    echo "sudo lsof -ti :8100,:8316,:8317,:8318,:8319,:8320,:8321,:8322,:8323,:8324,:8325,:8326 | xargs sudo kill -9"
fi

echo ""

# 8. 🍒 재시작 및 추가 정보 안내
echo -e "${CHERRY}🍒 CherryAI Quick Reference${NC}"
echo "=================================================="
echo -e "${GREEN}🔄 시스템 재시작:${NC}"
echo "   ./start.sh"
echo ""
echo -e "${BLUE}📖 추가 정보:${NC}"
echo "   • 로그 파일: $LOG_DIR/"
echo "   • PID 파일: $PID_DIR/"
echo "   • A2A 에이전트: 11개 (포트 8316-8326)"
echo "   • 오케스트레이터: 1개 (포트 8100)"
echo "   • 아키텍처: LLM First + A2A Protocol + MCP Integration"
echo ""
echo -e "${CHERRY}🍒 Thank you for using CherryAI! 🍒${NC}"
echo -e "${PURPLE}🌟 World's First A2A + MCP Integrated Platform 🌟${NC}"