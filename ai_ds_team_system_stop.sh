#!/bin/bash

# AI_DS_Team A2A System Stop Script
# CherryAI 프로젝트 - AI_DS_Team 통합 시스템 중지

echo "🛑 AI_DS_Team A2A System Stopping..."
echo "==================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# AI_DS_Team 서버 포트들
AI_DS_TEAM_PORTS=(8306 8307 8308 8309 8310 8311 8312 8313 8314)
CORE_PORTS=(8100)

# 함수: 포트로 프로세스 종료
kill_process_by_port() {
    local port=$1
    local service_name=$2
    
    # lsof로 포트 사용 프로세스 찾기
    local pid=$(lsof -ti :$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        echo -e "${BLUE}🔍 Port $port에서 실행 중인 프로세스 발견 (PID: $pid)${NC}"
        
        # SIGTERM으로 정상 종료 시도
        if kill $pid 2>/dev/null; then
            echo -e "${YELLOW}⏳ $service_name 정상 종료 중...${NC}"
            
            # 5초 대기
            local count=0
            while [ $count -lt 5 ]; do
                if ! kill -0 $pid 2>/dev/null; then
                    echo -e "${GREEN}✅ $service_name 정상 종료됨${NC}"
                    return 0
                fi
                sleep 1
                count=$((count + 1))
            done
            
            # 강제 종료
            echo -e "${RED}⚡ $service_name 강제 종료 중...${NC}"
            if kill -9 $pid 2>/dev/null; then
                echo -e "${GREEN}✅ $service_name 강제 종료됨${NC}"
                return 0
            else
                echo -e "${RED}❌ $service_name 종료 실패${NC}"
                return 1
            fi
        else
            echo -e "${RED}❌ $service_name 종료 신호 전송 실패${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}ℹ️  Port $port: 실행 중인 프로세스 없음${NC}"
        return 0
    fi
}

# 함수: PID 파일로 프로세스 종료
kill_process_by_pidfile() {
    local pidfile=$1
    local service_name=$2
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo -e "${BLUE}🔍 PID 파일에서 $service_name 프로세스 발견 (PID: $pid)${NC}"
            
            if kill $pid 2>/dev/null; then
                echo -e "${GREEN}✅ $service_name (PID: $pid) 종료됨${NC}"
            else
                echo -e "${RED}❌ $service_name (PID: $pid) 종료 실패${NC}"
            fi
        else
            echo -e "${YELLOW}ℹ️  $service_name: 유효하지 않은 PID${NC}"
        fi
        
        # PID 파일 제거
        rm -f "$pidfile"
        echo -e "${CYAN}🗑️  PID 파일 제거: $pidfile${NC}"
    fi
}

# 메인 종료 프로세스
echo -e "${CYAN}🔄 AI_DS_Team 에이전트 종료 중...${NC}"

# 1. AI_DS_Team 서버들 종료
declare -A AI_DS_SERVICES=(
    [8306]="AI_DS_Team_DataCleaning"
    [8307]="AI_DS_Team_DataLoader"
    [8308]="AI_DS_Team_DataVisualization"
    [8309]="AI_DS_Team_DataWrangling"
    [8310]="AI_DS_Team_FeatureEngineering"
    [8311]="AI_DS_Team_SQLDatabase"
    [8312]="AI_DS_Team_EDATools"
    [8313]="AI_DS_Team_H2OML"
    [8314]="AI_DS_Team_MLflowTools"
)

stopped_count=0
total_ai_agents=${#AI_DS_TEAM_PORTS[@]}

for port in "${AI_DS_TEAM_PORTS[@]}"; do
    service_name="${AI_DS_SERVICES[$port]}"
    if kill_process_by_port $port "$service_name"; then
        stopped_count=$((stopped_count + 1))
    fi
    
    # PID 파일도 확인
    pidfile="a2a_ds_servers/logs/${service_name}.pid"
    kill_process_by_pidfile "$pidfile" "$service_name"
    
    sleep 0.5
done

echo ""
echo -e "${CYAN}🔄 코어 A2A 서버 종료 중...${NC}"

# 2. 코어 서버들 종료
declare -A CORE_SERVICES=(
    [8100]="Universal_AI_Orchestrator"
)

for port in "${CORE_PORTS[@]}"; do
    service_name="${CORE_SERVICES[$port]}"
    kill_process_by_port $port "$service_name"
    
    # PID 파일도 확인
    pidfile="a2a_ds_servers/logs/${service_name}.pid"
    kill_process_by_pidfile "$pidfile" "$service_name"
done

echo ""
echo -e "${CYAN}🧹 추가 정리 작업...${NC}"

# 3. Python 프로세스 중 서버 관련 프로세스들 종료
echo -e "${BLUE}🔍 Python 서버 프로세스 검사 중...${NC}"

# ai_ds_team 관련 프로세스들 찾기
ai_ds_processes=$(ps aux | grep python | grep -E "(ai_ds_team.*server|orchestrator_server)" | grep -v grep | awk '{print $2}')

if [ -n "$ai_ds_processes" ]; then
    echo -e "${YELLOW}⚡ AI_DS_Team 관련 Python 프로세스 종료 중...${NC}"
    for pid in $ai_ds_processes; do
        if kill $pid 2>/dev/null; then
            echo -e "${GREEN}✅ Python 프로세스 종료됨 (PID: $pid)${NC}"
        fi
    done
else
    echo -e "${GREEN}✅ AI_DS_Team 관련 Python 프로세스 없음${NC}"
fi

# 4. 로그 디렉토리 정리 (선택사항)
echo -e "${CYAN}📁 로그 파일 정리...${NC}"

if [ -d "a2a_ds_servers/logs" ]; then
    # PID 파일들 제거
    find a2a_ds_servers/logs -name "*.pid" -delete 2>/dev/null
    echo -e "${GREEN}✅ PID 파일들 정리됨${NC}"
    
    # 오래된 로그 파일들 압축 (7일 이상)
    find a2a_ds_servers/logs -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null
    echo -e "${GREEN}✅ 오래된 로그 파일들 압축됨${NC}"
fi

# 5. 최종 상태 확인
echo ""
echo "==================================="
echo -e "${CYAN}📊 최종 상태 확인:${NC}"

echo -e "${CYAN}🔍 포트 상태 검사:${NC}"
active_ports=0

# AI_DS_Team 포트 확인
for port in "${AI_DS_TEAM_PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port: 여전히 사용 중${NC}"
        active_ports=$((active_ports + 1))
    else
        echo -e "${GREEN}✅ Port $port: 정리됨${NC}"
    fi
done

# 코어 포트 확인
for port in "${CORE_PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $port: 여전히 사용 중${NC}"
        active_ports=$((active_ports + 1))
    else
        echo -e "${GREEN}✅ Port $port: 정리됨${NC}"
    fi
done

echo ""
echo -e "${CYAN}📈 종료 요약:${NC}"
echo "AI_DS_Team 에이전트: $stopped_count/$total_ai_agents 종료됨"
echo "활성 포트: $active_ports개"

if [ $active_ports -eq 0 ]; then
    echo -e "${GREEN}🎉 AI_DS_Team A2A 시스템이 완전히 종료되었습니다!${NC}"
else
    echo -e "${YELLOW}⚠️  일부 서비스가 여전히 실행 중입니다.${NC}"
    echo -e "${YELLOW}💡 강제 종료가 필요하면 다음 명령을 실행하세요:${NC}"
    echo "sudo lsof -ti :8306,:8307,:8308,:8309,:8310,:8311,:8312,:8313,:8314,:8100 | xargs sudo kill -9"
fi

# 재시작 안내
echo ""
echo -e "${CYAN}🔄 시스템 재시작:${NC}"
echo "./ai_ds_team_system_start.sh"

echo ""
echo -e "${CYAN}📖 추가 정보:${NC}"
echo "- 로그 파일: a2a_ds_servers/logs/"
echo "- 데이터 아티팩트: a2a_ds_servers/artifacts/"
echo "- 시스템 상태 확인: ps aux | grep python | grep server" 