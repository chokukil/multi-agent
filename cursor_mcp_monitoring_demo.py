"""
🔧 Cursor Style MCP Monitoring Demo - MCP 도구 모니터링 패널 데모

새로 구현한 Cursor 스타일 MCP 도구 모니터링 패널을 테스트하고 시연하는 데모
10개 MCP 도구의 실시간 상태 모니터링과 활동 시뮬레이션을 제공합니다.

실행 방법:
streamlit run cursor_mcp_monitoring_demo.py --server.port 8503
"""

import streamlit as st
import time
import random
import threading
from datetime import datetime
import asyncio

# Cursor 스타일 MCP 모니터링 import
from ui.cursor_mcp_monitoring import get_cursor_mcp_monitoring


def initialize_demo():
    """데모 초기화"""
    if 'mcp_demo_initialized' not in st.session_state:
        st.session_state.mcp_demo_initialized = True
        st.session_state.simulation_running = False
        st.session_state.auto_simulation = False

def simulate_random_activity():
    """랜덤 MCP 도구 활동 시뮬레이션"""
    monitoring = get_cursor_mcp_monitoring()
    
    # 사용 가능한 도구들과 액션들
    tool_actions = {
        'Data Loader': [
            '파일 업로드 처리',
            'CSV 데이터 파싱',
            '데이터 검증 수행',
            '메타데이터 추출'
        ],
        'Data Cleaning': [
            '결측치 처리',
            '이상치 탐지',
            '데이터 정규화',
            '중복 제거'
        ],
        'EDA Tools': [
            '기초 통계 계산',
            '분포 분석',
            '상관관계 분석',
            '패턴 탐지'
        ],
        'Data Visualization': [
            '히스토그램 생성',
            '산점도 생성',
            '박스플롯 생성',
            '히트맵 생성'
        ],
        'Feature Engineering': [
            '특성 변환',
            '원핫 인코딩',
            '스케일링',
            '파생 변수 생성'
        ],
        'H2O Modeling': [
            'AutoML 실행',
            '모델 훈련',
            '성능 평가',
            '하이퍼파라미터 튜닝'
        ],
        'MLflow Agent': [
            '실험 로깅',
            '모델 등록',
            '아티팩트 저장',
            '메트릭 추적'
        ],
        'SQL Database': [
            'SQL 쿼리 실행',
            '테이블 조회',
            '데이터 집계',
            '인덱스 최적화'
        ],
        'Data Wrangling': [
            '데이터 피벗',
            '그룹화 연산',
            '데이터 병합',
            '형식 변환'
        ],
        'Pandas Analyst': [
            '데이터프레임 분석',
            '통계 요약',
            '시계열 분석',
            '데이터 탐색'
        ]
    }
    
    # 랜덤하게 3-5개 도구 선택
    num_tools = random.randint(3, 5)
    selected_tools = random.sample(list(tool_actions.keys()), num_tools)
    
    for tool_name in selected_tools:
        action = random.choice(tool_actions[tool_name])
        duration = random.uniform(2.0, 5.0)
        
        # 백그라운드에서 시뮬레이션 실행
        threading.Thread(
            target=monitoring.simulate_tool_activity,
            args=(tool_name, action, duration),
            daemon=True
        ).start()
        
        # 약간의 지연으로 동시 시작 방지
        time.sleep(random.uniform(0.2, 0.8))

def simulate_data_pipeline():
    """전체 데이터 파이프라인 시뮬레이션"""
    monitoring = get_cursor_mcp_monitoring()
    
    # 순차적 파이프라인 시뮬레이션
    pipeline_steps = [
        ('Data Loader', '대용량 CSV 파일 로드', 3.0),
        ('Data Cleaning', '데이터 품질 검사 및 정제', 4.0),
        ('EDA Tools', '탐색적 데이터 분석 수행', 3.5),
        ('Feature Engineering', '머신러닝 특성 생성', 4.5),
        ('Data Visualization', '분석 결과 시각화', 2.5),
        ('H2O Modeling', 'AutoML 모델 훈련', 6.0),
        ('MLflow Agent', '실험 결과 추적', 2.0)
    ]
    
    def run_pipeline():
        for tool_name, action, duration in pipeline_steps:
            monitoring.simulate_tool_activity(tool_name, action, duration)
            time.sleep(0.5)  # 다음 단계로 넘어가기 전 잠시 대기
    
    # 백그라운드에서 파이프라인 실행
    threading.Thread(target=run_pipeline, daemon=True).start()

def simulate_error_scenario():
    """에러 시나리오 시뮬레이션"""
    monitoring = get_cursor_mcp_monitoring()
    
    # 일부 도구에서 의도적으로 에러 발생
    error_scenarios = [
        ('Data Loader', '파일 형식 오류', 'CSV 파일이 손상되었습니다'),
        ('SQL Database', '연결 실패', '데이터베이스 서버에 연결할 수 없습니다'),
        ('H2O Modeling', '메모리 부족', '모델 훈련 중 메모리가 부족합니다')
    ]
    
    selected_error = random.choice(error_scenarios)
    tool_name, action, error_msg = selected_error
    
    # 작업 시작
    monitoring.update_tool_status(tool_name, 'active', action)
    monitoring.add_tool_log(tool_name, f"작업 시작: {action}", "info")
    
    # 진행률 시뮬레이션
    for i in range(5):
        progress = (i + 1) / 10  # 50%까지만 진행
        monitoring.update_tool_status(tool_name, 'active', action, progress)
        time.sleep(0.5)
    
    # 에러 발생
    monitoring.set_tool_error(tool_name, error_msg)

def toggle_tool_connection(tool_name: str):
    """도구 연결 상태 토글"""
    monitoring = get_cursor_mcp_monitoring()
    tool = monitoring._get_tool_by_name(tool_name)
    if tool:
        tool.is_connected = not tool.is_connected
        tool.status = 'idle' if tool.is_connected else 'offline'
        status = "연결됨" if tool.is_connected else "연결 해제됨"
        monitoring.add_tool_log(tool_name, f"연결 상태: {status}", "info")

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="MCP Monitoring Demo",
        page_icon="🔧",
        layout="wide"
    )
    
    initialize_demo()
    
    st.title("🔧 Cursor Style MCP Monitoring Demo")
    st.markdown("CherryAI의 10개 MCP 도구를 실시간으로 모니터링하는 Cursor 스타일 대시보드입니다.")
    
    # 제어 패널
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🎲 랜덤 활동", use_container_width=True):
            if not st.session_state.simulation_running:
                st.session_state.simulation_running = True
                with st.spinner("MCP 도구 활동 시뮬레이션 중..."):
                    simulate_random_activity()
                st.rerun()
    
    with col2:
        if st.button("🔄 데이터 파이프라인", use_container_width=True):
            if not st.session_state.simulation_running:
                st.session_state.simulation_running = True
                with st.spinner("데이터 파이프라인 시뮬레이션 중..."):
                    simulate_data_pipeline()
                st.session_state.simulation_running = False
                st.rerun()
    
    with col3:
        if st.button("❌ 에러 시나리오", use_container_width=True):
            with st.spinner("에러 시나리오 시뮬레이션 중..."):
                simulate_error_scenario()
                time.sleep(3)  # 에러 발생 대기
            st.rerun()
    
    with col4:
        auto_sim = st.checkbox("🔄 자동 시뮬레이션", help="10초마다 자동으로 랜덤 활동 시뮬레이션")
        st.session_state.auto_simulation = auto_sim
    
    with col5:
        if st.button("🧹 초기화", use_container_width=True):
            monitoring = get_cursor_mcp_monitoring()
            monitoring.clear_monitoring()
            st.session_state.simulation_running = False
            st.session_state.auto_simulation = False
            st.rerun()
    
    # MCP 도구 연결 상태 제어
    with st.expander("🔌 연결 상태 제어", expanded=False):
        cols = st.columns(5)
        tool_names = [
            'Data Loader', 'Data Cleaning', 'EDA Tools', 'Data Visualization', 'Feature Engineering',
            'H2O Modeling', 'MLflow Agent', 'SQL Database', 'Data Wrangling', 'Pandas Analyst'
        ]
        
        for i, tool_name in enumerate(tool_names):
            col_idx = i % 5
            with cols[col_idx]:
                if st.button(f"Toggle {tool_name}", key=f"toggle_{i}"):
                    toggle_tool_connection(tool_name)
                    st.rerun()
    
    # 구분선
    st.markdown("---")
    
    # 메인 모니터링 대시보드
    monitoring = get_cursor_mcp_monitoring()
    
    # 모니터링 세션이 시작되지 않았으면 시작
    if not monitoring.is_monitoring:
        monitoring.start_monitoring_session("🔧 MCP Tools Real-time Dashboard")
    else:
        # 기존 세션이 있으면 렌더링 업데이트
        monitoring._render_monitoring_dashboard()
    
    # 자동 시뮬레이션 모드
    if st.session_state.auto_simulation:
        # 10초마다 자동 시뮬레이션 (실제로는 페이지 새로고침 시)
        import time
        time.sleep(1)  # 짧은 지연
        if random.random() < 0.3:  # 30% 확률로 새로운 활동 시작
            simulate_random_activity()
        st.rerun()
    
    # 사이드바에 설명과 통계
    with st.sidebar:
        st.markdown("## 🔧 MCP 모니터링 기능")
        st.markdown("""
        ### ✨ 주요 특징
        - **실시간 상태**: 10개 MCP 도구의 실시간 상태 모니터링
        - **진행률 표시**: 각 도구의 작업 진행 상황 시각화
        - **성능 메트릭**: 요청 수, 성공률, 평균 응답시간
        - **실행 로그**: 도구별 상세 실행 로그
        - **연결 상태**: 각 도구의 서버 연결 상태 표시
        
        ### 🎮 사용 방법
        1. **랜덤 활동**: 3-5개 도구가 무작위로 작업 수행
        2. **데이터 파이프라인**: 순차적 데이터 처리 워크플로우
        3. **에러 시나리오**: 의도적 에러 발생으로 에러 처리 테스트
        4. **자동 시뮬레이션**: 지속적인 활동 시뮬레이션
        5. **연결 상태 제어**: 개별 도구 연결/해제
        
        ### 🔧 MCP 도구 목록
        - 📁 **Data Loader**: 파일 업로드 및 데이터 로드
        - 🧹 **Data Cleaning**: 데이터 정제 및 전처리
        - 🔍 **EDA Tools**: 탐색적 데이터 분석
        - 📊 **Data Visualization**: 차트 및 그래프 생성
        - ⚙️ **Feature Engineering**: 특성 생성 및 변환
        - 🤖 **H2O Modeling**: AutoML 모델 생성
        - 📈 **MLflow Agent**: 실험 추적 및 모델 관리
        - 🗄️ **SQL Database**: SQL 쿼리 및 데이터베이스
        - 🔧 **Data Wrangling**: 데이터 변환 및 조작
        - 🐼 **Pandas Analyst**: Pandas 기반 데이터 분석
        """)
        
        # 현재 상태 통계
        if monitoring.tools:
            st.markdown("### 📊 실시간 통계")
            
            total_tools = len(monitoring.tools)
            active_tools = len([t for t in monitoring.tools.values() if t.status == 'active'])
            completed_tools = len([t for t in monitoring.tools.values() if t.status == 'completed'])
            failed_tools = len([t for t in monitoring.tools.values() if t.status == 'failed'])
            offline_tools = len([t for t in monitoring.tools.values() if t.status == 'offline'])
            
            st.metric("전체 도구", total_tools)
            
            # 상태별 도구 수
            col1, col2 = st.columns(2)
            with col1:
                st.metric("활성", active_tools)
                st.metric("완료", completed_tools)
            with col2:
                st.metric("실패", failed_tools)
                st.metric("오프라인", offline_tools)
            
            # 전체 요청 수와 성공률
            total_requests = sum(t.total_requests for t in monitoring.tools.values())
            successful_requests = sum(t.successful_requests for t in monitoring.tools.values())
            
            if total_requests > 0:
                overall_success_rate = (successful_requests / total_requests) * 100
                st.metric("전체 성공률", f"{overall_success_rate:.1f}%")
                st.metric("총 요청", total_requests)
            
            # 평균 응답 시간
            response_times = [t.avg_response_time for t in monitoring.tools.values() if t.avg_response_time > 0]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                st.metric("평균 응답시간", f"{avg_response_time:.1f}s")
        
        # 내보내기 기능
        st.markdown("---")
        if st.button("📤 모니터링 데이터 내보내기", use_container_width=True):
            export_data = monitoring.export_monitoring_data()
            st.download_button(
                label="JSON 다운로드",
                data=str(export_data),
                file_name=f"mcp_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main() 