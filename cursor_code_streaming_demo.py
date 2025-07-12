"""
Cursor-Style Code Streaming UI Demo
A2A SDK 0.2.9 + SSE 기반 실시간 코드 스트리밍 데모
"""

import streamlit as st
import time
import random
import threading
from datetime import datetime

# 로컬 모듈 임포트
from ui.cursor_code_streaming import (
    get_cursor_code_streaming,
    CodeStreamingStatus,
    CodeBlockType
)

def initialize_demo():
    """데모 초기화"""
    if 'code_streaming_demo_initialized' not in st.session_state:
        st.session_state.code_streaming_demo_initialized = True
        st.session_state.current_request = None
        st.session_state.streaming_active = False
        st.session_state.cursor_code_streaming = {
            'current_plan': None,
            'streaming_active': False,
            'events': []
        }

def simulate_advanced_scenario():
    """고급 시나리오 시뮬레이션"""
    code_streaming = get_cursor_code_streaming()
    
    # 복잡한 ML 파이프라인 시나리오
    scenarios = [
        {
            'request': '머신러닝 모델 훈련 파이프라인 구현',
            'description': '데이터 로드부터 모델 훈련, 평가까지 전체 ML 파이프라인을 구현합니다'
        },
        {
            'request': '실시간 데이터 분석 대시보드 생성',
            'description': 'Streamlit을 활용한 실시간 데이터 분석 대시보드를 생성합니다'
        },
        {
            'request': '자동화된 데이터 전처리 시스템',
            'description': '다양한 데이터 타입을 자동으로 전처리하는 시스템을 구현합니다'
        }
    ]
    
    scenario = random.choice(scenarios)
    st.session_state.current_request = scenario['request']
    
    # 스트리밍 시작
    code_streaming.start_code_streaming(scenario['request'])
    
    return scenario

def simulate_a2a_integration():
    """A2A SDK 통합 시뮬레이션"""
    code_streaming = get_cursor_code_streaming()
    
    # A2A 기반 에이전트 협업 시나리오
    a2a_scenarios = [
        {
            'request': 'A2A 에이전트 간 협업 코드 생성',
            'description': '여러 A2A 에이전트가 협업하여 복잡한 분석을 수행하는 코드를 생성합니다'
        },
        {
            'request': 'SSE 기반 실시간 알림 시스템',
            'description': 'Server-Sent Events를 활용한 실시간 알림 시스템을 구현합니다'
        },
        {
            'request': 'TaskUpdater 기반 진행률 추적',
            'description': 'A2A TaskUpdater를 활용하여 실시간 진행률을 추적하는 시스템을 구현합니다'
        }
    ]
    
    scenario = random.choice(a2a_scenarios)
    st.session_state.current_request = scenario['request']
    
    # 스트리밍 시작
    code_streaming.start_code_streaming(scenario['request'])
    
    return scenario

def simulate_cursor_todo_style():
    """Cursor 스타일 할일 목록 시뮬레이션"""
    code_streaming = get_cursor_code_streaming()
    
    # Cursor 스타일 프로젝트 시나리오
    cursor_scenarios = [
        {
            'request': '프로젝트 구조 자동 생성',
            'description': 'Cursor 스타일의 체계적인 프로젝트 구조를 자동으로 생성합니다'
        },
        {
            'request': '코드 리팩토링 자동화',
            'description': '기존 코드를 분석하고 개선된 구조로 리팩토링합니다'
        },
        {
            'request': '테스트 코드 자동 생성',
            'description': '메인 코드에 대응하는 포괄적인 테스트 코드를 자동으로 생성합니다'
        }
    ]
    
    scenario = random.choice(cursor_scenarios)
    st.session_state.current_request = scenario['request']
    
    # 스트리밍 시작
    code_streaming.start_code_streaming(scenario['request'])
    
    return scenario

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor Code Streaming Demo",
        page_icon="⚡",
        layout="wide"
    )
    
    initialize_demo()
    
    st.title("⚡ Cursor Style Code Streaming Demo")
    st.markdown("A2A SDK 0.2.9 + SSE 기반 실시간 코드 스트리밍을 CherryAI에 적용한 데모입니다.")
    
    # 제어 패널
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 기본 코드 생성", use_container_width=True):
            if not st.session_state.streaming_active:
                st.session_state.streaming_active = True
                st.session_state.current_request = "데이터 분석 코드 생성"
                
                code_streaming = get_cursor_code_streaming()
                code_streaming.start_code_streaming("데이터 분석 코드 생성")
                
                st.rerun()
    
    with col2:
        if st.button("🔬 고급 시나리오", use_container_width=True):
            if not st.session_state.streaming_active:
                st.session_state.streaming_active = True
                scenario = simulate_advanced_scenario()
                st.success(f"시나리오 시작: {scenario['request']}")
                st.rerun()
    
    with col3:
        if st.button("🤖 A2A 통합", use_container_width=True):
            if not st.session_state.streaming_active:
                st.session_state.streaming_active = True
                scenario = simulate_a2a_integration()
                st.success(f"A2A 시나리오 시작: {scenario['request']}")
                st.rerun()
    
    with col4:
        if st.button("📋 Cursor Todo", use_container_width=True):
            if not st.session_state.streaming_active:
                st.session_state.streaming_active = True
                scenario = simulate_cursor_todo_style()
                st.success(f"Cursor 스타일 시작: {scenario['request']}")
                st.rerun()
    
    # 제어 버튼
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("⏸️ 스트리밍 중지", use_container_width=True):
            code_streaming = get_cursor_code_streaming()
            code_streaming.stop_streaming()
            st.session_state.streaming_active = False
            st.warning("스트리밍이 중지되었습니다.")
            st.rerun()
    
    with col6:
        if st.button("🔄 다시 시작", use_container_width=True):
            if st.session_state.current_request:
                code_streaming = get_cursor_code_streaming()
                code_streaming.clear_plan()
                code_streaming.start_code_streaming(st.session_state.current_request)
                st.session_state.streaming_active = True
                st.rerun()
    
    with col7:
        if st.button("🧹 초기화", use_container_width=True):
            code_streaming = get_cursor_code_streaming()
            code_streaming.clear_plan()
            st.session_state.streaming_active = False
            st.session_state.current_request = None
            st.rerun()
    
    with col8:
        auto_demo = st.checkbox("🔄 자동 데모", help="10초마다 자동으로 새로운 시나리오 실행")
    
    # 구분선
    st.markdown("---")
    
    # 메인 컨테이너
    main_container = st.container()
    
    # Cursor 스타일 코드 스트리밍 렌더링
    with main_container:
        code_streaming = get_cursor_code_streaming()
        code_streaming.initialize_container()
        
        # 현재 계획이 있으면 렌더링
        if st.session_state.cursor_code_streaming.get('current_plan'):
            code_streaming.render_code_plan()
        else:
            st.info("⚡ 위의 버튼을 클릭하여 실시간 코드 스트리밍을 시작해보세요!")
    
    # 자동 데모 모드
    if auto_demo and not st.session_state.streaming_active:
        time.sleep(10)
        # 랜덤 시나리오 실행
        scenario_functions = [
            simulate_advanced_scenario,
            simulate_a2a_integration,
            simulate_cursor_todo_style
        ]
        
        chosen_function = random.choice(scenario_functions)
        st.session_state.streaming_active = True
        scenario = chosen_function()
        st.rerun()
    
    # 사이드바에 설명과 통계
    with st.sidebar:
        st.markdown("## ⚡ 코드 스트리밍 기능")
        st.markdown("""
        ### ✨ 주요 특징
        - **A2A SDK 0.2.9 통합**: AgentExecutor, TaskUpdater 활용
        - **SSE 실시간 스트리밍**: Server-Sent Events 기반 실시간 업데이트
        - **Cursor 스타일 Todo**: 체계적인 진행률 추적 및 표시
        - **타이핑 효과**: 실시간 코드 생성 타이핑 애니메이션
        - **블록 단위 생성**: 함수, 클래스, 실행 코드 블록별 생성
        - **실행 라인 하이라이트**: 현재 실행 중인 라인 강조 표시
        
        ### 🎮 사용 방법
        1. **기본 코드 생성**: 단순한 데이터 분석 코드 생성
        2. **고급 시나리오**: 복잡한 ML 파이프라인 시뮬레이션
        3. **A2A 통합**: A2A SDK 기반 협업 코드 생성
        4. **Cursor Todo**: Cursor 스타일 프로젝트 구조 생성
        5. **자동 데모**: 10초마다 자동으로 다른 시나리오 실행
        
        ### 🔧 기술 구현
        - **A2A AgentExecutor**: 코드 생성 에이전트 실행
        - **TaskUpdater**: 실시간 진행률 업데이트
        - **SSE EventQueue**: 실시간 이벤트 스트리밍
        - **CodePlan**: Cursor 스타일 코드 계획 구조
        - **CodeBlock**: 개별 코드 블록 관리
        - **실시간 타이핑**: 문자별 순차 표시 효과
        """)
        
        # 현재 상태 표시
        if st.session_state.streaming_active:
            st.markdown("### 📊 현재 상태")
            st.metric("스트리밍 상태", "🔄 활성")
            
            if st.session_state.current_request:
                st.metric("현재 작업", st.session_state.current_request)
            
            # 계획 상태
            current_plan = st.session_state.cursor_code_streaming.get('current_plan')
            if current_plan:
                st.metric("계획 상태", current_plan.get('status', 'unknown'))
                
                # 블록 진행률
                blocks = current_plan.get('blocks', [])
                if blocks:
                    completed_blocks = len([b for b in blocks if b.get('status') == 'completed'])
                    total_blocks = len(blocks)
                    st.metric("블록 진행률", f"{completed_blocks}/{total_blocks}")
                    
                    # 각 블록 상태
                    st.markdown("#### 📋 블록 상태")
                    for i, block in enumerate(blocks):
                        status = block.get('status', 'pending')
                        title = block.get('title', f'블록 {i+1}')
                        
                        status_emoji = {
                            'pending': '⏳',
                            'generating': '🔄',
                            'completed': '✅',
                            'failed': '❌'
                        }.get(status, '⏳')
                        
                        st.write(f"{status_emoji} {title}")
        else:
            st.markdown("### 📊 현재 상태")
            st.metric("스트리밍 상태", "⏸️ 대기")
        
        # 성능 메트릭
        st.markdown("---")
        st.markdown("### 📈 성능 메트릭")
        
        # 랜덤 성능 데이터 (실제로는 실제 메트릭 수집)
        if st.session_state.streaming_active:
            typing_speed = random.uniform(15, 25)
            response_time = random.uniform(0.1, 0.3)
            success_rate = random.uniform(95, 100)
            
            st.metric("타이핑 속도", f"{typing_speed:.1f} chars/s")
            st.metric("응답 시간", f"{response_time:.2f}s")
            st.metric("성공률", f"{success_rate:.1f}%")
        
        # A2A SDK 정보
        st.markdown("---")
        st.markdown("### 🤖 A2A SDK 정보")
        st.markdown("""
        - **버전**: 0.2.9 (최신)
        - **프로토콜**: JSONRPC over HTTP
        - **실시간 통신**: SSE (Server-Sent Events)
        - **태스크 관리**: TaskUpdater + EventQueue
        - **에이전트 실행**: AgentExecutor 패턴
        - **메시지 형식**: TextPart, DataPart, FilePart
        """)
        
        # 내보내기 기능
        if st.session_state.cursor_code_streaming.get('current_plan'):
            st.markdown("---")
            if st.button("📤 코드 계획 내보내기", use_container_width=True):
                plan_data = st.session_state.cursor_code_streaming['current_plan']
                st.download_button(
                    label="JSON 다운로드",
                    data=str(plan_data),
                    file_name=f"code_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 