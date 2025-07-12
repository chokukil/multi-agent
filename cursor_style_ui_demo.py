"""
🎨 Cursor Style UI Demo - Cursor 스타일 에이전트 카드 데모

새로 구현한 Cursor 스타일 에이전트 카드를 테스트하고 시연하는 데모 애플리케이션
실시간 에이전트 상태 변화와 진행 상황을 시뮬레이션합니다.

실행 방법:
streamlit run cursor_style_ui_demo.py
"""

import streamlit as st
import time
import random
import threading
from datetime import datetime
import asyncio

# Cursor 스타일 카드 import
from ui.cursor_style_agent_cards import get_cursor_agent_cards, AgentCard, AgentStep

def initialize_demo():
    """데모 초기화"""
    if 'demo_initialized' not in st.session_state:
        st.session_state.demo_initialized = True
        st.session_state.simulation_running = False
        st.session_state.agent_count = 0
        st.session_state.cursor_agent_cards = {}
        st.session_state.current_agent_ids = []

def create_demo_agents():
    """데모용 에이전트들 생성"""
    cards_manager = get_cursor_agent_cards()
    
    # 샘플 에이전트들
    agents = [
        {
            'name': 'Pandas Agent',
            'icon': '🐼',
            'task': '데이터 분석 및 정제 수행 중...',
            'steps': [
                ('📊', '데이터 로드', '파일에서 데이터를 읽어옵니다'),
                ('🧹', '데이터 정제', '결측치와 이상치를 처리합니다'),
                ('📈', '통계 분석', '기초 통계량을 계산합니다'),
                ('🔍', '패턴 탐지', '데이터 패턴을 찾습니다'),
                ('📋', '결과 요약', '분석 결과를 정리합니다')
            ]
        },
        {
            'name': 'Visualization Agent', 
            'icon': '📊',
            'task': '차트 및 그래프 생성 중...',
            'steps': [
                ('🎯', '차트 선택', '적절한 차트 유형을 선택합니다'),
                ('🎨', '스타일 적용', '색상과 레이아웃을 설정합니다'), 
                ('📊', '차트 생성', 'Plotly 차트를 생성합니다'),
                ('✨', '인터랙션 추가', '상호작용 기능을 추가합니다')
            ]
        },
        {
            'name': 'Knowledge Agent',
            'icon': '🧠', 
            'task': '지식 베이스 학습 및 검색 중...',
            'steps': [
                ('🔍', '패턴 분석', '데이터에서 패턴을 분석합니다'),
                ('💡', '인사이트 추출', '의미있는 인사이트를 추출합니다'),
                ('📚', '지식 저장', '학습된 내용을 저장합니다')
            ]
        }
    ]
    
    agent_ids = []
    for agent in agents:
        # 에이전트 카드 생성
        agent_id = cards_manager.create_agent_card(
            agent_name=agent['name'],
            agent_icon=agent['icon'], 
            current_task=agent['task']
        )
        
        # 단계들 추가
        card = cards_manager.get_card(agent_id)
        if card:
            for icon, name, desc in agent['steps']:
                card.add_step(icon, name, desc)
        
        agent_ids.append(agent_id)
    
    return agent_ids

def simulate_agent_progress(agent_ids):
    """에이전트 진행 상황 시뮬레이션"""
    cards_manager = get_cursor_agent_cards()
    
    for agent_id in agent_ids:
        card = cards_manager.get_card(agent_id)
        if not card or not card.steps:
            continue
        
        # 랜덤하게 일부 에이전트만 진행시킴
        if random.random() < 0.7:  # 70% 확률로 진행
            current_step_index = 0
            for step in card.steps:
                if step.status == 'completed':
                    current_step_index += 1
                elif step.status == 'running':
                    # 실행 중인 단계가 있으면 완료시킴
                    card.complete_step(step.step_id, f"단계 완료: {step.name}")
                    break
                elif step.status == 'pending':
                    # 대기 중인 첫 번째 단계 시작
                    card.start_step(step.step_id)
                    card.add_step_detail(step.step_id, "단계 시작됨")
                    break
        
        # 에이전트 상태 업데이트
        if card.progress >= 1.0:
            cards_manager.update_card_status(agent_id, 'completed', '모든 작업 완료!')
        elif card.progress > 0:
            cards_manager.update_card_status(agent_id, 'working', card.current_task)

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor Style UI Demo",
        page_icon="🎨", 
        layout="wide"
    )
    
    initialize_demo()
    
    st.title("🎨 Cursor Style Agent Cards Demo")
    st.markdown("Cursor의 우아한 접힌/펼친 카드 UI를 CherryAI에 적용한 데모입니다.")
    
    # 제어 패널
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 에이전트 시작", use_container_width=True):
            if st.session_state.agent_count == 0:
                agent_ids = create_demo_agents()
                st.session_state.current_agent_ids = agent_ids
                st.session_state.agent_count = len(agent_ids)
                st.rerun()
    
    with col2:
        if st.button("▶️ 진행 시뮬레이션", use_container_width=True):
            if hasattr(st.session_state, 'current_agent_ids'):
                simulate_agent_progress(st.session_state.current_agent_ids)
                st.rerun()
    
    with col3:
        auto_progress = st.checkbox("🔄 자동 진행", help="5초마다 자동으로 진행 상황 업데이트")
    
    with col4:
        if st.button("🧹 초기화", use_container_width=True):
            # 모든 카드 제거
            cards_manager = get_cursor_agent_cards()
            if hasattr(st.session_state, 'current_agent_ids'):
                for agent_id in st.session_state.current_agent_ids:
                    cards_manager.remove_card(agent_id)
            st.session_state.agent_count = 0
            st.session_state.current_agent_ids = []
            st.rerun()
    
    # 구분선
    st.markdown("---")
    
    # Cursor 스타일 에이전트 카드 렌더링
    cards_manager = get_cursor_agent_cards()
    cards_manager.render_cards_container()
    
    # 자동 진행 모드
    if auto_progress and st.session_state.agent_count > 0:
        # 5초마다 자동으로 새로고침
        time.sleep(5)
        if hasattr(st.session_state, 'current_agent_ids'):
            simulate_agent_progress(st.session_state.current_agent_ids)
        st.rerun()
    
    # 사이드바에 설명
    with st.sidebar:
        st.markdown("## 🎯 데모 기능")
        st.markdown("""
        ### ✨ 주요 특징
        - **접힌/펼친 카드**: 클릭하여 세부사항 보기/숨기기
        - **실시간 진행률**: 각 단계별 진행 상황 표시
        - **상태 애니메이션**: Thinking, Working, Completed 상태 시각화
        - **경과 시간**: 실시간 타이머 표시
        - **단계별 로그**: 각 단계의 세부 실행 로그
        
        ### 🎮 사용 방법
        1. **에이전트 시작** 버튼으로 샘플 에이전트 생성
        2. **진행 시뮬레이션** 버튼으로 수동 진행
        3. **자동 진행** 체크박스로 자동 업데이트
        4. 각 카드의 **▼/▲** 버튼으로 펼치기/접기
        
        ### 🔧 기술 스택
        - **Frontend**: Streamlit + Custom CSS
        - **State Management**: Session State
        - **Animations**: CSS3 Animations
        - **Icons**: Unicode Emojis
        """)
        
        # 현재 상태 표시
        if st.session_state.agent_count > 0:
            st.markdown("### 📊 현재 상태")
            st.metric("활성 에이전트", st.session_state.agent_count)
            
            # 전체 진행률 계산
            cards_manager = get_cursor_agent_cards()
            if hasattr(st.session_state, 'current_agent_ids'):
                total_progress = 0
                for agent_id in st.session_state.current_agent_ids:
                    card = cards_manager.get_card(agent_id)
                    if card:
                        total_progress += card.progress
                
                avg_progress = total_progress / len(st.session_state.current_agent_ids)
                st.metric("전체 진행률", f"{avg_progress*100:.1f}%")

if __name__ == "__main__":
    main() 