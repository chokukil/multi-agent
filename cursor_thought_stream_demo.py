"""
🧠 Cursor Style Thought Stream Demo - LLM 사고 과정 스트리밍 데모

새로 구현한 Cursor 스타일 LLM 사고 과정 스트리밍을 테스트하고 시연하는 데모
실시간 사고 과정과 타이핑 효과를 시뮬레이션합니다.

실행 방법:
streamlit run cursor_thought_stream_demo.py --server.port 8502
"""

import streamlit as st
import time
import asyncio
import threading
from datetime import datetime

# Cursor 스타일 사고 스트림 import
from ui.cursor_thought_stream import get_cursor_thought_stream, ThoughtStreamDemo


def initialize_demo():
    """데모 초기화"""
    if 'thought_demo_initialized' not in st.session_state:
        st.session_state.thought_demo_initialized = True
        st.session_state.current_scenario = None
        st.session_state.demo_running = False
        st.session_state.cursor_thought_stream = {'is_active': False, 'thoughts': []}

def simulate_real_time_thinking():
    """실시간 사고 과정 시뮬레이션"""
    thought_stream = get_cursor_thought_stream()
    
    # 사고 세션 시작
    thought_stream.start_thinking_session("🧠 실시간 AI 사고 과정")
    
    # 단계별 사고 과정
    scenarios = [
        {
            'text': '사용자 질문을 분석하고 있습니다',
            'category': 'analysis',
            'typing_speed': 0.08,
            'processing_time': 2.0,
            'details': ['자연어 처리 완료', '의도 파악 성공', '컨텍스트 추출 완료']
        },
        {
            'text': '최적의 분석 방법을 선택하고 있습니다',
            'category': 'planning',
            'typing_speed': 0.06,
            'processing_time': 1.5,
            'details': ['데이터 타입 확인', '분석 알고리즘 선택', '워크플로우 계획']
        },
        {
            'text': '데이터 전처리를 수행하고 있습니다',
            'category': 'execution',
            'typing_speed': 0.05,
            'processing_time': 3.0,
            'details': ['결측치 처리', '이상치 탐지', '데이터 정규화']
        },
        {
            'text': '통계 분석을 실행하고 있습니다',
            'category': 'execution',
            'typing_speed': 0.07,
            'processing_time': 2.5,
            'details': ['기초 통계량 계산', '상관관계 분석', '분포 분석']
        },
        {
            'text': '결과를 종합하고 인사이트를 도출하고 있습니다',
            'category': 'synthesis',
            'typing_speed': 0.04,
            'processing_time': 1.8,
            'details': ['패턴 인식', '결론 도출', '실행 가능한 제안 생성']
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        # 사고 추가
        thought_id = thought_stream.add_thought("", scenario['category'])
        
        # 타이핑 효과로 텍스트 스트리밍
        for j in range(len(scenario['text']) + 1):
            partial_text = scenario['text'][:j]
            if j < len(scenario['text']):
                partial_text += "⚡"  # 타이핑 커서
            
            # 사고 업데이트
            thought = thought_stream._get_thought_by_id(thought_id)
            if thought:
                thought.text = partial_text
                thought_stream._render_thought_stream()
            
            time.sleep(scenario['typing_speed'])
        
        # 최종 텍스트 설정
        thought = thought_stream._get_thought_by_id(thought_id)
        if thought:
            thought.text = scenario['text']
            thought_stream._render_thought_stream()
        
        # 처리 상태로 변경
        thought_stream.update_thought_status(thought_id, 'processing')
        time.sleep(scenario['processing_time'])
        
        # 완료 처리
        thought_stream.complete_thought(
            thought_id, 
            result=f"단계 {i+1} 완료"
        )
        
        # 세부사항 추가
        for detail in scenario['details']:
            thought_stream.update_thought_status(thought_id, 'completed', [detail])
            time.sleep(0.3)
    
    # 세션 종료
    thought_stream.end_thinking_session("🎉 모든 분석이 성공적으로 완료되었습니다!")

def run_preset_scenario(scenario_name: str):
    """사전 정의된 시나리오 실행"""
    demo = ThoughtStreamDemo()
    demo.run_demo_scenario(scenario_name)

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor Thought Stream Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    initialize_demo()
    
    st.title("🧠 Cursor Style Thought Stream Demo")
    st.markdown("Cursor의 실시간 LLM 사고 과정 스트리밍을 CherryAI에 적용한 데모입니다.")
    
    # 제어 패널
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 실시간 사고 시작", use_container_width=True):
            if not st.session_state.demo_running:
                st.session_state.demo_running = True
                with st.spinner("사고 과정을 시뮬레이션하고 있습니다..."):
                    simulate_real_time_thinking()
                st.session_state.demo_running = False
                st.rerun()
    
    with col2:
        scenario_options = ["데이터 분석 워크플로우", "복잡한 ML 파이프라인"]
        selected_scenario = st.selectbox(
            "시나리오 선택", 
            ["선택하세요"] + scenario_options,
            key="scenario_selector"
        )
        
        if st.button("📋 시나리오 실행", use_container_width=True):
            if selected_scenario != "선택하세요":
                st.session_state.current_scenario = selected_scenario
                with st.spinner(f"{selected_scenario} 시뮬레이션 중..."):
                    run_preset_scenario(selected_scenario)
                st.rerun()
    
    with col3:
        if st.button("💭 단일 사고 테스트", use_container_width=True):
            thought_stream = get_cursor_thought_stream()
            
            # 단일 사고 테스트
            thought_stream.start_thinking_session("🧠 단일 사고 테스트")
            
            test_thoughts = [
                ("간단한 분석을 수행합니다", "analysis"),
                ("결과를 검토하고 있습니다", "execution"),
                ("최종 결론을 도출합니다", "synthesis")
            ]
            
            for text, category in test_thoughts:
                thought_id = thought_stream.add_thought(text, category)
                time.sleep(1.0)
                thought_stream.update_thought_status(thought_id, 'processing')
                time.sleep(0.8)
                thought_stream.complete_thought(thought_id)
            
            thought_stream.end_thinking_session("테스트 완료!")
            st.rerun()
    
    with col4:
        if st.button("🧹 초기화", use_container_width=True):
            thought_stream = get_cursor_thought_stream()
            thought_stream.clear_thoughts()
            st.session_state.demo_running = False
            st.session_state.current_scenario = None
            st.rerun()
    
    # 구분선
    st.markdown("---")
    
    # 메인 컨테이너
    main_container = st.container()
    
    # Cursor 스타일 사고 스트림 렌더링
    with main_container:
        thought_stream = get_cursor_thought_stream()
        
        # 아직 사고가 없으면 안내 메시지
        if not thought_stream.thoughts:
            st.info("🧠 위의 버튼을 클릭하여 AI 사고 과정 시뮬레이션을 시작해보세요!")
        else:
            # 기존 사고 스트림이 있으면 렌더링
            thought_stream.stream_placeholder = st.empty()
            thought_stream._apply_thought_styles()
            thought_stream._render_thought_stream()
    
    # 사이드바에 설명과 통계
    with st.sidebar:
        st.markdown("## 🧠 사고 스트림 기능")
        st.markdown("""
        ### ✨ 주요 특징
        - **실시간 타이핑**: 사고 과정을 실시간으로 타이핑 효과로 표시
        - **상태 변화**: ⏳ 사고중 → 🔄 처리중 → ✅ 완료
        - **카테고리별 분류**: 🔍 분석, 📋 계획, ⚙️ 실행, 🎯 종합
        - **경과 시간**: 각 사고의 실시간 타이머 표시
        - **세부 로그**: 완료된 사고의 상세 정보
        
        ### 🎮 사용 방법
        1. **실시간 사고 시작**: 타이핑 효과가 있는 실시간 시뮬레이션
        2. **시나리오 실행**: 사전 정의된 분석 워크플로우
        3. **단일 사고 테스트**: 간단한 3단계 사고 과정
        4. **초기화**: 모든 사고 내용 지우기
        
        ### 🔧 기술 구현
        - **ThoughtBubble**: 개별 사고 데이터 구조
        - **실시간 스트리밍**: 상태 기반 UI 업데이트
        - **타이핑 애니메이션**: 문자별 순차 표시
        - **상태 머신**: thinking → processing → completed
        """)
        
        # 현재 상태 통계
        thought_stream = get_cursor_thought_stream()
        if thought_stream.thoughts:
            st.markdown("### 📊 현재 상태")
            
            total_thoughts = len(thought_stream.thoughts)
            completed = len([t for t in thought_stream.thoughts if t.status == 'completed'])
            processing = len([t for t in thought_stream.thoughts if t.status == 'processing'])
            thinking = len([t for t in thought_stream.thoughts if t.status == 'thinking'])
            failed = len([t for t in thought_stream.thoughts if t.status == 'failed'])
            
            st.metric("전체 사고", total_thoughts)
            st.metric("완료", completed)
            
            if processing > 0:
                st.metric("처리 중", processing)
            if thinking > 0:
                st.metric("사고 중", thinking)
            if failed > 0:
                st.metric("실패", failed)
            
            # 평균 처리 시간
            completed_thoughts = [t for t in thought_stream.thoughts if t.status == 'completed']
            if completed_thoughts:
                avg_time = sum(t.elapsed_time for t in completed_thoughts) / len(completed_thoughts)
                st.metric("평균 처리 시간", f"{avg_time:.1f}s")
            
            # 카테고리 분포
            categories = {}
            for thought in thought_stream.thoughts:
                categories[thought.category] = categories.get(thought.category, 0) + 1
            
            if categories:
                st.markdown("### 📈 카테고리 분포")
                for category, count in categories.items():
                    category_emoji = {
                        'analysis': '🔍',
                        'planning': '📋',
                        'execution': '⚙️',
                        'synthesis': '🎯',
                        'general': '💭'
                    }.get(category, '💭')
                    st.write(f"{category_emoji} {category}: {count}")
        
        # 내보내기 기능
        if thought_stream.thoughts:
            st.markdown("---")
            if st.button("📤 사고 과정 내보내기", use_container_width=True):
                export_data = thought_stream.export_thoughts()
                st.download_button(
                    label="JSON 다운로드",
                    data=str(export_data),
                    file_name=f"thought_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 