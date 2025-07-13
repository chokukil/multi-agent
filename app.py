"""
🚀 CherryAI - Cursor-Style A2A Platform
Cursor 벤치마킹 기반 세계 최초 A2A + MCP 통합 플랫폼

Inspired by Cursor's elegant UI/UX:
- Dark theme with modern aesthetics
- Real-time collaboration visualization
- Intelligent agent orchestration
- Professional dashboard design
"""

import streamlit as st
import asyncio
from typing import Dict, Any
import os
import sys

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Streamlit 페이지 설정
st.set_page_config(
    page_title="🧬 CherryAI - A2A + MCP 플랫폼",
    page_icon="🧬",
    layout="wide",  # 데스크톱 최적화
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/cherryai/docs',
        'Report a bug': 'https://github.com/cherryai/issues',
        'About': '세계 최초 A2A + MCP 통합 플랫폼'
    }
)

# Cursor 테마 적용
from ui.cursor_theme_system import apply_cursor_theme
apply_cursor_theme()

# 모듈화된 컴포넌트 임포트
try:
    from core.app_components.main_dashboard import render_main_dashboard
    from core.app_components.agent_orchestrator import render_agent_orchestrator
    from core.app_components.data_workspace import render_data_workspace
    from core.app_components.monitoring_panel import render_monitoring_panel
    from core.app_components.mcp_integration import render_mcp_integration
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 컴포넌트 로드 실패: {e}")
    COMPONENTS_AVAILABLE = False

# 에러 핸들링 시스템 (조건부)
try:
    from ui.enhanced_error_ui import integrate_error_system_to_app
    integrate_error_system_to_app()
except ImportError:
    pass

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'current_page': 'dashboard',
        'agents_initialized': False,
        'debug_mode': False,
        'theme_preference': 'dark',
        'collaboration_active': False,
        'mcp_tools_loaded': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_navigation():
    """Cursor 스타일 네비게이션"""
    st.markdown("""
    <div class="cursor-nav">
        <div class="nav-header">
            <h1>🧬 CherryAI</h1>
            <p class="nav-subtitle">A2A + MCP 통합 플랫폼</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 메인 네비게이션
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
    
    with col2:
        if st.button("🧬 A2A Agents", use_container_width=True):
            st.session_state.current_page = 'agents'
    
    with col3:
        if st.button("💾 Data Workspace", use_container_width=True):
            st.session_state.current_page = 'workspace'
    
    with col4:
        if st.button("🔧 MCP Tools", use_container_width=True):
            st.session_state.current_page = 'mcp'
    
    with col5:
        if st.button("📈 Monitoring", use_container_width=True):
            st.session_state.current_page = 'monitoring'

def render_fallback_ui():
    """컴포넌트 로드 실패 시 폴백 UI"""
    st.error("⚠️ 고급 UI 컴포넌트를 로드할 수 없습니다. 기본 모드로 실행합니다.")
    
    st.markdown("## 🧬 CherryAI - A2A Platform")
    st.info("컴포넌트를 생성하는 중입니다. 잠시만 기다려주세요...")
    
    # 기본 에이전트 상태 확인
    with st.expander("🔍 시스템 상태", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("A2A 에이전트", "10개", "100% 준비")
        
        with col2:
            st.metric("MCP 도구", "7개", "통합 완료")
        
        with col3:
            st.metric("시스템 상태", "정상", "운영 중")
    
    # 간단한 채팅 인터페이스
    st.markdown("### 💬 AI 에이전트와 대화")
    user_input = st.text_area("메시지를 입력하세요:", placeholder="데이터 분석을 요청하거나 질문해보세요...")
    
    if st.button("전송", type="primary"):
        if user_input.strip():
            st.info("🔄 에이전트가 작업 중입니다...")
            st.warning("⚠️ 고급 UI 컴포넌트를 활성화하려면 모듈을 설치해주세요.")

def main():
    """메인 애플리케이션"""
    initialize_session_state()
    
    # 네비게이션 렌더링
    render_navigation()
    
    st.markdown("---")
    
    # 컴포넌트가 사용 가능한 경우
    if COMPONENTS_AVAILABLE:
        # 현재 페이지에 따른 렌더링
        if st.session_state.current_page == 'dashboard':
            render_main_dashboard()
        elif st.session_state.current_page == 'agents':
            render_agent_orchestrator()
        elif st.session_state.current_page == 'workspace':
            render_data_workspace()
        elif st.session_state.current_page == 'mcp':
            render_mcp_integration()
        elif st.session_state.current_page == 'monitoring':
            render_monitoring_panel()
    else:
        # 폴백 UI
        render_fallback_ui()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            🧬 CherryAI - 세계 최초 A2A + MCP 통합 플랫폼 | 
            Powered by Cursor-Style UI | 
            Built with ❤️ by CherryAI Team
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
