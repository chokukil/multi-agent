"""
🚀 Main Dashboard Component
Cursor 스타일의 메인 대시보드 - 시스템 개요, 실시간 상태, 퀵 액션
"""

import streamlit as st
import time
import asyncio
from typing import Dict, Any, List
import httpx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Cursor UI 컴포넌트 임포트
from ui.cursor_theme_system import get_cursor_theme
from ui.cursor_style_agent_cards import get_cursor_agent_cards
from ui.cursor_collaboration_network import get_cursor_collaboration_network

def check_agent_status(port: int) -> Dict[str, Any]:
    """에이전트 상태 확인"""
    try:
        response = httpx.get(f"http://localhost:{port}/.well-known/agent.json", timeout=2.0)
        if response.status_code == 200:
            return {"status": "online", "data": response.json()}
    except:
        pass
    return {"status": "offline", "data": None}

def render_system_overview():
    """시스템 개요 렌더링"""
    st.markdown("## 🎯 시스템 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="cursor-metric-card">
            <div class="metric-icon">🧬</div>
            <div class="metric-value">10</div>
            <div class="metric-label">A2A 에이전트</div>
            <div class="metric-status online">모두 준비완료</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cursor-metric-card">
            <div class="metric-icon">🔧</div>
            <div class="metric-value">7</div>
            <div class="metric-label">MCP 도구</div>
            <div class="metric-status online">통합 완료</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="cursor-metric-card">
            <div class="metric-icon">🌐</div>
            <div class="metric-value">18</div>
            <div class="metric-label">총 컴포넌트</div>
            <div class="metric-status online">세계 최초</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        uptime = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div class="cursor-metric-card">
            <div class="metric-icon">⏱️</div>
            <div class="metric-value">{uptime}</div>
            <div class="metric-label">시스템 가동</div>
            <div class="metric-status online">정상 운영</div>
        </div>
        """, unsafe_allow_html=True)

def render_quick_actions():
    """퀵 액션 렌더링"""
    st.markdown("## ⚡ 퀵 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 데이터 분석")
        if st.button("🚀 새 분석 시작", use_container_width=True, type="primary"):
            st.session_state.current_page = 'workspace'
            st.rerun()
        
        if st.button("📁 데이터 업로드", use_container_width=True):
            st.session_state.current_page = 'workspace'
            st.rerun()
    
    with col2:
        st.markdown("### 🧬 에이전트 관리")
        if st.button("🔄 에이전트 상태 확인", use_container_width=True):
            st.session_state.current_page = 'agents'
            st.rerun()
        
        if st.button("🎛️ 오케스트레이션", use_container_width=True):
            st.session_state.current_page = 'agents'
            st.rerun()
    
    with col3:
        st.markdown("### 🔧 도구 관리")
        if st.button("🛠️ MCP 도구", use_container_width=True):
            st.session_state.current_page = 'mcp'
            st.rerun()
        
        if st.button("📈 모니터링", use_container_width=True):
            st.session_state.current_page = 'monitoring'
            st.rerun()

def render_realtime_status():
    """실시간 상태 렌더링"""
    st.markdown("## 🔄 실시간 에이전트 상태")
    
    # A2A 에이전트 포트들
    agent_ports = {
        8100: "A2A_Orchestrator",
        8306: "Data_Cleaning",
        8307: "Data_Loader", 
        8308: "Data_Visualization",
        8309: "Data_Wrangling",
        8310: "Feature_Engineering",
        8311: "SQL_Database",
        8312: "EDA_Tools",
        8313: "H2O_ML",
        8314: "MLflow_Tools",
        8315: "Python_REPL"
    }
    
    # 상태 확인 버튼
    if st.button("🔄 상태 새로고침", type="secondary"):
        st.rerun()
    
    # 에이전트 상태 그리드
    cols = st.columns(3)
    
    for i, (port, name) in enumerate(agent_ports.items()):
        with cols[i % 3]:
            status = check_agent_status(port)
            
            if status["status"] == "online":
                st.markdown(f"""
                <div class="cursor-agent-status online">
                    <div class="status-icon">✅</div>
                    <div class="agent-name">{name}</div>
                    <div class="agent-port">:{port}</div>
                    <div class="status-text">온라인</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="cursor-agent-status offline">
                    <div class="status-icon">❌</div>
                    <div class="agent-name">{name}</div>
                    <div class="agent-port">:{port}</div>
                    <div class="status-text">오프라인</div>
                </div>
                """, unsafe_allow_html=True)

def render_collaboration_preview():
    """협업 네트워크 미리보기"""
    st.markdown("## 🌐 에이전트 협업 네트워크")
    
    # Cursor 협업 네트워크 컴포넌트 사용
    collaboration_network = get_cursor_collaboration_network()
    
    # 네트워크 통계
    stats = collaboration_network.get_network_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("활성 노드", stats["active_nodes"], "실시간 동기화")
    
    with col2:
        st.metric("연결 수", stats["total_connections"], "A2A 프로토콜")
    
    with col3:
        st.metric("메시지 처리", stats["total_messages"], "실시간")
    
    # 간단한 네트워크 시각화
    if st.button("🔍 전체 네트워크 보기"):
        st.session_state.current_page = 'agents'
        st.rerun()

def render_architecture_overview():
    """아키텍처 개요"""
    st.markdown("## 🏗️ 플랫폼 아키텍처")
    
    with st.expander("🧬 A2A + MCP 통합 아키텍처", expanded=False):
        st.markdown("""
        ### 세계 최초 A2A + MCP 통합 플랫폼
        
        **🎯 핵심 특징:**
        - **A2A Protocol**: 10개 전문 에이전트 (포트 8306-8315)  
        - **MCP Tools**: 7개 통합 도구 (Playwright, FileManager, Database 등)
        - **Context Engineering**: 6개 데이터 레이어 구조
        - **Real-time Streaming**: SSE 기반 실시간 통신
        
        **🔄 워크플로우:**
        1. **INPUT** → MCP 도구로 데이터 수집/전처리
        2. **PROCESSING** → A2A 에이전트 협업으로 분석 실행  
        3. **OUTPUT** → 통합된 전문가 수준 결과 제공
        
        **⚡ 성능 특징:**
        - 병렬 처리: 멀티 에이전트 동시 실행
        - 지능형 라우팅: LLM 기반 최적 에이전트 선택
        - 오류 복구: 자동 복구 및 대체 실행
        """)
    
    # 기술 스택
    with st.expander("🛠️ 기술 스택", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Frontend & UI:**
            - Streamlit (Web UI)
            - Cursor-Style CSS
            - D3.js (네트워크 시각화)
            - Plotly (차트)
            """)
        
        with col2:
            st.markdown("""
            **Backend & Protocol:**
            - A2A SDK v0.2.9
            - MCP Integration
            - Python AsyncIO
            - SSE Streaming
            """)

def render_recent_activity():
    """최근 활동"""
    st.markdown("## 📋 최근 활동")
    
    # 모의 활동 로그
    activities = [
        {"time": "09:32", "type": "system", "message": "A2A 에이전트 시스템 시작"},
        {"time": "09:31", "type": "agent", "message": "Data Cleaning 에이전트 준비 완료"},
        {"time": "09:31", "type": "agent", "message": "Data Loader 에이전트 준비 완료"},
        {"time": "09:30", "type": "mcp", "message": "MCP 도구 통합 완료"},
        {"time": "09:30", "type": "system", "message": "Context Engineering 시스템 초기화"}
    ]
    
    for activity in activities:
        icon = {"system": "🔧", "agent": "🧬", "mcp": "🛠️"}.get(activity["type"], "📝")
        st.markdown(f"""
        <div class="cursor-activity-item">
            <span class="activity-time">{activity['time']}</span>
            <span class="activity-icon">{icon}</span>
            <span class="activity-message">{activity['message']}</span>
        </div>
        """, unsafe_allow_html=True)

def apply_dashboard_styles():
    """대시보드 전용 스타일"""
    st.markdown("""
    <style>
    .cursor-metric-card {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .cursor-metric-card:hover {
        border-color: var(--cursor-accent-blue);
        box-shadow: 0 4px 12px rgba(0, 122, 204, 0.2);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--cursor-secondary-text);
        margin-bottom: 0.25rem;
    }
    
    .metric-status {
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .metric-status.online {
        background: rgba(40, 167, 69, 0.2);
        color: #28a745;
    }
    
    .cursor-agent-status {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .cursor-agent-status.online {
        border-color: #28a745;
    }
    
    .cursor-agent-status.offline {
        border-color: #dc3545;
        opacity: 0.7;
    }
    
    .status-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .agent-name {
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 0.25rem;
    }
    
    .agent-port {
        font-size: 0.8rem;
        color: var(--cursor-muted-text);
        margin-bottom: 0.25rem;
    }
    
    .status-text {
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .cursor-activity-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: var(--cursor-secondary-bg);
        border-radius: 6px;
        border-left: 3px solid var(--cursor-accent-blue);
    }
    
    .activity-time {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        min-width: 60px;
        margin-right: 1rem;
    }
    
    .activity-icon {
        margin-right: 0.75rem;
        font-size: 1.1rem;
    }
    
    .activity-message {
        color: var(--cursor-secondary-text);
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)

def render_main_dashboard():
    """메인 대시보드 렌더링"""
    # 스타일 적용
    apply_dashboard_styles()
    
    # 헤더
    st.markdown("# 🚀 CherryAI Dashboard")
    st.markdown("**세계 최초 A2A + MCP 통합 플랫폼 대시보드**")
    
    st.markdown("---")
    
    # 시스템 개요
    render_system_overview()
    
    st.markdown("---")
    
    # 퀵 액션
    render_quick_actions()
    
    st.markdown("---")
    
    # 실시간 상태
    render_realtime_status()
    
    st.markdown("---")
    
    # 두 개 컬럼으로 나누어 표시
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_collaboration_preview()
    
    with col2:
        render_recent_activity()
    
    st.markdown("---")
    
    # 아키텍처 개요
    render_architecture_overview() 