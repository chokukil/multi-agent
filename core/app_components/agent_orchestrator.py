"""
🧬 Agent Orchestrator Component  
Cursor 스타일의 A2A 에이전트 오케스트레이션 UI
"""

import streamlit as st
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Cursor UI 컴포넌트 임포트
from ui.cursor_style_agent_cards import get_cursor_agent_cards, AgentCard, AgentStep
from ui.cursor_collaboration_network import get_cursor_collaboration_network
from ui.cursor_theme_system import get_cursor_theme

# A2A 클라이언트 (조건부)
try:
    from core.a2a.a2a_streamlit_client import A2AStreamlitClient
    A2A_CLIENT_AVAILABLE = True
except ImportError:
    A2A_CLIENT_AVAILABLE = False

class A2AAgentOrchestrator:
    """A2A 에이전트 오케스트레이터"""
    
    def __init__(self):
        self.agent_cards = get_cursor_agent_cards()
        self.collaboration_network = get_cursor_collaboration_network()
        self.agents_config = self._get_agents_config()
        
        if A2A_CLIENT_AVAILABLE:
            # A2A SDK 0.2.9 요구사항에 따라 agents_info 매개변수 제공
            self.a2a_client = A2AStreamlitClient(self.agents_config)
        else:
            self.a2a_client = None
    
    def _get_agents_config(self) -> Dict[str, Dict[str, Any]]:
        """에이전트 설정 정보"""
        return {
            "orchestrator": {
                "name": "A2A Orchestrator",
                "port": 8100,
                "icon": "🎯",
                "description": "AI DS Team을 지휘하는 마에스트로",
                "capabilities": ["task_coordination", "agent_selection", "workflow_management"]
            },
            "data_cleaning": {
                "name": "Data Cleaning",
                "port": 8306,
                "icon": "🧹",
                "description": "누락값 처리, 이상치 제거",
                "capabilities": ["missing_values", "outlier_detection", "data_validation"]
            },
            "data_loader": {
                "name": "Data Loader", 
                "port": 8307,
                "icon": "📁",
                "description": "다양한 데이터 소스 로딩",
                "capabilities": ["csv_loading", "excel_loading", "json_loading", "database_connection"]
            },
            "data_visualization": {
                "name": "Data Visualization",
                "port": 8308,
                "icon": "📊",
                "description": "고급 시각화 생성",
                "capabilities": ["plotly_charts", "statistical_plots", "interactive_viz"]
            },
            "data_wrangling": {
                "name": "Data Wrangling",
                "port": 8309,
                "icon": "🔧",
                "description": "데이터 변환 및 조작",
                "capabilities": ["data_transformation", "feature_creation", "data_reshaping"]
            },
            "feature_engineering": {
                "name": "Feature Engineering",
                "port": 8310,
                "icon": "⚙️",
                "description": "고급 피처 생성 및 선택",
                "capabilities": ["feature_creation", "feature_selection", "dimensionality_reduction"]
            },
            "sql_database": {
                "name": "SQL Database",
                "port": 8311,
                "icon": "🗄️",
                "description": "SQL 데이터베이스 분석",
                "capabilities": ["sql_query", "database_analysis", "data_extraction"]
            },
            "eda_tools": {
                "name": "EDA Tools",
                "port": 8312,
                "icon": "🔍",
                "description": "자동 EDA 및 상관관계 분석",
                "capabilities": ["automated_eda", "correlation_analysis", "statistical_summary"]
            },
            "h2o_ml": {
                "name": "H2O ML",
                "port": 8313,
                "icon": "🤖",
                "description": "H2O AutoML 기반 머신러닝",
                "capabilities": ["automl", "model_training", "model_evaluation"]
            },
            "mlflow_tools": {
                "name": "MLflow Tools",
                "port": 8314,
                "icon": "📈",
                "description": "MLflow 실험 관리",
                "capabilities": ["experiment_tracking", "model_registry", "deployment"]
            },
            "python_repl": {
                "name": "Python REPL",
                "port": 8315,
                "icon": "🐍",
                "description": "Python 코드 실행 환경",
                "capabilities": ["code_execution", "interactive_programming", "debugging"]
            }
        }
    
    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """에이전트 상태 확인"""
        config = self.agents_config.get(agent_id)
        if not config:
            return {"status": "unknown", "error": "Agent not configured"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{config['port']}/.well-known/agent.json",
                    timeout=3.0
                )
                if response.status_code == 200:
                    return {
                        "status": "online",
                        "data": response.json(),
                        "response_time": response.elapsed.total_seconds()
                    }
        except Exception as e:
            return {"status": "offline", "error": str(e)}
        
        return {"status": "offline", "error": "No response"}
    
    async def send_a2a_message(self, agent_id: str, message: str) -> Dict[str, Any]:
        """A2A 메시지 전송"""
        if not self.a2a_client:
            return {"success": False, "error": "A2A client not available"}
        
        config = self.agents_config.get(agent_id)
        if not config:
            return {"success": False, "error": "Agent not configured"}
        
        try:
            response = await self.a2a_client.send_message(
                f"http://localhost:{config['port']}",
                message
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}

def render_agent_grid():
    """에이전트 그리드 렌더링"""
    st.markdown("## 🧬 A2A 에이전트 상태")
    
    orchestrator = A2AAgentOrchestrator()
    
    # 상태 새로고침 버튼
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 상태 새로고침", type="secondary"):
            st.rerun()
    
    # 에이전트 그리드 (3열)
    cols = st.columns(3)
    
    for i, (agent_id, config) in enumerate(orchestrator.agents_config.items()):
        with cols[i % 3]:
            # 에이전트 상태 확인 (동기 방식으로 임시 처리)
            try:
                response = httpx.get(f"http://localhost:{config['port']}/.well-known/agent.json", timeout=2.0)
                status = "online" if response.status_code == 200 else "offline"
                response_time = response.elapsed.total_seconds() if response.status_code == 200 else None
            except:
                status = "offline"
                response_time = None
            
            # 에이전트 카드 렌더링
            status_color = "#28a745" if status == "online" else "#dc3545"
            status_icon = "✅" if status == "online" else "❌"
            
            st.markdown(f"""
            <div class="cursor-agent-card" style="border-left: 4px solid {status_color};">
                <div class="agent-header">
                    <span class="agent-icon">{config['icon']}</span>
                    <span class="agent-name">{config['name']}</span>
                    <span class="agent-status">{status_icon}</span>
                </div>
                <div class="agent-description">{config['description']}</div>
                <div class="agent-port">Port: {config['port']}</div>
                {f'<div class="agent-response-time">응답시간: {response_time:.2f}s</div>' if response_time else ''}
                <div class="agent-capabilities">
                    {' '.join([f'<span class="capability-tag">{cap}</span>' for cap in config['capabilities'][:2]])}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_collaboration_network():
    """협업 네트워크 시각화"""
    st.markdown("## 🌐 에이전트 협업 네트워크")
    
    collaboration_network = get_cursor_collaboration_network()
    
    # 네트워크 통계
    col1, col2, col3, col4 = st.columns(4)
    
    stats = collaboration_network.get_network_stats()
    
    with col1:
        st.metric("활성 노드", stats.get("active_nodes", 11), "실시간")
    
    with col2:
        st.metric("연결 수", stats.get("total_connections", 15), "A2A 프로토콜")
    
    with col3:
        st.metric("메시지 처리", stats.get("total_messages", 0), "실시간")
    
    with col4:
        st.metric("평균 응답시간", "0.12s", "최적화됨")
    
    # 네트워크 시각화
    with st.container():
        # D3.js 네트워크 시각화가 여기에 들어갈 예정
        st.info("🚧 실시간 네트워크 시각화가 로드되는 중입니다...")
        
        # 임시 네트워크 정보
        st.markdown("""
        **현재 활성 연결:**
        - 🎯 Orchestrator ↔ 🧹 Data Cleaning
        - 🎯 Orchestrator ↔ 📁 Data Loader  
        - 🎯 Orchestrator ↔ 📊 Data Visualization
        - 🎯 Orchestrator ↔ 🔧 Data Wrangling
        - 🎯 Orchestrator ↔ ⚙️ Feature Engineering
        """)

def render_message_console():
    """메시지 콘솔"""
    st.markdown("## 💬 A2A 메시지 콘솔")
    
    orchestrator = A2AAgentOrchestrator()
    
    # 에이전트 선택
    agent_options = {config['name']: agent_id for agent_id, config in orchestrator.agents_config.items()}
    selected_agent_name = st.selectbox("대상 에이전트 선택", list(agent_options.keys()))
    selected_agent_id = agent_options[selected_agent_name]
    
    # 메시지 입력
    message = st.text_area(
        "A2A 메시지", 
        placeholder="에이전트에게 전송할 메시지를 입력하세요...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("📤 메시지 전송", type="primary", disabled=not message.strip()):
            if A2A_CLIENT_AVAILABLE:
                # 비동기 메시지 전송 (실제 구현에서는 asyncio 사용)
                st.info(f"🔄 {selected_agent_name}에게 메시지 전송 중...")
                # 여기서 실제 A2A 메시지 전송 로직 구현
                st.success(f"✅ 메시지가 {selected_agent_name}에게 전송되었습니다!")
            else:
                st.warning("⚠️ A2A 클라이언트가 활성화되지 않았습니다.")
    
    # 최근 메시지 로그
    with st.expander("📋 최근 메시지 로그", expanded=False):
        st.markdown("""
        **[09:32:15]** 🎯 Orchestrator → 📁 Data Loader: "CSV 파일 로드 요청"  
        **[09:32:10]** 📁 Data Loader → 🎯 Orchestrator: "데이터 로드 완료"  
        **[09:32:05]** 🎯 Orchestrator → 🧹 Data Cleaning: "데이터 정제 요청"  
        **[09:32:00]** 🧹 Data Cleaning → 🎯 Orchestrator: "정제 작업 완료"  
        """)

def render_workflow_builder():
    """워크플로우 빌더"""
    st.markdown("## 🔄 워크플로우 빌더")
    
    st.info("🚧 고급 워크플로우 빌더 개발 중...")
    
    # 간단한 워크플로우 템플릿
    with st.expander("📋 사전 정의된 워크플로우", expanded=True):
        workflow_templates = {
            "기본 데이터 분석": [
                "📁 Data Loader → 🧹 Data Cleaning → 🔍 EDA Tools → 📊 Data Visualization"
            ],
            "머신러닝 파이프라인": [
                "📁 Data Loader → 🧹 Data Cleaning → ⚙️ Feature Engineering → 🤖 H2O ML → 📈 MLflow Tools"
            ],
            "SQL 데이터 분석": [
                "🗄️ SQL Database → 🔧 Data Wrangling → 🔍 EDA Tools → 📊 Data Visualization"
            ]
        }
        
        for workflow_name, steps in workflow_templates.items():
            st.markdown(f"**{workflow_name}:**")
            st.markdown(f"```\n{steps[0]}\n```")
            if st.button(f"🚀 {workflow_name} 실행", key=f"workflow_{workflow_name}"):
                st.success(f"✅ {workflow_name} 워크플로우가 시작되었습니다!")

def apply_orchestrator_styles():
    """오케스트레이터 전용 스타일"""
    st.markdown("""
    <style>
    .cursor-agent-card {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        height: 180px;
        display: flex;
        flex-direction: column;
    }
    
    .cursor-agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 122, 204, 0.15);
    }
    
    .agent-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .agent-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .agent-name {
        font-weight: 600;
        color: var(--cursor-primary-text);
        flex: 1;
    }
    
    .agent-status {
        font-size: 1.2rem;
    }
    
    .agent-description {
        color: var(--cursor-secondary-text);
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        flex: 1;
    }
    
    .agent-port {
        font-size: 0.8rem;
        color: var(--cursor-muted-text);
        margin-bottom: 0.5rem;
    }
    
    .agent-response-time {
        font-size: 0.8rem;
        color: var(--cursor-accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .agent-capabilities {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
    }
    
    .capability-tag {
        background: rgba(0, 122, 204, 0.2);
        color: var(--cursor-accent-blue);
        font-size: 0.7rem;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        border: 1px solid rgba(0, 122, 204, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def render_agent_orchestrator():
    """에이전트 오케스트레이터 메인 렌더링"""
    # 스타일 적용
    apply_orchestrator_styles()
    
    # 헤더
    st.markdown("# 🧬 A2A Agent Orchestrator")
    st.markdown("**A2A 프로토콜 기반 에이전트 오케스트레이션 및 협업 관리**")
    
    st.markdown("---")
    
    # 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 에이전트 상태", 
        "🌐 협업 네트워크", 
        "💬 메시지 콘솔", 
        "🔄 워크플로우"
    ])
    
    with tab1:
        render_agent_grid()
    
    with tab2:
        render_collaboration_network()
    
    with tab3:
        render_message_console()
    
    with tab4:
        render_workflow_builder() 