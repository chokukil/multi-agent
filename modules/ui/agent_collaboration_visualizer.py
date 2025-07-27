"""
Agent Collaboration Visualizer - 실시간 에이전트 협업 시각화

실시간 에이전트 협업 과정을 시각적으로 표시:
- 개별 에이전트 진행률 바 (0-100%)
- 에이전트 아바타와 상태 표시기
- 현재 작업 설명과 상태 메시지
- 완료 체크마크와 실행 시간
- 에이전트 간 데이터 흐름 시각화
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from ..models import AgentProgressInfo, TaskState


class AgentCollaborationVisualizer:
    """실시간 에이전트 협업 시각화 컴포넌트"""
    
    def __init__(self):
        """시각화 컴포넌트 초기화"""
        self.agent_avatars = {
            8306: {"icon": "🧹", "name": "Data Cleaning", "color": "#17a2b8"},
            8307: {"icon": "📁", "name": "Data Loader", "color": "#6f42c1"},
            8308: {"icon": "📊", "name": "Data Visualization", "color": "#e83e8c"},
            8309: {"icon": "🔧", "name": "Data Wrangling", "color": "#fd7e14"},
            8310: {"icon": "⚙️", "name": "Feature Engineering", "color": "#20c997"},
            8311: {"icon": "🗄️", "name": "SQL Database", "color": "#6c757d"},
            8312: {"icon": "🔍", "name": "EDA Tools", "color": "#007bff"},
            8313: {"icon": "🤖", "name": "H2O ML", "color": "#28a745"},
            8314: {"icon": "📈", "name": "MLflow Tools", "color": "#ffc107"},
            8315: {"icon": "🐼", "name": "Pandas Analyst", "color": "#dc3545"}
        }
        
        self._inject_collaboration_styles()
    
    def _inject_collaboration_styles(self):
        """에이전트 협업 시각화를 위한 CSS 스타일 주입"""
        st.markdown("""
        <style>
        /* Agent Collaboration Visualization Styles */
        
        .agent-collaboration-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid #dee2e6;
        }
        
        .collaboration-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #dee2e6;
        }
        
        .collaboration-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #495057;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .overall-progress {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .overall-progress-bar {
            width: 200px;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        
        .overall-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
            border-radius: 4px;
            transition: width 0.3s ease;
            position: relative;
        }
        
        .overall-progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .overall-progress-text {
            font-weight: 600;
            color: #495057;
            min-width: 50px;
        }
        
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .agent-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .agent-card.working {
            border-left: 4px solid #007bff;
            animation: pulse-border 2s infinite;
        }
        
        .agent-card.completed {
            border-left: 4px solid #28a745;
        }
        
        .agent-card.failed {
            border-left: 4px solid #dc3545;
        }
        
        .agent-card.pending {
            border-left: 4px solid #ffc107;
        }
        
        @keyframes pulse-border {
            0%, 100% { border-left-color: #007bff; }
            50% { border-left-color: #0056b3; }
        }
        
        .agent-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }
        
        .agent-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .agent-avatar {
            font-size: 1.5rem;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, var(--agent-color, #6c757d) 0%, var(--agent-color-dark, #495057) 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .agent-name {
            font-weight: 600;
            color: #495057;
            font-size: 0.9rem;
        }
        
        .agent-port {
            font-size: 0.75rem;
            color: #6c757d;
            background: #f8f9fa;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }
        
        .agent-status {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .status-icon {
            font-size: 1.2rem;
        }
        
        .status-icon.working {
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .agent-progress {
            margin-bottom: 0.75rem;
        }
        
        .progress-bar-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .progress-bar {
            flex: 1;
            height: 6px;
            background: #e9ecef;
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
            position: relative;
        }
        
        .progress-fill.working {
            background: linear-gradient(90deg, #007bff 0%, #0056b3 100%);
        }
        
        .progress-fill.completed {
            background: linear-gradient(90deg, #28a745 0%, #1e7e34 100%);
        }
        
        .progress-fill.failed {
            background: linear-gradient(90deg, #dc3545 0%, #c82333 100%);
        }
        
        .progress-fill.working::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%);
            animation: progress-shimmer 1.5s infinite;
        }
        
        @keyframes progress-shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .progress-percentage {
            font-size: 0.8rem;
            font-weight: 600;
            color: #495057;
            min-width: 35px;
            text-align: right;
        }
        
        .agent-task {
            font-size: 0.8rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }
        
        .agent-metrics {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: #6c757d;
        }
        
        .execution-time {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .artifacts-count {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .data-flow-visualization {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 2px solid #dee2e6;
        }
        
        .data-flow-title {
            font-size: 1rem;
            font-weight: 600;
            color: #495057;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .data-flow-diagram {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .flow-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
            min-width: 80px;
        }
        
        .flow-node.active {
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }
        
        .flow-arrow {
            font-size: 1.5rem;
            color: #6c757d;
            animation: flow-pulse 2s infinite;
        }
        
        @keyframes flow-pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        .flow-node-icon {
            font-size: 1.2rem;
        }
        
        .flow-node-name {
            font-size: 0.7rem;
            color: #6c757d;
            text-align: center;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .agents-grid {
                grid-template-columns: 1fr;
            }
            
            .collaboration-header {
                flex-direction: column;
                gap: 1rem;
                align-items: stretch;
            }
            
            .overall-progress {
                justify-content: center;
            }
            
            .data-flow-diagram {
                flex-direction: column;
            }
            
            .flow-arrow {
                transform: rotate(90deg);
            }
        }
        
        /* Accessibility improvements */
        .agent-card:focus-within {
            outline: 2px solid #007bff;
            outline-offset: 2px;
        }
        
        /* High contrast mode */
        @media (prefers-contrast: high) {
            .agent-card {
                border: 2px solid currentColor;
            }
            
            .progress-bar,
            .overall-progress-bar {
                border: 1px solid currentColor;
            }
        }
        
        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {
            .agent-card.working,
            .status-icon.working,
            .progress-fill.working::after,
            .overall-progress-fill::after,
            .flow-arrow {
                animation: none;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_collaboration_dashboard(self, 
                                     agents: List[AgentProgressInfo],
                                     task_id: str = None,
                                     show_data_flow: bool = True) -> None:
        """
        실시간 에이전트 협업 대시보드 렌더링
        
        Args:
            agents: 에이전트 진행 상황 정보 리스트
            task_id: 작업 ID
            show_data_flow: 데이터 흐름 시각화 표시 여부
        """
        if not agents:
            return
        
        # 전체 진행률 계산
        total_progress = sum(agent.progress_percentage for agent in agents) / len(agents)
        active_agents = sum(1 for agent in agents if agent.status == TaskState.WORKING)
        completed_agents = sum(1 for agent in agents if agent.status == TaskState.COMPLETED)
        
        # 협업 대시보드 컨테이너
        st.markdown(f"""
        <div class="agent-collaboration-container" data-testid="agent-collaboration">
            <div class="collaboration-header">
                <div class="collaboration-title">
                    🤝 Agent Collaboration
                    {f'<span style="color: #6c757d; font-size: 0.9rem;">({task_id})</span>' if task_id else ''}
                </div>
                <div class="overall-progress">
                    <div class="overall-progress-bar">
                        <div class="overall-progress-fill" style="width: {total_progress}%"></div>
                    </div>
                    <div class="overall-progress-text">{total_progress:.0f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 에이전트 상태 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔄 Active", active_agents)
        
        with col2:
            st.metric("✅ Completed", completed_agents)
        
        with col3:
            st.metric("📊 Total Progress", f"{total_progress:.0f}%")
        
        with col4:
            elapsed_time = max(agent.execution_time for agent in agents) if agents else 0
            st.metric("⏱️ Elapsed", f"{elapsed_time:.1f}s")
        
        # 에이전트 카드 그리드
        self._render_agents_grid(agents)
        
        # 데이터 흐름 시각화
        if show_data_flow:
            self._render_data_flow_visualization(agents)
    
    def _render_agents_grid(self, agents: List[AgentProgressInfo]) -> None:
        """에이전트 카드 그리드 렌더링"""
        
        # 에이전트를 상태별로 정렬 (작업 중 > 완료 > 대기 > 실패)
        status_priority = {
            TaskState.WORKING: 1,
            TaskState.COMPLETED: 2,
            TaskState.PENDING: 3,
            TaskState.FAILED: 4
        }
        
        sorted_agents = sorted(agents, key=lambda a: status_priority.get(a.status, 5))
        
        st.markdown('<div class="agents-grid">', unsafe_allow_html=True)
        
        for agent in sorted_agents:
            self._render_agent_card(agent)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_agent_card(self, agent: AgentProgressInfo) -> None:
        """개별 에이전트 카드 렌더링 - Streamlit 네이티브 컴포넌트 사용"""
        
        agent_info = self.agent_avatars.get(agent.port, {
            "icon": "🔄",
            "name": f"Agent {agent.port}",
            "color": "#6c757d"
        })
        
        status_icon = self._get_status_icon(agent.status)
        
        # Streamlit 네이티브 컴포넌트로 에이전트 카드 렌더링
        with st.container():
            # 에이전트 헤더
            col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
            
            with col1:
                st.markdown(f"## {agent_info['icon']}")
            
            with col2:
                st.markdown(f"**{agent_info['name']}**")
                st.caption(f"Port {agent.port}")
            
            with col3:
                st.markdown(f"### {status_icon}")
            
            # 진행률 바
            progress_bar = st.progress(agent.progress_percentage / 100)
            
            # 진행률과 작업 상태
            col_progress, col_percentage = st.columns([0.8, 0.2])
            with col_progress:
                st.caption(agent.current_task or 'Ready to work')
            with col_percentage:
                st.caption(f"{agent.progress_percentage:.0f}%")
            
            # 메트릭스
            col_time, col_artifacts = st.columns(2)
            with col_time:
                st.caption(f"⏱️ {agent.execution_time:.1f}s")
            with col_artifacts:
                st.caption(f"📄 {len(agent.artifacts_generated)} artifacts")
            
            # 구분선
            st.markdown("---")
    
    def _render_data_flow_visualization(self, agents: List[AgentProgressInfo]) -> None:
        """데이터 흐름 시각화 렌더링 - Streamlit 네이티브 컴포넌트 사용"""
        
        # 활성 에이전트만 표시
        active_agents = [agent for agent in agents 
                        if agent.status in [TaskState.WORKING, TaskState.COMPLETED]]
        
        if not active_agents:
            return
        
        st.markdown("### 🔄 Data Flow Visualization")
        
        # 에이전트 플로우를 가로로 표시
        if len(active_agents) > 1:
            cols = st.columns(len(active_agents) * 2 - 1)  # 에이전트 + 화살표
            
            for i, agent in enumerate(active_agents):
                agent_info = self.agent_avatars.get(agent.port, {
                    "icon": "🔄",
                    "name": f"Agent {agent.port}"
                })
                
                # 에이전트 노드
                with cols[i * 2]:
                    st.markdown(f"""
                    **{agent_info['icon']}**  
                    {agent_info['name']}
                    """)
                
                # 화살표 (마지막이 아닌 경우)
                if i < len(active_agents) - 1:
                    with cols[i * 2 + 1]:
                        st.markdown("**→**")
        else:
            # 단일 에이전트인 경우
            agent = active_agents[0]
            agent_info = self.agent_avatars.get(agent.port, {
                "icon": "🔄",
                "name": f"Agent {agent.port}"
            })
            st.markdown(f"**{agent_info['icon']} {agent_info['name']}**")
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _get_status_icon(self, status: TaskState) -> str:
        """상태별 아이콘 반환"""
        status_icons = {
            TaskState.PENDING: "⏳",
            TaskState.WORKING: "⚡",
            TaskState.COMPLETED: "✅",
            TaskState.FAILED: "❌"
        }
        return status_icons.get(status, "❓")
    
    def _darken_color(self, hex_color: str, factor: float = 0.2) -> str:
        """색상을 어둡게 만드는 함수"""
        try:
            # #을 제거하고 RGB 값 추출
            hex_color = hex_color.lstrip('#')
            
            # RGB 값을 정수로 변환
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # 각 값을 factor만큼 어둡게
            r = int(r * (1 - factor))
            g = int(g * (1 - factor))
            b = int(b * (1 - factor))
            
            # 다시 hex 형태로 변환
            return f"#{r:02x}{g:02x}{b:02x}"
        
        except:
            # 오류 발생 시 기본값 반환
            return "#495057"
    
    def render_compact_progress_bar(self, 
                                  agents: List[AgentProgressInfo],
                                  show_details: bool = False) -> None:
        """
        컴팩트한 진행률 바 렌더링 (채팅 인터페이스용)
        
        Args:
            agents: 에이전트 진행 상황 정보 리스트
            show_details: 상세 정보 표시 여부
        """
        if not agents:
            return
        
        total_progress = sum(agent.progress_percentage for agent in agents) / len(agents)
        active_count = sum(1 for agent in agents if agent.status == TaskState.WORKING)
        
        # 컴팩트 진행률 바
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 6px;
            margin: 0.5rem 0;
            font-size: 0.8rem;
        ">
            <div style="
                width: 100px;
                height: 4px;
                background: #e9ecef;
                border-radius: 2px;
                overflow: hidden;
            ">
                <div style="
                    width: {total_progress}%;
                    height: 100%;
                    background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
                    border-radius: 2px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <span style="color: #495057; font-weight: 500;">
                {total_progress:.0f}% • {active_count} agents working
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # 상세 정보 표시
        if show_details:
            with st.expander("🔍 Agent Details", expanded=False):
                for agent in agents:
                    agent_info = self.agent_avatars.get(agent.port, {
                        "icon": "🔄",
                        "name": f"Agent {agent.port}"
                    })
                    
                    status_icon = self._get_status_icon(agent.status)
                    
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        padding: 0.25rem 0;
                        font-size: 0.8rem;
                    ">
                        <span>{agent_info["icon"]}</span>
                        <span style="flex: 1;">{agent_info["name"]}</span>
                        <span>{status_icon}</span>
                        <span style="min-width: 40px; text-align: right;">
                            {agent.progress_percentage:.0f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
    
    def create_mock_agents_for_demo(self) -> List[AgentProgressInfo]:
        """데모용 모의 에이전트 생성"""
        import random
        
        demo_agents = []
        ports = [8306, 8312, 8315, 8308]  # 일부 에이전트만 사용
        
        for i, port in enumerate(ports):
            if i == 0:
                status = TaskState.COMPLETED
                progress = 100
                task = "Data cleaning completed"
                execution_time = 5.2
            elif i == 1:
                status = TaskState.WORKING
                progress = random.randint(30, 80)
                task = "Performing exploratory data analysis..."
                execution_time = 3.1
            elif i == 2:
                status = TaskState.WORKING
                progress = random.randint(10, 50)
                task = "Processing data with pandas..."
                execution_time = 1.8
            else:
                status = TaskState.PENDING
                progress = 0
                task = "Waiting for data..."
                execution_time = 0
            
            agent = AgentProgressInfo(
                port=port,
                name=self.agent_avatars[port]["name"],
                status=status,
                progress_percentage=progress,
                current_task=task,
                execution_time=execution_time,
                artifacts_generated=["result.json"] if status == TaskState.COMPLETED else []
            )
            
            demo_agents.append(agent)
        
        return demo_agents