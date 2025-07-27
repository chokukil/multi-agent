"""
에이전트 협업 시각화 시스템

이 모듈은 멀티 에이전트 시스템의 실시간 상태와 협업 과정을 시각화하여
사용자가 분석 진행 상황을 직관적으로 이해할 수 있도록 하는 시스템을 제공합니다.

주요 기능:
- 실시간 에이전트 상태 대시보드
- 작업 진행률 및 완료 상태 표시
- 에이전트 간 데이터 흐름 시각화
- 성능 메트릭 실시간 모니터링
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ..integration.agent_result_collector import AgentStatus, AgentResult, CollectionSession

logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """시각화 유형"""
    STATUS_DASHBOARD = "status_dashboard"       # 상태 대시보드
    WORKFLOW_DIAGRAM = "workflow_diagram"       # 워크플로우 다이어그램
    TIMELINE_CHART = "timeline_chart"           # 타임라인 차트
    PERFORMANCE_METRICS = "performance_metrics"  # 성능 메트릭
    DATA_FLOW = "data_flow"                     # 데이터 흐름
    ERROR_ANALYSIS = "error_analysis"           # 에러 분석

class AgentState(Enum):
    """에이전트 상태"""
    IDLE = "idle"                 # 대기중
    INITIALIZING = "initializing" # 초기화중
    RUNNING = "running"           # 실행중
    PROCESSING = "processing"     # 처리중
    COMPLETING = "completing"     # 완료중
    COMPLETED = "completed"       # 완료됨
    ERROR = "error"              # 에러
    TIMEOUT = "timeout"          # 타임아웃

@dataclass
class AgentMetrics:
    """에이전트 메트릭"""
    agent_id: str
    agent_name: str
    
    # 상태 정보
    current_state: AgentState = AgentState.IDLE
    last_state_change: datetime = field(default_factory=datetime.now)
    
    # 실행 메트릭
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_duration: float = 0.0
    
    # 진행률
    progress_percentage: float = 0.0
    current_task: str = ""
    completed_tasks: int = 0
    total_tasks: int = 0
    
    # 리소스 사용량
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # 데이터 처리량
    data_processed: int = 0
    artifacts_generated: int = 0
    
    # 에러 정보
    error_count: int = 0
    last_error: Optional[str] = None
    
    # 의존성
    depends_on: List[str] = field(default_factory=list)
    dependent_agents: List[str] = field(default_factory=list)

@dataclass
class DataFlow:
    """데이터 흐름"""
    flow_id: str
    source_agent: str
    target_agent: str
    data_type: str
    data_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "active"

@dataclass
class ExecutionLogEntry:
    """실행 로그 엔트리"""
    log_id: str
    agent_id: str
    timestamp: datetime
    log_level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_step: str = ""
    duration_ms: float = 0.0
    memory_usage: float = 0.0
    artifacts_created: List[str] = field(default_factory=list)
    
@dataclass
class AgentExecutionSummary:
    """에이전트 실행 요약"""
    agent_id: str
    agent_name: str
    total_execution_time: float
    total_steps: int
    successful_steps: int
    failed_steps: int
    artifacts_generated: int
    final_contribution_score: float  # 0.0 ~ 1.0
    key_achievements: List[str] = field(default_factory=list)
    performance_issues: List[str] = field(default_factory=list)
    
@dataclass
class CollaborationSnapshot:
    """협업 스냅샷"""
    timestamp: datetime
    agent_states: Dict[str, AgentState]
    active_flows: List[DataFlow]
    overall_progress: float
    performance_metrics: Dict[str, float]

class AgentCollaborationVisualizer:
    """에이전트 협업 시각화기"""
    
    def __init__(self):
        # 에이전트 메트릭 저장소
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # 데이터 흐름 추적
        self.data_flows: List[DataFlow] = []
        self.flow_history: deque = deque(maxlen=1000)  # 최근 1000개 흐름
        
        # 상태 변경 이력
        self.state_history: deque = deque(maxlen=500)  # 최근 500개 상태 변경
        
        # 실시간 업데이트 콜백
        self.update_callbacks: List[Callable] = []
        
        # 성능 추적
        self.performance_history: deque = deque(maxlen=100)  # 최근 100개 샘플
        
        # 색상 매핑
        self.state_colors = {
            AgentState.IDLE: "#95a5a6",         # 회색
            AgentState.INITIALIZING: "#3498db",  # 파란색
            AgentState.RUNNING: "#2ecc71",      # 녹색
            AgentState.PROCESSING: "#f39c12",   # 주황색
            AgentState.COMPLETING: "#9b59b6",   # 보라색
            AgentState.COMPLETED: "#27ae60",    # 진한 녹색
            AgentState.ERROR: "#e74c3c",        # 빨간색
            AgentState.TIMEOUT: "#e67e22"       # 진한 주황색
        }
        
        # 시각화 설정
        self.dashboard_config = {
            'update_interval': 1.0,  # 1초
            'max_agents_display': 20,
            'chart_height': 400,
            'timeline_window': 300  # 5분
        }
        
        # 실행 로그
        self.execution_logs: deque = deque(maxlen=10000)  # 최근 10,000개 로그
        self.agent_summaries: Dict[str, AgentExecutionSummary] = {}
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_name: str,
                      depends_on: List[str] = None) -> AgentMetrics:
        """에이전트 등록"""
        
        if agent_id not in self.agent_metrics:
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                depends_on=depends_on or []
            )
            
            self.agent_metrics[agent_id] = metrics
            
            # 의존성 관계 업데이트
            for dep_id in metrics.depends_on:
                if dep_id in self.agent_metrics:
                    self.agent_metrics[dep_id].dependent_agents.append(agent_id)
            
            logger.info(f"📊 에이전트 등록: {agent_name} ({agent_id})")
        
        return self.agent_metrics[agent_id]
    
    def update_agent_state(self, 
                          agent_id: str, 
                          new_state: AgentState,
                          task: str = "",
                          progress: float = None):
        """에이전트 상태 업데이트"""
        
        if agent_id not in self.agent_metrics:
            logger.warning(f"미등록 에이전트: {agent_id}")
            return
        
        metrics = self.agent_metrics[agent_id]
        old_state = metrics.current_state
        
        # 상태 변경
        metrics.current_state = new_state
        metrics.last_state_change = datetime.now()
        
        if task:
            metrics.current_task = task
        
        if progress is not None:
            metrics.progress_percentage = max(0.0, min(100.0, progress))
        
        # 시작/종료 시간 기록
        if new_state == AgentState.RUNNING and metrics.start_time is None:
            metrics.start_time = datetime.now()
        elif new_state in [AgentState.COMPLETED, AgentState.ERROR, AgentState.TIMEOUT]:
            metrics.end_time = datetime.now()
            if metrics.start_time:
                metrics.execution_duration = (metrics.end_time - metrics.start_time).total_seconds()
        
        # 상태 변경 이력 기록
        self.state_history.append({
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'old_state': old_state,
            'new_state': new_state
        })
        
        # 콜백 호출
        self._trigger_updates()
    
    def add_data_flow(self,
                     source_agent: str,
                     target_agent: str,
                     data_type: str,
                     data_size: int):
        """데이터 흐름 추가"""
        
        flow = DataFlow(
            flow_id=f"{source_agent}->{target_agent}_{len(self.data_flows)}",
            source_agent=source_agent,
            target_agent=target_agent,
            data_type=data_type,
            data_size=data_size
        )
        
        self.data_flows.append(flow)
        self.flow_history.append(flow)
        
        logger.debug(f"🔄 데이터 흐름: {source_agent} → {target_agent} ({data_type})")
        
        # 콜백 호출
        self._trigger_updates()
    
    def update_agent_progress(self,
                            agent_id: str,
                            completed_tasks: int,
                            total_tasks: int):
        """에이전트 진행률 업데이트"""
        
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.completed_tasks = completed_tasks
            metrics.total_tasks = total_tasks
            
            if total_tasks > 0:
                metrics.progress_percentage = (completed_tasks / total_tasks) * 100
            
            self._trigger_updates()
    
    def update_agent_metrics(self,
                           agent_id: str,
                           cpu_usage: float = None,
                           memory_usage: float = None,
                           data_processed: int = None,
                           artifacts_generated: int = None):
        """에이전트 메트릭 업데이트"""
        
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            
            if cpu_usage is not None:
                metrics.cpu_usage = cpu_usage
            if memory_usage is not None:
                metrics.memory_usage = memory_usage
            if data_processed is not None:
                metrics.data_processed = data_processed
            if artifacts_generated is not None:
                metrics.artifacts_generated = artifacts_generated
    
    def record_agent_error(self, agent_id: str, error_message: str):
        """에이전트 에러 기록"""
        
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.error_count += 1
            metrics.last_error = error_message
            metrics.current_state = AgentState.ERROR
            metrics.last_state_change = datetime.now()
            
            logger.error(f"❌ 에이전트 에러 - {agent_id}: {error_message}")
            
            self._trigger_updates()
    
    def create_status_dashboard(self) -> Dict[str, Any]:
        """상태 대시보드 생성"""
        
        dashboard_data = {
            'timestamp': datetime.now(),
            'summary': self._generate_summary_metrics(),
            'agent_cards': self._generate_agent_cards(),
            'timeline_chart': self._generate_timeline_chart(),
            'progress_overview': self._generate_progress_overview()
        }
        
        return dashboard_data
    
    def render_streamlit_dashboard(self, container=None):
        """Streamlit 대시보드 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            # 헤더
            st.markdown("## 🤝 에이전트 협업 대시보드")
            
            # 요약 메트릭
            summary = self._generate_summary_metrics()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="활성 에이전트",
                    value=summary['active_agents'],
                    delta=f"{summary['total_agents']} 전체"
                )
            
            with col2:
                st.metric(
                    label="전체 진행률",
                    value=f"{summary['overall_progress']:.1f}%",
                    delta=f"{summary['completed_agents']} 완료"
                )
            
            with col3:
                st.metric(
                    label="평균 실행 시간",
                    value=f"{summary['avg_execution_time']:.1f}초",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="에러율",
                    value=f"{summary['error_rate']:.1f}%",
                    delta=f"{summary['total_errors']} 건"
                )
            
            # 진행률 차트
            st.markdown("### 📊 에이전트별 진행 상황")
            progress_fig = self._create_progress_chart()
            st.plotly_chart(progress_fig, use_container_width=True)
            
            # 타임라인 차트
            st.markdown("### 📈 실행 타임라인")
            timeline_fig = self._create_timeline_chart()
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # 에이전트 상태 카드
            st.markdown("### 🎯 에이전트 상태")
            self._render_agent_cards()
            
            # 데이터 흐름 다이어그램
            if self.data_flows:
                st.markdown("### 🔄 데이터 흐름")
                flow_fig = self._create_data_flow_diagram()
                st.plotly_chart(flow_fig, use_container_width=True)
            
            # 워크플로우 다이어그램
            st.markdown("### 🔀 워크플로우 다이어그램")
            workflow_fig = self.create_workflow_diagram()
            st.plotly_chart(workflow_fig, use_container_width=True)
            
            # 기여도 분석 차트
            if self.agent_summaries:
                st.markdown("### 📊 에이전트별 기여도 분석")
                contribution_fig = self.create_contribution_analysis_chart()
                st.plotly_chart(contribution_fig, use_container_width=True)
            
            # 에이전트 실행 요약
            self.render_agent_summaries()
            
            # 상세 실행 로그
            self.render_execution_logs()
    
    def _generate_summary_metrics(self) -> Dict[str, Any]:
        """요약 메트릭 생성"""
        
        total_agents = len(self.agent_metrics)
        if total_agents == 0:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'completed_agents': 0,
                'error_agents': 0,
                'overall_progress': 0.0,
                'avg_execution_time': 0.0,
                'total_errors': 0,
                'error_rate': 0.0
            }
        
        # 상태별 집계
        state_counts = defaultdict(int)
        total_progress = 0.0
        total_execution_time = 0.0
        execution_count = 0
        total_errors = 0
        
        for metrics in self.agent_metrics.values():
            state_counts[metrics.current_state] += 1
            total_progress += metrics.progress_percentage
            
            if metrics.execution_duration > 0:
                total_execution_time += metrics.execution_duration
                execution_count += 1
            
            total_errors += metrics.error_count
        
        active_states = [
            AgentState.INITIALIZING, 
            AgentState.RUNNING, 
            AgentState.PROCESSING
        ]
        
        active_agents = sum(state_counts[state] for state in active_states)
        completed_agents = state_counts[AgentState.COMPLETED]
        error_agents = state_counts[AgentState.ERROR]
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'completed_agents': completed_agents,
            'error_agents': error_agents,
            'overall_progress': total_progress / total_agents if total_agents > 0 else 0.0,
            'avg_execution_time': total_execution_time / execution_count if execution_count > 0 else 0.0,
            'total_errors': total_errors,
            'error_rate': (error_agents / total_agents * 100) if total_agents > 0 else 0.0
        }
    
    def _generate_agent_cards(self) -> List[Dict[str, Any]]:
        """에이전트 카드 데이터 생성"""
        
        cards = []
        
        for agent_id, metrics in self.agent_metrics.items():
            card = {
                'agent_id': agent_id,
                'agent_name': metrics.agent_name,
                'state': metrics.current_state.value,
                'state_color': self.state_colors.get(metrics.current_state, "#95a5a6"),
                'progress': metrics.progress_percentage,
                'current_task': metrics.current_task,
                'execution_time': metrics.execution_duration,
                'error_count': metrics.error_count,
                'last_error': metrics.last_error,
                'artifacts_generated': metrics.artifacts_generated,
                'data_processed': metrics.data_processed
            }
            cards.append(card)
        
        # 상태별로 정렬 (에러 > 실행중 > 완료 > 대기)
        state_priority = {
            AgentState.ERROR: 0,
            AgentState.RUNNING: 1,
            AgentState.PROCESSING: 2,
            AgentState.INITIALIZING: 3,
            AgentState.COMPLETING: 4,
            AgentState.COMPLETED: 5,
            AgentState.IDLE: 6,
            AgentState.TIMEOUT: 7
        }
        
        cards.sort(key=lambda x: (
            state_priority.get(AgentState(x['state']), 99),
            -x['progress']
        ))
        
        return cards
    
    def _create_progress_chart(self) -> go.Figure:
        """진행률 차트 생성"""
        
        agent_names = []
        progress_values = []
        colors = []
        
        for metrics in self.agent_metrics.values():
            agent_names.append(metrics.agent_name)
            progress_values.append(metrics.progress_percentage)
            colors.append(self.state_colors.get(metrics.current_state, "#95a5a6"))
        
        fig = go.Figure(data=[
            go.Bar(
                x=progress_values,
                y=agent_names,
                orientation='h',
                marker_color=colors,
                text=[f'{p:.1f}%' for p in progress_values],
                textposition='inside'
            )
        ])
        
        fig.update_layout(
            height=max(300, len(agent_names) * 30),
            xaxis_title="진행률 (%)",
            yaxis_title="에이전트",
            showlegend=False,
            margin=dict(l=150, r=20, t=20, b=40)
        )
        
        fig.update_xaxis(range=[0, 100])
        
        return fig
    
    def _create_timeline_chart(self) -> go.Figure:
        """타임라인 차트 생성"""
        
        fig = go.Figure()
        
        # 에이전트별 실행 시간 막대
        for i, (agent_id, metrics) in enumerate(self.agent_metrics.items()):
            if metrics.start_time:
                start_time = metrics.start_time
                end_time = metrics.end_time or datetime.now()
                
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time, end_time, start_time, start_time],
                    y=[i-0.4, i-0.4, i+0.4, i+0.4, i-0.4],
                    fill='toself',
                    fillcolor=self.state_colors.get(metrics.current_state, "#95a5a6"),
                    line=dict(color='rgba(0,0,0,0)'),
                    name=metrics.agent_name,
                    hovertemplate=f"{metrics.agent_name}<br>시작: %{{x}}<br>상태: {metrics.current_state.value}"
                ))
        
        # 상태 변경 이벤트
        for event in self.state_history:
            if event['agent_id'] in self.agent_metrics:
                agent_idx = list(self.agent_metrics.keys()).index(event['agent_id'])
                
                fig.add_trace(go.Scatter(
                    x=[event['timestamp']],
                    y=[agent_idx],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.state_colors.get(event['new_state'], "#95a5a6"),
                        symbol='circle'
                    ),
                    showlegend=False,
                    hovertemplate=f"상태 변경: {event['old_state'].value} → {event['new_state'].value}<br>%{{x}}"
                ))
        
        fig.update_layout(
            height=max(300, len(self.agent_metrics) * 50),
            xaxis_title="시간",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(self.agent_metrics))),
                ticktext=[m.agent_name for m in self.agent_metrics.values()]
            ),
            showlegend=False,
            margin=dict(l=150, r=20, t=20, b=40)
        )
        
        return fig
    
    def _create_data_flow_diagram(self) -> go.Figure:
        """데이터 흐름 다이어그램 생성"""
        
        # 노드 위치 계산
        agent_positions = self._calculate_node_positions()
        
        fig = go.Figure()
        
        # 에이전트 노드
        for agent_id, pos in agent_positions.items():
            metrics = self.agent_metrics.get(agent_id)
            if metrics:
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        color=self.state_colors.get(metrics.current_state, "#95a5a6")
                    ),
                    text=[metrics.agent_name],
                    textposition='bottom center',
                    showlegend=False
                ))
        
        # 데이터 흐름 화살표
        for flow in self.data_flows[-20:]:  # 최근 20개만 표시
            if (flow.source_agent in agent_positions and 
                flow.target_agent in agent_positions):
                
                source_pos = agent_positions[flow.source_agent]
                target_pos = agent_positions[flow.target_agent]
                
                fig.add_annotation(
                    x=target_pos[0],
                    y=target_pos[1],
                    ax=source_pos[0],
                    ay=source_pos[1],
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#3498db',
                    opacity=0.6
                )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        return fig
    
    def create_workflow_diagram(self) -> go.Figure:
        """워크플로우 다이어그램 생성 - 에이전트 실행 순서 및 의존성 표시"""
        
        fig = go.Figure()
        
        # 위치 계산을 위한 레벨 할당
        agent_levels = self._calculate_workflow_levels()
        
        # 노드 위치 계산
        positions = {}
        level_counts = defaultdict(int)
        
        for agent_id, level in agent_levels.items():
            level_counts[level] += 1
        
        level_indices = defaultdict(int)
        
        for agent_id, level in agent_levels.items():
            idx = level_indices[level]
            total_in_level = level_counts[level]
            
            # 수평 위치 (레벨 내에서 고르게 분포)
            x = (idx + 1) / (total_in_level + 1)
            # 수직 위치 (레벨에 따라)
            y = 1 - (level / (max(agent_levels.values()) + 1))
            
            positions[agent_id] = (x, y)
            level_indices[level] += 1
        
        # 의존성 화살표 그리기
        for agent_id, metrics in self.agent_metrics.items():
            if agent_id in positions:
                for dep_id in metrics.depends_on:
                    if dep_id in positions:
                        source_pos = positions[dep_id]
                        target_pos = positions[agent_id]
                        
                        # 의존성 화살표
                        fig.add_trace(go.Scatter(
                            x=[source_pos[0], target_pos[0]],
                            y=[source_pos[1], target_pos[1]],
                            mode='lines',
                            line=dict(
                                color='#bdc3c7',
                                width=2,
                                dash='dot'
                            ),
                            showlegend=False,
                            hoverinfo='none'
                        ))
                        
                        # 화살표 머리
                        fig.add_annotation(
                            x=target_pos[0],
                            y=target_pos[1],
                            ax=source_pos[0],
                            ay=source_pos[1],
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='#bdc3c7',
                            opacity=0.8
                        )
        
        # 에이전트 노드 그리기
        for agent_id, pos in positions.items():
            metrics = self.agent_metrics.get(agent_id)
            if metrics:
                # 노드 색상 및 상태
                color = self.state_colors.get(metrics.current_state, "#95a5a6")
                
                # 노드
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=color,
                        line=dict(color='white', width=2)
                    ),
                    text=[metrics.agent_name],
                    textposition='bottom center',
                    name=metrics.agent_name,
                    hovertemplate=(
                        f"<b>{metrics.agent_name}</b><br>"
                        f"상태: {metrics.current_state.value}<br>"
                        f"진행률: {metrics.progress_percentage:.1f}%<br>"
                        f"의존성: {len(metrics.depends_on)}개<br>"
                        f"실행시간: {metrics.execution_duration:.1f}초<br>"
                        "<extra></extra>"
                    )
                ))
                
                # 진행률 표시 (노드 위에)
                if metrics.progress_percentage > 0:
                    fig.add_trace(go.Scatter(
                        x=[pos[0]],
                        y=[pos[1] + 0.05],
                        mode='text',
                        text=[f"{metrics.progress_percentage:.0f}%"],
                        textfont=dict(size=10, color='black'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
        
        # 병렬 실행 그룹 표시
        parallel_groups = self._identify_parallel_groups(agent_levels)
        
        for i, group in enumerate(parallel_groups):
            if len(group) > 1:
                # 그룹의 바운딩 박스 계산
                x_coords = [positions[aid][0] for aid in group if aid in positions]
                y_coords = [positions[aid][1] for aid in group if aid in positions]
                
                if x_coords and y_coords:
                    # 배경 사각형
                    fig.add_shape(
                        type="rect",
                        x0=min(x_coords) - 0.05,
                        y0=min(y_coords) - 0.05,
                        x1=max(x_coords) + 0.05,
                        y1=max(y_coords) + 0.05,
                        fillcolor="rgba(52, 152, 219, 0.1)",
                        line=dict(
                            color="rgba(52, 152, 219, 0.3)",
                            width=2,
                            dash="dash"
                        )
                    )
                    
                    # 병렬 그룹 라벨
                    fig.add_annotation(
                        x=(min(x_coords) + max(x_coords)) / 2,
                        y=max(y_coords) + 0.08,
                        text=f"병렬 그룹 {i+1}",
                        showarrow=False,
                        font=dict(size=10, color="#3498db")
                    )
        
        # 에러 발생 지점 표시
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.current_state == AgentState.ERROR and agent_id in positions:
                pos = positions[agent_id]
                
                # 에러 마커
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers',
                    marker=dict(
                        size=50,
                        color='rgba(231, 76, 60, 0.3)',
                        line=dict(color='#e74c3c', width=3)
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
        
        # 레이아웃 설정
        fig.update_layout(
            title="에이전트 워크플로우 다이어그램",
            height=600,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[-0.1, 1.1]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                range=[-0.1, 1.1]
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _calculate_workflow_levels(self) -> Dict[str, int]:
        """워크플로우 레벨 계산 (의존성 기반)"""
        
        levels = {}
        
        # 의존성이 없는 에이전트는 레벨 0
        for agent_id, metrics in self.agent_metrics.items():
            if not metrics.depends_on:
                levels[agent_id] = 0
        
        # 의존성이 있는 에이전트의 레벨 계산
        max_iterations = len(self.agent_metrics)
        iteration = 0
        
        while len(levels) < len(self.agent_metrics) and iteration < max_iterations:
            for agent_id, metrics in self.agent_metrics.items():
                if agent_id not in levels:
                    # 모든 의존성이 레벨이 할당되었는지 확인
                    deps_resolved = all(dep in levels for dep in metrics.depends_on)
                    
                    if deps_resolved:
                        # 의존성 중 최대 레벨 + 1
                        if metrics.depends_on:
                            max_dep_level = max(levels[dep] for dep in metrics.depends_on)
                            levels[agent_id] = max_dep_level + 1
                        else:
                            levels[agent_id] = 0
            
            iteration += 1
        
        # 레벨이 할당되지 않은 에이전트는 마지막 레벨로
        max_level = max(levels.values()) if levels else 0
        for agent_id in self.agent_metrics:
            if agent_id not in levels:
                levels[agent_id] = max_level + 1
        
        return levels
    
    def _identify_parallel_groups(self, agent_levels: Dict[str, int]) -> List[List[str]]:
        """동일 레벨의 병렬 실행 가능한 에이전트 그룹 식별"""
        
        # 레벨별로 그룹화
        level_groups = defaultdict(list)
        for agent_id, level in agent_levels.items():
            level_groups[level].append(agent_id)
        
        # 각 레벨에서 실제로 병렬 실행 가능한 그룹 찾기
        parallel_groups = []
        
        for level, agents in level_groups.items():
            if len(agents) > 1:
                # 동일 레벨 내에서 서로 의존성이 없는 에이전트들 그룹화
                independent_groups = self._find_independent_groups(agents)
                parallel_groups.extend(independent_groups)
        
        return parallel_groups
    
    def _find_independent_groups(self, agents: List[str]) -> List[List[str]]:
        """서로 독립적인 에이전트 그룹 찾기"""
        
        groups = []
        used = set()
        
        for agent in agents:
            if agent not in used:
                group = [agent]
                used.add(agent)
                
                # 이 에이전트와 독립적인 다른 에이전트 찾기
                for other in agents:
                    if other not in used:
                        metrics = self.agent_metrics.get(other)
                        if metrics and agent not in metrics.depends_on:
                            # 그룹 내 다른 에이전트와도 독립적인지 확인
                            independent = True
                            for g_agent in group:
                                g_metrics = self.agent_metrics.get(g_agent)
                                if g_metrics and other in g_metrics.depends_on:
                                    independent = False
                                    break
                            
                            if independent:
                                group.append(other)
                                used.add(other)
                
                if len(group) > 1:
                    groups.append(group)
        
        return groups
    
    def add_execution_log(self,
                         agent_id: str,
                         log_level: str,
                         message: str,
                         details: Dict[str, Any] = None,
                         execution_step: str = "",
                         duration_ms: float = 0.0,
                         memory_usage: float = 0.0,
                         artifacts_created: List[str] = None):
        """실행 로그 추가"""
        
        log_entry = ExecutionLogEntry(
            log_id=f"{agent_id}_{len(self.execution_logs)}",
            agent_id=agent_id,
            timestamp=datetime.now(),
            log_level=log_level,
            message=message,
            details=details or {},
            execution_step=execution_step,
            duration_ms=duration_ms,
            memory_usage=memory_usage,
            artifacts_created=artifacts_created or []
        )
        
        self.execution_logs.append(log_entry)
        
        # 요약 정보 업데이트
        self._update_agent_summary(agent_id, log_entry)
        
        logger.debug(f"📄 실행 로그 추가 - {agent_id}: {message}")
    
    def _update_agent_summary(self, agent_id: str, log_entry: ExecutionLogEntry):
        """에이전트 요약 업데이트"""
        
        if agent_id not in self.agent_summaries:
            metrics = self.agent_metrics.get(agent_id)
            self.agent_summaries[agent_id] = AgentExecutionSummary(
                agent_id=agent_id,
                agent_name=metrics.agent_name if metrics else agent_id,
                total_execution_time=0.0,
                total_steps=0,
                successful_steps=0,
                failed_steps=0,
                artifacts_generated=0,
                final_contribution_score=0.0
            )
        
        summary = self.agent_summaries[agent_id]
        
        # 단계 카운트 업데이트
        if log_entry.execution_step:
            summary.total_steps += 1
            
            if log_entry.log_level == "ERROR":
                summary.failed_steps += 1
                if log_entry.message not in summary.performance_issues:
                    summary.performance_issues.append(log_entry.message)
            else:
                summary.successful_steps += 1
        
        # 실행 시간 누적
        summary.total_execution_time += log_entry.duration_ms / 1000.0
        
        # 아티팩트 카운트
        summary.artifacts_generated += len(log_entry.artifacts_created)
        
        # 주요 성취 추가
        if log_entry.log_level == "INFO" and log_entry.artifacts_created:
            achievement = f"{len(log_entry.artifacts_created)}개 아티팩트 생성: {log_entry.execution_step}"
            if achievement not in summary.key_achievements:
                summary.key_achievements.append(achievement)
        
        # 기여도 점수 계산
        if summary.total_steps > 0:
            success_rate = summary.successful_steps / summary.total_steps
            artifact_score = min(1.0, summary.artifacts_generated / 10.0)  # 최대 10개 기준
            summary.final_contribution_score = (success_rate * 0.6 + artifact_score * 0.4)
    
    def render_execution_logs(self, container=None, agent_filter: str = None, limit: int = 100):
        """상세 실행 로그 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            st.markdown("### 📄 상세 실행 로그")
            
            # 필터 옵션
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_agent = st.selectbox(
                    "에이전트 필터",
                    options=["All"] + list(self.agent_metrics.keys()),
                    index=0
                )
            
            with col2:
                log_level_filter = st.selectbox(
                    "로그 레벨",
                    options=["All", "ERROR", "WARNING", "INFO", "DEBUG"]
                )
            
            with col3:
                max_logs = st.slider("최대 로그 개수", 10, 500, limit)
            
            # 로그 필터링
            filtered_logs = list(self.execution_logs)
            
            if selected_agent != "All":
                filtered_logs = [log for log in filtered_logs if log.agent_id == selected_agent]
            
            if log_level_filter != "All":
                filtered_logs = [log for log in filtered_logs if log.log_level == log_level_filter]
            
            # 최근 로그부터 표시
            filtered_logs = filtered_logs[-max_logs:]
            filtered_logs.reverse()  # 최신 로그를 위에
            
            if not filtered_logs:
                st.info("표시할 로그가 없습니다.")
                return
            
            # 로그 표시
            for log in filtered_logs:
                # 로그 레벨에 따른 아이콘 및 색상
                level_config = {
                    "ERROR": {"icon": "❌", "color": "#e74c3c"},
                    "WARNING": {"icon": "⚠️", "color": "#f39c12"},
                    "INFO": {"icon": "ℹ️", "color": "#3498db"},
                    "DEBUG": {"icon": "🔍", "color": "#95a5a6"}
                }
                
                config = level_config.get(log.log_level, level_config["INFO"])
                
                # 로그 엔트리 표시
                with st.expander(
                    f"{config['icon']} {log.timestamp.strftime('%H:%M:%S')} - "
                    f"{log.agent_id} - {log.message[:100]}{'...' if len(log.message) > 100 else ''}",
                    expanded=log.log_level == "ERROR"
                ):
                    # 기본 정보
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**에이전트:** {log.agent_id}")
                        st.markdown(f"**레벨:** {log.log_level}")
                    
                    with col2:
                        st.markdown(f"**시간:** {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        if log.execution_step:
                            st.markdown(f"**단계:** {log.execution_step}")
                    
                    with col3:
                        if log.duration_ms > 0:
                            st.markdown(f"**실행시간:** {log.duration_ms:.1f}ms")
                        if log.memory_usage > 0:
                            st.markdown(f"**메모리:** {log.memory_usage:.1f}MB")
                    
                    # 메시지
                    st.markdown(f"**메시지:**\n{log.message}")
                    
                    # 상세 정보
                    if log.details:
                        st.markdown("**상세 정보:**")
                        st.json(log.details)
                    
                    # 생성된 아티팩트
                    if log.artifacts_created:
                        st.markdown("**생성된 아티팩트:**")
                        for artifact in log.artifacts_created:
                            st.markdown(f"- {artifact}")
    
    def render_agent_summaries(self, container=None):
        """에이전트별 실행 요약 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            st.markdown("### 🏆 에이전트별 실행 요약")
            
            if not self.agent_summaries:
                st.info("아직 실행된 에이전트가 없습니다.")
                return
            
            # 기여도 순으로 정렬
            sorted_summaries = sorted(
                self.agent_summaries.values(),
                key=lambda s: s.final_contribution_score,
                reverse=True
            )
            
            for i, summary in enumerate(sorted_summaries):
                # 순위 배지
                rank_colors = {
                    0: "🥇",  # 1등 - 금메달
                    1: "🥈",  # 2등 - 은메달
                    2: "🥉"   # 3등 - 동메달
                }
                rank_icon = rank_colors.get(i, "🏅")
                
                with st.expander(
                    f"{rank_icon} {summary.agent_name} - 기여도: {summary.final_contribution_score:.1%}",
                    expanded=i < 3  # 상위 3개만 확장
                ):
                    # 메트릭 요약
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "실행시간",
                            f"{summary.total_execution_time:.1f}초"
                        )
                    
                    with col2:
                        st.metric(
                            "성공률",
                            f"{summary.successful_steps}/{summary.total_steps}",
                            f"{(summary.successful_steps/summary.total_steps*100) if summary.total_steps > 0 else 0:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "아티팩트",
                            f"{summary.artifacts_generated}개"
                        )
                    
                    with col4:
                        st.metric(
                            "기여도",
                            f"{summary.final_contribution_score:.1%}"
                        )
                    
                    # 주요 성취
                    if summary.key_achievements:
                        st.markdown("**주요 성취:**")
                        for achievement in summary.key_achievements[:5]:  # 최대 5개
                            st.markdown(f"- {achievement}")
                    
                    # 성능 이슈
                    if summary.performance_issues:
                        st.markdown("**성능 이슈:**")
                        for issue in summary.performance_issues[:3]:  # 최대 3개
                            st.error(f"⚠️ {issue}")
    
    def create_contribution_analysis_chart(self) -> go.Figure:
        """에이전트별 기여도 분석 차트"""
        
        if not self.agent_summaries:
            return go.Figure()
        
        # 데이터 준비
        agents = []
        contributions = []
        artifacts = []
        success_rates = []
        
        for summary in self.agent_summaries.values():
            agents.append(summary.agent_name)
            contributions.append(summary.final_contribution_score)
            artifacts.append(summary.artifacts_generated)
            success_rates.append(
                summary.successful_steps / summary.total_steps 
                if summary.total_steps > 0 else 0
            )
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '기여도 점수',
                '아티팩트 생성 수',
                '성공률',
                '종합 성능'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 기여도 점수
        fig.add_trace(
            go.Bar(
                x=agents,
                y=contributions,
                name='기얬도',
                marker_color='#3498db'
            ),
            row=1, col=1
        )
        
        # 아티팩트 생성 수
        fig.add_trace(
            go.Bar(
                x=agents,
                y=artifacts,
                name='아티팩트',
                marker_color='#2ecc71'
            ),
            row=1, col=2
        )
        
        # 성공률
        fig.add_trace(
            go.Bar(
                x=agents,
                y=[rate * 100 for rate in success_rates],
                name='성공률(%)',
                marker_color='#f39c12'
            ),
            row=2, col=1
        )
        
        # 종합 성능 (산점도)
        fig.add_trace(
            go.Scatter(
                x=artifacts,
                y=[rate * 100 for rate in success_rates],
                mode='markers+text',
                text=agents,
                textposition='top center',
                marker=dict(
                    size=[c * 50 + 10 for c in contributions],  # 기여도에 비례한 크기
                    color=contributions,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="기얬도")
                ),
                name='종합 성능'
            ),
            row=2, col=2
        )
        
        # 레이아웃 설정
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="에이전트별 기얬도 분석",
            title_x=0.5
        )
        
        # x축 라벨 회전
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    tickangle=45,
                    row=i, col=j
                )
        
        # y축 라벨
        fig.update_yaxes(title_text="기얬도", row=1, col=1)
        fig.update_yaxes(title_text="아티팩트 수", row=1, col=2)
        fig.update_yaxes(title_text="성공률 (%)", row=2, col=1)
        fig.update_yaxes(title_text="성공률 (%)", row=2, col=2)
        fig.update_xaxes(title_text="아티팩트 수", row=2, col=2)
        
        return fig
    
    def _render_agent_cards(self):
        """에이전트 카드 렌더링 (Streamlit)"""
        
        cards = self._generate_agent_cards()
        
        # 3열로 표시
        cols = st.columns(3)
        
        for i, card in enumerate(cards):
            with cols[i % 3]:
                with st.container():
                    # 카드 헤더
                    st.markdown(f"""
                    <div style="background-color: {card['state_color']}20; 
                                border-left: 4px solid {card['state_color']}; 
                                padding: 10px; margin-bottom: 10px;">
                        <h4 style="margin: 0;">{card['agent_name']}</h4>
                        <p style="margin: 0; color: {card['state_color']};">
                            {card['state']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 진행률
                    if card['progress'] > 0:
                        st.progress(card['progress'] / 100)
                        st.caption(f"진행률: {card['progress']:.1f}%")
                    
                    # 현재 작업
                    if card['current_task']:
                        st.caption(f"🔄 {card['current_task']}")
                    
                    # 메트릭
                    if card['execution_time'] > 0:
                        st.caption(f"⏱️ 실행시간: {card['execution_time']:.1f}초")
                    
                    if card['artifacts_generated'] > 0:
                        st.caption(f"📊 생성 아티팩트: {card['artifacts_generated']}개")
                    
                    # 에러 정보
                    if card['error_count'] > 0:
                        st.error(f"⚠️ 에러: {card['error_count']}건")
                        if card['last_error']:
                            st.caption(card['last_error'][:100])
    
    def _calculate_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """노드 위치 계산 (간단한 원형 레이아웃)"""
        
        positions = {}
        agent_ids = list(self.agent_metrics.keys())
        n = len(agent_ids)
        
        if n == 0:
            return positions
        
        import math
        
        for i, agent_id in enumerate(agent_ids):
            angle = 2 * math.pi * i / n
            x = math.cos(angle)
            y = math.sin(angle)
            positions[agent_id] = (x, y)
        
        return positions
    
    def _generate_timeline_chart(self) -> Dict[str, Any]:
        """타임라인 차트 데이터 생성"""
        
        timeline_data = {
            'agents': [],
            'events': []
        }
        
        for agent_id, metrics in self.agent_metrics.items():
            agent_timeline = {
                'agent_id': agent_id,
                'agent_name': metrics.agent_name,
                'start_time': metrics.start_time.isoformat() if metrics.start_time else None,
                'end_time': metrics.end_time.isoformat() if metrics.end_time else None,
                'duration': metrics.execution_duration,
                'state': metrics.current_state.value,
                'state_color': self.state_colors.get(metrics.current_state, "#95a5a6")
            }
            timeline_data['agents'].append(agent_timeline)
        
        # 최근 상태 변경 이벤트
        for event in list(self.state_history)[-50:]:  # 최근 50개
            timeline_data['events'].append({
                'timestamp': event['timestamp'].isoformat(),
                'agent_id': event['agent_id'],
                'old_state': event['old_state'].value,
                'new_state': event['new_state'].value
            })
        
        return timeline_data
    
    def _generate_progress_overview(self) -> Dict[str, Any]:
        """진행률 개요 생성"""
        
        # 전체 진행률 계산
        total_progress = 0.0
        agent_count = len(self.agent_metrics)
        
        if agent_count > 0:
            for metrics in self.agent_metrics.values():
                total_progress += metrics.progress_percentage
            total_progress /= agent_count
        
        # 단계별 진행률
        phase_progress = {
            'initialization': 0.0,
            'processing': 0.0,
            'completion': 0.0
        }
        
        for metrics in self.agent_metrics.values():
            if metrics.current_state in [AgentState.IDLE, AgentState.INITIALIZING]:
                phase_progress['initialization'] += metrics.progress_percentage
            elif metrics.current_state in [AgentState.RUNNING, AgentState.PROCESSING]:
                phase_progress['processing'] += metrics.progress_percentage
            else:
                phase_progress['completion'] += metrics.progress_percentage
        
        # 정규화
        if agent_count > 0:
            for phase in phase_progress:
                phase_progress[phase] /= agent_count
        
        return {
            'overall_progress': total_progress,
            'phase_progress': phase_progress,
            'active_agents': sum(1 for m in self.agent_metrics.values() 
                               if m.current_state in [AgentState.RUNNING, AgentState.PROCESSING]),
            'completed_agents': sum(1 for m in self.agent_metrics.values() 
                                  if m.current_state == AgentState.COMPLETED)
        }
    
    def _trigger_updates(self):
        """업데이트 콜백 트리거"""
        
        for callback in self.update_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"업데이트 콜백 오류: {e}")
    
    def add_update_callback(self, callback: Callable):
        """업데이트 콜백 추가"""
        
        self.update_callbacks.append(callback)
    
    def get_snapshot(self) -> CollaborationSnapshot:
        """현재 협업 상태 스냅샷"""
        
        agent_states = {
            agent_id: metrics.current_state
            for agent_id, metrics in self.agent_metrics.items()
        }
        
        active_flows = [
            flow for flow in self.data_flows
            if flow.status == "active" and 
            (datetime.now() - flow.timestamp).total_seconds() < 60  # 1분 이내
        ]
        
        summary = self._generate_summary_metrics()
        
        return CollaborationSnapshot(
            timestamp=datetime.now(),
            agent_states=agent_states,
            active_flows=active_flows,
            overall_progress=summary['overall_progress'],
            performance_metrics={
                'avg_execution_time': summary['avg_execution_time'],
                'error_rate': summary['error_rate'],
                'active_agents': summary['active_agents']
            }
        )
    
    def export_metrics(self) -> Dict[str, Any]:
        """메트릭 내보내기"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'agents': {
                agent_id: {
                    'name': metrics.agent_name,
                    'state': metrics.current_state.value,
                    'progress': metrics.progress_percentage,
                    'execution_time': metrics.execution_duration,
                    'errors': metrics.error_count,
                    'artifacts': metrics.artifacts_generated,
                    'data_processed': metrics.data_processed
                }
                for agent_id, metrics in self.agent_metrics.items()
            },
            'summary': self._generate_summary_metrics(),
            'data_flows': len(self.data_flows),
            'state_changes': len(self.state_history)
        }