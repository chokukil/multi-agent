"""
Enhanced Real-time Agent Dashboard

실시간 에이전트 모니터링 및 제어 대시보드
- Capability Discovery: 에이전트 능력 자동 발견
- Real-time Observability: 실행 과정 실시간 가시화  
- Interruptibility: 작업 중단 및 제어 기능
- Cost-Aware: 토큰/비용 실시간 표시

Author: CherryAI Team
License: MIT License
"""

import streamlit as st
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
import json
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """에이전트 능력 정보"""
    name: str
    description: str
    skill_type: str
    confidence: float
    last_used: Optional[str] = None
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class AgentMetrics:
    """에이전트 메트릭"""
    agent_name: str
    status: str
    current_task: Optional[str] = None
    
    # 성능 메트릭
    response_time: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    
    # 비용 메트릭
    tokens_used: int = 0
    estimated_cost: float = 0.0
    
    # 실시간 상태
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_activity: Optional[str] = None
    
    # 업데이트 시간
    updated_at: str = None


@dataclass
class TaskExecution:
    """작업 실행 정보"""
    task_id: str
    agent_name: str
    task_description: str
    status: str
    start_time: str
    
    # 진행 상황
    progress: float = 0.0
    current_step: str = ""
    
    # 메트릭
    execution_time: float = 0.0
    tokens_consumed: int = 0
    
    # 결과
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    # 제어
    is_interruptible: bool = True
    can_pause: bool = False


class EnhancedAgentDashboard:
    """Enhanced Real-time Agent Dashboard"""
    
    def __init__(self):
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.capabilities_cache: Dict[str, List[AgentCapability]] = {}
        
        # 실시간 업데이트 제어
        self.is_monitoring = False
        self.update_interval = 2.0  # 초
        
        # 비용 설정
        self.token_costs = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
            "claude": 0.008 / 1000,  # $0.008 per 1K tokens
        }
        
        # UI 컴포넌트 초기화
        self._initialize_ui_components()
    
    def _initialize_ui_components(self):
        """UI 컴포넌트 초기화"""
        if 'enhanced_dashboard_state' not in st.session_state:
            st.session_state.enhanced_dashboard_state = {
                'selected_agents': [],
                'monitoring_enabled': False,
                'auto_refresh': True,
                'cost_threshold': 10.0,  # $10 알림 임계값
                'show_detailed_metrics': False
            }
    
    def render_dashboard(self):
        """메인 대시보드 렌더링"""
        st.title("🎛️ Enhanced Agent Dashboard")
        
        # 제어 패널
        self._render_control_panel()
        
        # 실시간 메트릭 요약
        self._render_metrics_summary()
        
        # 탭으로 구분된 상세 뷰
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Capability Discovery", 
            "👁️ Real-time Observability", 
            "⏸️ Task Control", 
            "💰 Cost Monitoring",
            "📊 Performance Analytics"
        ])
        
        with tab1:
            self._render_capability_discovery()
        
        with tab2:
            self._render_observability_view()
        
        with tab3:
            self._render_task_control()
        
        with tab4:
            self._render_cost_monitoring()
        
        with tab5:
            self._render_performance_analytics()
    
    def _render_control_panel(self):
        """제어 패널 렌더링"""
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                monitoring_enabled = st.checkbox(
                    "🔄 실시간 모니터링", 
                    value=st.session_state.enhanced_dashboard_state['monitoring_enabled'],
                    help="에이전트 활동을 실시간으로 모니터링합니다"
                )
                st.session_state.enhanced_dashboard_state['monitoring_enabled'] = monitoring_enabled
            
            with col2:
                auto_refresh = st.checkbox(
                    "🔃 자동 새로고침",
                    value=st.session_state.enhanced_dashboard_state['auto_refresh'],
                    help="대시보드를 자동으로 새로고침합니다"
                )
                st.session_state.enhanced_dashboard_state['auto_refresh'] = auto_refresh
            
            with col3:
                if st.button("🔍 에이전트 스캔"):
                    self._discover_agents()
            
            with col4:
                if st.button("🧹 캐시 정리"):
                    self._clear_cache()
    
    def _render_metrics_summary(self):
        """메트릭 요약 렌더링"""
        if not self.agent_metrics:
            st.info("📡 에이전트 데이터를 수집하고 있습니다...")
            return
        
        # 핵심 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_agents = len([m for m in self.agent_metrics.values() if m.status == "active"])
            st.metric("🤖 활성 에이전트", active_agents, len(self.agent_metrics))
        
        with col2:
            running_tasks = len([t for t in self.active_tasks.values() if t.status == "running"])
            st.metric("⚡ 실행 중인 작업", running_tasks, len(self.active_tasks))
        
        with col3:
            total_cost = sum(m.estimated_cost for m in self.agent_metrics.values())
            st.metric("💰 총 비용", f"${total_cost:.2f}", "오늘")
        
        with col4:
            avg_response = sum(m.response_time for m in self.agent_metrics.values()) / len(self.agent_metrics)
            st.metric("⏱️ 평균 응답시간", f"{avg_response:.1f}s", "실시간")
    
    def _render_capability_discovery(self):
        """능력 발견 렌더링"""
        st.header("🔍 Agent Capability Discovery")
        
        # 에이전트 능력 매트릭스
        if self.capabilities_cache:
            self._render_capability_matrix()
        else:
            st.info("에이전트 능력을 발견하고 있습니다...")
            if st.button("🔍 능력 스캔 시작"):
                with st.spinner("에이전트 능력을 분석하고 있습니다..."):
                    self._discover_agent_capabilities()
    
    def _render_capability_matrix(self):
        """능력 매트릭스 렌더링"""
        # 능력별 히트맵 데이터 준비
        skill_types = set()
        for caps in self.capabilities_cache.values():
            skill_types.update(cap.skill_type for cap in caps)
        
        skill_types = sorted(list(skill_types))
        agents = list(self.capabilities_cache.keys())
        
        # 매트릭스 생성
        matrix_data = []
        for agent in agents:
            row = []
            for skill in skill_types:
                # 해당 스킬의 평균 신뢰도 계산
                matching_caps = [cap for cap in self.capabilities_cache[agent] 
                               if cap.skill_type == skill]
                if matching_caps:
                    avg_confidence = sum(cap.confidence for cap in matching_caps) / len(matching_caps)
                    row.append(avg_confidence)
                else:
                    row.append(0.0)
            matrix_data.append(row)
        
        # 히트맵 생성
        fig = px.imshow(
            matrix_data,
            x=skill_types,
            y=agents,
            title="Agent Capability Heatmap",
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 능력 목록
        selected_agent = st.selectbox("상세 분석할 에이전트", agents)
        if selected_agent:
            self._render_agent_capabilities_detail(selected_agent)
    
    def _render_agent_capabilities_detail(self, agent_name: str):
        """에이전트 능력 상세 정보"""
        caps = self.capabilities_cache.get(agent_name, [])
        
        if not caps:
            st.warning(f"{agent_name}의 능력 정보가 없습니다.")
            return
        
        st.subheader(f"🤖 {agent_name} 능력 분석")
        
        # 능력 목록 테이블
        cap_data = []
        for cap in caps:
            cap_data.append({
                "능력명": cap.name,
                "설명": cap.description,
                "유형": cap.skill_type,
                "신뢰도": f"{cap.confidence:.1%}",
                "사용 횟수": cap.usage_count,
                "성공률": f"{cap.success_rate:.1%}",
                "마지막 사용": cap.last_used or "없음"
            })
        
        df = pd.DataFrame(cap_data)
        st.dataframe(df, use_container_width=True)
        
        # 능력 유형별 분포
        skill_counts = {}
        for cap in caps:
            skill_counts[cap.skill_type] = skill_counts.get(cap.skill_type, 0) + 1
        
        fig = px.pie(
            values=list(skill_counts.values()),
            names=list(skill_counts.keys()),
            title=f"{agent_name} 능력 유형 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_observability_view(self):
        """실시간 관찰가능성 뷰"""
        st.header("👁️ Real-time Observability")
        
        # 실시간 로그 스트림
        log_container = st.container()
        
        with log_container:
            st.subheader("📊 실시간 활동 피드")
            
            # 실시간 로그 표시를 위한 placeholder
            log_placeholder = st.empty()
            
            # 실시간 메트릭 차트
            self._render_realtime_metrics_chart()
            
            # 에이전트별 실행 상태
            self._render_agent_execution_status()
    
    def _render_realtime_metrics_chart(self):
        """실시간 메트릭 차트"""
        if not self.agent_metrics:
            st.info("메트릭 데이터를 수집하고 있습니다...")
            return
        
        # 응답시간 트렌드
        agents = list(self.agent_metrics.keys())
        response_times = [self.agent_metrics[agent].response_time for agent in agents]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agents,
            y=response_times,
            name="응답시간 (초)",
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="에이전트별 응답시간",
            xaxis_title="에이전트",
            yaxis_title="응답시간 (초)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_execution_status(self):
        """에이전트 실행 상태"""
        st.subheader("🔄 에이전트 실행 상태")
        
        for agent_name, metrics in self.agent_metrics.items():
            with st.expander(f"🤖 {agent_name} ({metrics.status})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("상태", metrics.status)
                    st.metric("총 요청", metrics.total_requests)
                    st.metric("성공률", f"{metrics.success_rate:.1%}")
                
                with col2:
                    st.metric("응답시간", f"{metrics.response_time:.1f}s")
                    st.metric("토큰 사용", metrics.tokens_used)
                    st.metric("예상 비용", f"${metrics.estimated_cost:.2f}")
                
                if metrics.current_task:
                    st.info(f"현재 작업: {metrics.current_task}")
                
                # 실시간 리소스 사용량
                col3, col4 = st.columns(2)
                with col3:
                    st.progress(metrics.cpu_usage / 100.0)
                    st.caption(f"CPU: {metrics.cpu_usage:.1f}%")
                
                with col4:
                    st.progress(metrics.memory_usage / 100.0)
                    st.caption(f"Memory: {metrics.memory_usage:.1f}%")
    
    def _render_task_control(self):
        """작업 제어 렌더링"""
        st.header("⏸️ Task Control & Interruptibility")
        
        if not self.active_tasks:
            st.info("현재 실행 중인 작업이 없습니다.")
            return
        
        # 실행 중인 작업 목록
        for task_id, task in self.active_tasks.items():
            with st.expander(f"📝 {task.task_description[:50]}...", expanded=True):
                self._render_task_control_panel(task)
    
    def _render_task_control_panel(self, task: TaskExecution):
        """개별 작업 제어 패널"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 진행 상황 표시
            st.progress(task.progress)
            st.caption(f"진행률: {task.progress:.1%} - {task.current_step}")
            
            # 작업 정보
            st.info(f"""
            **에이전트**: {task.agent_name}
            **시작 시간**: {task.start_time}
            **실행 시간**: {task.execution_time:.1f}초
            **토큰 소비**: {task.tokens_consumed}
            """)
        
        with col2:
            # 제어 버튼
            if task.is_interruptible:
                if st.button(f"⏸️ 중단", key=f"pause_{task.task_id}"):
                    self._interrupt_task(task.task_id)
                
                if task.can_pause and st.button(f"⏯️ 일시정지", key=f"pause_{task.task_id}"):
                    self._pause_task(task.task_id)
            else:
                st.warning("중단 불가능한 작업")
        
        with col3:
            # 상태 표시
            status_color = {
                "running": "🟢",
                "paused": "🟡", 
                "failed": "🔴",
                "completed": "✅"
            }
            
            st.markdown(f"**상태**: {status_color.get(task.status, '⚪')} {task.status}")
            
            if task.error:
                st.error(f"오류: {task.error}")
    
    def _render_cost_monitoring(self):
        """비용 모니터링 렌더링"""
        st.header("💰 Cost Monitoring & Awareness")
        
        # 비용 요약
        self._render_cost_summary()
        
        # 에이전트별 비용 분석
        self._render_agent_cost_analysis()
        
        # 비용 알림 설정
        self._render_cost_alert_settings()
    
    def _render_cost_summary(self):
        """비용 요약"""
        total_cost = sum(m.estimated_cost for m in self.agent_metrics.values())
        total_tokens = sum(m.tokens_used for m in self.agent_metrics.values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 총 비용", f"${total_cost:.2f}")
        
        with col2:
            st.metric("🪙 총 토큰", f"{total_tokens:,}")
        
        with col3:
            avg_cost = total_cost / len(self.agent_metrics) if self.agent_metrics else 0
            st.metric("📊 평균 비용", f"${avg_cost:.2f}")
        
        with col4:
            threshold = st.session_state.enhanced_dashboard_state['cost_threshold']
            remaining = max(0, threshold - total_cost)
            st.metric("⚠️ 임계값까지", f"${remaining:.2f}")
    
    def _render_agent_cost_analysis(self):
        """에이전트별 비용 분석"""
        if not self.agent_metrics:
            return
        
        # 비용 차트
        agents = list(self.agent_metrics.keys())
        costs = [self.agent_metrics[agent].estimated_cost for agent in agents]
        tokens = [self.agent_metrics[agent].tokens_used for agent in agents]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agents,
            y=costs,
            name="비용 ($)",
            yaxis="y1",
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Scatter(
            x=agents,
            y=tokens,
            mode='lines+markers',
            name="토큰 수",
            yaxis="y2",
            line=dict(color='lightblue')
        ))
        
        fig.update_layout(
            title="에이전트별 비용 및 토큰 사용량",
            xaxis_title="에이전트",
            yaxis=dict(title="비용 ($)", side="left"),
            yaxis2=dict(title="토큰 수", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_cost_alert_settings(self):
        """비용 알림 설정"""
        st.subheader("⚠️ 비용 알림 설정")
        
        current_threshold = st.session_state.enhanced_dashboard_state['cost_threshold']
        new_threshold = st.slider(
            "비용 임계값 ($)", 
            min_value=1.0, 
            max_value=100.0, 
            value=current_threshold,
            step=1.0
        )
        st.session_state.enhanced_dashboard_state['cost_threshold'] = new_threshold
        
        # 현재 총 비용
        total_cost = sum(m.estimated_cost for m in self.agent_metrics.values())
        
        if total_cost >= new_threshold:
            st.error(f"⚠️ 비용 임계값 초과! 현재: ${total_cost:.2f} / 임계값: ${new_threshold:.2f}")
        elif total_cost >= new_threshold * 0.8:
            st.warning(f"⚠️ 비용 임계값 80% 도달! 현재: ${total_cost:.2f} / 임계값: ${new_threshold:.2f}")
        else:
            st.success(f"✅ 비용 정상 범위 내 현재: ${total_cost:.2f} / 임계값: ${new_threshold:.2f}")
    
    def _render_performance_analytics(self):
        """성능 분석 렌더링"""
        st.header("📊 Performance Analytics")
        
        if not self.agent_metrics:
            st.info("성능 데이터를 수집하고 있습니다...")
            return
        
        # 성능 트렌드 차트
        self._render_performance_trends()
        
        # 에이전트 성능 비교
        self._render_agent_performance_comparison()
        
        # 성능 권장사항
        self._render_performance_recommendations()
    
    def _render_performance_trends(self):
        """성능 트렌드 차트"""
        # 시뮬레이션된 시계열 데이터 (실제로는 저장된 메트릭에서 가져와야 함)
        time_points = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                   end=datetime.now(), freq='5min')
        
        fig = go.Figure()
        
        for agent_name, metrics in self.agent_metrics.items():
            # 시뮬레이션된 응답시간 트렌드
            response_times = [metrics.response_time + (i % 3) * 0.1 for i in range(len(time_points))]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=response_times,
                mode='lines+markers',
                name=f"{agent_name} 응답시간",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="에이전트 성능 트렌드 (최근 1시간)",
            xaxis_title="시간",
            yaxis_title="응답시간 (초)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_performance_comparison(self):
        """에이전트 성능 비교"""
        # 성능 메트릭 테이블
        perf_data = []
        for agent_name, metrics in self.agent_metrics.items():
            perf_data.append({
                "에이전트": agent_name,
                "응답시간": f"{metrics.response_time:.1f}s",
                "성공률": f"{metrics.success_rate:.1%}",
                "총 요청": metrics.total_requests,
                "평균 비용": f"${metrics.estimated_cost:.3f}",
                "효율성 점수": self._calculate_efficiency_score(metrics)
            })
        
        df = pd.DataFrame(perf_data)
        st.dataframe(df, use_container_width=True)
        
        # 효율성 점수 차트
        fig = px.bar(
            df, 
            x="에이전트", 
            y="효율성 점수",
            title="에이전트 효율성 점수",
            color="효율성 점수",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_recommendations(self):
        """성능 권장사항"""
        st.subheader("💡 성능 최적화 권장사항")
        
        recommendations = self._generate_performance_recommendations()
        
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")
    
    def _calculate_efficiency_score(self, metrics: AgentMetrics) -> float:
        """효율성 점수 계산"""
        # 단순한 효율성 점수 공식 (실제로는 더 복잡한 로직 필요)
        if metrics.response_time == 0:
            return 0.0
        
        time_score = max(0, 10 - metrics.response_time)  # 빠를수록 높은 점수
        success_score = metrics.success_rate * 5  # 성공률이 높을수록 높은 점수
        cost_score = max(0, 5 - metrics.estimated_cost * 100)  # 비용이 낮을수록 높은 점수
        
        return min(10.0, time_score + success_score + cost_score)
    
    def _generate_performance_recommendations(self) -> List[str]:
        """성능 권장사항 생성"""
        recommendations = []
        
        if not self.agent_metrics:
            return ["에이전트 데이터를 수집한 후 권장사항을 제공하겠습니다."]
        
        # 느린 에이전트 식별
        slow_agents = [name for name, metrics in self.agent_metrics.items() 
                      if metrics.response_time > 5.0]
        if slow_agents:
            recommendations.append(
                f"응답시간이 느린 에이전트들을 최적화하세요: {', '.join(slow_agents)}"
            )
        
        # 높은 비용 에이전트 식별
        expensive_agents = [name for name, metrics in self.agent_metrics.items() 
                           if metrics.estimated_cost > 1.0]
        if expensive_agents:
            recommendations.append(
                f"비용이 높은 에이전트들의 토큰 사용을 최적화하세요: {', '.join(expensive_agents)}"
            )
        
        # 낮은 성공률 에이전트 식별
        unreliable_agents = [name for name, metrics in self.agent_metrics.items() 
                            if metrics.success_rate < 0.9]
        if unreliable_agents:
            recommendations.append(
                f"성공률이 낮은 에이전트들을 점검하세요: {', '.join(unreliable_agents)}"
            )
        
        if not recommendations:
            recommendations.append("🎉 모든 에이전트가 최적 상태로 작동하고 있습니다!")
        
        return recommendations
    
    # Helper Methods
    def _discover_agents(self):
        """에이전트 자동 발견"""
        st.info("🔍 에이전트를 발견하고 있습니다...")
        
        # 시뮬레이션된 에이전트 발견 (실제로는 A2A 프로토콜로 검색)
        discovered_agents = [
            "DataLoaderAgent", "DataCleaningAgent", "EDAAgent", 
            "VisualizationAgent", "MLAgent", "PandasAgent"
        ]
        
        for agent_name in discovered_agents:
            self.agent_metrics[agent_name] = AgentMetrics(
                agent_name=agent_name,
                status="discovered",
                response_time=2.0 + len(agent_name) * 0.1,
                success_rate=0.95,
                total_requests=0,
                tokens_used=0,
                estimated_cost=0.0,
                updated_at=datetime.now().isoformat()
            )
        
        st.success(f"✅ {len(discovered_agents)}개 에이전트를 발견했습니다!")
    
    def _discover_agent_capabilities(self):
        """에이전트 능력 발견"""
        for agent_name in self.agent_metrics.keys():
            # 시뮬레이션된 능력 데이터
            capabilities = [
                AgentCapability(
                    name=f"{agent_name}_capability_1",
                    description=f"{agent_name}의 주요 기능",
                    skill_type="data_processing",
                    confidence=0.9
                ),
                AgentCapability(
                    name=f"{agent_name}_capability_2", 
                    description=f"{agent_name}의 보조 기능",
                    skill_type="analysis",
                    confidence=0.8
                )
            ]
            self.capabilities_cache[agent_name] = capabilities
    
    def _interrupt_task(self, task_id: str):
        """작업 중단"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = "interrupted"
            st.success(f"✅ 작업 {task_id}이 중단되었습니다.")
    
    def _pause_task(self, task_id: str):
        """작업 일시정지"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = "paused"
            st.success(f"⏸️ 작업 {task_id}이 일시정지되었습니다.")
    
    def _clear_cache(self):
        """캐시 정리"""
        self.capabilities_cache.clear()
        self.agent_metrics.clear()
        self.active_tasks.clear()
        st.success("🧹 캐시가 정리되었습니다.")


# 전역 대시보드 인스턴스
_dashboard_instance = None

def get_enhanced_agent_dashboard() -> EnhancedAgentDashboard:
    """Enhanced Agent Dashboard 싱글톤 인스턴스 반환"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = EnhancedAgentDashboard()
    return _dashboard_instance


# Streamlit 앱으로 실행할 때
if __name__ == "__main__":
    dashboard = get_enhanced_agent_dashboard()
    dashboard.render_dashboard() 