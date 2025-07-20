"""
Realtime Analysis Progress - 실시간 분석 진행 상황 표시

요구사항 3.4에 따른 구현:
- Universal Engine 메타 추론 단계별 진행 상황
- A2A 에이전트별 작업 상태 실시간 모니터링
- 분석 성능 메트릭 실시간 표시
- 진행률 시각화 및 예상 완료 시간
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import json
import time
import threading
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComponentType(Enum):
    """컴포넌트 타입"""
    META_REASONING = "meta_reasoning"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_SELECTION = "agent_selection"
    WORKFLOW_EXECUTION = "workflow_execution"
    RESULT_INTEGRATION = "result_integration"
    RESPONSE_GENERATION = "response_generation"


@dataclass
class ProgressUpdate:
    """진행 상황 업데이트"""
    component: ComponentType
    stage: str
    status: TaskStatus
    progress_percent: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    estimated_completion: Optional[str] = None


@dataclass
class AgentTaskInfo:
    """에이전트 작업 정보"""
    agent_id: str
    agent_name: str
    task_description: str
    status: TaskStatus
    progress_percent: float
    start_time: str
    estimated_completion: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class RealtimeAnalysisProgress:
    """
    실시간 분석 진행 상황 모니터링
    - Universal Engine 각 단계별 진행 상황 시각화
    - A2A 에이전트별 작업 상태 추적
    - 성능 메트릭 실시간 표시
    - 예상 완료 시간 계산
    """
    
    def __init__(self):
        """RealtimeAnalysisProgress 초기화"""
        self.progress_history: List[ProgressUpdate] = []
        self.agent_tasks: Dict[str, AgentTaskInfo] = {}
        self.start_time: Optional[datetime] = None
        self.is_monitoring = False
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_task_duration": 0.0,
            "current_throughput": 0.0
        }
        logger.info("RealtimeAnalysisProgress initialized")
    
    def start_monitoring(self, total_components: int = 6):
        """모니터링 시작"""
        self.start_time = datetime.now()
        self.is_monitoring = True
        self.performance_metrics["total_tasks"] = total_components
        
        # 세션 상태 초기화
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = {
                "overall_progress": 0.0,
                "current_stage": "초기화",
                "component_progress": {},
                "agent_status": {},
                "performance_metrics": self.performance_metrics.copy()
            }
        
        logger.info(f"Monitoring started for {total_components} components")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"Monitoring stopped. Total duration: {duration:.2f}s")
    
    def update_progress(self, update: ProgressUpdate):
        """진행 상황 업데이트"""
        if not self.is_monitoring:
            return
        
        self.progress_history.append(update)
        
        # 세션 상태 업데이트
        if 'progress_data' in st.session_state:
            st.session_state.progress_data["component_progress"][update.component.value] = {
                "stage": update.stage,
                "status": update.status.value,
                "progress": update.progress_percent,
                "message": update.message,
                "timestamp": update.timestamp
            }
            
            # 전체 진행률 계산
            self._update_overall_progress()
    
    def update_agent_task(self, agent_task: AgentTaskInfo):
        """에이전트 작업 상태 업데이트"""
        self.agent_tasks[agent_task.agent_id] = agent_task
        
        # 세션 상태 업데이트
        if 'progress_data' in st.session_state:
            st.session_state.progress_data["agent_status"][agent_task.agent_id] = {
                "name": agent_task.agent_name,
                "task": agent_task.task_description,
                "status": agent_task.status.value,
                "progress": agent_task.progress_percent,
                "start_time": agent_task.start_time,
                "estimated_completion": agent_task.estimated_completion,
                "metrics": agent_task.performance_metrics
            }
    
    def render_progress_dashboard(self):
        """📊 진행 상황 대시보드 렌더링"""
        if not self.is_monitoring and not st.session_state.get('progress_data'):
            st.info("분석이 시작되지 않았습니다.")
            return
        
        progress_data = st.session_state.get('progress_data', {})
        
        # 헤더 섹션
        st.markdown("## 📊 실시간 분석 진행 상황")
        
        # 전체 진행률
        self._render_overall_progress(progress_data)
        
        # 컴포넌트별 진행 상황
        self._render_component_progress(progress_data)
        
        # A2A 에이전트 상태
        self._render_agent_status(progress_data)
        
        # 성능 메트릭
        self._render_performance_metrics(progress_data)
        
        # 시간 정보
        self._render_timing_information(progress_data)
    
    def _render_overall_progress(self, progress_data: Dict):
        """전체 진행률 표시"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            overall_progress = progress_data.get("overall_progress", 0.0)
            st.progress(overall_progress / 100)
            st.caption(f"전체 진행률: {overall_progress:.1f}%")
        
        with col2:
            current_stage = progress_data.get("current_stage", "대기 중")
            st.metric("현재 단계", current_stage)
        
        with col3:
            if self.start_time and self.is_monitoring:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                st.metric("경과 시간", f"{elapsed:.1f}초")
            else:
                st.metric("상태", "완료" if progress_data else "대기")
    
    def _render_component_progress(self, progress_data: Dict):
        """컴포넌트별 진행 상황 표시"""
        st.markdown("### 🧩 단계별 진행 상황")
        
        component_progress = progress_data.get("component_progress", {})
        
        if not component_progress:
            st.info("단계별 진행 정보가 없습니다.")
            return
        
        # 컴포넌트 정보
        component_info = {
            "meta_reasoning": ("🧠 메타 추론", "사용자 요청 분석 및 전략 수립"),
            "agent_discovery": ("🔍 에이전트 발견", "사용 가능한 A2A 에이전트 스캔"),
            "agent_selection": ("🎯 에이전트 선택", "최적 에이전트 조합 결정"),
            "workflow_execution": ("⚡ 워크플로우 실행", "선택된 에이전트들과 협업"),
            "result_integration": ("🔄 결과 통합", "다중 에이전트 결과 병합"),
            "response_generation": ("📝 응답 생성", "최종 사용자 응답 생성")
        }
        
        for component_key, (name, description) in component_info.items():
            component_data = component_progress.get(component_key, {})
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    status = component_data.get("status", "pending")
                    status_icon = {
                        "pending": "⏳",
                        "running": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                        "cancelled": "🚫"
                    }.get(status, "❓")
                    
                    st.write(f"{status_icon} **{name}**")
                    st.caption(description)
                
                with col2:
                    progress = component_data.get("progress", 0.0)
                    st.progress(progress / 100)
                    st.caption(f"{progress:.1f}%")
                
                with col3:
                    stage = component_data.get("stage", "대기")
                    st.write(f"**{stage}**")
                
                with col4:
                    timestamp = component_data.get("timestamp", "")
                    if timestamp:
                        time_str = self._format_timestamp(timestamp)
                        st.caption(time_str)
                
                # 상세 메시지
                message = component_data.get("message", "")
                if message:
                    st.caption(f"📋 {message}")
                
                st.markdown("---")
    
    def _render_agent_status(self, progress_data: Dict):
        """A2A 에이전트 상태 표시"""
        st.markdown("### 🤖 A2A 에이전트 활동")
        
        agent_status = progress_data.get("agent_status", {})
        
        if not agent_status:
            st.info("활성 에이전트가 없습니다.")
            return
        
        # 에이전트 상태 요약
        active_agents = len([a for a in agent_status.values() if a.get("status") == "running"])
        completed_agents = len([a for a in agent_status.values() if a.get("status") == "completed"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 에이전트", len(agent_status))
        with col2:
            st.metric("활성 에이전트", active_agents)
        with col3:
            st.metric("완료 에이전트", completed_agents)
        
        # 개별 에이전트 상태
        for agent_id, agent_data in agent_status.items():
            with st.expander(f"🤖 {agent_data.get('name', agent_id)}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    status = agent_data.get("status", "unknown")
                    status_colors = {
                        "pending": "🟡",
                        "running": "🟢",
                        "completed": "✅",
                        "failed": "🔴",
                        "cancelled": "⚫"
                    }
                    st.write(f"**상태:** {status_colors.get(status, '❓')} {status}")
                    
                    task = agent_data.get("task", "알 수 없음")
                    st.write(f"**작업:** {task}")
                    
                    progress = agent_data.get("progress", 0.0)
                    st.progress(progress / 100)
                    st.caption(f"진행률: {progress:.1f}%")
                
                with col2:
                    start_time = agent_data.get("start_time", "")
                    if start_time:
                        st.write(f"**시작 시간:** {self._format_timestamp(start_time)}")
                    
                    estimated_completion = agent_data.get("estimated_completion", "")
                    if estimated_completion:
                        st.write(f"**예상 완료:** {self._format_timestamp(estimated_completion)}")
                    
                    # 성능 메트릭
                    metrics = agent_data.get("metrics", {})
                    if metrics:
                        st.write("**성능 메트릭:**")
                        for metric_name, value in metrics.items():
                            st.write(f"• {metric_name}: {value}")
    
    def _render_performance_metrics(self, progress_data: Dict):
        """성능 메트릭 표시"""
        st.markdown("### 📈 성능 메트릭")
        
        metrics = progress_data.get("performance_metrics", {})
        
        if not metrics:
            st.info("성능 메트릭이 없습니다.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tasks = metrics.get("total_tasks", 0)
            completed_tasks = metrics.get("completed_tasks", 0)
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            st.metric("완료율", f"{completion_rate:.1f}%", f"{completed_tasks}/{total_tasks}")
        
        with col2:
            failed_tasks = metrics.get("failed_tasks", 0)
            error_rate = (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            st.metric("오류율", f"{error_rate:.1f}%", f"{failed_tasks}개 실패")
        
        with col3:
            avg_duration = metrics.get("average_task_duration", 0.0)
            st.metric("평균 작업 시간", f"{avg_duration:.2f}초")
        
        with col4:
            throughput = metrics.get("current_throughput", 0.0)
            st.metric("현재 처리율", f"{throughput:.2f}/s")
    
    def _render_timing_information(self, progress_data: Dict):
        """시간 정보 표시"""
        if not self.start_time:
            return
        
        st.markdown("### ⏰ 시간 정보")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            st.metric("경과 시간", f"{elapsed:.1f}초")
        
        with col2:
            # 예상 완료 시간 계산
            overall_progress = progress_data.get("overall_progress", 0.0)
            if overall_progress > 0:
                estimated_total = elapsed / (overall_progress / 100)
                remaining = estimated_total - elapsed
                st.metric("예상 남은 시간", f"{remaining:.1f}초")
            else:
                st.metric("예상 남은 시간", "계산 중...")
        
        with col3:
            if overall_progress > 0:
                estimated_completion = self.start_time + timedelta(seconds=elapsed / (overall_progress / 100))
                st.metric("예상 완료 시각", estimated_completion.strftime("%H:%M:%S"))
            else:
                st.metric("예상 완료 시각", "계산 중...")
    
    def render_progress_timeline(self):
        """📅 진행 상황 타임라인 렌더링"""
        st.markdown("### 📅 진행 상황 타임라인")
        
        if not self.progress_history:
            st.info("진행 상황 이력이 없습니다.")
            return
        
        # 타임라인 시각화
        timeline_data = []
        for update in self.progress_history[-20:]:  # 최근 20개만 표시
            timeline_data.append({
                "시간": self._format_timestamp(update.timestamp),
                "컴포넌트": update.component.value,
                "단계": update.stage,
                "상태": update.status.value,
                "진행률": f"{update.progress_percent:.1f}%",
                "메시지": update.message[:50] + "..." if len(update.message) > 50 else update.message
            })
        
        if timeline_data:
            import pandas as pd
            df = pd.DataFrame(timeline_data)
            st.dataframe(df, use_container_width=True, height=300)
    
    def create_progress_stream(self) -> AsyncGenerator[ProgressUpdate, None]:
        """진행 상황 스트림 생성"""
        async def progress_generator():
            while self.is_monitoring:
                if self.progress_history:
                    yield self.progress_history[-1]
                await asyncio.sleep(0.5)  # 0.5초마다 업데이트
        
        return progress_generator()
    
    def _update_overall_progress(self):
        """전체 진행률 업데이트"""
        if 'progress_data' not in st.session_state:
            return
        
        component_progress = st.session_state.progress_data["component_progress"]
        
        if not component_progress:
            return
        
        # 모든 컴포넌트의 평균 진행률 계산
        total_progress = sum(comp.get("progress", 0.0) for comp in component_progress.values())
        component_count = len(component_progress)
        
        overall_progress = total_progress / component_count if component_count > 0 else 0.0
        st.session_state.progress_data["overall_progress"] = overall_progress
        
        # 현재 단계 업데이트
        running_components = [
            comp for comp in component_progress.values() 
            if comp.get("status") == "running"
        ]
        
        if running_components:
            current_stage = running_components[0].get("stage", "진행 중")
            st.session_state.progress_data["current_stage"] = current_stage
        elif overall_progress >= 100.0:
            st.session_state.progress_data["current_stage"] = "완료"
    
    def _format_timestamp(self, timestamp: str) -> str:
        """타임스탬프 포맷팅"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """진행 상황 요약"""
        if not self.progress_history:
            return {"message": "No progress data available"}
        
        total_components = len(set(update.component for update in self.progress_history))
        completed_components = len(set(
            update.component for update in self.progress_history 
            if update.status == TaskStatus.COMPLETED
        ))
        
        elapsed_time = 0.0
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_components": total_components,
            "completed_components": completed_components,
            "completion_rate": (completed_components / total_components * 100) if total_components > 0 else 0,
            "elapsed_time": elapsed_time,
            "total_updates": len(self.progress_history),
            "active_agents": len([t for t in self.agent_tasks.values() if t.status == TaskStatus.RUNNING]),
            "performance_metrics": self.performance_metrics.copy()
        }
    
    def export_progress_report(self) -> Dict[str, Any]:
        """진행 상황 리포트 내보내기"""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": datetime.now().isoformat()
            },
            "progress_history": [
                {
                    "component": update.component.value,
                    "stage": update.stage,
                    "status": update.status.value,
                    "progress": update.progress_percent,
                    "message": update.message,
                    "timestamp": update.timestamp
                }
                for update in self.progress_history
            ],
            "agent_tasks": {
                agent_id: {
                    "name": task.agent_name,
                    "task": task.task_description,
                    "status": task.status.value,
                    "progress": task.progress_percent,
                    "performance": task.performance_metrics
                }
                for agent_id, task in self.agent_tasks.items()
            },
            "summary": self.get_progress_summary()
        }