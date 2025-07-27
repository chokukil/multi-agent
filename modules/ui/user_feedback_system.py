"""
사용자 피드백 시스템

이 모듈은 분석 과정별 명확한 상태 메시지, 예상 완료 시간 및 취소 기능,
만족도 조사 및 피드백 수집을 제공하는 사용자 피드백 시스템을 구현합니다.

주요 기능:
- 실시간 상태 메시지 및 진행률 표시
- 예상 완료 시간 계산 및 취소 기능
- 만족도 조사 및 피드백 수집
- 분석 단계별 상세 안내
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnalysisStage(Enum):
    """분석 단계"""
    INITIALIZING = "initializing"       # 초기화 중
    DATA_LOADING = "data_loading"       # 데이터 로딩
    AGENT_DISPATCH = "agent_dispatch"   # 에이전트 배치
    PROCESSING = "processing"           # 처리 중
    INTEGRATION = "integration"         # 통합 중
    FINAL_FORMATTING = "final_formatting"  # 최종 포맷팅
    COMPLETED = "completed"             # 완료
    ERROR = "error"                     # 에러
    CANCELLED = "cancelled"             # 취소됨

class SatisfactionLevel(Enum):
    """만족도 수준"""
    VERY_SATISFIED = 5      # 매우 만족
    SATISFIED = 4          # 만족
    NEUTRAL = 3            # 보통
    DISSATISFIED = 2       # 불만족
    VERY_DISSATISFIED = 1  # 매우 불만족

@dataclass
class AnalysisStatus:
    """분석 상태"""
    session_id: str
    current_stage: AnalysisStage
    progress_percentage: float  # 0.0 ~ 100.0
    
    # 시간 정보
    start_time: datetime
    estimated_completion_time: Optional[datetime] = None
    elapsed_time: float = 0.0
    remaining_time: float = 0.0
    
    # 상태 메시지
    status_message: str = ""
    detailed_message: str = ""
    current_agent: str = ""
    
    # 진행 정보
    completed_agents: int = 0
    total_agents: int = 0
    completed_steps: int = 0
    total_steps: int = 0
    
    # 메타데이터
    is_cancellable: bool = True
    show_cancel_confirmation: bool = False
    
@dataclass
class UserFeedback:
    """사용자 피드백"""
    feedback_id: str
    session_id: str
    timestamp: datetime
    
    # 만족도 평가
    overall_satisfaction: SatisfactionLevel
    ease_of_use: SatisfactionLevel
    result_quality: SatisfactionLevel
    response_time: SatisfactionLevel
    interface_quality: SatisfactionLevel
    
    # 텍스트 피드백
    positive_feedback: str = ""
    improvement_suggestions: str = ""
    additional_comments: str = ""
    
    # 사용 패턴
    session_duration: float = 0.0
    artifacts_viewed: int = 0
    interactions_count: int = 0
    
    # 기술적 정보
    user_agent: str = ""
    browser_info: Dict[str, Any] = field(default_factory=dict)

class UserFeedbackSystem:
    """사용자 피드백 시스템"""
    
    def __init__(self):
        # 상태 관리
        self.current_status: Optional[AnalysisStatus] = None
        self.status_history: deque = deque(maxlen=100)
        
        # 피드백 수집
        self.feedback_collection: List[UserFeedback] = []
        
        # 상태 메시지 템플릿
        self.status_messages = {
            AnalysisStage.INITIALIZING: {
                "title": "🚀 분석 시작",
                "message": "시스템을 초기화하고 있습니다",
                "detailed": "데이터를 검증하고 분석 환경을 준비중입니다."
            },
            AnalysisStage.DATA_LOADING: {
                "title": "📁 데이터 로딩",
                "message": "데이터를 불러오고 있습니다",
                "detailed": "업로드된 파일을 분석하고 데이터 품질을 확인중입니다."
            },
            AnalysisStage.AGENT_DISPATCH: {
                "title": "🤖 에이전트 배치",
                "message": "분석 에이전트들을 배치하고 있습니다",
                "detailed": "각 분석 영역별로 전문 에이전트를 할당하고 작업을 시작합니다."
            },
            AnalysisStage.PROCESSING: {
                "title": "⚡ 분석 중",
                "message": "데이터를 분석하고 있습니다",
                "detailed": "여러 에이전트가 동시에 데이터를 분석하고 인사이트를 도출중입니다."
            },
            AnalysisStage.INTEGRATION: {
                "title": "🔗 결과 통합",
                "message": "분석 결과를 통합하고 있습니다",
                "detailed": "각 에이전트의 결과를 검증하고 종합적인 인사이트를 생성중입니다."
            },
            AnalysisStage.FINAL_FORMATTING: {
                "title": "📝 최종 정리",
                "message": "최종 결과를 정리하고 있습니다",
                "detailed": "결과를 시각화하고 사용자 친화적인 형태로 포맷팅중입니다."
            },
            AnalysisStage.COMPLETED: {
                "title": "✅ 분석 완료",
                "message": "분석이 성공적으로 완료되었습니다",
                "detailed": "모든 분석이 완료되었습니다. 결과를 확인해보세요!"
            },
            AnalysisStage.ERROR: {
                "title": "❌ 오류 발생",
                "message": "분석 중 오류가 발생했습니다",
                "detailed": "문제를 해결하고 다시 시도해주세요."
            },
            AnalysisStage.CANCELLED: {
                "title": "🛑 분석 취소",
                "message": "사용자 요청으로 분석이 취소되었습니다",
                "detailed": "진행 중이던 분석이 안전하게 중단되었습니다."
            }
        }
        
        # 단계별 예상 소요 시간 (초)
        self.stage_durations = {
            AnalysisStage.INITIALIZING: 5,
            AnalysisStage.DATA_LOADING: 10,
            AnalysisStage.AGENT_DISPATCH: 8,
            AnalysisStage.PROCESSING: 120,  # 가장 긴 단계
            AnalysisStage.INTEGRATION: 30,
            AnalysisStage.FINAL_FORMATTING: 15
        }
        
        # 취소 핸들러
        self.cancel_handlers: List[Callable] = []
        
        # 성능 메트릭
        self.performance_metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'cancelled_sessions': 0,
            'avg_session_duration': 0.0,
            'avg_satisfaction': 0.0
        }
    
    def start_analysis(self, session_id: str, total_agents: int = 5, total_steps: int = 10) -> AnalysisStatus:
        """분석 시작"""
        
        status = AnalysisStatus(
            session_id=session_id,
            current_stage=AnalysisStage.INITIALIZING,
            progress_percentage=0.0,
            start_time=datetime.now(),
            total_agents=total_agents,
            total_steps=total_steps,
            status_message=self.status_messages[AnalysisStage.INITIALIZING]["message"],
            detailed_message=self.status_messages[AnalysisStage.INITIALIZING]["detailed"]
        )
        
        # 예상 완료 시간 계산
        total_estimated_duration = sum(self.stage_durations.values())
        status.estimated_completion_time = status.start_time + timedelta(seconds=total_estimated_duration)
        status.remaining_time = total_estimated_duration
        
        self.current_status = status
        self.status_history.append(status)
        
        logger.info(f"🚀 분석 시작 - 세션 {session_id}")
        
        return status
    
    def update_status(self,
                     stage: AnalysisStage = None,
                     progress: float = None,
                     current_agent: str = "",
                     completed_agents: int = None,
                     completed_steps: int = None,
                     custom_message: str = "") -> AnalysisStatus:
        """상태 업데이트"""
        
        if not self.current_status:
            logger.warning("분석이 시작되지 않았습니다")
            return None
        
        status = self.current_status
        
        # 단계 업데이트
        if stage and stage != status.current_stage:
            status.current_stage = stage
            status.status_message = custom_message or self.status_messages[stage]["message"]
            status.detailed_message = self.status_messages[stage]["detailed"]
            logger.info(f"📊 분석 단계 변경: {stage.value}")
            
        # 진행률 업데이트
        if progress is not None:
            status.progress_percentage = max(0.0, min(100.0, progress))
        
        # 에이전트 정보 업데이트
        if current_agent:
            status.current_agent = current_agent
            
        if completed_agents is not None:
            status.completed_agents = completed_agents
            
        if completed_steps is not None:
            status.completed_steps = completed_steps
        
        # 시간 정보 업데이트
        now = datetime.now()
        status.elapsed_time = (now - status.start_time).total_seconds()
        
        # 남은 시간 추정 (진행률 기반)
        if status.progress_percentage > 0:
            estimated_total_time = status.elapsed_time / (status.progress_percentage / 100.0)
            status.remaining_time = max(0, estimated_total_time - status.elapsed_time)
            status.estimated_completion_time = status.start_time + timedelta(seconds=estimated_total_time)
        
        # 취소 가능 여부 (통합 단계부터는 취소 불가)
        status.is_cancellable = stage not in [
            AnalysisStage.INTEGRATION,
            AnalysisStage.FINAL_FORMATTING,
            AnalysisStage.COMPLETED,
            AnalysisStage.ERROR,
            AnalysisStage.CANCELLED
        ]
        
        # 히스토리에 추가
        self.status_history.append(status)
        
        return status
    
    def cancel_analysis(self) -> bool:
        """분석 취소"""
        
        if not self.current_status or not self.current_status.is_cancellable:
            return False
        
        # 취소 핸들러 실행
        for handler in self.cancel_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"취소 핸들러 오류: {e}")
        
        # 상태 업데이트
        self.update_status(
            stage=AnalysisStage.CANCELLED,
            progress=0.0
        )
        
        # 메트릭 업데이트
        self.performance_metrics['cancelled_sessions'] += 1
        
        logger.info(f"🛑 분석 취소됨 - 세션 {self.current_status.session_id}")
        
        return True
    
    def complete_analysis(self) -> AnalysisStatus:
        """분석 완료"""
        
        if not self.current_status:
            return None
        
        status = self.update_status(
            stage=AnalysisStage.COMPLETED,
            progress=100.0
        )
        
        # 성능 메트릭 업데이트
        self.performance_metrics['total_sessions'] += 1
        self.performance_metrics['successful_sessions'] += 1
        
        session_duration = status.elapsed_time
        current_avg = self.performance_metrics['avg_session_duration']
        total_successful = self.performance_metrics['successful_sessions']
        
        # 평균 세션 시간 업데이트
        self.performance_metrics['avg_session_duration'] = (
            (current_avg * (total_successful - 1) + session_duration) / total_successful
        )
        
        logger.info(f"✅ 분석 완료 - 세션 {status.session_id}, 소요시간: {session_duration:.1f}초")
        
        return status
    
    def handle_error(self, error_message: str) -> AnalysisStatus:
        """에러 처리"""
        
        if not self.current_status:
            return None
        
        status = self.update_status(
            stage=AnalysisStage.ERROR,
            custom_message=f"오류: {error_message}"
        )
        
        logger.error(f"❌ 분석 오류 - 세션 {status.session_id}: {error_message}")
        
        return status
    
    def render_status_display(self, container=None):
        """상태 표시 렌더링"""
        
        if container is None:
            container = st.container()
        
        if not self.current_status:
            with container:
                st.info("분석을 시작하려면 데이터를 업로드하고 분석 버튼을 클릭하세요.")
            return
        
        status = self.current_status
        stage_config = self.status_messages[status.current_stage]
        
        with container:
            # 메인 상태 표시
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"## {stage_config['title']}")
                st.markdown(f"**{status.status_message}**")
                st.caption(status.detailed_message)
                
                if status.current_agent:
                    st.caption(f"🤖 현재 작업: {status.current_agent}")
            
            with col2:
                # 진행률 표시
                progress_color = self._get_progress_color(status.current_stage)
                st.metric(
                    "진행률",
                    f"{status.progress_percentage:.1f}%",
                    f"{status.completed_agents}/{status.total_agents} 에이전트"
                )
            
            with col3:
                # 시간 정보
                if status.remaining_time > 0:
                    remaining_str = self._format_duration(status.remaining_time)
                    st.metric("남은 시간", remaining_str)
                else:
                    elapsed_str = self._format_duration(status.elapsed_time)
                    st.metric("경과 시간", elapsed_str)
            
            # 진행률 바
            progress_bar = st.progress(status.progress_percentage / 100.0)
            
            # 단계별 진행 상황
            self._render_stage_progress()
            
            # 취소 버튼
            if status.is_cancellable and status.current_stage not in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED]:
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col2:
                    if st.button("🛑 분석 취소", type="secondary", use_container_width=True):
                        status.show_cancel_confirmation = True
                
                # 취소 확인 다이얼로그
                if status.show_cancel_confirmation:
                    with st.expander("⚠️ 정말로 분석을 취소하시겠습니까?", expanded=True):
                        st.warning("진행 중인 분석이 중단되고 결과를 받을 수 없습니다.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("예, 취소합니다", type="primary", use_container_width=True):
                                self.cancel_analysis()
                                st.experimental_rerun()
                        
                        with col2:
                            if st.button("아니오, 계속합니다", use_container_width=True):
                                status.show_cancel_confirmation = False
                                st.experimental_rerun()
    
    def render_feedback_collection(self, container=None):
        """피드백 수집 렌더링"""
        
        if container is None:
            container = st.container()
        
        if not self.current_status or self.current_status.current_stage != AnalysisStage.COMPLETED:
            return
        
        with container:
            st.markdown("## 📝 분석 결과에 대한 피드백")
            st.markdown("여러분의 소중한 의견을 들려주세요!")
            
            with st.form("feedback_form"):
                # 만족도 평가
                st.markdown("### 📊 만족도 평가")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    overall_satisfaction = st.select_slider(
                        "전반적 만족도",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        format_func=lambda x: ["매우 불만족", "불만족", "보통", "만족", "매우 만족"][x-1]
                    )
                    
                    ease_of_use = st.select_slider(
                        "사용 편의성",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        format_func=lambda x: ["매우 어려움", "어려움", "보통", "쉬움", "매우 쉬움"][x-1]
                    )
                
                with col2:
                    result_quality = st.select_slider(
                        "결과 품질",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        format_func=lambda x: ["매우 나쁨", "나쁨", "보통", "좋음", "매우 좋음"][x-1]
                    )
                    
                    response_time = st.select_slider(
                        "응답 속도",
                        options=[1, 2, 3, 4, 5],
                        value=4,
                        format_func=lambda x: ["매우 느림", "느림", "보통", "빠름", "매우 빠름"][x-1]
                    )
                
                interface_quality = st.select_slider(
                    "인터페이스 품질",
                    options=[1, 2, 3, 4, 5],
                    value=4,
                    format_func=lambda x: ["매우 나쁨", "나쁨", "보통", "좋음", "매우 좋음"][x-1]
                )
                
                # 텍스트 피드백
                st.markdown("### 💬 의견 및 제안")
                
                positive_feedback = st.text_area(
                    "좋았던 점이 있다면 알려주세요",
                    placeholder="분석 결과나 사용 경험에서 만족스러웠던 부분을 자유롭게 작성해주세요.",
                    height=100
                )
                
                improvement_suggestions = st.text_area(
                    "개선사항이 있다면 제안해주세요",
                    placeholder="더 나은 서비스를 위한 개선 아이디어나 불편했던 점을 알려주세요.",
                    height=100
                )
                
                additional_comments = st.text_area(
                    "기타 의견",
                    placeholder="추가로 전달하고 싶은 내용이 있으시면 자유롭게 작성해주세요.",
                    height=80
                )
                
                # 제출 버튼
                submitted = st.form_submit_button("📤 피드백 제출", type="primary", use_container_width=True)
                
                if submitted:
                    # 피드백 저장
                    feedback = UserFeedback(
                        feedback_id=f"feedback_{len(self.feedback_collection)}_{int(time.time())}",
                        session_id=self.current_status.session_id,
                        timestamp=datetime.now(),
                        overall_satisfaction=SatisfactionLevel(overall_satisfaction),
                        ease_of_use=SatisfactionLevel(ease_of_use),
                        result_quality=SatisfactionLevel(result_quality),
                        response_time=SatisfactionLevel(response_time),
                        interface_quality=SatisfactionLevel(interface_quality),
                        positive_feedback=positive_feedback,
                        improvement_suggestions=improvement_suggestions,
                        additional_comments=additional_comments,
                        session_duration=self.current_status.elapsed_time
                    )
                    
                    self.feedback_collection.append(feedback)
                    
                    # 평균 만족도 업데이트
                    self._update_satisfaction_metrics()
                    
                    st.success("✅ 피드백이 성공적으로 제출되었습니다. 소중한 의견 감사합니다!")
                    logger.info(f"📝 피드백 수집 완료 - 세션 {feedback.session_id}")
    
    def _render_stage_progress(self):
        """단계별 진행률 표시"""
        
        if not self.current_status:
            return
        
        stages = list(self.status_messages.keys())[:-3]  # ERROR, CANCELLED 제외
        current_stage_index = list(stages).index(self.current_status.current_stage) if self.current_status.current_stage in stages else -1
        
        # 단계 진행 표시
        cols = st.columns(len(stages))
        
        for i, (col, stage) in enumerate(zip(cols, stages)):
            with col:
                if i < current_stage_index:
                    # 완료된 단계
                    st.markdown(f"✅ **{self.status_messages[stage]['title']}**")
                elif i == current_stage_index:
                    # 현재 단계
                    st.markdown(f"🔄 **{self.status_messages[stage]['title']}**")
                else:
                    # 대기 중인 단계
                    st.markdown(f"⏳ {self.status_messages[stage]['title']}")
    
    def _get_progress_color(self, stage: AnalysisStage) -> str:
        """단계별 진행률 색상"""
        
        color_mapping = {
            AnalysisStage.INITIALIZING: "#3498db",
            AnalysisStage.DATA_LOADING: "#9b59b6",
            AnalysisStage.AGENT_DISPATCH: "#e67e22",
            AnalysisStage.PROCESSING: "#f39c12",
            AnalysisStage.INTEGRATION: "#2ecc71",
            AnalysisStage.FINAL_FORMATTING: "#1abc9c",
            AnalysisStage.COMPLETED: "#27ae60",
            AnalysisStage.ERROR: "#e74c3c",
            AnalysisStage.CANCELLED: "#95a5a6"
        }
        
        return color_mapping.get(stage, "#3498db")
    
    def _format_duration(self, seconds: float) -> str:
        """시간 포맷팅"""
        
        if seconds < 60:
            return f"{int(seconds)}초"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}분 {remaining_seconds}초"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}시간 {minutes}분"
    
    def _update_satisfaction_metrics(self):
        """만족도 메트릭 업데이트"""
        
        if not self.feedback_collection:
            return
        
        total_satisfaction = sum(
            feedback.overall_satisfaction.value
            for feedback in self.feedback_collection
        )
        
        self.performance_metrics['avg_satisfaction'] = total_satisfaction / len(self.feedback_collection)
    
    def add_cancel_handler(self, handler: Callable):
        """취소 핸들러 추가"""
        
        self.cancel_handlers.append(handler)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환"""
        
        return {
            'total_sessions': self.performance_metrics['total_sessions'],
            'success_rate': (
                self.performance_metrics['successful_sessions'] / 
                max(1, self.performance_metrics['total_sessions'])
            ),
            'cancellation_rate': (
                self.performance_metrics['cancelled_sessions'] / 
                max(1, self.performance_metrics['total_sessions'])
            ),
            'avg_session_duration': self.performance_metrics['avg_session_duration'],
            'avg_satisfaction': self.performance_metrics['avg_satisfaction'],
            'total_feedback_count': len(self.feedback_collection)
        }
    
    def export_feedback_data(self) -> List[Dict[str, Any]]:
        """피드백 데이터 내보내기"""
        
        return [
            {
                'feedback_id': feedback.feedback_id,
                'session_id': feedback.session_id,
                'timestamp': feedback.timestamp.isoformat(),
                'overall_satisfaction': feedback.overall_satisfaction.value,
                'ease_of_use': feedback.ease_of_use.value,
                'result_quality': feedback.result_quality.value,
                'response_time': feedback.response_time.value,
                'interface_quality': feedback.interface_quality.value,
                'positive_feedback': feedback.positive_feedback,
                'improvement_suggestions': feedback.improvement_suggestions,
                'additional_comments': feedback.additional_comments,
                'session_duration': feedback.session_duration
            }
            for feedback in self.feedback_collection
        ]