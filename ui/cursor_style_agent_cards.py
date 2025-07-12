"""
🎨 Cursor Style Agent Cards - Cursor 벤치마킹 실시간 에이전트 상태 카드

Cursor의 우아한 접힌/펼친 카드 UI를 CherryAI에 적용:
- 접힌 상태: 간결한 요약 정보 (에이전트명, 상태, 시간)
- 펼친 상태: 세부 진행 단계와 실시간 로그
- 실시간 애니메이션: 상태 변화와 진행률 시각화
- 반응형 디자인: 모바일과 데스크탑 모두 지원

Author: CherryAI Team  
License: MIT License
"""

import streamlit as st
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid

from ui.cursor_theme_system import get_cursor_theme, apply_cursor_theme


@dataclass
class AgentStep:
    """에이전트 실행 단계"""
    step_id: str
    icon: str
    name: str
    description: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    details: List[str] = None
    result: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []

    @property
    def duration(self) -> float:
        """단계 실행 시간 (초)"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def status_emoji(self) -> str:
        """상태별 이모지"""
        return {
            'pending': '⏳',
            'running': '🔄', 
            'completed': '✅',
            'failed': '❌'
        }.get(self.status, '❓')


@dataclass  
class AgentCard:
    """Cursor 스타일 에이전트 카드"""
    agent_id: str
    agent_name: str
    agent_icon: str
    status: str  # 'thinking', 'working', 'completed', 'failed', 'waiting'
    current_task: str
    start_time: float
    
    # 진행 단계들
    steps: List[AgentStep] = None
    
    # 실시간 데이터
    progress: float = 0.0  # 0.0 ~ 1.0
    current_step_id: Optional[str] = None
    
    # UI 상태
    is_expanded: bool = False
    
    # 결과 데이터
    artifacts: List[Dict] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.artifacts is None:
            self.artifacts = []

    @property
    def elapsed_time(self) -> float:
        """경과 시간 (초)"""
        return time.time() - self.start_time

    @property
    def status_emoji(self) -> str:
        """상태별 이모지"""
        return {
            'thinking': '💭',
            'working': '🔄',
            'completed': '✅', 
            'failed': '❌',
            'waiting': '⏱️'
        }.get(self.status, '❓')

    @property
    def status_color(self) -> str:
        """상태별 색상"""
        return {
            'thinking': '#fd7e14',  # orange
            'working': '#007acc',   # blue
            'completed': '#28a745', # green
            'failed': '#dc3545',    # red
            'waiting': '#6c757d'    # gray
        }.get(self.status, '#6c757d')

    def get_current_step(self) -> Optional[AgentStep]:
        """현재 실행 중인 단계"""
        if self.current_step_id:
            for step in self.steps:
                if step.step_id == self.current_step_id:
                    return step
        return None

    def add_step(self, icon: str, name: str, description: str) -> str:
        """새 단계 추가"""
        step_id = str(uuid.uuid4())
        step = AgentStep(
            step_id=step_id,
            icon=icon,
            name=name, 
            description=description,
            status='pending'
        )
        self.steps.append(step)
        return step_id

    def start_step(self, step_id: str):
        """단계 시작"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'running'
                step.start_time = time.time()
                self.current_step_id = step_id
                break

    def complete_step(self, step_id: str, result: Optional[str] = None):
        """단계 완료"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'completed'
                step.end_time = time.time()
                if result:
                    step.result = result
                break
        
        # 다음 단계가 있으면 자동 시작
        step_index = next((i for i, s in enumerate(self.steps) if s.step_id == step_id), -1)
        if step_index >= 0 and step_index < len(self.steps) - 1:
            next_step = self.steps[step_index + 1]
            self.start_step(next_step.step_id)
        
        # 진행률 업데이트
        completed_steps = sum(1 for step in self.steps if step.status == 'completed')
        self.progress = completed_steps / len(self.steps) if self.steps else 0.0

    def fail_step(self, step_id: str, error: str):
        """단계 실패"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'failed'
                step.end_time = time.time()
                break
        self.status = 'failed'
        self.error_message = error

    def add_step_detail(self, step_id: str, detail: str):
        """단계 세부사항 추가"""
        for step in self.steps:
            if step.step_id == step_id:
                step.details.append(f"[{datetime.now().strftime('%H:%M:%S')}] {detail}")
                break


class CursorStyleAgentCards:
    """Cursor 스타일 에이전트 카드 매니저"""
    
    def __init__(self):
        self.cards: Dict[str, AgentCard] = {}
        self._initialize_session_state()

    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'cursor_agent_cards' not in st.session_state:
            st.session_state.cursor_agent_cards = {}
        if 'expanded_cards' not in st.session_state:
            st.session_state.expanded_cards = set()

    def create_agent_card(self, agent_name: str, agent_icon: str, current_task: str) -> str:
        """새 에이전트 카드 생성"""
        # 세션 상태 재초기화 확인
        self._initialize_session_state()
        
        agent_id = str(uuid.uuid4())
        card = AgentCard(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_icon=agent_icon,
            status='thinking',
            current_task=current_task,
            start_time=time.time()
        )
        self.cards[agent_id] = card
        
        # 안전한 세션 상태 접근
        try:
            st.session_state.cursor_agent_cards[agent_id] = asdict(card)
        except AttributeError:
            self._initialize_session_state()
            st.session_state.cursor_agent_cards[agent_id] = asdict(card)
        
        return agent_id

    def update_card_status(self, agent_id: str, status: str, current_task: str = None):
        """카드 상태 업데이트"""
        if agent_id in self.cards:
            self.cards[agent_id].status = status
            if current_task:
                self.cards[agent_id].current_task = current_task
            
            # 안전한 세션 상태 접근
            try:
                st.session_state.cursor_agent_cards[agent_id] = asdict(self.cards[agent_id])
            except (AttributeError, KeyError):
                self._initialize_session_state()
                st.session_state.cursor_agent_cards[agent_id] = asdict(self.cards[agent_id])

    def render_cards_container(self):
        """카드 컨테이너 렌더링"""
        if not self.cards:
            st.info("🤖 아직 활성화된 에이전트가 없습니다.")
            return

        st.markdown("### 🤖 Active Agents")
        
        # 커스텀 CSS 적용
        self._apply_cursor_styles()
        
        # 카드들을 상태별로 정렬 (진행 중 > 완료 > 실패)
        sorted_cards = sorted(
            self.cards.values(),
            key=lambda x: (
                0 if x.status in ['thinking', 'working'] else 
                1 if x.status == 'completed' else 2
            )
        )
        
        for card in sorted_cards:
            self._render_agent_card(card)

    def _apply_cursor_styles(self):
        """Cursor 스타일 CSS 적용"""
        # 새로운 통합 테마 시스템 사용
        apply_cursor_theme()
        
        # 에이전트 카드 특화 스타일
        st.markdown("""
        <style>
        /* 에이전트 카드 전용 스타일 */
        .cursor-agent-card {
            background: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 8px;
            margin: 10px 0;
            transition: var(--cursor-transition);
        }
        
        .cursor-agent-card:hover {
            border-color: var(--cursor-accent-blue);
            box-shadow: 0 4px 12px rgba(0, 122, 204, 0.15);
        }
        
        .cursor-agent-card.thinking {
            border-left: 4px solid var(--cursor-accent-blue);
            animation: cursor-pulse 2s infinite;
        }
        
        .cursor-agent-card.working {
            border-left: 4px solid #2e7d32;
            animation: cursor-glow 3s ease-in-out infinite;
        }
        
        .cursor-agent-card.completed {
            border-left: 4px solid #388e3c;
        }
        
        .cursor-agent-card.failed {
            border-left: 4px solid #d32f2f;
        }
        
        .cursor-agent-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem;
            border-bottom: 1px solid var(--cursor-border-light);
            cursor: pointer;
        }
        
        .cursor-agent-header:hover {
            background: var(--cursor-tertiary-bg);
        }
        
        .cursor-agent-info {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .cursor-agent-icon {
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--cursor-tertiary-bg);
        }
        
        .cursor-agent-details h4 {
            margin: 0;
            color: var(--cursor-primary-text);
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .cursor-agent-details p {
            margin: 0.25rem 0 0 0;
            color: var(--cursor-muted-text);
            font-size: 0.9rem;
        }
        
        .cursor-agent-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .cursor-expand-btn {
            background: none;
            border: none;
            color: var(--cursor-accent-blue);
            cursor: pointer;
            padding: 0.25rem;
            border-radius: 4px;
            transition: var(--cursor-transition);
        }
        
        .cursor-expand-btn:hover {
            background: var(--cursor-accent-blue);
            color: var(--cursor-primary-text);
        }
        
        .cursor-agent-content {
            padding: 1rem;
            border-top: 1px solid var(--cursor-border-light);
            animation: cursor-slide-in 0.3s ease;
        }
        
        .cursor-step-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .cursor-step-item {
            display: flex;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--cursor-border-light);
        }
        
        .cursor-step-item:last-child {
            border-bottom: none;
        }
        
        .cursor-step-icon {
            margin-right: 0.5rem;
            font-size: 1rem;
        }
        
        .cursor-step-content {
            flex: 1;
        }
        
        .cursor-step-name {
            font-weight: 500;
            color: var(--cursor-primary-text);
            margin: 0;
        }
        
        .cursor-step-description {
            color: var(--cursor-muted-text);
            font-size: 0.85rem;
            margin: 0.25rem 0 0 0;
        }
        
        .cursor-step-status {
            margin-left: 0.5rem;
        }
        
        .cursor-step-item.pending {
            opacity: 0.7;
        }
        
        .cursor-step-item.running {
            background: rgba(0, 122, 204, 0.1);
            border-radius: 4px;
            padding: 0.5rem;
            margin: 0.25rem 0;
        }
        
        .cursor-step-item.completed {
            opacity: 0.8;
        }
        
        .cursor-step-item.failed {
            background: rgba(211, 47, 47, 0.1);
            border-radius: 4px;
            padding: 0.5rem;
            margin: 0.25rem 0;
        }
        
        .cursor-agent-metrics {
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--cursor-border-light);
        }
        
        .cursor-metric {
            text-align: center;
        }
        
        .cursor-metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--cursor-primary-text);
            margin: 0;
        }
        
        .cursor-metric-label {
            font-size: 0.75rem;
            color: var(--cursor-muted-text);
            margin: 0.25rem 0 0 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_agent_card(self, card: AgentCard):
        """개별 에이전트 카드 렌더링"""
        # 안전한 세션 상태 접근
        try:
            is_expanded = card.agent_id in st.session_state.expanded_cards
        except (AttributeError, KeyError):
            self._initialize_session_state()
            is_expanded = False
        
        # 카드 컨테이너
        with st.container():
            # 접힌 상태 헤더 렌더링
            header_col1, header_col2 = st.columns([1, 20])
            
            with header_col1:
                # 확장/축소 버튼
                expand_icon = "▲" if is_expanded else "▼"
                if st.button(expand_icon, key=f"expand_{card.agent_id}", help="상세 정보 보기/숨기기"):
                    try:
                        if is_expanded:
                            st.session_state.expanded_cards.discard(card.agent_id)
                        else:
                            st.session_state.expanded_cards.add(card.agent_id)
                        st.rerun()
                    except (AttributeError, KeyError):
                        self._initialize_session_state()
                        st.session_state.expanded_cards.add(card.agent_id)
                        st.rerun()
            
            with header_col2:
                # 에이전트 헤더 정보
                self._render_card_header(card)
        
        # 펼친 상태 세부 정보
        if is_expanded:
            self._render_card_details(card)

    def _render_card_header(self, card: AgentCard):
        """카드 헤더 렌더링"""
        col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
        
        with col1:
            st.markdown(f"<div class='agent-icon'>{card.agent_icon}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{card.agent_name}**")
            st.markdown(f"<small>{card.current_task}</small>", unsafe_allow_html=True)
        
        with col3:
            # 상태 표시
            status_class = f"status-{card.status}"
            st.markdown(f"""
            <div class="agent-status {status_class}">
                {card.status_emoji} {card.status.title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # 경과 시간 및 진행률
            elapsed = card.elapsed_time
            minutes, seconds = divmod(elapsed, 60)
            time_str = f"{int(minutes)}:{int(seconds):02d}"
            
            st.markdown(f"<div class='elapsed-time'>⏱️ {time_str}</div>", unsafe_allow_html=True)
            
            if card.progress > 0:
                st.progress(card.progress, text=f"{card.progress*100:.0f}%")

    def _render_card_details(self, card: AgentCard):
        """카드 세부 정보 렌더링"""
        with st.container():
            st.markdown("---")
            
            # 단계별 진행 상황
            if card.steps:
                st.markdown("**📋 실행 단계**")
                
                for step in card.steps:
                    self._render_step_item(step)
            
            # 생성된 아티팩트
            if card.artifacts:
                st.markdown("**🎯 생성된 결과**")
                for i, artifact in enumerate(card.artifacts):
                    with st.expander(f"결과 {i+1}: {artifact.get('name', 'Unnamed')}"):
                        self._render_artifact(artifact)
            
            # 에러 메시지
            if card.error_message:
                st.error(f"❌ {card.error_message}")

    def _render_step_item(self, step: AgentStep):
        """단계 아이템 렌더링"""
        col1, col2, col3 = st.columns([1, 6, 2])
        
        with col1:
            st.markdown(f"<div class='step-icon'>{step.status_emoji}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{step.name}**")
            if step.description:
                st.markdown(f"<small>{step.description}</small>", unsafe_allow_html=True)
            
            # 단계 세부사항
            if step.details:
                with st.expander("세부 로그", expanded=False):
                    for detail in step.details[-5:]:  # 최근 5개만 표시
                        st.text(detail)
        
        with col3:
            if step.duration > 0:
                st.markdown(f"<div class='step-time'>{step.duration:.1f}s</div>", unsafe_allow_html=True)

    def _render_artifact(self, artifact: Dict):
        """아티팩트 렌더링"""
        artifact_type = artifact.get('type', 'text')
        content = artifact.get('content', '')
        
        if artifact_type == 'text':
            st.markdown(content)
        elif artifact_type == 'code':
            st.code(content, language=artifact.get('language', 'python'))
        elif artifact_type == 'data':
            # 데이터프레임 표시
            if isinstance(content, dict) and 'dataframe' in content:
                st.dataframe(content['dataframe'])
        elif artifact_type == 'chart':
            # 차트 표시
            st.plotly_chart(content, use_container_width=True)

    def get_card(self, agent_id: str) -> Optional[AgentCard]:
        """카드 조회"""
        return self.cards.get(agent_id)

    def remove_card(self, agent_id: str):
        """카드 제거"""
        if agent_id in self.cards:
            del self.cards[agent_id]
        
        # 안전한 세션 상태 접근
        try:
            if agent_id in st.session_state.cursor_agent_cards:
                del st.session_state.cursor_agent_cards[agent_id]
            st.session_state.expanded_cards.discard(agent_id)
        except (AttributeError, KeyError):
            self._initialize_session_state()


# 전역 인스턴스
_cursor_cards_instance = None

def get_cursor_agent_cards() -> CursorStyleAgentCards:
    """Cursor 스타일 에이전트 카드 싱글톤 인스턴스 반환"""
    global _cursor_cards_instance
    if _cursor_cards_instance is None:
        _cursor_cards_instance = CursorStyleAgentCards()
    return _cursor_cards_instance 