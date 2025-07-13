#!/usr/bin/env python3
"""
🎭 Streaming Orchestrator

Streamlit UI와 통합 메시지 브로커를 연결하는 실시간 스트리밍 오케스트레이터
ChatGPT/Claude 스타일의 실시간 채팅 UX 제공

Features:
- Streamlit 실시간 연동
- 통합 메시지 브로커 통신
- ChatGPT/Claude 스타일 UX
- 실시간 스트리밍 처리
- 세션 및 상태 관리
- 에러 처리 및 복구
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st

# 스트리밍 컴포넌트들
from .unified_message_broker import get_unified_message_broker, UnifiedMessage, MessagePriority
from ui.streaming.realtime_chat_container import RealtimeChatContainer
from ui.components.unified_chat_interface import UnifiedChatInterface

logger = logging.getLogger(__name__)

class ChatStyle(Enum):
    """채팅 인터페이스 스타일"""
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CUSTOM = "custom"

class StreamingStatus(Enum):
    """스트리밍 상태"""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class StreamingSession:
    """스트리밍 세션 정보"""
    session_id: str
    broker_session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: StreamingStatus = StreamingStatus.IDLE
    total_messages: int = 0
    active_streams: int = 0
    error_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    chat_style: ChatStyle = ChatStyle.CHATGPT
    enable_typing_indicator: bool = True
    enable_progress_bar: bool = True
    auto_scroll: bool = True
    max_retries: int = 3
    stream_timeout: int = 60
    chunk_size: int = 1024
    show_agent_names: bool = True
    show_timestamps: bool = False

class StreamingOrchestrator:
    """스트리밍 오케스트레이터"""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.unified_broker = get_unified_message_broker()
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.chat_containers: Dict[str, RealtimeChatContainer] = {}
        self.chat_interfaces: Dict[str, UnifiedChatInterface] = {}
        
        # 스트리밍 통계
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_messages': 0,
            'successful_streams': 0,
            'failed_streams': 0,
            'uptime_start': datetime.now()
        }
    
    async def initialize(self):
        """오케스트레이터 초기화"""
        logger.info("🎭 Streaming Orchestrator 초기화 시작...")
        
        # 브로커 초기화
        await self.unified_broker.initialize()
        
        logger.info("✅ Streaming Orchestrator 초기화 완료")
    
    def get_or_create_session(
        self, 
        user_id: str, 
        session_key: Optional[str] = None
    ) -> StreamingSession:
        """세션 가져오기 또는 생성"""
        
        if session_key is None:
            session_key = f"streamlit_{user_id}_{int(time.time())}"
        
        if session_key not in self.active_sessions:
            # 새 세션 생성
            session = StreamingSession(
                session_id=session_key,
                broker_session_id=f"broker_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.active_sessions[session_key] = session
            self.stats['total_sessions'] += 1
            self.stats['active_sessions'] = len(self.active_sessions)
            
            logger.info(f"🆕 스트리밍 세션 생성: {session_key}")
        
        return self.active_sessions[session_key]
    
    def get_chat_container(self, session_id: str) -> RealtimeChatContainer:
        """채팅 컨테이너 가져오기 또는 생성"""
        if session_id not in self.chat_containers:
            self.chat_containers[session_id] = RealtimeChatContainer()
        return self.chat_containers[session_id]
    
    def get_chat_interface(self, session_id: str) -> UnifiedChatInterface:
        """채팅 인터페이스 가져오기 또는 생성"""
        if session_id not in self.chat_interfaces:
            self.chat_interfaces[session_id] = UnifiedChatInterface()
        return self.chat_interfaces[session_id]
    
    async def stream_user_query(
        self,
        session_id: str,
        user_query: str,
        capabilities: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """사용자 쿼리 스트리밍 처리"""
        
        session = self.active_sessions.get(session_id)
        if not session:
            yield {
                'type': 'error',
                'content': f'Session not found: {session_id}',
                'final': True
            }
            return
        
        try:
            session.status = StreamingStatus.CONNECTING
            session.active_streams += 1
            session.total_messages += 1
            session.last_activity = datetime.now()
            
            # 브로커 세션 생성 (필요시)
            await self.unified_broker.create_session(user_query, session.broker_session_id)
            
            # 시작 이벤트
            yield {
                'type': 'stream_start',
                'content': {
                    'session_id': session_id,
                    'query': user_query,
                    'capabilities': capabilities
                },
                'final': False
            }
            
            session.status = StreamingStatus.STREAMING
            
            # 멀티 에이전트 쿼리 오케스트레이션
            async for event in self.unified_broker.orchestrate_multi_agent_query(
                session.broker_session_id,
                user_query,
                capabilities
            ):
                # 브로커 이벤트를 UI 이벤트로 변환
                ui_event = await self._convert_broker_event_to_ui(event, session)
                yield ui_event
                
                if ui_event.get('final'):
                    break
            
            session.status = StreamingStatus.COMPLETED
            self.stats['successful_streams'] += 1
            
            # 완료 이벤트
            yield {
                'type': 'stream_complete',
                'content': {
                    'session_id': session_id,
                    'total_messages': session.total_messages,
                    'duration': (datetime.now() - session.created_at).total_seconds()
                },
                'final': True
            }
            
        except Exception as e:
            logger.error(f"❌ 스트리밍 처리 오류: {e}")
            session.status = StreamingStatus.ERROR
            session.error_count += 1
            self.stats['failed_streams'] += 1
            
            yield {
                'type': 'error',
                'content': {
                    'error': str(e),
                    'session_id': session_id,
                    'retry_count': session.error_count
                },
                'final': True
            }
        
        finally:
            session.active_streams = max(0, session.active_streams - 1)
            session.last_activity = datetime.now()
    
    async def _convert_broker_event_to_ui(
        self, 
        broker_event: Dict[str, Any], 
        session: StreamingSession
    ) -> Dict[str, Any]:
        """브로커 이벤트를 UI 이벤트로 변환"""
        
        event_type = broker_event.get('event', 'unknown')
        event_data = broker_event.get('data', {})
        
        # 기본 UI 이벤트 구조
        ui_event = {
            'type': 'agent_response',
            'content': {},
            'final': event_data.get('final', False),
            'timestamp': datetime.now().isoformat()
        }
        
        # 이벤트 타입별 변환
        if event_type == 'orchestration_start':
            ui_event.update({
                'type': 'orchestration_start',
                'content': {
                    'message': '🤖 AI 에이전트들이 협업을 시작합니다...',
                    'agents': event_data.get('selected_agents', []),
                    'capabilities': event_data.get('capabilities', [])
                }
            })
        
        elif event_type == 'routing':
            if self.config.show_agent_names:
                ui_event.update({
                    'type': 'agent_routing',
                    'content': {
                        'message': f"🔄 {event_data.get('to', '에이전트')}가 처리 중...",
                        'from_agent': event_data.get('from'),
                        'to_agent': event_data.get('to'),
                        'agent_type': event_data.get('type')
                    }
                })
        
        elif event_type in ['a2a_response', 'mcp_sse_response', 'mcp_stdio_response']:
            # 에이전트 응답 처리
            agent_id = event_data.get('agent_id', 'unknown')
            content = event_data.get('content', {})
            
            # A2A 응답 파싱
            if event_type == 'a2a_response' and isinstance(content, dict):
                text_content = content.get('content', '')
                if isinstance(text_content, str):
                    try:
                        parsed_content = json.loads(text_content)
                        text_content = parsed_content
                    except json.JSONDecodeError:
                        pass
                
                ui_event['content'] = {
                    'agent_id': agent_id,
                    'text': text_content,
                    'message_type': 'agent_response'
                }
            
            # MCP 응답 처리
            elif event_type in ['mcp_sse_response', 'mcp_stdio_response']:
                mcp_content = content.get('content', {})
                
                if event_type == 'mcp_stdio_response':
                    # STDIO 응답의 중첩 구조 처리
                    stdio_data = mcp_content.get('data', {})
                    if 'result' in stdio_data:
                        text_content = stdio_data['result']
                    elif 'output' in stdio_data:
                        text_content = stdio_data['output']
                    else:
                        text_content = str(mcp_content)
                else:
                    text_content = mcp_content
                
                ui_event['content'] = {
                    'agent_id': agent_id,
                    'text': text_content,
                    'message_type': 'mcp_response'
                }
        
        elif event_type.endswith('_error'):
            ui_event.update({
                'type': 'error',
                'content': {
                    'error': event_data.get('error', 'Unknown error'),
                    'agent_id': event_data.get('agent_id'),
                    'message': f"❌ {event_data.get('agent_id', '에이전트')} 오류: {event_data.get('error', 'Unknown error')}"
                }
            })
        
        return ui_event
    
    def render_streaming_chat(
        self,
        session_id: str,
        container_key: str = "main_chat"
    ):
        """Streamlit에서 스트리밍 채팅 인터페이스 렌더링"""
        
        # 채팅 컨테이너 가져오기
        chat_container = self.get_chat_container(session_id)
        chat_interface = self.get_chat_interface(session_id)
        
        # 채팅 인터페이스 렌더링
        with st.container():
            # 채팅 스타일 적용
            if self.config.chat_style == ChatStyle.CHATGPT:
                st.markdown("""
                <style>
                .chatgpt-style {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                </style>
                """, unsafe_allow_html=True)
            
            # 채팅 기록 표시
            chat_container.render_chat_history()
            
            # 입력 인터페이스
            chat_interface.render_chat_input(
                key=f"chat_input_{container_key}",
                placeholder="AI 에이전트들에게 질문하세요...",
                on_submit=lambda query: self._handle_user_input(session_id, query)
            )
            
            # 상태 표시
            session = self.active_sessions.get(session_id)
            if session and session.status != StreamingStatus.IDLE:
                self._render_status_indicator(session)
    
    def _handle_user_input(self, session_id: str, user_query: str):
        """사용자 입력 처리"""
        if not user_query.strip():
            return
        
        # 채팅 컨테이너에 사용자 메시지 추가
        chat_container = self.get_chat_container(session_id)
        chat_container.add_user_message(user_query)
        
        # 스트리밍 처리를 위한 플레이스홀더 생성
        response_placeholder = chat_container.add_assistant_message("🤖 처리 중...")
        
        # 비동기 스트리밍 처리 시작
        asyncio.create_task(
            self._process_streaming_response(session_id, user_query, response_placeholder)
        )
    
    async def _process_streaming_response(
        self,
        session_id: str,
        user_query: str,
        response_placeholder
    ):
        """스트리밍 응답 처리"""
        
        accumulated_response = ""
        chat_container = self.get_chat_container(session_id)
        
        try:
            async for event in self.stream_user_query(session_id, user_query):
                event_type = event.get('type')
                content = event.get('content', {})
                
                if event_type == 'agent_response':
                    # 에이전트 응답 누적
                    text = content.get('text', '')
                    if isinstance(text, str):
                        accumulated_response += text
                    elif isinstance(text, dict):
                        accumulated_response += json.dumps(text, ensure_ascii=False, indent=2)
                    
                    # 실시간 업데이트
                    chat_container.update_streaming_message(
                        response_placeholder,
                        accumulated_response
                    )
                
                elif event_type == 'orchestration_start':
                    chat_container.update_streaming_message(
                        response_placeholder,
                        content.get('message', '처리 중...')
                    )
                
                elif event_type == 'agent_routing' and self.config.show_agent_names:
                    routing_message = content.get('message', '')
                    chat_container.update_streaming_message(
                        response_placeholder,
                        routing_message
                    )
                
                elif event_type == 'error':
                    error_message = f"❌ 오류: {content.get('error', 'Unknown error')}"
                    chat_container.update_streaming_message(
                        response_placeholder,
                        error_message
                    )
                
                if event.get('final'):
                    break
            
            # 최종 응답 완료
            if accumulated_response:
                chat_container.finalize_streaming_message(response_placeholder)
            
        except Exception as e:
            logger.error(f"❌ 스트리밍 응답 처리 오류: {e}")
            chat_container.update_streaming_message(
                response_placeholder,
                f"❌ 처리 중 오류가 발생했습니다: {str(e)}"
            )
            chat_container.finalize_streaming_message(response_placeholder)
    
    def _render_status_indicator(self, session: StreamingSession):
        """상태 표시기 렌더링"""
        
        status_colors = {
            StreamingStatus.IDLE: "🟢",
            StreamingStatus.CONNECTING: "🟡", 
            StreamingStatus.STREAMING: "🔵",
            StreamingStatus.COMPLETED: "✅",
            StreamingStatus.ERROR: "🔴"
        }
        
        status_messages = {
            StreamingStatus.IDLE: "대기 중",
            StreamingStatus.CONNECTING: "연결 중...",
            StreamingStatus.STREAMING: "처리 중...",
            StreamingStatus.COMPLETED: "완료",
            StreamingStatus.ERROR: "오류 발생"
        }
        
        status_icon = status_colors.get(session.status, "⚪")
        status_text = status_messages.get(session.status, "알 수 없음")
        
        with st.sidebar:
            st.markdown(f"**상태**: {status_icon} {status_text}")
            
            if session.active_streams > 0:
                st.markdown(f"**활성 스트림**: {session.active_streams}")
            
            if self.config.enable_progress_bar and session.status == StreamingStatus.STREAMING:
                st.progress(0.5, "AI 에이전트들이 협업 중...")
    
    def render_session_info(self, session_id: str):
        """세션 정보 렌더링 (사이드바용)"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        with st.sidebar:
            st.markdown("### 📊 세션 정보")
            st.markdown(f"**세션 ID**: `{session_id[:8]}...`")
            st.markdown(f"**총 메시지**: {session.total_messages}")
            st.markdown(f"**오류 횟수**: {session.error_count}")
            
            duration = datetime.now() - session.created_at
            st.markdown(f"**진행 시간**: {duration.total_seconds():.1f}초")
    
    def render_orchestrator_stats(self):
        """오케스트레이터 통계 렌더링"""
        with st.sidebar:
            st.markdown("### 📈 시스템 통계")
            st.markdown(f"**총 세션**: {self.stats['total_sessions']}")
            st.markdown(f"**활성 세션**: {self.stats['active_sessions']}")
            st.markdown(f"**성공률**: {(self.stats['successful_streams'] / max(1, self.stats['successful_streams'] + self.stats['failed_streams'])) * 100:.1f}%")
    
    async def cleanup_expired_sessions(self, max_idle_minutes: int = 30):
        """만료된 세션 정리"""
        cutoff_time = datetime.now().timestamp() - (max_idle_minutes * 60)
        
        to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity.timestamp() < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            # 관련 컨테이너들도 정리
            if session_id in self.chat_containers:
                del self.chat_containers[session_id]
            if session_id in self.chat_interfaces:
                del self.chat_interfaces[session_id]
            
            del self.active_sessions[session_id]
            logger.info(f"🧹 만료된 스트리밍 세션 정리: {session_id}")
        
        self.stats['active_sessions'] = len(self.active_sessions)
        
        # 브로커 세션도 정리
        await self.unified_broker.cleanup_expired_sessions(max_idle_minutes // 60)
    
    async def shutdown(self):
        """오케스트레이터 종료"""
        logger.info("🔚 Streaming Orchestrator 종료 시작...")
        
        # 모든 세션 정리
        self.active_sessions.clear()
        self.chat_containers.clear()
        self.chat_interfaces.clear()
        
        # 브로커 종료
        await self.unified_broker.shutdown()
        
        logger.info("✅ Streaming Orchestrator 종료 완료")


# 전역 오케스트레이터 인스턴스
_streaming_orchestrator = None

def get_streaming_orchestrator(config: Optional[StreamingConfig] = None) -> StreamingOrchestrator:
    """전역 스트리밍 오케스트레이터 인스턴스"""
    global _streaming_orchestrator
    if _streaming_orchestrator is None:
        _streaming_orchestrator = StreamingOrchestrator(config)
    return _streaming_orchestrator


# Streamlit 편의 함수들
def init_streaming_chat(
    user_id: str = "default_user",
    chat_style: ChatStyle = ChatStyle.CHATGPT,
    container_key: str = "main_chat"
) -> str:
    """Streamlit에서 스트리밍 채팅 초기화"""
    
    # 설정
    config = StreamingConfig(chat_style=chat_style)
    orchestrator = get_streaming_orchestrator(config)
    
    # 세션 생성/가져오기
    session = orchestrator.get_or_create_session(user_id, container_key)
    
    # 비동기 초기화 (필요시)
    if not hasattr(st.session_state, 'orchestrator_initialized'):
        asyncio.create_task(orchestrator.initialize())
        st.session_state.orchestrator_initialized = True
    
    return session.session_id

def render_streaming_interface(
    session_id: str,
    show_session_info: bool = True,
    show_stats: bool = True
):
    """Streamlit 스트리밍 인터페이스 렌더링"""
    orchestrator = get_streaming_orchestrator()
    
    # 메인 채팅 인터페이스
    orchestrator.render_streaming_chat(session_id)
    
    # 사이드바 정보
    if show_session_info:
        orchestrator.render_session_info(session_id)
    
    if show_stats:
        orchestrator.render_orchestrator_stats()


if __name__ == "__main__":
    # 테스트 코드
    async def demo():
        config = StreamingConfig(
            chat_style=ChatStyle.CHATGPT,
            enable_typing_indicator=True,
            show_agent_names=True
        )
        
        orchestrator = StreamingOrchestrator(config)
        await orchestrator.initialize()
        
        print("🎭 Streaming Orchestrator Demo")
        
        # 테스트 세션
        session = orchestrator.get_or_create_session("test_user")
        print(f"🆕 테스트 세션: {session.session_id}")
        
        # 스트리밍 테스트
        async for event in orchestrator.stream_user_query(
            session.session_id,
            "안녕하세요! 데이터 분석을 도와주세요.",
            ["data_processing", "visualization"]
        ):
            print(f"📨 이벤트: {event}")
            if event.get('final'):
                break
        
        await orchestrator.shutdown()
    
    asyncio.run(demo()) 