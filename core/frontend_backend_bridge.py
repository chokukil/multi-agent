#!/usr/bin/env python3
"""
🌉 CherryAI 프론트엔드-백엔드 융합 브릿지

ChatGPT/Claude 수준의 UI/UX와 강력한 백엔드 시스템을 완전히 융합하는 브릿지

Key Features:
- UI 컴포넌트와 백엔드 시스템 완전 통합
- 실시간 SSE 스트리밍 연동
- LLM First Engine과 UI 융합
- Knowledge Bank와 채팅 인터페이스 연동
- 세션 관리와 백엔드 동기화
- A2A 에이전트와 UI 상태 동기화
- MCP 도구와 실시간 피드백 연동

Architecture:
- Bridge Controller: 전체 브릿지 제어
- Component Connectors: UI-백엔드 커넥터들
- State Synchronizer: 상태 동기화 관리
- Event Coordinator: 이벤트 조정 및 전달
- Performance Monitor: 융합 성능 모니터링
"""

import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# UI 컴포넌트들 임포트
from ui.components.chat_interface import (
    get_chat_interface, ChatInterface, MessageRole
)
from ui.components.rich_content_renderer import (
    get_rich_content_renderer, RichContentRenderer, ContentType
)
from ui.components.session_manager import (
    get_session_manager, get_session_manager_ui, 
    SessionManager, SessionManagerUI, SessionType
)
from ui.components.streaming_manager import (
    get_sse_streaming_manager, SSEStreamingManager, StreamingStatus
)
from ui.components.shortcuts_system import (
    get_shortcuts_manager, ShortcutsManager, ShortcutContext
)

# 백엔드 시스템들 임포트
try:
    from core.shared_knowledge_bank import (
        get_shared_knowledge_bank, add_user_file_knowledge, search_relevant_knowledge
    )
    KNOWLEDGE_BANK_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BANK_AVAILABLE = False

try:
    from core.llm_first_engine import (
        get_llm_first_engine, analyze_intent, make_decision, assess_quality
    )
    LLM_FIRST_AVAILABLE = True
except ImportError:
    LLM_FIRST_AVAILABLE = False

try:
    from core.main_app_engine import (
        get_main_engine, CherryAIMainEngine
    )
    MAIN_ENGINE_AVAILABLE = True
except ImportError:
    MAIN_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

class BridgeStatus(Enum):
    """브릿지 상태"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class EventType(Enum):
    """이벤트 타입"""
    USER_MESSAGE = "user_message"
    AI_RESPONSE = "ai_response"
    FILE_UPLOAD = "file_upload"
    SESSION_CHANGE = "session_change"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class BridgeEvent:
    """브릿지 이벤트"""
    event_type: EventType
    data: Any
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class FrontendBackendBridge:
    """
    🌉 프론트엔드-백엔드 융합 브릿지
    
    UI와 백엔드 시스템을 완전히 통합하는 메인 브릿지
    """
    
    def __init__(self):
        """브릿지 초기화"""
        self.status = BridgeStatus.INITIALIZING
        
        # UI 컴포넌트들
        self.chat_interface: Optional[ChatInterface] = None
        self.rich_renderer: Optional[RichContentRenderer] = None
        self.session_manager: Optional[SessionManager] = None
        self.session_ui: Optional[SessionManagerUI] = None
        self.streaming_manager: Optional[SSEStreamingManager] = None
        self.shortcuts_manager: Optional[ShortcutsManager] = None
        
        # 백엔드 시스템들
        self.knowledge_bank = None
        self.llm_first_engine = None
        self.main_engine: Optional[CherryAIMainEngine] = None
        
        # 이벤트 처리
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue: List[BridgeEvent] = []
        
        # 상태 관리
        self.current_session_id: Optional[str] = None
        self.active_streaming_session: Optional[str] = None
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_messages": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0.0,
            "stream_sessions": 0
        }
        
        logger.info("🌉 프론트엔드-백엔드 브릿지 초기화 시작")
    
    async def initialize(self) -> bool:
        """브릿지 완전 초기화"""
        try:
            # UI 컴포넌트들 초기화
            self.chat_interface = get_chat_interface()
            self.rich_renderer = get_rich_content_renderer()
            self.session_manager = get_session_manager()
            self.session_ui = get_session_manager_ui()
            self.streaming_manager = get_sse_streaming_manager()
            self.shortcuts_manager = get_shortcuts_manager()
            
            # 백엔드 시스템들 초기화
            if KNOWLEDGE_BANK_AVAILABLE:
                self.knowledge_bank = get_shared_knowledge_bank()
            
            if LLM_FIRST_AVAILABLE:
                self.llm_first_engine = get_llm_first_engine()
            
            if MAIN_ENGINE_AVAILABLE:
                self.main_engine = get_main_engine()
                await self.main_engine.initialize()
            
            # 이벤트 핸들러 등록
            self._register_event_handlers()
            
            # 기본 세션 생성 (필요한 경우)
            if not self.session_manager.current_session_id:
                session_id = self.session_manager.create_session(
                    name="새로운 대화",
                    session_type=SessionType.CHAT
                )
                self.current_session_id = session_id
            else:
                self.current_session_id = self.session_manager.current_session_id
            
            self.status = BridgeStatus.READY
            logger.info("🌉 프론트엔드-백엔드 브릿지 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"브릿지 초기화 실패: {e}")
            self.status = BridgeStatus.ERROR
            return False
    
    def _register_event_handlers(self) -> None:
        """이벤트 핸들러 등록"""
        # 사용자 메시지 처리
        self.register_event_handler(EventType.USER_MESSAGE, self._handle_user_message)
        
        # 파일 업로드 처리
        self.register_event_handler(EventType.FILE_UPLOAD, self._handle_file_upload)
        
        # 세션 변경 처리
        self.register_event_handler(EventType.SESSION_CHANGE, self._handle_session_change)
        
        # 스트리밍 이벤트 처리
        self.register_event_handler(EventType.STREAM_START, self._handle_stream_start)
        self.register_event_handler(EventType.STREAM_END, self._handle_stream_end)
        
        # 에러 처리
        self.register_event_handler(EventType.ERROR_OCCURRED, self._handle_error)
    
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """이벤트 핸들러 등록"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: BridgeEvent) -> None:
        """이벤트 발행"""
        self.event_queue.append(event)
        
        # 등록된 핸들러들 실행
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"이벤트 핸들러 실행 실패: {e}")
    
    async def _handle_user_message(self, event: BridgeEvent) -> None:
        """사용자 메시지 처리"""
        try:
            user_input = event.data.get("message", "")
            uploaded_files = event.data.get("files", [])
            
            if not user_input:
                return
            
            # 채팅 인터페이스에 사용자 메시지 추가
            self.chat_interface.add_message(MessageRole.USER, user_input)
            
            # Knowledge Bank에 컨텍스트 저장 (가능한 경우)
            if KNOWLEDGE_BANK_AVAILABLE and self.knowledge_bank:
                try:
                    conversation_context = self.chat_interface.get_conversation_context()
                    # 여기서 Knowledge Bank에 컨텍스트 저장 로직 추가
                except Exception as e:
                    logger.warning(f"Knowledge Bank 저장 실패: {e}")
            
            # 메인 엔진으로 처리 요청
            if self.main_engine:
                # 스트리밍 세션 생성
                streaming_session_id = self.streaming_manager.create_streaming_session()
                self.active_streaming_session = streaming_session_id
                
                # 스트리밍 placeholder 생성
                streaming_placeholder = st.empty()
                
                # AI 메시지 스트리밍 시작
                ai_message_id = self.chat_interface.add_streaming_message(MessageRole.ASSISTANT)
                
                # 스트리밍 이벤트 발행
                await self.emit_event(BridgeEvent(
                    event_type=EventType.STREAM_START,
                    data={"streaming_session_id": streaming_session_id},
                    timestamp=datetime.now(),
                    session_id=self.current_session_id
                ))
                
                # 메인 엔진으로 처리
                success = await self._process_with_main_engine(
                    user_input, uploaded_files, streaming_placeholder, ai_message_id
                )
                
                # 스트리밍 완료
                self.chat_interface.complete_streaming_message(ai_message_id)
                
                # 스트리밍 종료 이벤트 발행
                await self.emit_event(BridgeEvent(
                    event_type=EventType.STREAM_END,
                    data={"success": success, "message_id": ai_message_id},
                    timestamp=datetime.now(),
                    session_id=self.current_session_id
                ))
                
                # 성능 메트릭 업데이트
                self.performance_metrics["total_messages"] += 1
                if success:
                    self.performance_metrics["successful_responses"] += 1
                else:
                    self.performance_metrics["failed_responses"] += 1
            
        except Exception as e:
            logger.error(f"사용자 메시지 처리 실패: {e}")
            await self.emit_event(BridgeEvent(
                event_type=EventType.ERROR_OCCURRED,
                data={"error": str(e), "context": "user_message"},
                timestamp=datetime.now(),
                session_id=self.current_session_id
            ))
    
    async def _process_with_main_engine(self, 
                                      user_input: str, 
                                      uploaded_files: List[Any],
                                      streaming_placeholder,
                                      ai_message_id: str) -> bool:
        """메인 엔진으로 처리 - 실제 SSE 스트리밍 통합"""
        try:
            # 메인 엔진의 실시간 SSE 스트리밍 처리
            accumulated_response = ""
            
            # 실제 스트리밍 처리 (sleep 없는 진짜 스트리밍)
            async for chunk in self.main_engine.process_user_request(user_input, uploaded_files):
                if chunk.strip():
                    # 직접 UI 업데이트 (가짜 chunk generator 제거)
                    self.chat_interface.update_streaming_message(ai_message_id, chunk)
                    accumulated_response += chunk
                    
                    # Streamlit 실시간 업데이트
                    if streaming_placeholder:
                        with streaming_placeholder.container():
                            st.markdown(self._format_streaming_content(accumulated_response), 
                                      unsafe_allow_html=True)
            
            # 리치 콘텐츠 렌더링 (필요한 경우)
            if self._contains_rich_content(accumulated_response):
                await self._render_rich_content(accumulated_response)
            
            return True
            
        except Exception as e:
            logger.error(f"메인 엔진 처리 실패: {e}")
            
            # 에러 메시지 표시
            error_message = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
            self.chat_interface.update_streaming_message(ai_message_id, error_message)
            
            return False
    
    def _format_streaming_content(self, content: str) -> str:
        """스트리밍 콘텐츠 포맷팅 - LLM First 원칙"""
        # HTML 이스케이프 제거, LLM 생성 콘텐츠 그대로 렌더링
        import re
        
        # 줄바꿈 처리
        content = content.replace('\n', '<br>')
        
        # 마크다운 처리
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', content)
        content = re.sub(r'`([^`]+)`', r'<code style="background: #f1f3f4; padding: 2px 4px; border-radius: 3px; font-family: monospace;">\1</code>', content)
        
        return content
    
    def _render_context_integration(self) -> None:
        """6-Layer Context System 및 Knowledge Bank 통합 렌더링"""
        try:
            # 6-Layer Context System 패널
            self.chat_interface.render_context_layers_panel()
            
            # Knowledge Bank UI 통합
            from core.knowledge_bank_ui_integration import get_knowledge_bank_ui_integrator
            integrator = get_knowledge_bank_ui_integrator()
            
            # 현재 세션 컨텍스트 업데이트
            if hasattr(st.session_state, 'messages') and st.session_state.messages:
                last_message = st.session_state.messages[-1] if st.session_state.messages else None
                if last_message:
                    # 비동기 메서드를 동기로 실행
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            integrator.update_context_knowledge({
                                "last_user_input": last_message.get('content', ''),
                                "session_id": self.current_session_id
                            })
                        )
                        loop.close()
                    except Exception as e:
                        logger.warning(f"컨텍스트 지식 업데이트 실패: {e}")
            
            # Knowledge Bank 사이드바 렌더링
            integrator.render_knowledge_sidebar()
            
        except Exception as e:
            logger.warning(f"Context integration 렌더링 실패: {e}")
    
    # 가짜 chunk generator 제거 - 실제 SSE 스트리밍 사용
    
    def _contains_rich_content(self, content: str) -> bool:
        """리치 콘텐츠 포함 여부 확인"""
        # 간단한 패턴 매칭으로 리치 콘텐츠 감지
        rich_patterns = [
            "```",  # 코드 블록
            "| ",   # 테이블
            "![",   # 이미지
            "data:",  # 데이터 URL
            "chart_", # 차트 관련
            "plot_"   # 플롯 관련
        ]
        
        return any(pattern in content for pattern in rich_patterns)
    
    async def _render_rich_content(self, content: str) -> None:
        """리치 콘텐츠 렌더링"""
        try:
            # 리치 콘텐츠 렌더러를 통한 고급 렌더링
            rich_content = self.rich_renderer.create_content(
                data=content,
                title="AI 응답"
            )
            self.rich_renderer.render_content(rich_content)
            
        except Exception as e:
            logger.error(f"리치 콘텐츠 렌더링 실패: {e}")
    
    async def _handle_file_upload(self, event: BridgeEvent) -> None:
        """파일 업로드 처리"""
        try:
            files = event.data.get("files", [])
            
            if not files:
                return
            
            # Knowledge Bank에 파일 추가 (가능한 경우)
            if KNOWLEDGE_BANK_AVAILABLE and self.knowledge_bank:
                for file in files:
                    try:
                        # 파일을 Knowledge Bank에 추가
                        add_user_file_knowledge(
                            file_path=file.get("path", ""),
                            file_content=file.get("content", ""),
                            agent_context=f"session_{self.current_session_id}"
                        )
                        logger.info(f"파일이 Knowledge Bank에 추가됨: {file.get('name', 'Unknown')}")
                        
                    except Exception as e:
                        logger.error(f"파일 Knowledge Bank 추가 실패: {e}")
            
            # 파일 처리 메시지 표시
            file_names = [f.get("name", "Unknown") for f in files]
            message = f"📁 {len(files)}개 파일이 업로드되었습니다: {', '.join(file_names)}"
            self.chat_interface.add_message(MessageRole.ASSISTANT, message)
            
        except Exception as e:
            logger.error(f"파일 업로드 처리 실패: {e}")
    
    async def _handle_session_change(self, event: BridgeEvent) -> None:
        """세션 변경 처리"""
        try:
            new_session_id = event.data.get("session_id")
            
            if not new_session_id:
                return
            
            # 현재 세션 저장
            if self.current_session_id:
                await self._save_current_session()
            
            # 새 세션 로드
            session_data = self.session_manager.load_session(new_session_id)
            if session_data:
                self.current_session_id = new_session_id
                
                # 채팅 히스토리 복원
                self.chat_interface.clear_messages()
                for message_data in session_data.messages:
                    role = MessageRole(message_data["role"])
                    content = message_data["content"]
                    self.chat_interface.add_message(role, content)
                
                logger.info(f"세션 변경됨: {new_session_id}")
            
        except Exception as e:
            logger.error(f"세션 변경 처리 실패: {e}")
    
    async def _handle_stream_start(self, event: BridgeEvent) -> None:
        """스트리밍 시작 처리"""
        streaming_session_id = event.data.get("streaming_session_id")
        self.performance_metrics["stream_sessions"] += 1
        logger.info(f"스트리밍 시작: {streaming_session_id}")
    
    async def _handle_stream_end(self, event: BridgeEvent) -> None:
        """스트리밍 종료 처리"""
        success = event.data.get("success", False)
        message_id = event.data.get("message_id")
        
        if success:
            logger.info(f"스트리밍 성공 완료: {message_id}")
        else:
            logger.warning(f"스트리밍 실패: {message_id}")
        
        # 현재 세션 저장
        await self._save_current_session()
    
    async def _handle_error(self, event: BridgeEvent) -> None:
        """에러 처리"""
        error = event.data.get("error", "Unknown error")
        context = event.data.get("context", "Unknown context")
        
        logger.error(f"브릿지 에러 [{context}]: {error}")
        
        # 사용자에게 에러 메시지 표시
        error_message = f"⚠️ 오류가 발생했습니다: {error}"
        self.chat_interface.add_message(MessageRole.ASSISTANT, error_message)
    
    async def _save_current_session(self) -> None:
        """현재 세션 저장"""
        try:
            if not self.current_session_id:
                return
            
            # 현재 채팅 히스토리 가져오기
            messages = self.chat_interface.get_messages()
            message_dicts = [msg.to_dict() for msg in messages]
            
            # 세션 데이터 업데이트
            session_data = self.session_manager.load_session(self.current_session_id)
            if session_data:
                session_data.messages = message_dicts
                self.session_manager.save_session(session_data)
                
        except Exception as e:
            logger.error(f"세션 저장 실패: {e}")
    
    def render_complete_interface(self) -> Optional[str]:
        """완전한 통합 인터페이스 렌더링"""
        try:
            # 브릿지 상태 확인
            if self.status != BridgeStatus.READY:
                st.error("🌉 브릿지가 준비되지 않았습니다. 페이지를 새로고침해주세요.")
                return None
            
            # 바로가기 시스템 활성화
            self.shortcuts_manager.set_active_contexts({
                ShortcutContext.GLOBAL,
                ShortcutContext.CHAT,
                ShortcutContext.SESSION
            })
            
            # 세션 관리 사이드바
            self.session_ui.render_session_sidebar()
            
            # 메인 채팅 인터페이스
            user_input = self.chat_interface.render_complete_interface()
            
            # 6-Layer Context System 및 Knowledge Bank 통합
            self._render_context_integration()
            
            # 사용자 입력 처리
            if user_input:
                # 파일 업로드 상태 확인
                uploaded_files = []
                if hasattr(st.session_state, "uploaded_files"):
                    uploaded_files = st.session_state.uploaded_files
                
                # 사용자 메시지 이벤트 발행
                asyncio.create_task(self.emit_event(BridgeEvent(
                    event_type=EventType.USER_MESSAGE,
                    data={"message": user_input, "files": uploaded_files},
                    timestamp=datetime.now(),
                    session_id=self.current_session_id
                )))
            
            return user_input
            
        except Exception as e:
            logger.error(f"인터페이스 렌더링 실패: {e}")
            st.error(f"인터페이스 오류: {str(e)}")
            return None
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """브릿지 상태 정보"""
        return {
            "status": self.status.value,
            "current_session": self.current_session_id,
            "active_streaming": self.active_streaming_session,
            "components_ready": {
                "chat_interface": self.chat_interface is not None,
                "rich_renderer": self.rich_renderer is not None,
                "session_manager": self.session_manager is not None,
                "streaming_manager": self.streaming_manager is not None,
                "shortcuts_manager": self.shortcuts_manager is not None
            },
            "backend_systems": {
                "knowledge_bank": KNOWLEDGE_BANK_AVAILABLE and self.knowledge_bank is not None,
                "llm_first_engine": LLM_FIRST_AVAILABLE and self.llm_first_engine is not None,
                "main_engine": MAIN_ENGINE_AVAILABLE and self.main_engine is not None
            },
            "performance_metrics": self.performance_metrics
        }

# 비동기 브릿지 실행을 위한 헬퍼 함수들
async def run_bridge_async(bridge: FrontendBackendBridge, user_input: str, files: List[Any] = None):
    """브릿지 비동기 실행"""
    try:
        await bridge.emit_event(BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": user_input, "files": files or []},
            timestamp=datetime.now(),
            session_id=bridge.current_session_id
        ))
    except Exception as e:
        logger.error(f"브릿지 비동기 실행 실패: {e}")

def sync_run_bridge(bridge: FrontendBackendBridge, user_input: str, files: List[Any] = None):
    """브릿지 동기 실행 (Streamlit 호환)"""
    try:
        # Streamlit에서는 이벤트 루프가 이미 실행 중일 수 있으므로 조심스럽게 처리
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_bridge_async(bridge, user_input, files))
        loop.close()
    except Exception as e:
        logger.error(f"브릿지 동기 실행 실패: {e}")

# 전역 인스턴스 관리
_frontend_backend_bridge_instance = None

def get_frontend_backend_bridge() -> FrontendBackendBridge:
    """프론트엔드-백엔드 브릿지 싱글톤 인스턴스 반환"""
    global _frontend_backend_bridge_instance
    if _frontend_backend_bridge_instance is None:
        _frontend_backend_bridge_instance = FrontendBackendBridge()
    return _frontend_backend_bridge_instance

async def initialize_frontend_backend_bridge() -> FrontendBackendBridge:
    """프론트엔드-백엔드 브릿지 초기화"""
    global _frontend_backend_bridge_instance
    _frontend_backend_bridge_instance = FrontendBackendBridge()
    await _frontend_backend_bridge_instance.initialize()
    return _frontend_backend_bridge_instance 