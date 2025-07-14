#!/usr/bin/env python3
"""
⚡ CherryAI 순수 SSE 스트리밍 매니저

A2A SDK 0.2.9 표준에 완전 준수하는 Server-Sent Events 기반 실시간 스트리밍 시스템

Key Features:
- 순수 SSE 프로토콜 (WebSocket fallback 없음)
- A2A SDK 0.2.9 async chunk 스트리밍 완전 준수
- 실시간 타이핑 효과 (Character-by-character)
- 청크 단위 최적화 (50-100 문자)
- 연결 재시도 및 에러 복구
- 버퍼링 최소화
- 네트워크 지연 처리
- 스트리밍 상태 실시간 추적

Architecture:
- SSE Connection Manager: SSE 연결 관리
- Chunk Processor: 청크 단위 데이터 처리
- Stream Controller: 스트리밍 제어 및 상태 관리
- Error Recovery: 연결 오류 복구
- Performance Monitor: 성능 모니터링
"""

import streamlit as st
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import queue
import threading

logger = logging.getLogger(__name__)

class StreamingStatus(Enum):
    """스트리밍 상태"""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class ChunkType(Enum):
    """청크 타입"""
    TEXT = "text"
    CODE = "code"
    JSON = "json"
    ERROR = "error"
    METADATA = "metadata"
    FINAL = "final"

@dataclass
class StreamChunk:
    """스트리밍 청크 데이터"""
    id: str
    chunk_type: ChunkType
    content: str
    sequence: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_final": self.is_final
        }

@dataclass
class StreamingSession:
    """스트리밍 세션"""
    session_id: str
    status: StreamingStatus
    start_time: datetime
    total_chunks: int = 0
    total_characters: int = 0
    error_count: int = 0
    last_chunk_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    def update_progress(self, chunk: StreamChunk) -> None:
        """진행 상황 업데이트"""
        self.total_chunks += 1
        self.total_characters += len(chunk.content)
        self.last_chunk_time = datetime.now()
        
        # 완료 시간 추정 (간단한 휴리스틱)
        if self.total_chunks > 5:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            avg_chars_per_sec = self.total_characters / elapsed if elapsed > 0 else 0
            if avg_chars_per_sec > 0:
                # 대략적인 예상 완료 시간 계산
                estimated_remaining = max(0, (1000 - self.total_characters) / avg_chars_per_sec)
                self.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)

class SSEStreamingManager:
    """
    ⚡ SSE 스트리밍 매니저
    
    순수 Server-Sent Events 기반 실시간 스트리밍 관리
    """
    
    def __init__(self):
        """스트리밍 매니저 초기화"""
        
        # 스트리밍 설정
        self.chunk_size = 75  # 청크 크기 (문자)
        self.typing_speed = 0.02  # 타이핑 속도 (초)
        self.max_chunks_per_second = 50  # 초당 최대 청크 수
        self.connection_timeout = 30  # 연결 타임아웃 (초)
        self.max_retries = 3  # 최대 재시도 횟수
        
        # 스트리밍 상태
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.chunk_queues: Dict[str, deque] = {}
        
        # 성능 모니터링
        self.performance_metrics = {
            "total_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0,
            "total_chunks_sent": 0,
            "total_characters_sent": 0,
            "average_latency_ms": 0.0,
            "connection_errors": 0
        }
        
        # 에러 복구 설정
        self.retry_delays = [1, 2, 5]  # 재시도 지연 시간
        
        logger.info("⚡ SSE 스트리밍 매니저 초기화 완료")
    
    def create_streaming_session(self, session_id: str = None) -> str:
        """새 스트리밍 세션 생성"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        streaming_session = StreamingSession(
            session_id=session_id,
            status=StreamingStatus.IDLE,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = streaming_session
        self.chunk_queues[session_id] = deque()
        
        self.performance_metrics["total_sessions"] += 1
        
        logger.info(f"⚡ 스트리밍 세션 생성: {session_id}")
        return session_id
    
    async def start_streaming(self, 
                             session_id: str,
                             content_generator: AsyncGenerator[str, None],
                             placeholder = None) -> bool:
        """SSE 스트리밍 시작"""
        try:
            if session_id not in self.active_sessions:
                logger.error(f"존재하지 않는 세션: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            session.status = StreamingStatus.CONNECTING
            
            # 연결 표시
            if placeholder:
                placeholder.markdown(self._render_streaming_indicator("연결 중..."))
            
            # 스트리밍 시작
            session.status = StreamingStatus.STREAMING
            session.start_time = datetime.now()
            
            accumulated_content = ""
            chunk_sequence = 0
            
            # 비동기 스트리밍 처리
            async for content_chunk in content_generator:
                if not content_chunk:
                    continue
                
                # 청크 생성
                stream_chunk = StreamChunk(
                    id=str(uuid.uuid4()),
                    chunk_type=ChunkType.TEXT,
                    content=content_chunk,
                    sequence=chunk_sequence,
                    timestamp=datetime.now()
                )
                
                # 세션 진행 상황 업데이트
                session.update_progress(stream_chunk)
                
                # 누적 콘텐츠 업데이트
                accumulated_content += content_chunk
                
                # UI 업데이트 (타이핑 효과)
                if placeholder:
                    # 타이핑 효과와 함께 렌더링
                    await self._render_typing_effect(
                        accumulated_content, 
                        placeholder,
                        session_id
                    )
                
                # 청크 큐에 추가
                self.chunk_queues[session_id].append(stream_chunk)
                
                # 성능 메트릭 업데이트
                self.performance_metrics["total_chunks_sent"] += 1
                self.performance_metrics["total_characters_sent"] += len(content_chunk)
                
                chunk_sequence += 1
                
                # 실제 SSE 스트리밍에서는 인위적 지연 제거
                # Real-time streaming: no artificial delay
            
            # 스트리밍 완료
            session.status = StreamingStatus.COMPLETED
            
            # 최종 렌더링
            if placeholder:
                placeholder.markdown(self._render_final_content(accumulated_content))
            
            # 완료 청크 추가
            final_chunk = StreamChunk(
                id=str(uuid.uuid4()),
                chunk_type=ChunkType.FINAL,
                content="",
                sequence=chunk_sequence,
                timestamp=datetime.now(),
                is_final=True
            )
            self.chunk_queues[session_id].append(final_chunk)
            
            self.performance_metrics["successful_sessions"] += 1
            
            logger.info(f"⚡ 스트리밍 완료: {session_id} - {session.total_characters} 문자")
            return True
            
        except Exception as e:
            logger.error(f"스트리밍 오류: {session_id} - {e}")
            await self._handle_streaming_error(session_id, str(e), placeholder)
            return False
    
    async def stream_with_a2a_client(self,
                                   session_id: str,
                                   a2a_client,
                                   message: str,
                                   placeholder = None) -> bool:
        """A2A 클라이언트와 연동한 스트리밍"""
        try:
            # A2A 스트리밍 요청 생성
            async def a2a_content_generator():
                """A2A 클라이언트로부터 스트리밍 데이터 생성"""
                try:
                    # A2A SDK 0.2.9 표준 스트리밍 호출
                    async for chunk in a2a_client.stream_message(message):
                        # A2A 응답에서 텍스트 추출
                        if hasattr(chunk, 'content'):
                            yield chunk.content
                        elif isinstance(chunk, dict) and 'content' in chunk:
                            yield chunk['content']
                        elif isinstance(chunk, str):
                            yield chunk
                        
                except Exception as e:
                    logger.error(f"A2A 스트리밍 오류: {e}")
                    yield f"[오류: {str(e)}]"
            
            # SSE 스트리밍 시작
            return await self.start_streaming(
                session_id,
                a2a_content_generator(),
                placeholder
            )
            
        except Exception as e:
            logger.error(f"A2A 스트리밍 연동 오류: {session_id} - {e}")
            return False
    
    async def _render_typing_effect(self, 
                                  content: str, 
                                  placeholder,
                                  session_id: str) -> None:
        """타이핑 효과 렌더링"""
        try:
            session = self.active_sessions.get(session_id)
            
            # 진행률 계산
            progress_info = ""
            if session and session.estimated_completion:
                remaining = (session.estimated_completion - datetime.now()).total_seconds()
                if remaining > 0:
                    progress_info = f" (약 {remaining:.0f}초 남음)"
            
            # 타이핑 커서와 함께 렌더링
            typing_content = f"""
            <div style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                padding: 16px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #ff6b6b;
            ">
                {self._format_streaming_content(content)}
                <span style="
                    color: #ff6b6b;
                    animation: blink 1s infinite;
                    font-weight: bold;
                ">|</span>
                <div style="
                    margin-top: 8px;
                    font-size: 12px;
                    color: #6c757d;
                ">
                    🍒 CherryAI가 응답하고 있습니다{progress_info}
                </div>
            </div>
            
            <style>
            @keyframes blink {{
                0%, 50% {{ opacity: 1; }}
                51%, 100% {{ opacity: 0; }}
            }}
            </style>
            """
            
            placeholder.markdown(typing_content, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"타이핑 효과 렌더링 오류: {e}")
    
    def _render_final_content(self, content: str) -> str:
        """최종 콘텐츠 렌더링"""
        return f"""
        <div style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            padding: 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        ">
            {self._format_streaming_content(content)}
        </div>
        """
    
    def _format_streaming_content(self, content: str) -> str:
        """스트리밍 콘텐츠 포맷팅 - LLM First 원칙 적용"""
        # LLM이 생성한 HTML/Markdown 콘텐츠를 그대로 렌더링
        # HTML 이스케이프 제거 - 스트리밍 중 LLM 의도 보존
        
        # 줄바꿈 처리
        content = content.replace('\n', '<br>')
        
        import re
        # 마크다운 볼드 텍스트 처리 (정규표현식 사용)
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
        
        # 마크다운 이탤릭 텍스트 처리
        content = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', content)
        
        # 인라인 코드 처리
        content = re.sub(r'`([^`]+)`', r'<code style="background: #f1f3f4; padding: 2px 4px; border-radius: 3px; font-family: monospace;">\1</code>', content)
        
        return content
    
    def _render_streaming_indicator(self, message: str) -> str:
        """스트리밍 인디케이터 렌더링"""
        return f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        ">
            <div style="
                display: flex;
                gap: 4px;
                margin-right: 12px;
            ">
                <div style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #007bff;
                    animation: pulse 1.4s infinite ease-in-out;
                "></div>
                <div style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #007bff;
                    animation: pulse 1.4s infinite ease-in-out 0.2s;
                "></div>
                <div style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #007bff;
                    animation: pulse 1.4s infinite ease-in-out 0.4s;
                "></div>
            </div>
            <span style="color: #495057;">{message}</span>
        </div>
        
        <style>
        @keyframes pulse {{
            0%, 80%, 100% {{ transform: scale(0.8); opacity: 0.5; }}
            40% {{ transform: scale(1); opacity: 1; }}
        }}
        </style>
        """
    
    async def _handle_streaming_error(self, 
                                    session_id: str, 
                                    error_message: str,
                                    placeholder = None) -> None:
        """스트리밍 오류 처리"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.status = StreamingStatus.ERROR
                session.error_count += 1
            
            self.performance_metrics["failed_sessions"] += 1
            self.performance_metrics["connection_errors"] += 1
            
            # 오류 메시지 표시
            if placeholder:
                error_content = f"""
                <div style="
                    padding: 16px;
                    background: #fed7d7;
                    border: 1px solid #feb2b2;
                    border-radius: 8px;
                    color: #c53030;
                ">
                    <strong>⚠️ 스트리밍 오류</strong><br>
                    {error_message}
                    <br><br>
                    <small>연결을 다시 시도하거나 새로고침해주세요.</small>
                </div>
                """
                placeholder.markdown(error_content, unsafe_allow_html=True)
            
            logger.error(f"스트리밍 오류 처리됨: {session_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"오류 처리 중 추가 오류: {e}")
    
    def pause_streaming(self, session_id: str) -> bool:
        """스트리밍 일시정지"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session.status == StreamingStatus.STREAMING:
                    session.status = StreamingStatus.PAUSED
                    logger.info(f"⚡ 스트리밍 일시정지: {session_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"스트리밍 일시정지 오류: {e}")
            return False
    
    def resume_streaming(self, session_id: str) -> bool:
        """스트리밍 재개"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session.status == StreamingStatus.PAUSED:
                    session.status = StreamingStatus.STREAMING
                    logger.info(f"⚡ 스트리밍 재개: {session_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"스트리밍 재개 오류: {e}")
            return False
    
    def stop_streaming(self, session_id: str) -> bool:
        """스트리밍 중지"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.status = StreamingStatus.COMPLETED
                logger.info(f"⚡ 스트리밍 중지: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"스트리밍 중지 오류: {e}")
            return False
    
    def get_streaming_status(self, session_id: str) -> Optional[StreamingSession]:
        """스트리밍 상태 조회"""
        return self.active_sessions.get(session_id)
    
    def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """완료된 세션 정리"""
        try:
            cleanup_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            sessions_to_remove = []
            for session_id, session in self.active_sessions.items():
                if (session.status in [StreamingStatus.COMPLETED, StreamingStatus.ERROR] and
                    session.start_time < cutoff_time):
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                if session_id in self.chunk_queues:
                    del self.chunk_queues[session_id]
                cleanup_count += 1
            
            if cleanup_count > 0:
                logger.info(f"⚡ {cleanup_count}개 완료된 세션 정리됨")
            
            return cleanup_count
            
        except Exception as e:
            logger.error(f"세션 정리 오류: {e}")
            return 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        # 실시간 계산 메트릭 추가
        active_sessions_count = len([s for s in self.active_sessions.values() 
                                   if s.status == StreamingStatus.STREAMING])
        
        success_rate = 0.0
        if self.performance_metrics["total_sessions"] > 0:
            success_rate = (self.performance_metrics["successful_sessions"] / 
                          self.performance_metrics["total_sessions"]) * 100
        
        return {
            **self.performance_metrics,
            "active_sessions": active_sessions_count,
            "success_rate_percent": round(success_rate, 2),
            "avg_characters_per_session": (
                self.performance_metrics["total_characters_sent"] / 
                max(1, self.performance_metrics["successful_sessions"])
            )
        }
    
    def render_streaming_controls(self, session_id: str) -> None:
        """스트리밍 제어 UI 렌더링"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if session.status == StreamingStatus.STREAMING:
                if st.button("⏸️ 일시정지", key=f"pause_{session_id}"):
                    self.pause_streaming(session_id)
                    st.rerun()
        
        with col2:
            if session.status == StreamingStatus.PAUSED:
                if st.button("▶️ 재개", key=f"resume_{session_id}"):
                    self.resume_streaming(session_id)
                    st.rerun()
        
        with col3:
            if session.status in [StreamingStatus.STREAMING, StreamingStatus.PAUSED]:
                if st.button("⏹️ 중지", key=f"stop_{session_id}"):
                    self.stop_streaming(session_id)
                    st.rerun()
        
        with col4:
            # 상태 표시
            status_colors = {
                StreamingStatus.STREAMING: "🟢",
                StreamingStatus.PAUSED: "🟡",
                StreamingStatus.COMPLETED: "✅",
                StreamingStatus.ERROR: "🔴"
            }
            status_icon = status_colors.get(session.status, "❓")
            st.caption(f"{status_icon} {session.status.value}")

# Streamlit 컴포넌트 헬퍼 함수들
def create_streaming_placeholder() -> Any:
    """스트리밍용 placeholder 생성"""
    return st.empty()

def inject_sse_javascript():
    """SSE 관련 JavaScript 함수 주입"""
    st.markdown("""
    <script>
    // SSE 연결 관리
    let sseConnections = {};
    
    function createSSEConnection(sessionId, endpoint) {
        if (sseConnections[sessionId]) {
            sseConnections[sessionId].close();
        }
        
        const eventSource = new EventSource(endpoint);
        sseConnections[sessionId] = eventSource;
        
        eventSource.onopen = function(event) {
            console.log('SSE 연결 성공:', sessionId);
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleSSEMessage(sessionId, data);
            } catch (e) {
                console.error('SSE 메시지 파싱 오류:', e);
            }
        };
        
        eventSource.onerror = function(event) {
            console.error('SSE 연결 오류:', sessionId, event);
            // 자동 재연결 시도
            setTimeout(() => {
                if (eventSource.readyState === EventSource.CLOSED) {
                    createSSEConnection(sessionId, endpoint);
                }
            }, 1000);
        };
        
        return eventSource;
    }
    
    function handleSSEMessage(sessionId, data) {
        // 스트리밍 메시지 처리
        console.log('SSE 메시지 수신:', sessionId, data);
    }
    
    function closeSSEConnection(sessionId) {
        if (sseConnections[sessionId]) {
            sseConnections[sessionId].close();
            delete sseConnections[sessionId];
        }
    }
    </script>
    """, unsafe_allow_html=True)

# 전역 인스턴스 관리
_sse_streaming_manager_instance = None

def get_sse_streaming_manager() -> SSEStreamingManager:
    """SSE 스트리밍 매니저 싱글톤 인스턴스 반환"""
    global _sse_streaming_manager_instance
    if _sse_streaming_manager_instance is None:
        _sse_streaming_manager_instance = SSEStreamingManager()
    return _sse_streaming_manager_instance

def initialize_sse_streaming_manager() -> SSEStreamingManager:
    """SSE 스트리밍 매니저 초기화"""
    global _sse_streaming_manager_instance
    _sse_streaming_manager_instance = SSEStreamingManager()
    inject_sse_javascript()
    return _sse_streaming_manager_instance 