#!/usr/bin/env python3
"""
🍒 CherryAI 실시간 스트리밍 핸들러

ChatGPT/Claude 스타일의 실시간 메시지 처리 및 UI 업데이트
- 실시간 스트리밍 메시지 처리
- 타이핑 인디케이터 
- 메시지 청킹 및 버퍼링
- 에러 처리 및 복구
"""

import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import streamlit as st

logger = logging.getLogger(__name__)

class StreamState(Enum):
    """스트림 상태"""
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    TYPING = "typing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class StreamChunk:
    """스트림 청크 데이터"""
    chunk_id: str
    content: str
    chunk_type: str = "text"  # text, json, error, status
    source_agent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False

@dataclass
class StreamSession:
    """스트리밍 세션"""
    session_id: str
    query: str
    state: StreamState = StreamState.IDLE
    chunks: List[StreamChunk] = field(default_factory=list)
    accumulated_content: str = ""
    current_agent: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    typing_start: Optional[float] = None
    error_message: Optional[str] = None

class RealtimeStreamingHandler:
    """실시간 스트리밍 핸들러"""
    
    def __init__(self):
        # 활성 스트림 세션들
        self.active_streams: Dict[str, StreamSession] = {}
        
        # UI 업데이트 콜백들
        self.ui_callbacks: List[Callable] = []
        
        # 설정
        self.config = {
            'chunk_delay_ms': 50,  # 청크 간 지연 (ms)
            'typing_indicator_delay': 1.0,  # 타이핑 인디케이터 지연 (초)
            'max_chunk_size': 100,  # 최대 청크 크기
            'buffer_timeout': 5.0,  # 버퍼 타임아웃 (초)
            'enable_typing_animation': True,
            'enable_progress_updates': True
        }
        
        # 통계
        self.stats = {
            'total_streams': 0,
            'successful_streams': 0,
            'failed_streams': 0,
            'total_chunks': 0,
            'avg_response_time': 0.0
        }
    
    def create_stream_session(self, query: str, session_id: Optional[str] = None) -> str:
        """새 스트리밍 세션 생성"""
        if session_id is None:
            session_id = f"stream_{uuid.uuid4().hex[:8]}"
        
        session = StreamSession(
            session_id=session_id,
            query=query
        )
        
        self.active_streams[session_id] = session
        self.stats['total_streams'] += 1
        
        logger.info(f"🎬 스트리밍 세션 생성: {session_id}")
        return session_id
    
    async def process_stream_async(
        self, 
        session_id: str, 
        stream_generator: AsyncGenerator[Dict[str, Any], None],
        ui_container: Optional[Any] = None
    ) -> str:
        """비동기 스트림 처리"""
        
        session = self.active_streams.get(session_id)
        if not session:
            raise ValueError(f"Stream session not found: {session_id}")
        
        try:
            session.state = StreamState.CONNECTING
            
            # 연결 상태 UI 업데이트
            if ui_container and self.config['enable_progress_updates']:
                with ui_container:
                    st.info("🔄 A2A + MCP 시스템에 연결 중...")
            
            session.state = StreamState.STREAMING
            accumulated_response = ""
            chunk_buffer = []
            last_ui_update = time.time()
            
            # 스트림 이벤트 처리
            async for stream_event in stream_generator:
                try:
                    # 이벤트 파싱
                    chunk = await self._parse_stream_event(stream_event, session)
                    if not chunk:
                        continue
                    
                    # 청크 추가
                    session.chunks.append(chunk)
                    chunk_buffer.append(chunk)
                    self.stats['total_chunks'] += 1
                    
                    # 에이전트 변경 감지
                    if chunk.source_agent and chunk.source_agent != session.current_agent:
                        session.current_agent = chunk.source_agent
                        if ui_container:
                            with ui_container:
                                st.info(f"🤖 {chunk.source_agent} 에이전트가 처리 중...")
                    
                    # 콘텐츠 누적
                    if chunk.content:
                        accumulated_response += chunk.content
                        session.accumulated_content = accumulated_response
                    
                    # 실시간 UI 업데이트 (버퍼링으로 성능 최적화)
                    current_time = time.time()
                    if (current_time - last_ui_update > 0.1 or  # 100ms마다 또는
                        chunk.is_final or  # 최종 청크이면
                        len(chunk_buffer) >= 5):  # 5개 청크마다
                        
                        await self._update_ui_realtime(session, ui_container, chunk_buffer)
                        chunk_buffer = []
                        last_ui_update = current_time
                    
                    # 타이핑 인디케이터
                    if self.config['enable_typing_animation'] and chunk.content:
                        await self._show_typing_indicator(session, ui_container)
                    
                    # 최종 청크 처리
                    if chunk.is_final:
                        session.state = StreamState.COMPLETED
                        break
                        
                except Exception as e:
                    logger.error(f"❌ 스트림 청크 처리 오류: {e}")
                    continue
            
            # 최종 UI 업데이트
            if chunk_buffer:
                await self._update_ui_realtime(session, ui_container, chunk_buffer)
            
            # 완료 처리
            response_time = time.time() - session.start_time
            self._update_stats(response_time, success=True)
            
            if ui_container:
                with ui_container:
                    st.success(f"✅ 완료 ({response_time:.1f}초)")
            
            logger.info(f"🎉 스트리밍 완료: {session_id} ({response_time:.1f}초)")
            return accumulated_response
            
        except Exception as e:
            session.state = StreamState.ERROR
            session.error_message = str(e)
            self._update_stats(0, success=False)
            
            logger.error(f"❌ 스트리밍 처리 실패: {e}")
            
            if ui_container:
                with ui_container:
                    st.error(f"❌ 스트리밍 오류: {str(e)}")
            
            raise e
        
        finally:
            # 세션 정리
            if session_id in self.active_streams:
                del self.active_streams[session_id]
    
    async def _parse_stream_event(self, event: Dict[str, Any], session: StreamSession) -> Optional[StreamChunk]:
        """스트림 이벤트를 청크로 파싱"""
        try:
            event_type = event.get('event', event.get('type', 'unknown'))
            data = event.get('data', event.get('content', {}))
            
            # 에이전트 응답 이벤트 처리
            if event_type in ['a2a_response', 'mcp_sse_response', 'mcp_stdio_response']:
                agent_id = data.get('agent_id', 'unknown')
                content = data.get('content', {})
                
                # 콘텐츠 추출
                if isinstance(content, dict):
                    text = (content.get('text', '') or 
                           content.get('response', '') or 
                           content.get('message', '') or 
                           str(content))
                else:
                    text = str(content)
                
                return StreamChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=text,
                    chunk_type="agent_response",
                    source_agent=agent_id,
                    is_final=data.get('final', False)
                )
            
            # 상태 업데이트 이벤트
            elif event_type in ['status_update', 'routing']:
                status_text = data.get('message', str(data))
                
                return StreamChunk(
                    chunk_id=str(uuid.uuid4()),
                    content="",  # 상태는 콘텐츠에 포함하지 않음
                    chunk_type="status",
                    source_agent=data.get('agent_id', 'system'),
                    is_final=False
                )
            
            # 에러 이벤트
            elif event_type == 'error':
                error_msg = data.get('error', str(data))
                
                return StreamChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=f"⚠️ {error_msg}",
                    chunk_type="error",
                    source_agent=data.get('agent_id', 'system'),
                    is_final=True
                )
            
            return None
            
        except Exception as e:
            logger.error(f"스트림 이벤트 파싱 오류: {e}")
            return None
    
    async def _update_ui_realtime(
        self, 
        session: StreamSession, 
        ui_container: Optional[Any],
        chunks: List[StreamChunk]
    ):
        """실시간 UI 업데이트"""
        if not ui_container:
            return
        
        try:
            with ui_container:
                # 현재 누적된 응답 표시
                if session.accumulated_content:
                    # ChatGPT 스타일 메시지 박스
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #da3633 0%, #a21e1e 100%);
                        color: white;
                        border-radius: 10px;
                        padding: 1rem;
                        margin: 0.5rem 0;
                        margin-right: 20%;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    ">
                        <strong>🍒 CherryAI:</strong><br>
                        {session.accumulated_content}
                    </div>
                    """, unsafe_allow_html=True)
                
                # 현재 처리 중인 에이전트 표시
                if session.current_agent and session.state == StreamState.STREAMING:
                    st.info(f"🤖 {session.current_agent} 에이전트가 작업 중...")
                
                # 타이핑 인디케이터
                if (session.state == StreamState.TYPING and 
                    self.config['enable_typing_animation']):
                    st.markdown("💭 응답을 생성하는 중...")
        
        except Exception as e:
            logger.error(f"UI 업데이트 오류: {e}")
    
    async def _show_typing_indicator(self, session: StreamSession, ui_container: Optional[Any]):
        """타이핑 인디케이터 표시"""
        if not self.config['enable_typing_animation'] or not ui_container:
            return
        
        try:
            session.state = StreamState.TYPING
            session.typing_start = time.time()
            
            # 짧은 지연으로 타이핑 효과
            await asyncio.sleep(self.config['typing_indicator_delay'])
            
        except Exception as e:
            logger.error(f"타이핑 인디케이터 오류: {e}")
    
    def _update_stats(self, response_time: float, success: bool):
        """통계 업데이트"""
        if success:
            self.stats['successful_streams'] += 1
            # 평균 응답 시간 계산
            total_successful = self.stats['successful_streams']
            current_avg = self.stats['avg_response_time']
            self.stats['avg_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        else:
            self.stats['failed_streams'] += 1
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """스트리밍 통계 반환"""
        total_streams = self.stats['total_streams']
        success_rate = 0.0
        if total_streams > 0:
            success_rate = (self.stats['successful_streams'] / total_streams) * 100
        
        return {
            "total_streams": total_streams,
            "successful_streams": self.stats['successful_streams'],
            "failed_streams": self.stats['failed_streams'],
            "success_rate": round(success_rate, 1),
            "avg_response_time": round(self.stats['avg_response_time'], 2),
            "total_chunks": self.stats['total_chunks'],
            "active_streams": len(self.active_streams)
        }

# 전역 스트리밍 핸들러 인스턴스
_streaming_handler = None

def get_streaming_handler() -> RealtimeStreamingHandler:
    """스트리밍 핸들러 싱글톤 인스턴스 반환"""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = RealtimeStreamingHandler()
    return _streaming_handler

def process_query_with_streaming(
    query: str,
    broker_stream_generator: AsyncGenerator[Dict[str, Any], None],
    ui_container: Optional[Any] = None
) -> str:
    """쿼리를 실시간 스트리밍으로 처리 (동기 래퍼)"""
    
    handler = get_streaming_handler()
    session_id = handler.create_stream_session(query)
    
    try:
        # 비동기 처리를 동기적으로 실행
        try:
            loop = asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 태스크로 처리
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    handler.process_stream_async(session_id, broker_stream_generator, ui_container)
                )
                return future.result(timeout=60)
        except RuntimeError:
            # 실행 중인 루프가 없으면 새로 생성
            return asyncio.run(
                handler.process_stream_async(session_id, broker_stream_generator, ui_container)
            )
    
    except Exception as e:
        logger.error(f"스트리밍 처리 동기 래퍼 오류: {e}")
        return f"스트리밍 처리 중 오류가 발생했습니다: {str(e)}" 