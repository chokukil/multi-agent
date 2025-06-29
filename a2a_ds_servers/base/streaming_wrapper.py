"""
Streaming AI Data Science Team A2A Wrapper

스트리밍을 지원하는 AI DS Team 라이브러리 A2A 래퍼
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Type, AsyncIterator

from a2a.server.agent_execution.agent_executor import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils.message import new_agent_text_message

from .ai_ds_team_wrapper import AIDataScienceTeamWrapper
from .utils import (
    extract_user_input,
    format_streaming_chunk,
    create_error_response
)

logger = logging.getLogger(__name__)


class StreamingAIDataScienceWrapper(AIDataScienceTeamWrapper):
    """
    스트리밍을 지원하는 AI Data Science Team A2A 래퍼
    
    AI DS Team 에이전트의 스트리밍 기능을 A2A 프로토콜과 호환되도록 제공합니다.
    """
    
    def __init__(
        self,
        agent_class: Type[Any],
        agent_config: Optional[Dict[str, Any]] = None,
        agent_name: str = "AI DS Streaming Agent",
        streaming_config: Optional[Dict[str, Any]] = None
    ):
        """
        스트리밍 AI DS Team 래퍼 초기화
        
        Args:
            agent_class: AI DS Team 에이전트 클래스
            agent_config: 에이전트 설정
            agent_name: 에이전트 이름
            streaming_config: 스트리밍 설정
        """
        super().__init__(agent_class, agent_config, agent_name)
        self.streaming_config = streaming_config or {}
        self.chunk_size = self.streaming_config.get("chunk_size", 100)
        self.stream_delay = self.streaming_config.get("stream_delay", 0.1)
        
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        스트리밍 방식으로 A2A 실행
        
        Args:
            context: A2A 요청 컨텍스트
            event_queue: A2A 이벤트 큐
        """
        try:
            logger.info(f"🚀 Starting streaming {self.agent_name} execution")
            
            # 사용자 입력 추출
            user_input = extract_user_input(context)
            if not user_input:
                await self._send_error_response(
                    event_queue, 
                    "사용자 입력이 비어있습니다."
                )
                return
            
            # 스트리밍 실행 시작
            await self._execute_streaming(event_queue, user_input)
            
            logger.info(f"✅ Streaming {self.agent_name} execution completed")
            
        except Exception as e:
            logger.error(f"❌ Streaming {self.agent_name} execution failed: {e}", exc_info=True)
            await self._send_error_response(event_queue, str(e))
    
    async def _execute_streaming(self, event_queue: EventQueue, user_input: str) -> None:
        """
        스트리밍 방식으로 에이전트를 실행합니다.
        
        Args:
            event_queue: A2A 이벤트 큐
            user_input: 사용자 입력
        """
        try:
            # 시작 메시지
            await self._send_streaming_chunk(
                event_queue, 
                f"🔄 {self.agent_name} 스트리밍 시작...", 
                0, 
                False
            )
            
            # AI DS Team 에이전트가 스트리밍을 지원하는지 확인
            if hasattr(self.agent, 'stream') or hasattr(self.agent, 'astream'):
                await self._execute_native_streaming(event_queue, user_input)
            else:
                # 스트리밍을 지원하지 않는 경우 시뮬레이션
                await self._execute_simulated_streaming(event_queue, user_input)
                
        except Exception as e:
            logger.error(f"❌ Streaming execution error: {e}")
            await self._send_streaming_chunk(
                event_queue,
                f"❌ 스트리밍 실행 중 오류가 발생했습니다: {str(e)}",
                999,
                True
            )
    
    async def _execute_native_streaming(self, event_queue: EventQueue, user_input: str) -> None:
        """
        AI DS Team 에이전트의 네이티브 스트리밍을 사용합니다.
        
        Args:
            event_queue: A2A 이벤트 큐
            user_input: 사용자 입력
        """
        try:
            chunk_id = 1
            
            # 비동기 스트리밍 메서드 우선 사용
            if hasattr(self.agent, 'astream'):
                logger.info("🔧 Using agent.astream() for native streaming")
                async for chunk in self.agent.astream(user_input):
                    content = self._extract_chunk_content(chunk)
                    if content:
                        await self._send_streaming_chunk(
                            event_queue, 
                            content, 
                            chunk_id, 
                            False
                        )
                        chunk_id += 1
                        await asyncio.sleep(self.stream_delay)
            
            # 동기 스트리밍 메서드 사용
            elif hasattr(self.agent, 'stream'):
                logger.info("🔧 Using agent.stream() for native streaming")
                stream_result = self.agent.stream(user_input)
                
                # 스트리밍 결과가 제너레이터인 경우
                if hasattr(stream_result, '__iter__'):
                    for chunk in stream_result:
                        content = self._extract_chunk_content(chunk)
                        if content:
                            await self._send_streaming_chunk(
                                event_queue, 
                                content, 
                                chunk_id, 
                                False
                            )
                            chunk_id += 1
                            await asyncio.sleep(self.stream_delay)
            
            # 최종 완료 메시지
            await self._send_streaming_chunk(
                event_queue,
                f"✅ {self.agent_name} 스트리밍 완료",
                chunk_id,
                True
            )
            
        except Exception as e:
            logger.error(f"❌ Native streaming error: {e}")
            raise
    
    async def _execute_simulated_streaming(self, event_queue: EventQueue, user_input: str) -> None:
        """
        스트리밍을 지원하지 않는 에이전트의 경우 시뮬레이션합니다.
        
        Args:
            event_queue: A2A 이벤트 큐
            user_input: 사용자 입력
        """
        try:
            logger.info("🔧 Using simulated streaming for non-streaming agent")
            
            # 진행 상황 시뮬레이션
            progress_messages = [
                "📊 데이터 분석 중...",
                "🔍 패턴 탐지 중...",
                "📈 결과 생성 중...",
                "✨ 최종 처리 중..."
            ]
            
            chunk_id = 1
            for message in progress_messages:
                await self._send_streaming_chunk(
                    event_queue, 
                    message, 
                    chunk_id, 
                    False
                )
                chunk_id += 1
                await asyncio.sleep(self.stream_delay * 2)  # 조금 더 긴 지연
            
            # 실제 에이전트 실행
            result = await self._execute_agent(user_input)
            
            # 결과를 청크로 분할하여 스트리밍
            result_content = str(result)
            chunks = self._split_content_into_chunks(result_content)
            
            for i, chunk in enumerate(chunks):
                is_final = (i == len(chunks) - 1)
                await self._send_streaming_chunk(
                    event_queue, 
                    chunk, 
                    chunk_id, 
                    is_final
                )
                chunk_id += 1
                if not is_final:
                    await asyncio.sleep(self.stream_delay)
            
        except Exception as e:
            logger.error(f"❌ Simulated streaming error: {e}")
            raise
    
    def _extract_chunk_content(self, chunk: Any) -> str:
        """
        스트리밍 청크에서 내용을 추출합니다.
        
        Args:
            chunk: 스트리밍 청크
            
        Returns:
            str: 추출된 내용
        """
        try:
            # 딕셔너리 형태의 청크
            if isinstance(chunk, dict):
                # 메시지 필드들 확인
                for key in ['content', 'text', 'message', 'output']:
                    if key in chunk:
                        return str(chunk[key])
                        
                # messages 배열인 경우
                if 'messages' in chunk and isinstance(chunk['messages'], list):
                    if chunk['messages']:
                        last_msg = chunk['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            return str(last_msg.content)
                        elif isinstance(last_msg, str):
                            return last_msg
            
            # 문자열 청크
            elif isinstance(chunk, str):
                return chunk
            
            # 객체의 content 속성
            elif hasattr(chunk, 'content'):
                return str(chunk.content)
            
            # 기타 경우 문자열 변환
            else:
                return str(chunk)
                
        except Exception as e:
            logger.error(f"Error extracting chunk content: {e}")
            return str(chunk)
        
        return ""
    
    def _split_content_into_chunks(self, content: str) -> list[str]:
        """
        긴 내용을 청크로 분할합니다.
        
        Args:
            content: 분할할 내용
            
        Returns:
            list[str]: 분할된 청크 리스트
        """
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    async def _send_streaming_chunk(
        self, 
        event_queue: EventQueue, 
        content: str, 
        chunk_id: int, 
        is_final: bool
    ) -> None:
        """
        스트리밍 청크를 전송합니다.
        
        Args:
            event_queue: A2A 이벤트 큐
            content: 청크 내용
            chunk_id: 청크 ID
            is_final: 마지막 청크 여부
        """
        try:
            # 스트리밍 청크 포맷
            chunk_data = format_streaming_chunk(
                content, 
                chunk_id, 
                is_final, 
                self.agent_name
            )
            
            # 표시용 메시지 생성
            if is_final:
                display_message = f"[FINAL] {content}"
            else:
                display_message = f"[{chunk_id}] {content}"
            
            # A2A 메시지로 전송
            message = new_agent_text_message(display_message)
            await event_queue.enqueue_event(message)
            
            logger.debug(f"📤 Sent streaming chunk {chunk_id}: {content[:50]}...")
            
        except Exception as e:
            logger.error(f"Error sending streaming chunk: {e}")
    
    async def _send_error_response(self, event_queue: EventQueue, error_message: str) -> None:
        """스트리밍 오류 응답을 전송합니다."""
        try:
            error_response = create_error_response(error_message, self.agent_name)
            message = new_agent_text_message(f"❌ [STREAMING ERROR] {error_response['content']}")
            await event_queue.enqueue_event(message)
        except Exception as e:
            logger.error(f"Error sending streaming error response: {e}") 