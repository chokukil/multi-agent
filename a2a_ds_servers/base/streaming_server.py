"""
🍒 CherryAI - A2A Streaming Server
A2A SDK 0.2.9 표준 스트리밍 서버

A2A 에이전트들이 SSE를 통해 실시간 스트리밍을 제공하는 서버 기반 클래스
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import RequestContext
from a2a.types import TaskState
from dataclasses import dataclass

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    buffer_size: int = 1024
    timeout_seconds: int = 30
    heartbeat_interval: int = 10
    max_message_size: int = 1024 * 1024  # 1MB

class A2AStreamingServer:
    """A2A SDK 0.2.9 표준 스트리밍 서버"""
    
    def __init__(self, agent_executor, config: Optional[StreamingConfig] = None):
        """
        Args:
            agent_executor: A2A 에이전트 실행기
            config: 스트리밍 설정
        """
        self.agent_executor = agent_executor
        self.config = config or StreamingConfig()
        self.app = FastAPI(
            title="A2A Streaming Server",
            description="A2A SDK 0.2.9 표준 스트리밍 서버",
            version="1.0.0"
        )
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # SSE 엔드포인트 등록
        self._register_sse_endpoints()
    
    def _register_sse_endpoints(self):
        """SSE 엔드포인트 등록"""
        
        @self.app.post("/stream/{session_id}")
        async def stream_endpoint(session_id: str, request_data: Dict[str, Any]):
            """A2A 표준 SSE 스트리밍 엔드포인트"""
            
            try:
                # A2A 표준 요청 검증
                if not self._validate_a2a_request(request_data):
                    raise HTTPException(status_code=400, detail="Invalid A2A request format")
                
                # 스트리밍 응답 생성
                return StreamingResponse(
                    self._create_sse_stream(session_id, request_data),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
                
            except Exception as e:
                logger.error(f"SSE 엔드포인트 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stream/{session_id}/status")
        async def stream_status(session_id: str):
            """스트림 상태 확인"""
            
            if session_id in self.active_streams:
                stream_info = self.active_streams[session_id]
                return {
                    "session_id": session_id,
                    "status": stream_info.get("status", "unknown"),
                    "started_at": stream_info.get("started_at"),
                    "last_update": stream_info.get("last_update"),
                    "message_count": stream_info.get("message_count", 0)
                }
            else:
                return {"session_id": session_id, "status": "not_found"}
    
    def _validate_a2a_request(self, request_data: Dict[str, Any]) -> bool:
        """A2A 표준 요청 형식 검증"""
        
        required_fields = ["messageId", "role", "parts"]
        
        for field in required_fields:
            if field not in request_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # parts 검증
        parts = request_data.get("parts", [])
        if not isinstance(parts, list) or not parts:
            logger.error("Invalid or empty parts")
            return False
        
        for part in parts:
            if not isinstance(part, dict) or "kind" not in part:
                logger.error(f"Invalid part format: {part}")
                return False
        
        return True
    
    async def _create_sse_stream(
        self, 
        session_id: str, 
        request_data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """A2A 표준 SSE 스트림 생성"""
        
        # 스트림 상태 추적
        self.active_streams[session_id] = {
            "status": "starting",
            "started_at": time.time(),
            "last_update": time.time(),
            "message_count": 0
        }
        
        try:
            # 작업 시작 이벤트
            yield self._create_sse_event(
                event_type="start",
                data={
                    "message": "A2A 에이전트 분석을 시작합니다...",
                    "session_id": session_id,
                    "final": False
                }
            )
            
            self._update_stream_status(session_id, "processing")
            
            # A2A 에이전트 실행 및 중간 결과 스트리밍
            async for chunk in self._execute_agent_with_streaming(session_id, request_data):
                yield self._create_sse_event(
                    event_type="progress",
                    data=chunk
                )
                
                self._update_stream_status(session_id, "streaming")
                
                # 하트비트 체크
                if time.time() - self.active_streams[session_id]["last_update"] > self.config.heartbeat_interval:
                    yield self._create_sse_event(
                        event_type="heartbeat",
                        data={"timestamp": time.time()}
                    )
            
            # 완료 이벤트
            yield self._create_sse_event(
                event_type="complete",
                data={
                    "message": "분석이 완료되었습니다.",
                    "session_id": session_id,
                    "final": True
                }
            )
            
            self._update_stream_status(session_id, "completed")
            
        except Exception as e:
            logger.error(f"스트리밍 중 오류 발생: {e}")
            
            # 에러 이벤트
            yield self._create_sse_event(
                event_type="error",
                data={
                    "error": str(e),
                    "session_id": session_id,
                    "final": True
                }
            )
            
            self._update_stream_status(session_id, "error")
            
        finally:
            # 스트림 정리
            self._cleanup_stream(session_id)
    
    async def _execute_agent_with_streaming(
        self, 
        session_id: str, 
        request_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """에이전트 실행 및 스트리밍"""
        
        try:
            # RequestContext 생성 (A2A SDK 표준)
            context = RequestContext(
                method="POST",  # HTTP 메소드
                message=self._create_message_from_request(request_data)
            )
            
            # TaskUpdater 생성 (실시간 상태 업데이트용)
            task_updater = TaskUpdater(session_id)
            
            # 에이전트 실행 시작
            await task_updater.update_status(
                TaskState.working,
                message="에이전트 실행 중..."
            )
            
            # 에이전트 스트리밍 실행
            if hasattr(self.agent_executor, 'stream'):
                # 스트리밍 지원 에이전트
                async for chunk in self.agent_executor.stream(context):
                    chunk_data = {
                        "content": getattr(chunk, 'content', str(chunk)),
                        "metadata": getattr(chunk, 'metadata', {}),
                        "timestamp": time.time(),
                        "final": False
                    }
                    
                    # TaskUpdater로 진행 상황 업데이트
                    await task_updater.update_status(
                        TaskState.working,
                        message=f"처리 중: {chunk_data['content'][:50]}..."
                    )
                    
                    yield chunk_data
                    
            else:
                # 일반 에이전트 (non-streaming)
                result = await self.agent_executor.execute(context)
                
                # 결과를 청크로 분할하여 스트리밍 시뮬레이션
                content = str(result)
                chunk_size = 100
                
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    
                    yield {
                        "content": chunk,
                        "metadata": {"chunk_index": i // chunk_size},
                        "timestamp": time.time(),
                        "final": i + chunk_size >= len(content)
                    }
                    
                    # 스트리밍 시뮬레이션을 위한 지연
                    await asyncio.sleep(0.1)
            
            # 완료 상태 업데이트
            await task_updater.update_status(TaskState.completed)
            
        except Exception as e:
            logger.error(f"에이전트 실행 오류: {e}")
            
            # 에러 상태 업데이트
            if 'task_updater' in locals():
                await task_updater.update_status(TaskState.failed, str(e))
            
            # 에러 이벤트 생성
            yield {
                "error": str(e),
                "timestamp": time.time(),
                "final": True
            }
    
    def _create_message_from_request(self, request_data: Dict[str, Any]):
        """A2A 요청에서 메시지 객체 생성"""
        
        # A2A SDK 메시지 형식으로 변환
        # 실제 구현에서는 A2A SDK의 Message 클래스 사용
        
        class MockMessage:
            def __init__(self, parts):
                self.parts = parts
        
        class MockPart:
            def __init__(self, part_data):
                self.root = MockPartRoot(part_data)
        
        class MockPartRoot:
            def __init__(self, part_data):
                self.kind = part_data.get("kind", "text")
                self.text = part_data.get("text", "")
        
        parts = []
        for part_data in request_data.get("parts", []):
            parts.append(MockPart(part_data))
        
        return MockMessage(parts)
    
    def _create_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """A2A 표준 SSE 이벤트 생성"""
        
        # JSON 직렬화
        data_json = json.dumps(data, ensure_ascii=False)
        
        # SSE 형식
        return f"event: {event_type}\ndata: {data_json}\n\n"
    
    def _update_stream_status(self, session_id: str, status: str):
        """스트림 상태 업데이트"""
        
        if session_id in self.active_streams:
            self.active_streams[session_id]["status"] = status
            self.active_streams[session_id]["last_update"] = time.time()
            self.active_streams[session_id]["message_count"] += 1
    
    def _cleanup_stream(self, session_id: str):
        """스트림 정리"""
        
        if session_id in self.active_streams:
            logger.info(f"스트림 정리: {session_id}")
            del self.active_streams[session_id]
    
    def get_app(self) -> FastAPI:
        """FastAPI 앱 반환"""
        return self.app
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """활성 스트림 상태 반환"""
        return self.active_streams.copy()
    
    async def cleanup_inactive_streams(self, timeout_seconds: int = 300):
        """비활성 스트림 정리"""
        
        current_time = time.time()
        inactive_streams = []
        
        for session_id, stream_info in self.active_streams.items():
            last_update = stream_info.get("last_update", 0)
            if current_time - last_update > timeout_seconds:
                inactive_streams.append(session_id)
        
        for session_id in inactive_streams:
            self._cleanup_stream(session_id)
            logger.info(f"비활성 스트림 정리: {session_id}")
    
    async def shutdown(self):
        """서버 종료"""
        
        # 모든 활성 스트림 정리
        for session_id in list(self.active_streams.keys()):
            self._cleanup_stream(session_id)
        
        logger.info("A2A 스트리밍 서버 종료 완료") 