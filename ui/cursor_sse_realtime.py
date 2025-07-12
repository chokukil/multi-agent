"""
Cursor-Style SSE Real-time Update System
A2A SDK 0.2.9 표준 준수 Server-Sent Events 기반 실시간 동기화 시스템
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Callable, Set, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st

# A2A SDK 0.2.9 imports
try:
    from a2a.types import TextPart, DataPart, FilePart, Part
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
    from a2a.server.application import A2AFastAPIApplication
    from a2a.server.request_handler import DefaultRequestHandler
    from a2a.server.agent_executor import AgentExecutor
    from a2a.server.request_context import RequestContext
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logging.warning("A2A SDK not available - using mock implementation")
    
    # Mock implementations
    class Part:
        def __init__(self, **kwargs):
            self.root = type('Root', (), kwargs)()
    
    class TextPart(Part):
        def __init__(self, text: str):
            super().__init__(text=text, kind='text')
    
    class TaskUpdater:
        async def update_status(self, state: str, message: str = ""):
            pass
    
    class RequestContext:
        def __init__(self):
            self.message = None
            self.task_id = None
            self.session_id = None
    
    class AgentExecutor:
        async def execute(self, context: RequestContext, task_updater: TaskUpdater):
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSEEventType(Enum):
    """SSE 이벤트 타입"""
    AGENT_STATUS_UPDATE = "agent_status_update"
    THOUGHT_PROCESS_UPDATE = "thought_process_update"
    MCP_TOOL_UPDATE = "mcp_tool_update"
    CODE_STREAMING_UPDATE = "code_streaming_update"
    SYSTEM_METRICS_UPDATE = "system_metrics_update"
    ERROR_NOTIFICATION = "error_notification"
    HEARTBEAT = "heartbeat"
    CONNECTION_STATUS = "connection_status"

class ConnectionStatus(Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class SSEMessage:
    """SSE 메시지 구조"""
    id: str
    event_type: SSEEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    component_id: Optional[str] = None
    
    def to_sse_format(self) -> str:
        """SSE 포맷으로 변환"""
        sse_data = {
            "id": self.id,
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp
        }
        if self.component_id:
            sse_data["component_id"] = self.component_id
        
        return f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

@dataclass
class SSESubscription:
    """SSE 구독 정보"""
    subscription_id: str
    component_ids: Set[str] = field(default_factory=set)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class A2ASSEEventQueue:
    """A2A SDK EventQueue SSE 통합"""
    
    def __init__(self, sse_manager):
        self.sse_manager = sse_manager
        self.event_queue = asyncio.Queue()
        self.is_running = False
        self.processing_task = None
        
        if A2A_AVAILABLE:
            self.task_store = InMemoryTaskStore()
        
    async def start(self):
        """이벤트 큐 처리 시작"""
        if not self.is_running:
            self.is_running = True
            self.processing_task = asyncio.create_task(self._process_events())
            logger.info("A2A SSE EventQueue integration started")
    
    async def stop(self):
        """이벤트 큐 처리 중지"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            logger.info("A2A SSE EventQueue integration stopped")
    
    async def enqueue_event(self, event_type: SSEEventType, data: Dict[str, Any], 
                           component_id: Optional[str] = None):
        """이벤트 큐에 이벤트 추가"""
        message = SSEMessage(
            id=str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            component_id=component_id
        )
        await self.event_queue.put(message)
    
    async def _process_events(self):
        """이벤트 큐 처리 루프"""
        while self.is_running:
            try:
                # 이벤트 큐에서 메시지 가져오기
                message = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # SSE를 통해 클라이언트에게 브로드캐스트
                await self.sse_manager.broadcast_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing A2A SSE event: {e}")
    
    async def handle_task_update(self, task_id: str, status: str, message: str):
        """TaskUpdater 이벤트 처리"""
        await self.enqueue_event(
            SSEEventType.AGENT_STATUS_UPDATE,
            {
                "task_id": task_id,
                "status": status,
                "message": message,
                "timestamp": time.time()
            }
        )

class CursorSSEManager:
    """Cursor 스타일 SSE 관리자"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.subscriptions: Dict[str, SSESubscription] = {}
        self.app = None
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # A2A 통합
        self.a2a_integration = A2ASSEEventQueue(self)
        
        # 메시지 핸들러
        self.message_handlers: Dict[SSEEventType, List[Callable]] = {
            event_type: [] for event_type in SSEEventType
        }
        
        # 이벤트 브로드캐스트 큐
        self.broadcast_queue = asyncio.Queue()
        self.broadcast_task = None
        
        # 컴포넌트 상태 저장
        self.component_states: Dict[str, Dict[str, Any]] = {}
        
        # 연결 상태 콜백
        self.connection_callbacks: List[Callable] = []
    
    def create_fastapi_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        app = FastAPI(title="Cursor SSE Real-time System")
        
        # CORS 설정
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/sse/stream")
        async def sse_stream(request: Request, components: str = ""):
            """SSE 스트리밍 엔드포인트"""
            subscription_id = str(uuid.uuid4())
            component_ids = set(components.split(",")) if components else set()
            
            # 구독 정보 저장
            self.subscriptions[subscription_id] = SSESubscription(
                subscription_id=subscription_id,
                component_ids=component_ids
            )
            
            logger.info(f"New SSE subscription: {subscription_id} for components: {component_ids}")
            
            return StreamingResponse(
                self._generate_sse_stream(subscription_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        
        @app.get("/sse/status")
        async def get_status():
            """시스템 상태 확인"""
            return {
                "status": "running" if self.is_running else "stopped",
                "subscriptions": len(self.subscriptions),
                "components": list(self.component_states.keys())
            }
        
        @app.post("/sse/broadcast")
        async def broadcast_event(event_data: Dict[str, Any]):
            """이벤트 브로드캐스트"""
            try:
                event_type = SSEEventType(event_data.get("type", "system_metrics_update"))
                message = SSEMessage(
                    id=str(uuid.uuid4()),
                    event_type=event_type,
                    data=event_data.get("data", {}),
                    component_id=event_data.get("component_id")
                )
                await self.broadcast_message(message)
                return {"success": True}
            except Exception as e:
                logger.error(f"Error broadcasting event: {e}")
                return {"success": False, "error": str(e)}
        
        return app
    
    async def _generate_sse_stream(self, subscription_id: str) -> AsyncIterator[str]:
        """SSE 스트림 생성"""
        try:
            # 연결 확인 메시지 전송
            yield f"data: {json.dumps({'type': 'connection_status', 'status': 'connected', 'subscription_id': subscription_id})}\n\n"
            
            # 구독 정보 가져오기
            subscription = self.subscriptions.get(subscription_id)
            if not subscription:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid subscription'})}\n\n"
                return
            
            last_heartbeat = time.time()
            
            while subscription_id in self.subscriptions:
                try:
                    # 하트비트 전송 (30초마다)
                    current_time = time.time()
                    if current_time - last_heartbeat > 30:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': current_time})}\n\n"
                        last_heartbeat = current_time
                    
                    # 브로드캐스트 큐에서 메시지 확인
                    try:
                        message = await asyncio.wait_for(self.broadcast_queue.get(), timeout=1.0)
                        
                        # 구독 필터링
                        if (not subscription.component_ids or 
                            message.component_id in subscription.component_ids):
                            yield message.to_sse_format()
                        
                    except asyncio.TimeoutError:
                        continue
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error generating SSE stream: {e}")
        finally:
            # 구독 정리
            if subscription_id in self.subscriptions:
                del self.subscriptions[subscription_id]
                logger.info(f"Removed SSE subscription: {subscription_id}")
    
    async def start_server(self):
        """SSE 서버 시작"""
        if self.is_running:
            return
            
        self.is_running = True
        self.app = self.create_fastapi_app()
        
        # A2A 통합 시작
        await self.a2a_integration.start()
        
        # 브로드캐스트 작업 시작
        self.broadcast_task = asyncio.create_task(self._broadcast_worker())
        
        logger.info(f"SSE server started on {self.host}:{self.port}")
    
    async def stop_server(self):
        """SSE 서버 중지"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # A2A 통합 중지
        await self.a2a_integration.stop()
        
        # 브로드캐스트 작업 중지
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        
        # 모든 구독 정리
        self.subscriptions.clear()
        
        logger.info("SSE server stopped")
    
    async def broadcast_message(self, message: SSEMessage):
        """메시지 브로드캐스트"""
        await self.broadcast_queue.put(message)
        
        # 컴포넌트 상태 업데이트
        if message.component_id:
            self.component_states[message.component_id] = {
                "last_update": message.timestamp,
                "last_event": message.event_type.value,
                "data": message.data
            }
    
    async def _broadcast_worker(self):
        """브로드캐스트 워커"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)  # 작은 지연
            except asyncio.CancelledError:
                break
    
    def subscribe_to_component(self, subscription_id: str, component_id: str):
        """컴포넌트 구독"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].component_ids.add(component_id)
    
    def unsubscribe_from_component(self, subscription_id: str, component_id: str):
        """컴포넌트 구독 해제"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].component_ids.discard(component_id)
    
    def add_message_handler(self, event_type: SSEEventType, handler: Callable):
        """메시지 핸들러 추가"""
        self.message_handlers[event_type].append(handler)
    
    def remove_message_handler(self, event_type: SSEEventType, handler: Callable):
        """메시지 핸들러 제거"""
        if handler in self.message_handlers[event_type]:
            self.message_handlers[event_type].remove(handler)
    
    def add_connection_callback(self, callback: Callable):
        """연결 콜백 추가"""
        self.connection_callbacks.append(callback)
    
    def get_subscription_count(self) -> int:
        """구독 수 반환"""
        return len(self.subscriptions)
    
    def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """구독 정보 반환"""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            return {
                "subscription_id": subscription.subscription_id,
                "component_ids": list(subscription.component_ids),
                "last_activity": subscription.last_activity,
                "metadata": subscription.metadata
            }
        return None

class CursorSSERealTimeSync:
    """Cursor 스타일 SSE 실시간 동기화"""
    
    def __init__(self, sse_manager: CursorSSEManager):
        self.sse_manager = sse_manager
        self.sync_rules: Dict[str, List[str]] = {}
        self.component_states: Dict[str, Dict[str, Any]] = {}
        
        # 기본 동기화 규칙 설정
        self._setup_default_sync_rules()
        
        # 메시지 핸들러 등록
        self._register_message_handlers()
    
    def _setup_default_sync_rules(self):
        """기본 동기화 규칙 설정"""
        self.sync_rules.update({
            "agent_cards": ["thought_stream", "mcp_monitoring"],
            "thought_stream": ["agent_cards", "code_streaming"],
            "mcp_monitoring": ["agent_cards", "system_metrics"],
            "code_streaming": ["thought_stream", "collaboration_network"],
            "collaboration_network": ["agent_cards", "mcp_monitoring"]
        })
    
    def _register_message_handlers(self):
        """메시지 핸들러 등록"""
        self.sse_manager.add_message_handler(
            SSEEventType.AGENT_STATUS_UPDATE,
            self._handle_agent_update
        )
        self.sse_manager.add_message_handler(
            SSEEventType.THOUGHT_PROCESS_UPDATE,
            self._handle_thought_update
        )
        self.sse_manager.add_message_handler(
            SSEEventType.MCP_TOOL_UPDATE,
            self._handle_mcp_update
        )
        self.sse_manager.add_message_handler(
            SSEEventType.CODE_STREAMING_UPDATE,
            self._handle_code_update
        )
    
    async def _handle_agent_update(self, message: SSEMessage):
        """에이전트 업데이트 처리"""
        component_id = message.component_id or "agent_cards"
        await self._sync_to_related_components(component_id, message.data)
    
    async def _handle_thought_update(self, message: SSEMessage):
        """사고 과정 업데이트 처리"""
        component_id = message.component_id or "thought_stream"
        await self._sync_to_related_components(component_id, message.data)
    
    async def _handle_mcp_update(self, message: SSEMessage):
        """MCP 도구 업데이트 처리"""
        component_id = message.component_id or "mcp_monitoring"
        await self._sync_to_related_components(component_id, message.data)
    
    async def _handle_code_update(self, message: SSEMessage):
        """코드 스트리밍 업데이트 처리"""
        component_id = message.component_id or "code_streaming"
        await self._sync_to_related_components(component_id, message.data)
    
    async def _sync_to_related_components(self, source_component: str, data: Dict[str, Any]):
        """관련 컴포넌트로 동기화"""
        if source_component in self.sync_rules:
            related_components = self.sync_rules[source_component]
            
            # 동기화 메시지 생성
            sync_message = SSEMessage(
                id=str(uuid.uuid4()),
                event_type=SSEEventType.SYSTEM_METRICS_UPDATE,
                data={
                    "sync_source": source_component,
                    "sync_data": data,
                    "timestamp": time.time()
                }
            )
            
            # 관련 컴포넌트에 브로드캐스트
            for related_component in related_components:
                sync_message.component_id = related_component
                await self.sse_manager.broadcast_message(sync_message)
    
    def update_sync_rules(self, component: str, related_components: List[str]):
        """동기화 규칙 업데이트"""
        self.sync_rules[component] = related_components
    
    def get_component_state(self, component_id: str) -> Optional[Dict[str, Any]]:
        """컴포넌트 상태 가져오기"""
        return self.component_states.get(component_id)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """모든 컴포넌트 상태 가져오기"""
        return self.component_states.copy()

class CursorSSERealtimeManager:
    """Cursor 스타일 SSE 실시간 관리자"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.sse_manager = CursorSSEManager(host, port)
        self.sync_system = CursorSSERealTimeSync(self.sse_manager)
    
    async def start(self):
        """실시간 시스템 시작"""
        await self.sse_manager.start_server()
    
    async def stop(self):
        """실시간 시스템 중지"""
        await self.sse_manager.stop_server()
    
    async def send_agent_update(self, agent_id: str, status: str, data: Dict[str, Any]):
        """에이전트 업데이트 전송"""
        await self.sse_manager.a2a_integration.enqueue_event(
            SSEEventType.AGENT_STATUS_UPDATE,
            {
                "agent_id": agent_id,
                "status": status,
                "data": data,
                "timestamp": time.time()
            }
        )
    
    async def send_thought_update(self, thought_id: str, status: str, content: str):
        """사고 과정 업데이트 전송"""
        await self.sse_manager.a2a_integration.enqueue_event(
            SSEEventType.THOUGHT_PROCESS_UPDATE,
            {
                "thought_id": thought_id,
                "status": status,
                "content": content,
                "timestamp": time.time()
            }
        )
    
    async def send_mcp_update(self, tool_id: str, status: str, metrics: Dict[str, Any]):
        """MCP 도구 업데이트 전송"""
        await self.sse_manager.a2a_integration.enqueue_event(
            SSEEventType.MCP_TOOL_UPDATE,
            {
                "tool_id": tool_id,
                "status": status,
                "metrics": metrics,
                "timestamp": time.time()
            }
        )
    
    async def send_code_update(self, block_id: str, status: str, content: str):
        """코드 스트리밍 업데이트 전송"""
        await self.sse_manager.a2a_integration.enqueue_event(
            SSEEventType.CODE_STREAMING_UPDATE,
            {
                "block_id": block_id,
                "status": status,
                "content": content,
                "timestamp": time.time()
            }
        )
    
    def get_connection_count(self) -> int:
        """연결 수 반환"""
        return self.sse_manager.get_subscription_count()
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "running": self.sse_manager.is_running,
            "subscriptions": self.sse_manager.get_subscription_count(),
            "components": list(self.sse_manager.component_states.keys()),
            "sync_rules": self.sync_system.sync_rules,
            "timestamp": time.time()
        }

# 전역 인스턴스
_cursor_realtime_manager = None

def get_cursor_sse_realtime() -> CursorSSERealtimeManager:
    """Cursor SSE 실시간 관리자 싱글톤 인스턴스"""
    global _cursor_realtime_manager
    if _cursor_realtime_manager is None:
        _cursor_realtime_manager = CursorSSERealtimeManager()
    return _cursor_realtime_manager

def initialize_sse_realtime_in_streamlit():
    """Streamlit에서 SSE 실시간 시스템 초기화"""
    if 'sse_realtime_manager' not in st.session_state:
        st.session_state.sse_realtime_manager = get_cursor_sse_realtime()
    
    return st.session_state.sse_realtime_manager

def update_sse_realtime_status():
    """SSE 실시간 상태 업데이트"""
    manager = get_cursor_sse_realtime()
    return manager.get_system_status() 