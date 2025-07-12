"""
A2A SSE Streaming System Unit Tests
pytest 기반 단위 테스트 - 개별 함수/클래스 기능 검증
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List
import uuid

# 시스템 경로 설정
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ui'))

# 테스트 대상 임포트
from a2a_sse_streaming_system import (
    A2ASSEStreamingExecutor, 
    A2ASSEStreamingSystem,
    SSEEvent,
    A2AMessage,
    A2ATaskStatus,
    A2AArtifact,
    TaskState,
    get_a2a_sse_streaming_system
)

from cursor_sse_realtime import (
    CursorSSEManager,
    CursorSSERealtimeManager,
    SSEEventType,
    SSEMessage,
    SSESubscription,
    A2ASSEEventQueue,
    get_cursor_sse_realtime
)


class TestSSEEvent:
    """SSE 이벤트 클래스 테스트"""
    
    def test_sse_event_creation(self):
        """SSE 이벤트 생성 테스트"""
        event = SSEEvent(
            event_id="test-123",
            event_type="test_event",
            data={"message": "test"}
        )
        
        assert event.event_id == "test-123"
        assert event.event_type == "test_event"
        assert event.data == {"message": "test"}
        assert isinstance(event.timestamp, float)
    
    def test_sse_event_to_sse_format(self):
        """SSE 포맷 변환 테스트"""
        event = SSEEvent(
            event_id="test-123",
            event_type="test_event",
            data={"message": "test"}
        )
        
        sse_format = event.to_sse_format()
        
        assert sse_format.startswith("data: ")
        assert sse_format.endswith("\n\n")
        
        # JSON 파싱 테스트
        json_data = json.loads(sse_format.replace("data: ", "").strip())
        assert json_data["message"] == "test"


class TestSSEMessage:
    """SSE 메시지 클래스 테스트"""
    
    def test_sse_message_creation(self):
        """SSE 메시지 생성 테스트"""
        message = SSEMessage(
            id="msg-123",
            event_type=SSEEventType.AGENT_STATUS_UPDATE,
            data={"status": "working"}
        )
        
        assert message.id == "msg-123"
        assert message.event_type == SSEEventType.AGENT_STATUS_UPDATE
        assert message.data == {"status": "working"}
        assert isinstance(message.timestamp, float)
    
    def test_sse_message_to_sse_format(self):
        """SSE 메시지 포맷 변환 테스트"""
        message = SSEMessage(
            id="msg-123",
            event_type=SSEEventType.AGENT_STATUS_UPDATE,
            data={"status": "working"},
            component_id="test-component"
        )
        
        sse_format = message.to_sse_format()
        
        assert sse_format.startswith("data: ")
        assert sse_format.endswith("\n\n")
        
        # JSON 파싱 테스트
        json_data = json.loads(sse_format.replace("data: ", "").strip())
        assert json_data["id"] == "msg-123"
        assert json_data["type"] == "agent_status_update"
        assert json_data["data"]["status"] == "working"
        assert json_data["component_id"] == "test-component"


class TestA2ATaskStatus:
    """A2A 태스크 상태 테스트"""
    
    def test_task_status_creation(self):
        """태스크 상태 생성 테스트"""
        status = A2ATaskStatus(
            state=TaskState.WORKING,
            message=None
        )
        
        assert status.state == TaskState.WORKING
        assert status.message is None
        assert isinstance(status.timestamp, str)
    
    def test_task_status_to_dict(self):
        """태스크 상태 딕셔너리 변환 테스트"""
        status = A2ATaskStatus(
            state=TaskState.COMPLETED,
            message=None
        )
        
        status_dict = status.to_dict()
        
        assert status_dict["state"] == "completed"
        assert "timestamp" in status_dict
        # message가 None일 때는 딕셔너리에 포함되지 않음
        assert "message" not in status_dict


class TestA2AArtifact:
    """A2A 아티팩트 테스트"""
    
    def test_artifact_creation(self):
        """아티팩트 생성 테스트"""
        artifact = A2AArtifact(
            artifact_id="artifact-123",
            name="test-artifact",
            parts=[{"kind": "text", "text": "test content"}],
            metadata={"type": "analysis"}
        )
        
        assert artifact.artifact_id == "artifact-123"
        assert artifact.name == "test-artifact"
        assert len(artifact.parts) == 1
        assert artifact.metadata["type"] == "analysis"
    
    def test_artifact_to_dict(self):
        """아티팩트 딕셔너리 변환 테스트"""
        artifact = A2AArtifact(
            artifact_id="artifact-123",
            name="test-artifact",
            parts=[{"kind": "text", "text": "test content"}]
        )
        
        artifact_dict = artifact.to_dict()
        
        assert artifact_dict["artifactId"] == "artifact-123"
        assert artifact_dict["name"] == "test-artifact"
        assert len(artifact_dict["parts"]) == 1
        assert artifact_dict["parts"][0]["text"] == "test content"


class TestA2ASSEStreamingExecutor:
    """A2A SSE 스트리밍 실행자 테스트"""
    
    @pytest.fixture
    def executor(self):
        """테스트용 실행자 인스턴스"""
        return A2ASSEStreamingExecutor()
    
    @pytest.fixture
    def mock_context(self):
        """모의 요청 컨텍스트"""
        context = Mock()
        context.message = Mock()
        context.message.parts = [Mock()]
        context.message.parts[0].root = Mock()
        context.message.parts[0].root.text = "test input"
        context.task_id = "task-123"
        context.session_id = "session-123"
        return context
    
    @pytest.fixture
    def mock_task_updater(self):
        """모의 태스크 업데이터"""
        updater = AsyncMock()
        updater.update_status = AsyncMock()
        updater.add_artifact = AsyncMock()
        return updater
    
    def test_executor_initialization(self, executor):
        """실행자 초기화 테스트"""
        assert executor.is_streaming == False
        assert executor.stream_queue is not None
        assert executor.task_store is not None
    
    @pytest.mark.asyncio
    async def test_executor_execute(self, executor, mock_context, mock_task_updater):
        """실행자 실행 테스트"""
        # execute 메서드 호출
        await executor.execute(mock_context, mock_task_updater)
        
        # TaskUpdater 호출 검증
        mock_task_updater.update_status.assert_called()
        
        # 스트리밍 상태 검증
        assert executor.is_streaming == False  # 완료 후 False
    
    @pytest.mark.asyncio
    async def test_executor_cancel(self, executor, mock_context):
        """실행자 취소 테스트"""
        # 스트리밍 시작
        executor.is_streaming = True
        
        # 취소 호출
        await executor.cancel(mock_context)
        
        # 스트리밍 중지 검증
        assert executor.is_streaming == False
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, executor, mock_context):
        """스트림 처리 테스트"""
        events = []
        
        # 스트림 처리 이벤트 수집
        async for event in executor._stream_processing("test input", mock_context):
            events.append(event)
            if len(events) >= 3:  # 처음 3개 이벤트만 수집
                break
        
        # 이벤트 검증
        assert len(events) >= 3
        assert all(isinstance(event, SSEEvent) for event in events)
        assert events[0].event_type == "status_update"


class TestA2ASSEStreamingSystem:
    """A2A SSE 스트리밍 시스템 테스트"""
    
    @pytest.fixture
    def system(self):
        """테스트용 시스템 인스턴스"""
        return A2ASSEStreamingSystem()
    
    def test_system_initialization(self, system):
        """시스템 초기화 테스트"""
        assert system.executor is not None
        assert system.app is not None
        assert system.active_streams == {}
        # A2ASSEStreamingSystem에는 host, port 속성이 없음
    
    def test_create_fastapi_app(self, system):
        """FastAPI 앱 생성 테스트"""
        app = system._create_fastapi_app()
        
        # 라우트 검증
        routes = [route.path for route in app.routes]
        assert "/stream" in routes
        assert "/.well-known/agent.json" in routes
    
    @pytest.mark.asyncio
    async def test_handle_streaming_request(self, system):
        """스트리밍 요청 처리 테스트"""
        request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test message"}]
                }
            }
        }
        
        response = await system.handle_streaming_request(request)
        
        # StreamingResponse 검증
        assert hasattr(response, 'body_iterator')
        assert response.media_type == "text/event-stream"
    
    def test_get_singleton(self):
        """싱글톤 인스턴스 테스트"""
        system1 = get_a2a_sse_streaming_system()
        system2 = get_a2a_sse_streaming_system()
        
        assert system1 is system2


class TestCursorSSEManager:
    """Cursor SSE 관리자 테스트"""
    
    @pytest.fixture
    def manager(self):
        """테스트용 관리자 인스턴스"""
        return CursorSSEManager()
    
    def test_manager_initialization(self, manager):
        """관리자 초기화 테스트"""
        assert manager.host == "localhost"
        assert manager.port == 8765
        assert manager.is_running == False
        assert len(manager.subscriptions) == 0
        assert manager.a2a_integration is not None
    
    def test_create_fastapi_app(self, manager):
        """FastAPI 앱 생성 테스트"""
        app = manager.create_fastapi_app()
        
        # 라우트 검증
        routes = [route.path for route in app.routes]
        assert "/sse/stream" in routes
        assert "/sse/status" in routes
        assert "/sse/broadcast" in routes
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, manager):
        """메시지 브로드캐스트 테스트"""
        message = SSEMessage(
            id="msg-123",
            event_type=SSEEventType.AGENT_STATUS_UPDATE,
            data={"status": "working"}
        )
        
        # 브로드캐스트 실행
        await manager.broadcast_message(message)
        
        # 큐에 메시지 추가 검증
        assert manager.broadcast_queue.qsize() > 0
    
    def test_subscription_management(self, manager):
        """구독 관리 테스트"""
        subscription_id = "sub-123"
        
        # 구독 추가
        manager.subscribe_to_component(subscription_id, "test-component")
        
        # 구독 해제
        manager.unsubscribe_from_component(subscription_id, "test-component")
        
        # 구독 수 검증
        assert manager.get_subscription_count() == 0


class TestA2ASSEEventQueue:
    """A2A SSE 이벤트 큐 테스트"""
    
    @pytest.fixture
    def event_queue(self):
        """테스트용 이벤트 큐"""
        manager = Mock()
        manager.broadcast_message = AsyncMock()
        return A2ASSEEventQueue(manager)
    
    @pytest.mark.asyncio
    async def test_event_queue_start_stop(self, event_queue):
        """이벤트 큐 시작/중지 테스트"""
        # 시작
        await event_queue.start()
        assert event_queue.is_running == True
        
        # 중지
        await event_queue.stop()
        assert event_queue.is_running == False
    
    @pytest.mark.asyncio
    async def test_enqueue_event(self, event_queue):
        """이벤트 큐 추가 테스트"""
        await event_queue.enqueue_event(
            SSEEventType.AGENT_STATUS_UPDATE,
            {"status": "working"},
            "test-component"
        )
        
        # 큐 크기 검증
        assert event_queue.event_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_handle_task_update(self, event_queue):
        """태스크 업데이트 처리 테스트"""
        await event_queue.handle_task_update(
            "task-123",
            "working",
            "Processing request..."
        )
        
        # 큐에 이벤트 추가 검증
        assert event_queue.event_queue.qsize() == 1


class TestCursorSSERealtimeManager:
    """Cursor SSE 실시간 관리자 테스트"""
    
    @pytest.fixture
    def realtime_manager(self):
        """테스트용 실시간 관리자"""
        return CursorSSERealtimeManager()
    
    def test_realtime_manager_initialization(self, realtime_manager):
        """실시간 관리자 초기화 테스트"""
        assert realtime_manager.sse_manager is not None
        assert realtime_manager.sync_system is not None
    
    @pytest.mark.asyncio
    async def test_send_agent_update(self, realtime_manager):
        """에이전트 업데이트 전송 테스트"""
        with patch.object(realtime_manager.sse_manager.a2a_integration, 'enqueue_event') as mock_enqueue:
            await realtime_manager.send_agent_update(
                "agent-123",
                "working",
                {"progress": 0.5}
            )
            
            mock_enqueue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_thought_update(self, realtime_manager):
        """사고 과정 업데이트 전송 테스트"""
        with patch.object(realtime_manager.sse_manager.a2a_integration, 'enqueue_event') as mock_enqueue:
            await realtime_manager.send_thought_update(
                "thought-123",
                "processing",
                "Analyzing data..."
            )
            
            mock_enqueue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_mcp_update(self, realtime_manager):
        """MCP 도구 업데이트 전송 테스트"""
        with patch.object(realtime_manager.sse_manager.a2a_integration, 'enqueue_event') as mock_enqueue:
            await realtime_manager.send_mcp_update(
                "mcp-123",
                "active",
                {"cpu": 0.8, "memory": 0.6}
            )
            
            mock_enqueue.assert_called_once()
    
    def test_get_system_status(self, realtime_manager):
        """시스템 상태 조회 테스트"""
        status = realtime_manager.get_system_status()
        
        assert "running" in status
        assert "subscriptions" in status
        assert "components" in status
        assert "sync_rules" in status
        assert "timestamp" in status
    
    def test_get_singleton(self):
        """싱글톤 인스턴스 테스트"""
        manager1 = get_cursor_sse_realtime()
        manager2 = get_cursor_sse_realtime()
        
        assert manager1 is manager2


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_sse_flow(self):
        """End-to-End SSE 플로우 테스트"""
        # 시스템 초기화
        streaming_system = get_a2a_sse_streaming_system()
        realtime_manager = get_cursor_sse_realtime()
        
        # 요청 생성
        request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test message"}]
                }
            }
        }
        
        # 스트리밍 응답 생성
        response = await streaming_system.handle_streaming_request(request)
        
        # 응답 검증
        assert hasattr(response, 'body_iterator')
        assert response.media_type == "text/event-stream"
        
        # 실시간 업데이트 전송
        await realtime_manager.send_agent_update(
            "test-agent",
            "working",
            {"message": "Processing request"}
        )
        
        # 시스템 상태 검증
        status = realtime_manager.get_system_status()
        assert status["running"] == False  # 아직 시작되지 않음
    
    @pytest.mark.asyncio
    async def test_a2a_protocol_compliance(self):
        """A2A 프로토콜 준수 테스트"""
        system = get_a2a_sse_streaming_system()
        
        # Agent Card 검증
        app = system.get_app()
        
        # A2A 표준 엔드포인트 확인
        routes = [route.path for route in app.routes]
        assert "/.well-known/agent.json" in routes
        
        # SSE 스트리밍 엔드포인트 확인
        assert "/stream" in routes
    
    def test_sse_message_format_compliance(self):
        """SSE 메시지 포맷 준수 테스트"""
        # SSE 메시지 생성
        message = SSEMessage(
            id="test-123",
            event_type=SSEEventType.AGENT_STATUS_UPDATE,
            data={"status": "working", "progress": 0.5}
        )
        
        # SSE 포맷 변환
        sse_format = message.to_sse_format()
        
        # 포맷 검증
        assert sse_format.startswith("data: ")
        assert sse_format.endswith("\n\n")
        
        # JSON 파싱 검증
        json_str = sse_format.replace("data: ", "").strip()
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "test-123"
        assert parsed["type"] == "agent_status_update"
        assert parsed["data"]["status"] == "working"
        assert parsed["data"]["progress"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 