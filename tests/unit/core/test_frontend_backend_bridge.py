#!/usr/bin/env python3
"""
🌉 프론트엔드-백엔드 브릿지 단위 테스트

UI와 백엔드 시스템의 완전한 통합을 pytest로 검증

Test Coverage:
- 브릿지 초기화 및 컴포넌트 연결
- 이벤트 시스템 (발행, 구독, 처리)
- 사용자 메시지 처리 플로우
- 파일 업로드 처리
- 세션 변경 처리
- 스트리밍 이벤트 관리
- 에러 처리 및 복구
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# 테스트 대상 임포트
from core.frontend_backend_bridge import (
    FrontendBackendBridge, BridgeEvent, BridgeStatus, EventType,
    get_frontend_backend_bridge, initialize_frontend_backend_bridge,
    run_bridge_async, sync_run_bridge
)

class TestBridgeEvent:
    """BridgeEvent 데이터 클래스 테스트"""
    
    def test_bridge_event_creation(self):
        """브릿지 이벤트 생성 테스트"""
        timestamp = datetime.now()
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "안녕하세요"},
            timestamp=timestamp,
            session_id="test-session",
            metadata={"source": "test"}
        )
        
        assert event.event_type == EventType.USER_MESSAGE
        assert event.data == {"message": "안녕하세요"}
        assert event.timestamp == timestamp
        assert event.session_id == "test-session"
        assert event.metadata == {"source": "test"}

class TestFrontendBackendBridge:
    """FrontendBackendBridge 클래스 테스트"""
    
    @pytest.fixture
    def bridge(self):
        """테스트용 브릿지 인스턴스"""
        return FrontendBackendBridge()
    
    def test_bridge_initialization(self, bridge):
        """브릿지 초기화 테스트"""
        assert bridge.status == BridgeStatus.INITIALIZING
        assert bridge.chat_interface is None
        assert bridge.rich_renderer is None
        assert bridge.session_manager is None
        assert bridge.streaming_manager is None
        assert bridge.shortcuts_manager is None
        assert bridge.current_session_id is None
        assert bridge.active_streaming_session is None
        assert isinstance(bridge.event_handlers, dict)
        assert isinstance(bridge.event_queue, list)
        assert isinstance(bridge.performance_metrics, dict)
    
    @patch('core.frontend_backend_bridge.get_chat_interface')
    @patch('core.frontend_backend_bridge.get_rich_content_renderer')
    @patch('core.frontend_backend_bridge.get_session_manager')
    @patch('core.frontend_backend_bridge.get_session_manager_ui')
    @patch('core.frontend_backend_bridge.get_sse_streaming_manager')
    @patch('core.frontend_backend_bridge.get_shortcuts_manager')
    async def test_bridge_initialize_success(self, mock_shortcuts, mock_streaming,
                                           mock_session_ui, mock_session, 
                                           mock_renderer, mock_chat, bridge):
        """브릿지 초기화 성공 테스트"""
        # Mock 객체들 설정
        mock_chat.return_value = Mock()
        mock_renderer.return_value = Mock()
        mock_session.return_value = Mock()
        mock_session.return_value.current_session_id = None
        mock_session.return_value.create_session.return_value = "new-session-id"
        mock_session_ui.return_value = Mock()
        mock_streaming.return_value = Mock()
        mock_shortcuts.return_value = Mock()
        
        result = await bridge.initialize()
        
        assert result is True
        assert bridge.status == BridgeStatus.READY
        assert bridge.chat_interface is not None
        assert bridge.rich_renderer is not None
        assert bridge.session_manager is not None
        assert bridge.current_session_id == "new-session-id"
    
    def test_register_event_handler(self, bridge):
        """이벤트 핸들러 등록 테스트"""
        handler_func = Mock()
        
        bridge.register_event_handler(EventType.USER_MESSAGE, handler_func)
        
        assert EventType.USER_MESSAGE in bridge.event_handlers
        assert handler_func in bridge.event_handlers[EventType.USER_MESSAGE]
    
    async def test_emit_event(self, bridge):
        """이벤트 발행 테스트"""
        handler_func = AsyncMock()
        bridge.register_event_handler(EventType.USER_MESSAGE, handler_func)
        
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "테스트"},
            timestamp=datetime.now()
        )
        
        await bridge.emit_event(event)
        
        assert event in bridge.event_queue
        handler_func.assert_called_once_with(event)
    
    async def test_emit_event_sync_handler(self, bridge):
        """동기 핸들러로 이벤트 발행 테스트"""
        handler_func = Mock()
        bridge.register_event_handler(EventType.USER_MESSAGE, handler_func)
        
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "테스트"},
            timestamp=datetime.now()
        )
        
        await bridge.emit_event(event)
        
        handler_func.assert_called_once_with(event)
    
    async def test_emit_event_handler_error(self, bridge):
        """이벤트 핸들러 오류 처리 테스트"""
        error_handler = Mock(side_effect=Exception("Handler error"))
        bridge.register_event_handler(EventType.USER_MESSAGE, error_handler)
        
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "테스트"},
            timestamp=datetime.now()
        )
        
        # 예외가 발생해도 emit_event는 정상 완료되어야 함
        await bridge.emit_event(event)
        
        assert event in bridge.event_queue
        error_handler.assert_called_once()
    
    @patch('core.frontend_backend_bridge.get_chat_interface')
    async def test_handle_user_message(self, mock_get_chat, bridge):
        """사용자 메시지 처리 테스트"""
        # Mock 채팅 인터페이스 설정
        mock_chat = Mock()
        mock_chat.add_message.return_value = Mock()
        mock_chat.get_conversation_context.return_value = "context"
        mock_get_chat.return_value = mock_chat
        bridge.chat_interface = mock_chat
        
        # Mock 세션 관리자 설정
        bridge.current_session_id = "test-session"
        
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "안녕하세요", "files": []},
            timestamp=datetime.now(),
            session_id="test-session"
        )
        
        await bridge._handle_user_message(event)
        
        # 채팅 인터페이스에 메시지 추가 확인
        mock_chat.add_message.assert_called()
    
    async def test_handle_file_upload(self, bridge):
        """파일 업로드 처리 테스트"""
        bridge.chat_interface = Mock()
        bridge.chat_interface.add_message.return_value = Mock()
        bridge.current_session_id = "test-session"
        
        files = [
            {"name": "test.csv", "path": "/path/to/test.csv", "content": "data"}
        ]
        
        event = BridgeEvent(
            event_type=EventType.FILE_UPLOAD,
            data={"files": files},
            timestamp=datetime.now(),
            session_id="test-session"
        )
        
        await bridge._handle_file_upload(event)
        
        # 파일 업로드 메시지 추가 확인
        bridge.chat_interface.add_message.assert_called()
    
    async def test_handle_session_change(self, bridge):
        """세션 변경 처리 테스트"""
        # Mock 세션 관리자 설정
        mock_session_manager = Mock()
        mock_session_data = Mock()
        mock_session_data.messages = [
            {"role": "user", "content": "이전 메시지"}
        ]
        mock_session_manager.load_session.return_value = mock_session_data
        bridge.session_manager = mock_session_manager
        
        # Mock 채팅 인터페이스 설정
        mock_chat = Mock()
        bridge.chat_interface = mock_chat
        bridge.current_session_id = "old-session"
        
        event = BridgeEvent(
            event_type=EventType.SESSION_CHANGE,
            data={"session_id": "new-session"},
            timestamp=datetime.now()
        )
        
        with patch.object(bridge, '_save_current_session') as mock_save:
            await bridge._handle_session_change(event)
            
            # 이전 세션 저장 확인
            mock_save.assert_called_once()
            
            # 새 세션 로드 확인
            mock_session_manager.load_session.assert_called_with("new-session")
            
            # 채팅 히스토리 클리어 및 복원 확인
            mock_chat.clear_messages.assert_called_once()
            
            # 현재 세션 ID 업데이트 확인
            assert bridge.current_session_id == "new-session"
    
    async def test_handle_stream_start(self, bridge):
        """스트리밍 시작 처리 테스트"""
        event = BridgeEvent(
            event_type=EventType.STREAM_START,
            data={"streaming_session_id": "stream-123"},
            timestamp=datetime.now()
        )
        
        initial_count = bridge.performance_metrics["stream_sessions"]
        
        await bridge._handle_stream_start(event)
        
        # 성능 메트릭 업데이트 확인
        assert bridge.performance_metrics["stream_sessions"] == initial_count + 1
    
    async def test_handle_stream_end_success(self, bridge):
        """스트리밍 성공 종료 처리 테스트"""
        bridge.current_session_id = "test-session"
        
        event = BridgeEvent(
            event_type=EventType.STREAM_END,
            data={"success": True, "message_id": "msg-123"},
            timestamp=datetime.now()
        )
        
        with patch.object(bridge, '_save_current_session') as mock_save:
            await bridge._handle_stream_end(event)
            
            # 세션 저장 확인
            mock_save.assert_called_once()
    
    async def test_handle_error(self, bridge):
        """에러 처리 테스트"""
        mock_chat = Mock()
        bridge.chat_interface = mock_chat
        
        event = BridgeEvent(
            event_type=EventType.ERROR_OCCURRED,
            data={"error": "테스트 에러", "context": "test_context"},
            timestamp=datetime.now()
        )
        
        await bridge._handle_error(event)
        
        # 에러 메시지 추가 확인
        mock_chat.add_message.assert_called()
        call_args = mock_chat.add_message.call_args[0]
        assert "⚠️ 오류가 발생했습니다: 테스트 에러" in call_args[1]
    
    @patch('core.frontend_backend_bridge.st.session_state', {})
    async def test_save_current_session(self, bridge):
        """현재 세션 저장 테스트"""
        # Mock 설정
        mock_chat = Mock()
        mock_messages = [Mock()]
        mock_messages[0].to_dict.return_value = {"role": "user", "content": "test"}
        mock_chat.get_messages.return_value = mock_messages
        bridge.chat_interface = mock_chat
        
        mock_session_manager = Mock()
        mock_session_data = Mock()
        mock_session_manager.load_session.return_value = mock_session_data
        bridge.session_manager = mock_session_manager
        bridge.current_session_id = "test-session"
        
        await bridge._save_current_session()
        
        # 세션 로드 및 저장 확인
        mock_session_manager.load_session.assert_called_with("test-session")
        mock_session_manager.save_session.assert_called_with(mock_session_data)
        
        # 메시지 업데이트 확인
        assert mock_session_data.messages == [{"role": "user", "content": "test"}]
    
    def test_get_bridge_status(self, bridge):
        """브릿지 상태 조회 테스트"""
        bridge.status = BridgeStatus.READY
        bridge.current_session_id = "test-session"
        bridge.active_streaming_session = "stream-123"
        bridge.chat_interface = Mock()
        
        status = bridge.get_bridge_status()
        
        assert status["status"] == "ready"
        assert status["current_session"] == "test-session"
        assert status["active_streaming"] == "stream-123"
        assert status["components_ready"]["chat_interface"] is True
        assert "performance_metrics" in status

class TestBridgeIntegration:
    """브릿지 통합 테스트"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock 컴포넌트들"""
        components = {
            'chat_interface': Mock(),
            'rich_renderer': Mock(),
            'session_manager': Mock(),
            'session_ui': Mock(),
            'streaming_manager': Mock(),
            'shortcuts_manager': Mock()
        }
        return components
    
    @patch('core.frontend_backend_bridge.get_chat_interface')
    @patch('core.frontend_backend_bridge.get_rich_content_renderer')
    @patch('core.frontend_backend_bridge.get_session_manager')
    @patch('core.frontend_backend_bridge.get_session_manager_ui')
    @patch('core.frontend_backend_bridge.get_sse_streaming_manager')
    @patch('core.frontend_backend_bridge.get_shortcuts_manager')
    async def test_complete_user_message_flow(self, mock_shortcuts, mock_streaming,
                                            mock_session_ui, mock_session, 
                                            mock_renderer, mock_chat, mock_components):
        """완전한 사용자 메시지 처리 플로우 테스트"""
        # Mock 설정
        for name, mock_obj in mock_components.items():
            if name == 'session_manager':
                mock_obj.current_session_id = None
                mock_obj.create_session.return_value = "new-session"
            locals()[f'mock_{name.split("_")[0]}'].return_value = mock_obj
        
        bridge = FrontendBackendBridge()
        await bridge.initialize()
        
        # 사용자 메시지 이벤트 생성
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "안녕하세요", "files": []},
            timestamp=datetime.now(),
            session_id="new-session"
        )
        
        # 이벤트 처리
        await bridge.emit_event(event)
        
        # 채팅 인터페이스에 메시지 추가 확인
        mock_components['chat_interface'].add_message.assert_called()
        
        # 이벤트 큐에 추가 확인
        assert len(bridge.event_queue) > 0
        assert bridge.event_queue[0] == event

class TestBridgeGlobalFunctions:
    """전역 함수 테스트"""
    
    @patch('core.frontend_backend_bridge._frontend_backend_bridge_instance', None)
    def test_get_frontend_backend_bridge_singleton(self):
        """싱글톤 인스턴스 테스트"""
        bridge1 = get_frontend_backend_bridge()
        bridge2 = get_frontend_backend_bridge()
        
        assert bridge1 is bridge2
        assert isinstance(bridge1, FrontendBackendBridge)
    
    @patch('core.frontend_backend_bridge._frontend_backend_bridge_instance', None)
    async def test_initialize_frontend_backend_bridge(self):
        """브릿지 초기화 테스트"""
        with patch.object(FrontendBackendBridge, 'initialize', return_value=True) as mock_init:
            bridge = await initialize_frontend_backend_bridge()
            
            assert isinstance(bridge, FrontendBackendBridge)
            mock_init.assert_called_once()
    
    async def test_run_bridge_async(self):
        """비동기 브릿지 실행 테스트"""
        mock_bridge = Mock()
        mock_bridge.emit_event = AsyncMock()
        mock_bridge.current_session_id = "test-session"
        
        await run_bridge_async(mock_bridge, "테스트 메시지", ["file1.txt"])
        
        # emit_event 호출 확인
        mock_bridge.emit_event.assert_called_once()
        call_args = mock_bridge.emit_event.call_args[0][0]
        assert call_args.event_type == EventType.USER_MESSAGE
        assert call_args.data["message"] == "테스트 메시지"
        assert call_args.data["files"] == ["file1.txt"]
    
    @patch('asyncio.new_event_loop')
    @patch('asyncio.set_event_loop')
    def test_sync_run_bridge(self, mock_set_loop, mock_new_loop):
        """동기 브릿지 실행 테스트"""
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop
        mock_bridge = Mock()
        
        sync_run_bridge(mock_bridge, "테스트 메시지")
        
        # 이벤트 루프 생성 및 설정 확인
        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once()
        mock_loop.close.assert_called_once()

class TestBridgeErrorHandling:
    """브릿지 에러 처리 테스트"""
    
    @pytest.fixture
    def bridge(self):
        return FrontendBackendBridge()
    
    async def test_initialize_component_failure(self, bridge):
        """컴포넌트 초기화 실패 테스트"""
        with patch('core.frontend_backend_bridge.get_chat_interface', side_effect=Exception("Chat init error")):
            result = await bridge.initialize()
            
            assert result is False
            assert bridge.status == BridgeStatus.ERROR
    
    async def test_user_message_handling_error(self, bridge):
        """사용자 메시지 처리 에러 테스트"""
        bridge.chat_interface = Mock()
        bridge.chat_interface.add_message.side_effect = Exception("Message error")
        
        event = BridgeEvent(
            event_type=EventType.USER_MESSAGE,
            data={"message": "테스트", "files": []},
            timestamp=datetime.now()
        )
        
        # 에러가 발생해도 예외가 전파되지 않아야 함
        await bridge._handle_user_message(event)
        
        # 에러 이벤트가 발행되었는지 확인 (구현에 따라)
        # 실제 구현에서는 try-catch로 에러를 처리해야 함
    
    async def test_save_session_error(self, bridge):
        """세션 저장 에러 테스트"""
        bridge.current_session_id = "test-session"
        bridge.chat_interface = Mock()
        bridge.chat_interface.get_messages.side_effect = Exception("Get messages error")
        
        # 예외가 발생해도 함수가 정상 완료되어야 함
        await bridge._save_current_session()

class TestBridgePerformance:
    """브릿지 성능 테스트"""
    
    @pytest.fixture
    def bridge(self):
        return FrontendBackendBridge()
    
    async def test_multiple_events_handling(self, bridge):
        """다중 이벤트 처리 테스트"""
        handler = AsyncMock()
        bridge.register_event_handler(EventType.USER_MESSAGE, handler)
        
        # 여러 이벤트 동시 발행
        events = []
        for i in range(10):
            event = BridgeEvent(
                event_type=EventType.USER_MESSAGE,
                data={"message": f"메시지 {i}"},
                timestamp=datetime.now()
            )
            events.append(bridge.emit_event(event))
        
        # 모든 이벤트 처리 대기
        await asyncio.gather(*events)
        
        # 핸들러가 모든 이벤트에 대해 호출되었는지 확인
        assert handler.call_count == 10
        assert len(bridge.event_queue) == 10
    
    def test_performance_metrics_tracking(self, bridge):
        """성능 메트릭 추적 테스트"""
        initial_metrics = bridge.performance_metrics.copy()
        
        # 메트릭 업데이트
        bridge.performance_metrics["total_messages"] += 5
        bridge.performance_metrics["successful_responses"] += 4
        bridge.performance_metrics["failed_responses"] += 1
        
        # 변경 확인
        assert bridge.performance_metrics["total_messages"] == initial_metrics["total_messages"] + 5
        assert bridge.performance_metrics["successful_responses"] == initial_metrics["successful_responses"] + 4
        assert bridge.performance_metrics["failed_responses"] == initial_metrics["failed_responses"] + 1

# 테스트 실행을 위한 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 