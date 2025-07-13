"""
🍒 CherryAI - Streaming Components Unit Tests
새로운 스트리밍 컴포넌트들에 대한 pytest 기반 단위 테스트

테스트 대상:
- RealtimeChatContainer
- UnifiedChatInterface  
- A2ASSEClient
- A2AStreamingServer
"""

import pytest
import asyncio
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import aiohttp
from aiohttp import web
import streamlit as st

# 시스템 경로 설정
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 테스트 대상 임포트
from ui.streaming.realtime_chat_container import RealtimeChatContainer, StreamingMessage
from ui.components.unified_chat_interface import UnifiedChatInterface
from core.streaming.a2a_sse_client import A2ASSEClient, A2AStreamEvent, A2AMessageType
from a2a_ds_servers.base.streaming_server import A2AStreamingServer, StreamingConfig


class TestStreamingMessage:
    """StreamingMessage 데이터 클래스 테스트"""
    
    def test_streaming_message_creation(self):
        """StreamingMessage 생성 테스트"""
        message = StreamingMessage(
            message_id="test-123",
            source="a2a",
            agent_type="pandas",
            content="Test content",
            metadata={"type": "analysis"},
            status="streaming",
            timestamp=time.time(),
            is_final=False
        )
        
        assert message.message_id == "test-123"
        assert message.source == "a2a"
        assert message.agent_type == "pandas"
        assert message.content == "Test content"
        assert message.metadata["type"] == "analysis"
        assert message.status == "streaming"
        assert not message.is_final
    
    def test_streaming_message_final_flag(self):
        """StreamingMessage final 플래그 테스트"""
        message = StreamingMessage(
            message_id="test-final",
            source="mcp",
            agent_type="visualization",
            content="Final content",
            metadata={},
            status="completed",
            timestamp=time.time(),
            is_final=True
        )
        
        assert message.is_final
        assert message.status == "completed"


class TestRealtimeChatContainer:
    """RealtimeChatContainer 클래스 테스트"""
    
    @pytest.fixture
    def mock_streamlit_session_state(self):
        """Streamlit 세션 상태 모킹"""
        mock_state = {}
        with patch.object(st, 'session_state', mock_state):
            yield mock_state
    
    @pytest.fixture
    def chat_container(self, mock_streamlit_session_state):
        """RealtimeChatContainer 인스턴스"""
        return RealtimeChatContainer("test_chat")
    
    def test_container_initialization(self, chat_container, mock_streamlit_session_state):
        """컨테이너 초기화 테스트"""
        assert chat_container.container_key == "test_chat"
        assert isinstance(chat_container.active_streams, dict)
        assert isinstance(chat_container.messages, list)
        
        # 세션 상태 키 확인
        assert "test_chat_messages" in mock_streamlit_session_state
        assert "test_chat_active_streams" in mock_streamlit_session_state
    
    def test_add_user_message(self, chat_container):
        """사용자 메시지 추가 테스트"""
        message_id = chat_container.add_user_message("Hello, CherryAI!")
        
        assert len(chat_container.messages) == 1
        message = chat_container.messages[0]
        
        assert message.message_id == message_id
        assert message.source == "user"
        assert message.content == "Hello, CherryAI!"
        assert message.status == "completed"
        assert message.is_final
    
    def test_add_assistant_message(self, chat_container):
        """어시스턴트 메시지 추가 테스트"""
        metadata = {"analysis_type": "eda"}
        message_id = chat_container.add_assistant_message("Analysis complete", metadata)
        
        assert len(chat_container.messages) == 1
        message = chat_container.messages[0]
        
        assert message.message_id == message_id
        assert message.source == "assistant"
        assert message.content == "Analysis complete"
        assert message.metadata == metadata
        assert message.status == "completed"
        assert message.is_final
    
    def test_add_streaming_message(self, chat_container):
        """스트리밍 메시지 추가 테스트"""
        message_id = chat_container.add_streaming_message("a2a", "pandas", "Starting analysis...")
        
        assert len(chat_container.messages) == 1
        assert message_id in chat_container.active_streams
        
        message = chat_container.messages[0]
        assert message.source == "a2a"
        assert message.agent_type == "pandas"
        assert message.content == "Starting analysis..."
        assert message.status == "streaming"
        assert not message.is_final
    
    def test_update_streaming_message(self, chat_container):
        """스트리밍 메시지 업데이트 테스트"""
        message_id = chat_container.add_streaming_message("a2a", "pandas", "")
        
        # 첫 번째 청크 업데이트
        chat_container.update_streaming_message(message_id, "First chunk", False)
        message = chat_container.messages[0]
        assert message.content == "First chunk"
        assert not message.is_final
        
        # 두 번째 청크 업데이트
        chat_container.update_streaming_message(message_id, " Second chunk", False)
        message = chat_container.messages[0]
        assert message.content == "First chunk Second chunk"
        
        # 최종 청크 업데이트
        chat_container.update_streaming_message(message_id, " Final chunk", True)
        message = chat_container.messages[0]
        assert message.content == "First chunk Second chunk Final chunk"
        assert message.is_final
        assert message.status == "completed"
    
    def test_finalize_streaming_message(self, chat_container):
        """스트리밍 메시지 완료 테스트"""
        message_id = chat_container.add_streaming_message("a2a", "pandas", "Content")
        
        chat_container.finalize_streaming_message(message_id)
        
        message = chat_container.messages[0]
        assert message.is_final
        assert message.status == "completed"
        assert message_id not in chat_container.active_streams
    
    def test_clear_messages(self, chat_container):
        """메시지 클리어 테스트"""
        chat_container.add_user_message("Test 1")
        chat_container.add_user_message("Test 2")
        
        assert len(chat_container.messages) == 2
        
        chat_container.clear_messages()
        
        assert len(chat_container.messages) == 0
        assert len(chat_container.active_streams) == 0
    
    def test_get_active_streams_count(self, chat_container):
        """활성 스트림 카운트 테스트"""
        assert chat_container.get_active_streams_count() == 0
        
        chat_container.add_streaming_message("a2a", "pandas", "Stream 1")
        assert chat_container.get_active_streams_count() == 1
        
        chat_container.add_streaming_message("mcp", "visualization", "Stream 2")
        assert chat_container.get_active_streams_count() == 2


class TestUnifiedChatInterface:
    """UnifiedChatInterface 클래스 테스트"""
    
    @pytest.fixture
    def mock_streamlit_deps(self):
        """Streamlit 종속성 모킹"""
        # Mock session_state as object with attributes and dict-like behavior
        class MockSessionState:
            def __init__(self):
                self._data = {}
                self.file_upload_completed = False
                self.welcome_shown = False
                self.uploaded_files_for_chat = []
                self.ui_minimized = False
            
            def __contains__(self, key):
                return hasattr(self, key) or key in self._data
            
            def __getitem__(self, key):
                if hasattr(self, key):
                    return getattr(self, key)
                return self._data[key]
            
            def __setitem__(self, key, value):
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self._data[key] = value
        
        mock_session_state = MockSessionState()
        
        with patch.multiple(
            'streamlit',
            session_state=mock_session_state,
            container=MagicMock(),
            columns=MagicMock(return_value=[MagicMock(), MagicMock()]),
            success=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            error=MagicMock()
        ):
            yield
    
    @pytest.fixture
    def mock_components(self):
        """UI 컴포넌트 모킹"""
        with patch('ui.components.unified_chat_interface.create_file_upload_manager') as mock_file_manager, \
             patch('ui.components.unified_chat_interface.create_question_input') as mock_question_input:
            
            mock_file_manager.return_value = Mock()
            mock_question_input.return_value = Mock()
            yield mock_file_manager, mock_question_input
    
    @pytest.fixture
    def unified_interface(self, mock_streamlit_deps, mock_components):
        """UnifiedChatInterface 인스턴스"""
        return UnifiedChatInterface()
    
    def test_interface_initialization(self, unified_interface):
        """인터페이스 초기화 테스트"""
        assert hasattr(unified_interface, 'chat_container')
        assert hasattr(unified_interface, 'file_manager')
        assert hasattr(unified_interface, 'question_input')
        assert isinstance(unified_interface.chat_container, RealtimeChatContainer)
    
    def test_session_state_initialization(self, unified_interface):
        """세션 상태 초기화 테스트"""
        # MockSessionState 생성
        class MockSessionState:
            def __init__(self):
                self._data = {}
            
            def __contains__(self, key):
                return hasattr(self, key) or key in self._data
            
            def __setattr__(self, key, value):
                if key.startswith('_'):
                    super().__setattr__(key, value)
                else:
                    self._data[key] = value
        
        mock_state = MockSessionState()
        
        with patch.object(st, 'session_state', mock_state):
            unified_interface._initialize_session_state()
            
            assert 'file_upload_completed' in mock_state
            assert 'welcome_shown' in mock_state
            assert 'uploaded_files_for_chat' in mock_state
            assert 'ui_minimized' in mock_state
    
    def test_get_chat_container(self, unified_interface):
        """채팅 컨테이너 접근 테스트"""
        container = unified_interface.get_chat_container()
        assert container is unified_interface.chat_container
        assert isinstance(container, RealtimeChatContainer)
    
    def test_clear_all(self, unified_interface):
        """전체 클리어 테스트"""
        # 메시지 추가
        unified_interface.chat_container.add_user_message("Test message")
        
        # 클리어 실행
        unified_interface.clear_all()
        
        # 메시지가 클리어되었는지 확인
        assert len(unified_interface.chat_container.messages) == 0
    
    def test_toggle_minimized_mode(self, unified_interface):
        """최소화 모드 토글 테스트"""
        # MockSessionState 생성
        class MockSessionState:
            def __init__(self):
                self.ui_minimized = False
                self._data = {}
            
            def get(self, key, default=None):
                if hasattr(self, key):
                    return getattr(self, key)
                return self._data.get(key, default)
        
        mock_state = MockSessionState()
        
        with patch.object(st, 'session_state', mock_state):
            unified_interface.toggle_minimized_mode()
            assert mock_state.ui_minimized is True
            
            unified_interface.toggle_minimized_mode()
            assert mock_state.ui_minimized is False


class TestA2ASSEClient:
    """A2ASSEClient 클래스 테스트"""
    
    @pytest.fixture
    def client_config(self):
        """클라이언트 설정"""
        return {
            'base_url': 'http://localhost:8000',
            'agents': {
                'pandas': 'http://localhost:8001',
                'orchestrator': 'http://localhost:8002'
            }
        }
    
    @pytest.fixture
    def sse_client(self, client_config):
        """A2ASSEClient 인스턴스"""
        return A2ASSEClient(client_config['base_url'], client_config['agents'])
    
    def test_client_initialization(self, sse_client, client_config):
        """클라이언트 초기화 테스트"""
        assert sse_client.base_url == client_config['base_url']
        assert sse_client.agents == client_config['agents']
        assert isinstance(sse_client.active_connections, dict)
    
    def test_get_agent_urls(self, sse_client, client_config):
        """에이전트 URL 조회 테스트"""
        urls = sse_client.get_agent_urls()
        assert urls == client_config['agents']
    
    def test_add_agent(self, sse_client):
        """에이전트 추가 테스트"""
        sse_client.add_agent('visualization', 'http://localhost:8003')
        
        assert 'visualization' in sse_client.agents
        assert sse_client.agents['visualization'] == 'http://localhost:8003'
    
    def test_remove_agent(self, sse_client):
        """에이전트 제거 테스트"""
        sse_client.remove_agent('pandas')
        
        assert 'pandas' not in sse_client.agents
    
    def test_parse_sse_event_valid(self, sse_client):
        """유효한 SSE 이벤트 파싱 테스트"""
        raw_data = 'data: {"type": "progress", "content": "Processing...", "final": false}'
        
        event = sse_client._parse_sse_event(raw_data, 'pandas')
        
        assert event is not None
        assert event.agent == 'pandas'
        assert event.source == 'a2a'
        assert not event.final
    
    def test_parse_sse_event_invalid(self, sse_client):
        """유효하지 않은 SSE 이벤트 파싱 테스트"""
        raw_data = 'invalid json data'
        
        event = sse_client._parse_sse_event(raw_data, 'pandas')
        
        assert event is None
    
    def test_create_error_event(self, sse_client):
        """에러 이벤트 생성 테스트"""
        error_event = sse_client._create_error_event('pandas', 'Connection failed')
        
        assert error_event.agent == 'pandas'
        assert error_event.event_type == A2AMessageType.ERROR
        assert 'Connection failed' in error_event.data['error']
    
    @pytest.mark.asyncio
    async def test_close(self, sse_client):
        """클라이언트 종료 테스트"""
        # Mock 연결 추가
        mock_session = Mock()
        mock_session.closed = False  # aiohttp.ClientSession의 closed 속성 모킹
        mock_session.close = AsyncMock()
        sse_client.active_connections['test'] = mock_session
        
        await sse_client.close()
        
        mock_session.close.assert_called_once()
        assert len(sse_client.active_connections) == 0


class TestA2AStreamingServer:
    """A2AStreamingServer 클래스 테스트"""
    
    @pytest.fixture
    def mock_agent_executor(self):
        """Mock 에이전트 실행기"""
        executor = Mock()
        executor.execute = AsyncMock()
        executor.cancel = AsyncMock()
        return executor
    
    @pytest.fixture
    def streaming_config(self):
        """스트리밍 설정"""
        return StreamingConfig(
            buffer_size=512,
            timeout_seconds=15,
            heartbeat_interval=5,
            max_message_size=512 * 1024
        )
    
    @pytest.fixture
    def streaming_server(self, mock_agent_executor, streaming_config):
        """A2AStreamingServer 인스턴스"""
        return A2AStreamingServer(mock_agent_executor, streaming_config)
    
    def test_server_initialization(self, streaming_server, streaming_config):
        """서버 초기화 테스트"""
        assert streaming_server.config == streaming_config
        assert hasattr(streaming_server, 'app')
        assert isinstance(streaming_server.active_streams, dict)
    
    def test_validate_a2a_request_valid(self, streaming_server):
        """유효한 A2A 요청 검증 테스트"""
        valid_request = {
            'parts': [{'kind': 'text', 'text': 'Test query'}],
            'messageId': 'test-123',
            'role': 'user'
        }
        
        assert streaming_server._validate_a2a_request(valid_request)
    
    def test_validate_a2a_request_invalid(self, streaming_server):
        """유효하지 않은 A2A 요청 검증 테스트"""
        invalid_request = {
            'messageId': 'test-123'
            # parts와 role 누락
        }
        
        assert not streaming_server._validate_a2a_request(invalid_request)
    
    def test_create_sse_event(self, streaming_server):
        """SSE 이벤트 생성 테스트"""
        data = {'content': 'Test content', 'progress': 0.5}
        sse_event = streaming_server._create_sse_event('progress', data)
        
        assert 'data:' in sse_event
        assert 'Test content' in sse_event
        assert '0.5' in sse_event
    
    def test_update_stream_status(self, streaming_server):
        """스트림 상태 업데이트 테스트"""
        session_id = 'test-session'
        # 먼저 스트림 생성
        streaming_server.active_streams[session_id] = {
            'status': 'starting', 
            'message_count': 0, 
            'last_update': time.time()
        }
        
        streaming_server._update_stream_status(session_id, 'processing')
        
        assert session_id in streaming_server.active_streams
        assert streaming_server.active_streams[session_id]['status'] == 'processing'
    
    def test_cleanup_stream(self, streaming_server):
        """스트림 정리 테스트"""
        session_id = 'test-session'
        streaming_server.active_streams[session_id] = {'status': 'completed'}
        
        streaming_server._cleanup_stream(session_id)
        
        assert session_id not in streaming_server.active_streams
    
    def test_get_app(self, streaming_server):
        """FastAPI 앱 조회 테스트"""
        app = streaming_server.get_app()
        assert app is streaming_server.app
    
    def test_get_active_streams(self, streaming_server):
        """활성 스트림 조회 테스트"""
        test_stream = {'status': 'active', 'start_time': time.time()}
        streaming_server.active_streams['test-id'] = test_stream
        
        active_streams = streaming_server.get_active_streams()
        assert 'test-id' in active_streams
        assert active_streams['test-id'] == test_stream
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_streams(self, streaming_server):
        """비활성 스트림 정리 테스트"""
        # 오래된 스트림 추가 (last_update 기준)
        old_stream = {
            'status': 'streaming',
            'last_update': time.time() - 400,  # 400초 전
            'message_count': 0
        }
        streaming_server.active_streams['old-stream'] = old_stream
        
        # 최근 스트림 추가 (last_update 기준)
        recent_stream = {
            'status': 'streaming',
            'last_update': time.time() - 100,  # 100초 전
            'message_count': 0
        }
        streaming_server.active_streams['recent-stream'] = recent_stream
        
        await streaming_server.cleanup_inactive_streams(300)  # 300초 타임아웃
        
        # 오래된 스트림은 제거되고 최근 스트림은 유지
        assert 'old-stream' not in streaming_server.active_streams
        assert 'recent-stream' in streaming_server.active_streams
    
    @pytest.mark.asyncio
    async def test_shutdown(self, streaming_server):
        """서버 종료 테스트"""
        # 활성 스트림 추가
        streaming_server.active_streams['test-1'] = {'status': 'active'}
        streaming_server.active_streams['test-2'] = {'status': 'active'}
        
        await streaming_server.shutdown()
        
        # 모든 스트림이 정리되었는지 확인
        assert len(streaming_server.active_streams) == 0


class TestStreamingIntegration:
    """스트리밍 컴포넌트 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_chat_container_with_sse_client_integration(self):
        """채팅 컨테이너와 SSE 클라이언트 통합 테스트"""
        # Mock Streamlit 환경
        with patch.object(st, 'session_state', {}):
            chat_container = RealtimeChatContainer("integration_test")
            
            # 스트리밍 메시지 시작
            message_id = chat_container.add_streaming_message("a2a", "pandas", "")
            
            # SSE 이벤트 시뮬레이션
            chunks = ["Analyzing data...", " Processing statistics...", " Complete!"]
            for i, chunk in enumerate(chunks):
                is_final = (i == len(chunks) - 1)
                chat_container.update_streaming_message(message_id, chunk, is_final)
            
            # 최종 메시지 확인
            message = chat_container.messages[0]
            expected_content = "Analyzing data... Processing statistics... Complete!"
            assert message.content == expected_content
            assert message.is_final
            assert message.status == "completed"
    
    def test_streaming_message_lifecycle(self):
        """스트리밍 메시지 생명주기 테스트"""
        with patch.object(st, 'session_state', {}):
            chat_container = RealtimeChatContainer("lifecycle_test")
            
            # 1. 스트리밍 시작
            message_id = chat_container.add_streaming_message("a2a", "orchestrator", "Planning...")
            assert chat_container.get_active_streams_count() == 1
            
            # 2. 진행 상황 업데이트
            chat_container.update_streaming_message(message_id, " Step 1 complete.")
            chat_container.update_streaming_message(message_id, " Step 2 complete.")
            
            # 3. 완료
            chat_container.finalize_streaming_message(message_id)
            assert chat_container.get_active_streams_count() == 0
            
            # 4. 최종 상태 검증
            message = chat_container.messages[0]
            assert message.is_final
            assert message.status == "completed"
            assert "Planning... Step 1 complete. Step 2 complete." == message.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 