#!/usr/bin/env python3
"""
🧪 CherryAI SSE Streaming Integration Tests

Tests for the real SSE streaming integration fixes
- Unified Message Broker integration
- A2A agent streaming
- Real-time frontend updates
- No artificial delays
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
from datetime import datetime

# Import the components we fixed
from core.main_app_engine import MainAppEngine
from core.frontend_backend_bridge import FrontendBackendBridge
from core.streaming.unified_message_broker import UnifiedMessageBroker


class MockProcessingContext:
    """Mock processing context for testing"""
    def __init__(self):
        self.user_input = "test query"
        self.uploaded_files = []
        self.agent_decision = Mock()
        self.agent_decision.required_capabilities = ["data_analysis"]


class TestSSEStreamingIntegration:
    """Test suite for SSE streaming integration fixes"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.main_engine = MainAppEngine()
        self.bridge = FrontendBackendBridge()
    
    @pytest.mark.asyncio
    async def test_main_engine_uses_real_streaming(self):
        """Test that MainAppEngine uses real SSE streaming"""
        context = MockProcessingContext()
        
        # Mock the unified message broker
        with patch('core.main_app_engine.get_unified_message_broker') as mock_broker_getter:
            mock_broker = AsyncMock()
            mock_broker_getter.return_value = mock_broker
            mock_broker.create_session.return_value = "test_session_123"
            
            # Mock streaming events
            mock_events = [
                {'event': 'orchestration_start', 'data': {'selected_agents': ['data_loader', 'eda_tools']}},
                {'event': 'agent_response', 'data': {'agent': 'data_loader', 'content': 'Data loaded successfully'}},
                {'event': 'stream_chunk', 'data': {'content': 'Analysis in progress...'}},
                {'event': 'stream_chunk', 'data': {'content': 'Results generated'}},
            ]
            
            mock_broker.orchestrate_multi_agent_query.return_value = mock_events.__iter__()
            
            # Execute the streaming method
            results = []
            async for chunk in self.main_engine._execute_with_agents(context):
                results.append(chunk)
            
            # Verify real streaming integration
            mock_broker.create_session.assert_called_once_with(context.user_input)
            mock_broker.orchestrate_multi_agent_query.assert_called_once()
            
            # Verify output contains expected content
            assert any("data_loader, eda_tools" in result for result in results)
            assert any("data_loader: Data loaded successfully" in result for result in results)
            assert any("Analysis in progress..." in result for result in results)
            assert any("Results generated" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_frontend_bridge_real_streaming(self):
        """Test that frontend bridge processes real streaming"""
        test_input = "Analyze my data"
        uploaded_files = []
        
        # Mock the main engine streaming
        mock_chunks = [
            "🧠 사용자 의도를 분석하고 있습니다...",
            "✅ 의도 분석 완료: data_analysis",
            "🤖 최적의 에이전트를 선택하고 있습니다...",
            "✅ 에이전트 선택 완료: multi_agent",
            "🚀 A2A + MCP 하이브리드 시스템이 작업을 수행하고 있습니다...",
            "📊 data_loader: 데이터 로드 완료",
            "📊 eda_tools: 탐색적 데이터 분석 진행 중",
            "분석 결과: 유의미한 패턴 발견"
        ]
        
        with patch.object(self.bridge.main_engine, 'process_user_request') as mock_process:
            mock_process.return_value = mock_chunks.__iter__()
            
            with patch('streamlit.markdown') as mock_markdown:
                placeholder = Mock()
                placeholder.container.return_value.__enter__ = Mock(return_value=Mock())
                placeholder.container.return_value.__exit__ = Mock(return_value=None)
                
                ai_message_id = "test_msg_123"
                
                # Test the updated method
                result = await self.bridge._process_with_main_engine(
                    test_input, uploaded_files, placeholder, ai_message_id
                )
                
                # Verify success
                assert result is True
                
                # Verify all chunks were processed
                assert self.bridge.chat_interface.update_streaming_message.call_count == len(mock_chunks)
                
                # Verify real-time updates
                assert placeholder.container.call_count == len(mock_chunks)
    
    def test_no_fake_chunk_generator(self):
        """Test that fake chunk generator is removed"""
        # The old _create_chunk_generator method should be commented out
        # and not create artificial delays
        
        # Check that the method is commented out in the bridge
        assert not hasattr(self.bridge, '_create_chunk_generator') or \
               str(self.bridge._create_chunk_generator).startswith("# 가짜 chunk generator")
    
    @pytest.mark.asyncio
    async def test_streaming_format_preserves_content(self):
        """Test that streaming format preserves LLM content"""
        test_content = "<strong>Important:</strong> Analysis shows **significant** trends in `sales_data`"
        
        formatted = self.bridge._format_streaming_content(test_content)
        
        # Should preserve HTML
        assert "<strong>Important:</strong>" in formatted
        
        # Should convert markdown
        assert "<strong>significant</strong>" in formatted
        assert '<code style="background: #f1f3f4; padding: 2px 4px; border-radius: 3px; font-family: monospace;">sales_data</code>' in formatted
        
        # Should handle line breaks
        test_with_newlines = "Line 1\nLine 2"
        formatted_newlines = self.bridge._format_streaming_content(test_with_newlines)
        assert "Line 1<br>Line 2" in formatted_newlines


class TestUnifiedMessageBrokerIntegration:
    """Test suite for Unified Message Broker integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock the broker since we're testing integration
        self.mock_broker = Mock()
    
    @pytest.mark.asyncio
    async def test_broker_session_creation(self):
        """Test that sessions are properly created"""
        with patch('core.streaming.unified_message_broker.get_unified_message_broker') as mock_getter:
            mock_getter.return_value = self.mock_broker
            self.mock_broker.create_session = AsyncMock(return_value="session_123")
            
            from core.streaming.unified_message_broker import get_unified_message_broker
            broker = get_unified_message_broker()
            
            session_id = await broker.create_session("test query")
            
            assert session_id == "session_123"
            self.mock_broker.create_session.assert_called_once_with("test query")
    
    @pytest.mark.asyncio 
    async def test_multi_agent_orchestration(self):
        """Test multi-agent orchestration streaming"""
        test_events = [
            {
                'event': 'orchestration_start',
                'data': {
                    'session_id': 'test_session',
                    'selected_agents': ['data_loader', 'eda_tools', 'data_visualization'],
                    'capabilities': ['data_analysis', 'visualization'],
                    'final': False
                }
            },
            {
                'event': 'agent_response',
                'data': {
                    'agent': 'data_loader',
                    'content': 'Successfully loaded 1000 rows of data',
                    'timestamp': datetime.now().isoformat(),
                    'final': False
                }
            },
            {
                'event': 'stream_chunk',
                'data': {
                    'content': 'Performing exploratory data analysis...',
                    'source': 'eda_tools',
                    'final': False
                }
            }
        ]
        
        with patch('core.streaming.unified_message_broker.get_unified_message_broker') as mock_getter:
            self.mock_broker.orchestrate_multi_agent_query = AsyncMock(return_value=test_events)
            mock_getter.return_value = self.mock_broker
            
            from core.streaming.unified_message_broker import get_unified_message_broker
            broker = get_unified_message_broker()
            
            results = []
            async for event in broker.orchestrate_multi_agent_query("session_123", "analyze data"):
                results.append(event)
            
            assert len(results) == 3
            assert results[0]['event'] == 'orchestration_start'
            assert 'data_loader' in results[0]['data']['selected_agents']
            assert results[1]['event'] == 'agent_response'
            assert results[2]['event'] == 'stream_chunk'


class TestPerformanceImprovements:
    """Test suite for performance improvements"""
    
    def test_no_blocking_operations(self):
        """Test that blocking operations are removed"""
        from ui.components.chat_interface import ChatInterface
        from ui.components.streaming_manager import SSEStreamingManager
        from ui.main_ui_controller import MainUIController
        
        chat_interface = ChatInterface()
        streaming_manager = SSEStreamingManager()
        ui_controller = MainUIController()
        
        # Test that methods don't use time.sleep
        import inspect
        
        # Check chat interface methods
        for name, method in inspect.getmembers(chat_interface, predicate=inspect.ismethod):
            if hasattr(method, '__func__'):
                source = inspect.getsource(method.__func__)
                assert "time.sleep" not in source, f"Method {name} still uses time.sleep"
        
        # Check streaming manager
        source = inspect.getsource(streaming_manager._format_streaming_content)
        assert "await asyncio.sleep" not in source
        
        # Check UI controller
        source = inspect.getsource(ui_controller.display_streaming_response)
        assert "time.sleep" not in source
    
    def test_real_time_methods_exist(self):
        """Test that new real-time methods exist"""
        from ui.components.chat_interface import ChatInterface
        
        chat_interface = ChatInterface()
        
        # New methods should exist
        assert hasattr(chat_interface, 'update_streaming_message_realtime')
        assert hasattr(chat_interface, 'finalize_streaming_message')
        
        # Old simulate_typing_effect should be replaced
        if hasattr(chat_interface, 'simulate_typing_effect'):
            # If it exists, it should be the old method that's not used
            pass
        
        # The display_streaming_response should be simplified
        from ui.main_ui_controller import MainUIController
        ui_controller = MainUIController()
        
        # Method should exist and be streamlined
        assert hasattr(ui_controller, 'display_streaming_response')


class TestContextIntegration:
    """Test suite for context integration"""
    
    def test_context_panel_rendering(self):
        """Test that context panel can be rendered"""
        from ui.components.chat_interface import ChatInterface
        
        chat_interface = ChatInterface()
        
        # Should have the new context rendering method
        assert hasattr(chat_interface, 'render_context_layers_panel')
        assert hasattr(chat_interface, '_check_context_layer_data')
    
    def test_frontend_bridge_context_integration(self):
        """Test that frontend bridge integrates context"""
        bridge = FrontendBackendBridge()
        
        # Should have the new context integration method
        assert hasattr(bridge, '_render_context_integration')
        
        # Method should handle errors gracefully
        with patch('streamlit.error') as mock_error:
            bridge._render_context_integration()
            # Should not raise exceptions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])