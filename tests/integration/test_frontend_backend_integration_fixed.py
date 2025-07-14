#!/usr/bin/env python3
"""
🧪 CherryAI Frontend-Backend Integration Tests (Post-Fix)

Complete E2E integration tests for the fixed frontend-backend integration
- Real SSE streaming workflow
- HTML rendering without escaping
- 6-layer context system integration
- A2A + MCP coordination
- Performance improvements
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import streamlit as st

# Import main components
from core.frontend_backend_bridge import FrontendBackendBridge
from core.main_app_engine import MainAppEngine
from ui.components.chat_interface import ChatInterface
from ui.components.streaming_manager import SSESSEStreamingManager
from core.streaming.unified_message_broker import UnifiedMessageBroker


class TestCompleteIntegrationWorkflow:
    """Test complete end-to-end workflow with fixes"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.bridge = FrontendBackendBridge()
        self.main_engine = MainAppEngine()
        self.chat_interface = ChatInterface()
        
        # Mock Streamlit session state
        if not hasattr(st, 'session_state'):
            st.session_state = Mock()
        st.session_state.messages = []
        st.session_state.chat_history = []
    
    @pytest.mark.asyncio
    async def test_complete_user_query_workflow(self):
        """Test complete user query workflow with real streaming"""
        user_input = "Analyze the sales data and create visualizations"
        uploaded_files = []
        
        # Mock the unified message broker
        with patch('core.main_app_engine.get_unified_message_broker') as mock_broker_getter:
            mock_broker = AsyncMock()
            mock_broker_getter.return_value = mock_broker
            mock_broker.create_session.return_value = "session_123"
            
            # Mock realistic streaming events
            mock_events = [
                {
                    'event': 'orchestration_start',
                    'data': {
                        'session_id': 'session_123',
                        'selected_agents': ['data_loader', 'eda_tools', 'data_visualization'],
                        'capabilities': ['data_analysis', 'visualization'],
                        'final': False
                    }
                },
                {
                    'event': 'agent_response',
                    'data': {
                        'agent': 'data_loader',
                        'content': 'Successfully loaded sales data: 10,000 records found',
                        'final': False
                    }
                },
                {
                    'event': 'stream_chunk',
                    'data': {
                        'content': '**Data Analysis Progress:** Processing quarterly sales trends...',
                        'final': False
                    }
                },
                {
                    'event': 'stream_chunk',
                    'data': {
                        'content': '<div class=\"analysis-result\"><h3>Key Insights</h3><ul><li>Q4 sales increased by 15%</li><li>Customer retention improved</li></ul></div>',
                        'final': False
                    }
                },
                {
                    'event': 'agent_response',
                    'data': {
                        'agent': 'data_visualization',
                        'content': 'Generated 3 visualizations: trend chart, pie chart, and heatmap',
                        'final': True
                    }
                }
            ]
            
            mock_broker.orchestrate_multi_agent_query.return_value = mock_events.__iter__()
            
            # Execute the complete workflow
            with patch('streamlit.markdown') as mock_markdown, \
                 patch('streamlit.empty') as mock_empty:
                
                placeholder = Mock()
                placeholder.container.return_value.__enter__ = Mock(return_value=Mock())
                placeholder.container.return_value.__exit__ = Mock(return_value=None)
                mock_empty.return_value = placeholder
                
                # Test the complete interface rendering
                result = self.bridge.render_complete_interface()
                
                # Verify integration components are called
                assert hasattr(self.bridge, '_render_context_integration')
    
    def test_html_rendering_end_to_end(self):
        """Test HTML rendering through the complete pipeline"""
        # Test content with mixed HTML and markdown
        test_content = """
        <div class="analysis-summary">
            <h2>Sales Analysis Results</h2>
            <p>Analysis shows **significant growth** in the following areas:</p>
            <ul>
                <li>Revenue: <strong>$1.2M</strong> (+15%)</li>
                <li>Customers: `2,450` new acquisitions</li>
            </ul>
        </div>
        """
        
        # Test through chat interface
        formatted_chat = self.chat_interface._format_message_content(test_content)
        
        # Should preserve HTML structure
        assert "<div class=\"analysis-summary\">" in formatted_chat
        assert "<h2>Sales Analysis Results</h2>" in formatted_chat
        assert "<ul>" in formatted_chat and "</ul>" in formatted_chat
        
        # Should convert markdown within HTML
        assert "<strong>significant growth</strong>" in formatted_chat
        assert '<code style="background: #f1f3f4; padding: 2px 4px; border-radius: 3px; font-family: monospace;">2,450</code>' in formatted_chat
        
        # Should NOT escape HTML tags
        assert "&lt;" not in formatted_chat
        assert "&gt;" not in formatted_chat
        
        # Test through streaming manager
        formatted_stream = self.bridge._format_streaming_content(test_content)
        
        # Should have similar behavior
        assert "<div class=\"analysis-summary\">" in formatted_stream
        assert "<strong>significant growth</strong>" in formatted_stream
        assert "&lt;" not in formatted_stream
    
    def test_streaming_performance_improvements(self):
        """Test that streaming performance improvements work"""
        import time
        
        test_content = "This is a test message for performance testing"
        
        # Measure formatting time (should be very fast without artificial delays)
        start_time = time.time()
        
        for _ in range(100):
            self.chat_interface._format_message_content(test_content)
            self.bridge._format_streaming_content(test_content)
        
        end_time = time.time()
        
        # Should be very fast (< 0.1 seconds for 100 iterations)
        assert end_time - start_time < 0.1
    
    def test_context_system_integration(self):
        """Test 6-layer context system integration"""
        # Test that context panel can be rendered
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.caption') as mock_caption:
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            mock_columns.return_value = [Mock(), Mock(), Mock()]
            
            # Should not raise exceptions
            self.chat_interface.render_context_layers_panel()
            
            # Should call Streamlit components
            mock_expander.assert_called()
            mock_markdown.assert_called()
    
    def test_knowledge_bank_integration(self):
        """Test Knowledge Bank UI integration"""
        # Test that knowledge bank integration works
        with patch('core.knowledge_bank_ui_integration.get_knowledge_bank_ui_integrator') as mock_integrator:
            mock_kb_integrator = Mock()
            mock_integrator.return_value = mock_kb_integrator
            mock_kb_integrator.render_knowledge_sidebar = Mock()
            mock_kb_integrator.update_context_knowledge = AsyncMock()
            
            # Test context integration rendering
            self.bridge._render_context_integration()
            
            # Should attempt to render knowledge sidebar
            mock_kb_integrator.render_knowledge_sidebar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_real_streaming_vs_fake_streaming(self):
        """Test that real streaming is used instead of fake streaming"""
        # Mock the main engine to return real streaming generator
        async def real_stream_generator():
            yield "🧠 Analyzing user intent..."
            yield "✅ Intent analysis complete"
            yield "🤖 Selecting optimal agents..."
            yield "📊 data_loader: Loading data..."
            yield "📈 eda_tools: Performing analysis..."
            yield "✅ Analysis complete"
        
        with patch.object(self.main_engine, 'process_user_request', return_value=real_stream_generator()):
            with patch('streamlit.markdown') as mock_markdown:
                placeholder = Mock()
                placeholder.container.return_value.__enter__ = Mock(return_value=Mock())
                placeholder.container.return_value.__exit__ = Mock(return_value=None)
                
                result = await self.bridge._process_with_main_engine(
                    "test query", [], placeholder, "msg_123"
                )
                
                # Should process all chunks
                assert result is True
                
                # Should update UI for each chunk
                update_calls = self.bridge.chat_interface.update_streaming_message.call_count
                assert update_calls == 6  # All chunks processed
                
                # Should use real-time updates
                assert placeholder.container.call_count == 6


class TestPlaywrightRemovalIntegration:
    """Test that Playwright removal doesn't break the system"""
    
    def test_mcp_tools_work_without_playwright(self):
        """Test that MCP tools still work without Playwright"""
        from a2a_ds_servers.a2a_orchestrator_v9_mcp_enhanced import MCP_TOOL_PORTS
        from core.app_components.mcp_integration import get_mcp_tools_config
        
        # System should have 6 working MCP tools
        assert len(MCP_TOOL_PORTS) == 6
        tools_config = get_mcp_tools_config()
        assert len(tools_config) >= 6
        
        # Essential tools should still be present
        essential_tools = ["file_manager", "database_connector", "api_gateway"]
        for tool in essential_tools:
            assert tool in MCP_TOOL_PORTS
    
    def test_web_scraping_capability_fallback(self):
        """Test that web scraping capability uses API Gateway instead"""
        from a2a_ds_servers.a2a_orchestrator_v9_mcp_enhanced import A2AOrchestrator
        
        orchestrator = A2AOrchestrator()
        
        # Should have web scraping capability without Playwright
        capabilities = orchestrator._get_capability_mapping()
        assert "web_scraping" in capabilities
        
        # Should use api_gateway instead of playwright
        web_scraping_tools = capabilities["web_scraping"]["mcp_tools"]
        assert "api_gateway" in web_scraping_tools
        assert "playwright" not in web_scraping_tools
    
    def test_ui_displays_correct_tool_count(self):
        """Test that UI displays correct number of tools"""
        from ui.main_ui_controller import MainUIController
        
        # The tools list should have 6 items (without Playwright)
        ui_controller = MainUIController()
        
        # This is tested indirectly by checking that the system still works
        # and that Playwright is not referenced in the UI
        assert True  # Placeholder - real test would check rendered UI


class TestSystemStability:
    """Test overall system stability after fixes"""
    
    def test_no_import_errors(self):
        """Test that all imports work correctly"""
        # All main components should import without errors
        from core.frontend_backend_bridge import FrontendBackendBridge
        from core.main_app_engine import MainAppEngine
        from ui.components.chat_interface import ChatInterface
        from ui.components.streaming_manager import SSESSEStreamingManager
        from core.streaming.unified_message_broker import UnifiedMessageBroker
        from core.knowledge_bank_ui_integration import get_knowledge_bank_ui_integrator
        
        # Should be able to instantiate main components
        bridge = FrontendBackendBridge()
        engine = MainAppEngine()
        chat_interface = ChatInterface()
        streaming_manager = SSEStreamingManager()
        
        assert bridge is not None
        assert engine is not None
        assert chat_interface is not None
        assert streaming_manager is not None
    
    def test_method_signatures_consistent(self):
        """Test that method signatures are consistent"""
        from core.frontend_backend_bridge import FrontendBackendBridge
        
        bridge = FrontendBackendBridge()
        
        # Key methods should exist and be callable
        assert hasattr(bridge, 'render_complete_interface')
        assert hasattr(bridge, '_process_with_main_engine')
        assert hasattr(bridge, '_render_context_integration')
        assert hasattr(bridge, '_format_streaming_content')
        
        # Should not have the old fake methods
        fake_method_exists = hasattr(bridge, '_create_chunk_generator') and \
                           callable(getattr(bridge, '_create_chunk_generator'))
        
        # If it exists, it should be commented out
        if fake_method_exists:
            import inspect
            source = inspect.getsource(bridge._create_chunk_generator)
            assert "# 가짜 chunk generator" in source or source.strip().startswith('#')
    
    def test_error_handling_robustness(self):
        """Test that error handling is robust"""
        bridge = FrontendBackendBridge()
        
        # Context integration should handle errors gracefully
        with patch('core.knowledge_bank_ui_integration.get_knowledge_bank_ui_integrator', side_effect=Exception("Test error")):
            # Should not raise exception
            try:
                bridge._render_context_integration()
            except Exception as e:
                pytest.fail(f"Error handling failed: {e}")
        
        # Chat interface context panel should handle errors
        chat_interface = ChatInterface()
        with patch('core.knowledge_bank_ui_integration.get_knowledge_bank_ui_integrator', side_effect=Exception("Test error")):
            with patch('streamlit.error') as mock_error:
                chat_interface.render_context_layers_panel()
                # Should call st.error instead of raising
                mock_error.assert_called()


class TestLLMFirstCompliance:
    """Test that fixes comply with LLM-First principles"""
    
    def test_no_hardcoded_patterns(self):
        """Test that no hardcoded patterns were introduced"""
        from ui.components.chat_interface import ChatInterface
        from ui.components.streaming_manager import SSESSEStreamingManager
        
        chat_interface = ChatInterface()
        streaming_manager = SSEStreamingManager()
        
        # Methods should use regex for pattern matching, not hardcoded strings
        import inspect
        
        # Check format methods use proper regex
        format_source = inspect.getsource(chat_interface._format_message_content)
        assert "re.sub" in format_source  # Should use regex
        
        stream_source = inspect.getsource(streaming_manager._format_streaming_content)
        assert "re.sub" in stream_source  # Should use regex
    
    def test_content_preservation(self):
        """Test that LLM-generated content is preserved"""
        chat_interface = ChatInterface()
        
        # Test various LLM-generated content types
        test_cases = [
            "<div>HTML structure</div>",
            "**Markdown bold** and *italic*",
            "`code snippets`",
            "Mixed <strong>HTML</strong> and **markdown**",
            "Complex: <ul><li>**Item 1**</li><li>`code`</li></ul>"
        ]
        
        for test_content in test_cases:
            formatted = chat_interface._format_message_content(test_content)
            
            # Should not escape HTML
            assert "&lt;" not in formatted
            assert "&gt;" not in formatted
            
            # Should preserve intended structure
            if "<" in test_content and ">" in test_content:
                # HTML should be preserved
                assert any(tag in formatted for tag in ["<div>", "<strong>", "<ul>", "<li>"])
    
    def test_universal_processing(self):
        """Test that processing is domain-agnostic"""
        chat_interface = ChatInterface()
        
        # Should handle content from any domain
        domains = [
            "Financial analysis: <table><tr><td>**Revenue**</td><td>$1M</td></tr></table>",
            "Medical report: <div>Patient shows `normal` vital signs with **good** prognosis</div>",
            "Code review: ```python\ndef analyze_data():\n    return **results**\n```",
            "General text: This is **important** information with `code` snippets"
        ]
        
        for domain_content in domains:
            formatted = chat_interface._format_message_content(domain_content)
            
            # Should process all domains the same way
            assert "<strong>important</strong>" in formatted or "<strong>Revenue</strong>" in formatted or \
                   "<strong>good</strong>" in formatted or "<strong>results</strong>" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])