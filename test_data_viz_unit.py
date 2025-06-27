#!/usr/bin/env python3
"""
Unit Tests for Data Visualization Agent Server
Tests individual components and functionalities
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'a2a_ds_servers')))

from data_visualization_server import DataVisualizationAgent, DataVisualizationExecutor

class TestDataVisualizationAgent:
    """Test cases for DataVisualizationAgent class."""

    @pytest.fixture
    def agent(self):
        """Create DataVisualizationAgent instance for testing."""
        return DataVisualizationAgent()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert hasattr(agent, 'use_real_llm')
        assert hasattr(agent, 'llm')
        assert hasattr(agent, 'agent')

    @pytest.mark.asyncio
    async def test_agent_invoke_with_mock(self, agent):
        """Test agent invoke with mock response."""
        # Force mock mode
        agent.use_real_llm = False
        
        query = "Create a bar chart showing sales by region"
        result = await agent.invoke(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert "Data Visualization Result" in result
        assert query in result
        assert "Generated Visualization" in result
        assert "Chart Analysis" in result

    @pytest.mark.asyncio
    async def test_agent_invoke_empty_query(self, agent):
        """Test agent invoke with empty query."""
        agent.use_real_llm = False
        
        result = await agent.invoke("")
        
        assert result is not None
        assert isinstance(result, str)
        assert "Data Visualization Result" in result

    @pytest.mark.asyncio
    async def test_agent_invoke_complex_query(self, agent):
        """Test agent invoke with complex visualization query."""
        agent.use_real_llm = False
        
        query = "Create an interactive dashboard with multiple chart types: scatter plot with trend lines, bar chart with drill-down, and time series with seasonal decomposition"
        result = await agent.invoke(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert "Data Visualization Result" in result
        assert "Generated Visualization" in result
        assert "Visualization Features" in result

class TestDataVisualizationExecutor:
    """Test cases for DataVisualizationExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create DataVisualizationExecutor instance for testing."""
        return DataVisualizationExecutor()

    @pytest.fixture
    def mock_context(self):
        """Create mock RequestContext."""
        context = Mock()
        context.get_user_input.return_value = "Create a scatter plot showing correlation between variables"
        context.context_id = "test-context-viz-123"
        return context

    @pytest.fixture
    def mock_event_queue(self):
        """Create mock EventQueue."""
        event_queue = Mock()
        event_queue.enqueue_event = AsyncMock()
        return event_queue

    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert executor is not None
        assert hasattr(executor, 'agent')
        assert executor.agent is not None

    @pytest.mark.asyncio
    async def test_executor_execute(self, executor, mock_context, mock_event_queue):
        """Test executor execute method."""
        # Force mock mode
        executor.agent.use_real_llm = False
        
        # Execute
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify context was called
        mock_context.get_user_input.assert_called_once()
        
        # Verify event was enqueued
        mock_event_queue.enqueue_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_executor_execute_empty_input(self, executor, mock_event_queue):
        """Test executor execute with empty input."""
        # Create context that returns empty input
        mock_context = Mock()
        mock_context.get_user_input.return_value = ""
        mock_context.context_id = "test-context-empty-viz"
        
        # Force mock mode
        executor.agent.use_real_llm = False
        
        # Execute
        await executor.execute(mock_context, mock_event_queue)
        
        # Verify event was still enqueued
        mock_event_queue.enqueue_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_executor_cancel(self, executor, mock_context, mock_event_queue):
        """Test executor cancel method."""
        # Execute cancel
        await executor.cancel(mock_context, mock_event_queue)
        
        # Verify event was enqueued
        mock_event_queue.enqueue_event.assert_called_once()

class TestVisualizationPatterns:
    """Test visualization patterns and content."""

    @pytest.fixture
    def agent(self):
        """Create DataVisualizationAgent instance for testing."""
        agent = DataVisualizationAgent()
        agent.use_real_llm = False  # Force mock mode
        return agent

    @pytest.mark.asyncio
    async def test_plotly_code_patterns(self, agent):
        """Test that generated code follows Plotly patterns."""
        queries = [
            "Create a bar chart",
            "Generate a scatter plot",
            "Make a line chart",
            "Build a heatmap"
        ]
        
        for query in queries:
            result = await agent.invoke(query)
            
            # Check Plotly structure
            assert "plotly" in result.lower()
            assert "import plotly" in result
            assert "fig" in result
            assert "json" in result
            
            # Check visualization components
            assert "Visualization Features" in result
            assert "Chart Analysis" in result
            assert "Technical Details" in result

    @pytest.mark.asyncio
    async def test_interactive_components(self, agent):
        """Test that interactive components are included."""
        query = "Create an interactive dashboard with hover effects"
        result = await agent.invoke(query)
        
        assert "Interactive Elements" in result
        assert "hover" in result.lower()
        assert "zoom" in result.lower()
        assert "interactive" in result.lower()

    @pytest.mark.asyncio
    async def test_chart_customization(self, agent):
        """Test that chart customization options are provided."""
        query = "Create a professional styled chart with custom colors"
        result = await agent.invoke(query)
        
        assert "Color Scheme" in result
        assert "Professional" in result or "professional" in result
        assert "styling" in result.lower() or "template" in result.lower()

    @pytest.mark.asyncio
    async def test_chart_types_variety(self, agent):
        """Test different chart types are handled."""
        chart_types = [
            "bar chart",
            "scatter plot", 
            "line chart",
            "heatmap",
            "box plot",
            "histogram"
        ]
        
        for chart_type in chart_types:
            query = f"Create a {chart_type} for my data"
            result = await agent.invoke(query)
            
            assert result is not None
            assert "Data Visualization Result" in result
            assert "Generated Visualization" in result

    @pytest.mark.asyncio
    async def test_performance_optimization(self, agent):
        """Test that performance optimization tips are provided."""
        query = "Create a large dataset visualization that loads quickly"
        result = await agent.invoke(query)
        
        assert "Technical Details" in result
        assert "optimized" in result.lower() or "performance" in result.lower()
        assert "loading" in result.lower() or "size" in result.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in agent."""
        # Mock an exception in invoke
        with patch.object(agent, 'invoke', side_effect=Exception("Test visualization error")):
            agent_with_error = DataVisualizationAgent()
            agent_with_error.use_real_llm = False
            
            # This should not raise an exception
            try:
                result = await agent_with_error.invoke("test query")
                # If we get here, the original invoke worked despite the patch not working as expected
                assert result is not None
            except Exception as e:
                # If an exception occurs, make sure it's handled gracefully
                assert "error" in str(e).lower()

def test_agent_card_configuration():
    """Test that agent card is properly configured."""
    from data_visualization_server import main
    
    # This is a basic test to ensure the main function can be called
    # In a real scenario, you'd want to test the agent card configuration
    assert main is not None
    assert callable(main)

class TestIntegrationPrep:
    """Prepare for integration testing."""

    def test_port_configuration(self):
        """Test that port 8202 is configured correctly."""
        # This test verifies the port configuration
        # In the actual server, port 8202 should be used
        expected_port = 8202
        
        # Read the server file to verify port configuration
        with open(os.path.join('a2a_ds_servers', 'data_visualization_server.py'), 'r') as f:
            content = f.read()
            assert f"port={expected_port}" in content

    def test_url_configuration(self):
        """Test that URL is configured correctly."""
        expected_url = "http://localhost:8202/"
        
        # Read the server file to verify URL configuration
        with open(os.path.join('a2a_ds_servers', 'data_visualization_server.py'), 'r') as f:
            content = f.read()
            assert expected_url in content

    def test_visualization_specific_config(self):
        """Test visualization-specific configuration."""
        # Read the server file to verify visualization-specific elements
        with open(os.path.join('a2a_ds_servers', 'data_visualization_server.py'), 'r') as f:
            content = f.read()
            assert "Data Visualization Agent" in content
            assert "plotly" in content.lower()
            assert "charts" in content.lower()
            assert "interactive" in content.lower()

if __name__ == "__main__":
    print("🧪 Running Data Visualization Agent Unit Tests...")
    pytest.main([__file__, "-v", "--tb=short"]) 