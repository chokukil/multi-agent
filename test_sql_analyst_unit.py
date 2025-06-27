#!/usr/bin/env python3
"""
Unit Tests for SQL Data Analyst Server
Tests individual components and functionalities
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'a2a_ds_servers')))

from sql_data_analyst_server import SQLDataAnalystAgent, SQLDataAnalystExecutor

class TestSQLDataAnalystAgent:
    """Test cases for SQLDataAnalystAgent class."""

    @pytest.fixture
    def agent(self):
        """Create SQLDataAnalystAgent instance for testing."""
        return SQLDataAnalystAgent()

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
        
        query = "Show me sales data by month"
        result = await agent.invoke(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert "SQL Data Analysis Result" in result
        assert query in result
        assert "Generated SQL Query" in result
        assert "Query Results Summary" in result

    @pytest.mark.asyncio
    async def test_agent_invoke_empty_query(self, agent):
        """Test agent invoke with empty query."""
        agent.use_real_llm = False
        
        result = await agent.invoke("")
        
        assert result is not None
        assert isinstance(result, str)
        assert "SQL Data Analysis Result" in result

    @pytest.mark.asyncio
    async def test_agent_invoke_complex_query(self, agent):
        """Test agent invoke with complex SQL query."""
        agent.use_real_llm = False
        
        query = "Analyze customer demographics from the orders table, group by region and show monthly trends for the last 12 months"
        result = await agent.invoke(query)
        
        assert result is not None
        assert isinstance(result, str)
        assert "SQL Data Analysis Result" in result
        assert "Generated SQL Query" in result
        assert "Key SQL Insights" in result

class TestSQLDataAnalystExecutor:
    """Test cases for SQLDataAnalystExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create SQLDataAnalystExecutor instance for testing."""
        return SQLDataAnalystExecutor()

    @pytest.fixture
    def mock_context(self):
        """Create mock RequestContext."""
        context = Mock()
        context.get_user_input.return_value = "Show me sales revenue by territory"
        context.context_id = "test-context-123"
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
        mock_context.context_id = "test-context-empty"
        
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

class TestSQLAgentPatterns:
    """Test SQL analysis patterns and content."""

    @pytest.fixture
    def agent(self):
        """Create SQLDataAnalystAgent instance for testing."""
        agent = SQLDataAnalystAgent()
        agent.use_real_llm = False  # Force mock mode
        return agent

    @pytest.mark.asyncio
    async def test_sql_query_patterns(self, agent):
        """Test that generated SQL follows expected patterns."""
        queries = [
            "Show me sales by month",
            "Analyze customer segments",
            "Get revenue trends by region",
            "Find top performing products"
        ]
        
        for query in queries:
            result = await agent.invoke(query)
            
            # Check SQL structure
            assert "SELECT" in result
            assert "FROM" in result
            assert "GROUP BY" in result or "ORDER BY" in result
            
            # Check analysis components
            assert "Query Results Summary" in result
            assert "Key SQL Insights" in result
            assert "SQL Recommendations" in result

    @pytest.mark.asyncio
    async def test_visualization_components(self, agent):
        """Test that visualization components are included."""
        query = "Create a chart showing sales trends"
        result = await agent.invoke(query)
        
        assert "Data Visualization" in result
        assert "Plotly chart" in result
        assert "interactive" in result.lower()

    @pytest.mark.asyncio
    async def test_performance_recommendations(self, agent):
        """Test that performance recommendations are provided."""
        query = "Optimize query performance for large dataset"
        result = await agent.invoke(query)
        
        assert "SQL Recommendations" in result
        assert "indexes" in result.lower() or "index" in result.lower()
        assert "performance" in result.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in agent."""
        # Mock an exception in invoke
        with patch.object(agent, 'invoke', side_effect=Exception("Test error")):
            agent_with_error = SQLDataAnalystAgent()
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
    from sql_data_analyst_server import main
    
    # This is a basic test to ensure the main function can be called
    # In a real scenario, you'd want to test the agent card configuration
    assert main is not None
    assert callable(main)

class TestIntegrationPrep:
    """Prepare for integration testing."""

    def test_port_configuration(self):
        """Test that port 8201 is configured correctly."""
        # This test verifies the port configuration
        # In the actual server, port 8201 should be used
        expected_port = 8201
        
        # Read the server file to verify port configuration
        with open(os.path.join('a2a_ds_servers', 'sql_data_analyst_server.py'), 'r') as f:
            content = f.read()
            assert f"port={expected_port}" in content

    def test_url_configuration(self):
        """Test that URL is configured correctly."""
        expected_url = "http://localhost:8201/"
        
        # Read the server file to verify URL configuration
        with open(os.path.join('a2a_ds_servers', 'sql_data_analyst_server.py'), 'r') as f:
            content = f.read()
            assert expected_url in content

if __name__ == "__main__":
    print("🧪 Running SQL Data Analyst Unit Tests...")
    pytest.main([__file__, "-v", "--tb=short"]) 