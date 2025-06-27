#!/usr/bin/env python3
"""
Unit tests for EDA Tools Agent A2A Server
Tests the agent initialization, core functionality, and EDA-specific methods.
"""

import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all required imports work correctly."""
    try:
        import a2a_ds_servers.eda_tools_server as eda_server
        assert hasattr(eda_server, 'EDAToolsAgent')
        assert hasattr(eda_server, 'EDAToolsExecutor')
        assert hasattr(eda_server, 'main')
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_eda_tools_agent_initialization():
    """Test EDA Tools Agent initialization."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    assert agent is not None
    assert hasattr(agent, 'use_real_llm')
    assert hasattr(agent, 'llm')
    assert hasattr(agent, 'agent')
    print("✅ EDA Tools Agent initialization test passed")

@pytest.mark.asyncio
async def test_eda_mock_invoke():
    """Test EDA analysis with mock responses."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    # Test with sample EDA query
    query = "Analyze this dataset and provide comprehensive statistical summary"
    result = await agent.invoke(query)
    
    assert result is not None
    assert isinstance(result, str)
    assert "EDA Analysis" in result
    assert "Dataset Overview" in result
    assert "Statistical Summary" in result
    assert "Correlation Analysis" in result
    print("✅ EDA mock invoke test passed")

@pytest.mark.asyncio
async def test_eda_quality_assessment():
    """Test data quality assessment functionality."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Check data quality and identify missing values"
    result = await agent.invoke(query)
    
    assert "Data Quality Assessment" in result
    assert "Missing Values Analysis" in result
    assert "missing" in result.lower()
    assert "data types" in result.lower()
    print("✅ Data quality assessment test passed")

@pytest.mark.asyncio
async def test_correlation_analysis():
    """Test correlation analysis functionality."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Generate correlation matrix and identify strong correlations"
    result = await agent.invoke(query)
    
    assert "Correlation Analysis" in result
    assert "correlation" in result.lower()
    assert any(word in result.lower() for word in ["pearson", "correlation", "matrix"])
    print("✅ Correlation analysis test passed")

@pytest.mark.asyncio
async def test_outlier_detection():
    """Test outlier detection functionality."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Detect outliers and anomalies in the dataset"
    result = await agent.invoke(query)
    
    assert "Outlier Detection" in result
    assert "outlier" in result.lower()
    assert any(word in result.lower() for word in ["anomal", "outlier", "detection"])
    print("✅ Outlier detection test passed")

@pytest.mark.asyncio
async def test_statistical_summary():
    """Test statistical summary generation."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Provide detailed statistical summary with descriptive statistics"
    result = await agent.invoke(query)
    
    assert "Statistical Summary" in result
    assert any(word in result.lower() for word in ["mean", "std", "statistics", "distribution"])
    print("✅ Statistical summary test passed")

@pytest.mark.asyncio
async def test_eda_tools_executor_initialization():
    """Test EDA Tools Executor initialization."""
    from a2a_ds_servers.eda_tools_server import EDAToolsExecutor
    
    executor = EDAToolsExecutor()
    assert executor is not None
    assert hasattr(executor, 'agent')
    assert executor.agent is not None
    print("✅ EDA Tools Executor initialization test passed")

@pytest.mark.asyncio
async def test_executor_execute_method():
    """Test executor execute method."""
    from a2a_ds_servers.eda_tools_server import EDAToolsExecutor
    
    executor = EDAToolsExecutor()
    
    # Mock RequestContext and EventQueue
    mock_context = Mock()
    mock_context.get_user_input.return_value = "Perform comprehensive EDA analysis"
    
    mock_event_queue = AsyncMock()
    
    # Test execute method
    await executor.execute(mock_context, mock_event_queue)
    
    # Verify event_queue.enqueue_event was called
    mock_event_queue.enqueue_event.assert_called_once()
    print("✅ Executor execute method test passed")

@pytest.mark.asyncio
async def test_executor_cancel_method():
    """Test executor cancel method."""
    from a2a_ds_servers.eda_tools_server import EDAToolsExecutor
    
    executor = EDAToolsExecutor()
    
    # Mock context and event queue
    mock_context = Mock()
    mock_context.context_id = "test_context_123"
    
    mock_event_queue = AsyncMock()
    
    # Test cancel method
    await executor.cancel(mock_context, mock_event_queue)
    
    # Verify cancel message was sent
    mock_event_queue.enqueue_event.assert_called_once()
    print("✅ Executor cancel method test passed")

@pytest.mark.asyncio
async def test_eda_error_handling():
    """Test error handling in EDA analysis."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    # Test with potentially problematic input
    with patch.object(agent, 'invoke', side_effect=Exception("Test error")):
        # The actual invoke method should handle errors gracefully
        agent_fresh = EDAToolsAgent()
        result = await agent_fresh.invoke("This might cause an error")
        
        # Should still return a valid response (mock mode)
        assert result is not None
        assert isinstance(result, str)
    
    print("✅ Error handling test passed")

def test_eda_tools_response_content():
    """Test that EDA responses contain expected content."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    # Test the mock response content
    query = "Comprehensive exploratory data analysis"
    result = asyncio.run(agent.invoke(query))
    
    # Check for key EDA components
    expected_components = [
        "Dataset Overview",
        "Statistical Summary", 
        "Correlation Analysis",
        "Data Quality Assessment",
        "Missing Values Analysis",
        "Outlier Detection",
        "EDA Tools Applied",
        "Generated Artifacts",
        "Key Insights",
        "Next Steps"
    ]
    
    for component in expected_components:
        assert component in result, f"Missing component: {component}"
    
    print("✅ EDA response content test passed")

def test_eda_artifacts_structure():
    """Test that EDA analysis mentions appropriate artifacts."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Generate EDA artifacts and visualizations"
    result = asyncio.run(agent.invoke(query))
    
    # Check for artifact mentions
    expected_artifacts = [
        "eda_statistics",
        "correlation_matrix",
        "missing_values",
        "distributions",
        "outliers"
    ]
    
    result_lower = result.lower()
    for artifact in expected_artifacts:
        assert artifact in result_lower, f"Missing artifact mention: {artifact}"
    
    print("✅ EDA artifacts structure test passed")

def test_eda_tool_methods():
    """Test that EDA response mentions correct EDA tools."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Use all available EDA tools for analysis"
    result = asyncio.run(agent.invoke(query))
    
    # Check for EDA tool mentions from the original agent
    expected_tools = [
        "describe_dataset",
        "visualize_missing",
        "generate_correlation_funnel",
        "explain_data"
    ]
    
    for tool in expected_tools:
        assert tool in result, f"Missing EDA tool: {tool}"
    
    print("✅ EDA tool methods test passed")

@pytest.mark.asyncio 
async def test_eda_data_types_analysis():
    """Test data types analysis functionality."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Analyze data types and distribution patterns"
    result = await agent.invoke(query)
    
    assert "Data Types Distribution" in result
    assert any(word in result.lower() for word in ["numerical", "categorical", "boolean", "datetime"])
    print("✅ Data types analysis test passed")

@pytest.mark.asyncio
async def test_eda_insights_recommendations():
    """Test that EDA provides insights and recommendations."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Provide actionable insights and recommendations"
    result = await agent.invoke(query)
    
    assert "Key Insights" in result
    assert "Recommendations" in result
    assert "Next Steps" in result
    print("✅ EDA insights and recommendations test passed")

@pytest.mark.asyncio
async def test_real_llm_integration_mock():
    """Test real LLM integration (mocked)."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        with patch('core.llm_factory.create_llm_instance') as mock_llm:
            with patch('ai_data_science_team.ds_agents.EDAToolsAgent') as mock_agent_class:
                # Setup mocks
                mock_llm.return_value = Mock()
                mock_agent = Mock()
                mock_agent.invoke_agent.return_value = None
                mock_agent.response = {
                    'artifacts': {'test': 'data'},
                    'ai_message': 'Test analysis complete',
                    'tool_calls': ['describe_dataset', 'visualize_missing']
                }
                mock_agent.get_artifacts.return_value = {'test': 'data'}
                mock_agent.get_ai_message.return_value = 'Test analysis complete'  
                mock_agent.get_tool_calls.return_value = ['describe_dataset', 'visualize_missing']
                mock_agent_class.return_value = mock_agent
                
                agent = EDAToolsAgent()
                assert agent.use_real_llm == True
                
                result = await agent.invoke("Test EDA query")
                assert "EDA Analysis Complete" in result
                assert "Test analysis complete" in result
    
    print("✅ Real LLM integration mock test passed")

def test_eda_performance_metrics():
    """Test that EDA analysis includes performance-related information."""
    from a2a_ds_servers.eda_tools_server import EDAToolsAgent
    
    agent = EDAToolsAgent()
    
    query = "Analyze dataset performance and memory usage"
    result = asyncio.run(agent.invoke(query))
    
    # Check for performance-related content
    performance_indicators = ["memory", "shape", "rows", "columns", "kb", "mb"]
    result_lower = result.lower()
    
    found_indicators = [indicator for indicator in performance_indicators if indicator in result_lower]
    assert len(found_indicators) >= 2, f"Expected performance indicators, found: {found_indicators}"
    
    print("✅ EDA performance metrics test passed")

def run_all_tests():
    """Run all unit tests."""
    print("🧪 Starting EDA Tools Agent Unit Tests")
    print("=" * 50)
    
    # Test functions
    test_functions = [
        test_imports,
        test_eda_tools_agent_initialization,
        test_eda_tools_response_content,
        test_eda_artifacts_structure,
        test_eda_tool_methods,
        test_eda_performance_metrics
    ]
    
    # Async test functions  
    async_test_functions = [
        test_eda_mock_invoke,
        test_eda_quality_assessment,
        test_correlation_analysis,
        test_outlier_detection,
        test_statistical_summary,
        test_eda_tools_executor_initialization,
        
        test_executor_execute_method,
        test_executor_cancel_method,
        test_eda_error_handling,
        test_eda_data_types_analysis,
        test_eda_insights_recommendations,
        test_real_llm_integration_mock
    ]
    
    # Run sync tests
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    # Run async tests
    for test_func in async_test_functions:
        try:
            asyncio.run(test_func())
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    print("=" * 50)
    print("🎉 All EDA Tools Agent unit tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 