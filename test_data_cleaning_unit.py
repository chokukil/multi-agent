#!/usr/bin/env python3
"""
Unit tests for Data Cleaning Agent A2A Server
Tests the agent initialization, core functionality, and data cleaning methods.
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
        import a2a_ds_servers.data_cleaning_server as dc_server
        assert hasattr(dc_server, 'DataCleaningAgent')
        assert hasattr(dc_server, 'DataCleaningExecutor')
        assert hasattr(dc_server, 'main')
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_data_cleaning_agent_initialization():
    """Test Data Cleaning Agent initialization."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    assert agent is not None
    assert hasattr(agent, 'use_real_llm')
    assert hasattr(agent, 'llm')
    assert hasattr(agent, 'agent')
    print("✅ Data Cleaning Agent initialization test passed")

@pytest.mark.asyncio
async def test_data_cleaning_mock_invoke():
    """Test data cleaning with mock responses."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    # Test with sample data cleaning query
    query = "Clean the dataset by removing outliers and handling missing values"
    result = await agent.invoke(query)
    
    assert result is not None
    assert isinstance(result, str)
    assert "Data Cleaning" in result
    assert "Data Quality Assessment" in result
    assert "Data Cleaning Pipeline" in result
    print("✅ Data cleaning mock invoke test passed")

@pytest.mark.asyncio
async def test_missing_value_handling():
    """Test missing value handling functionality."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Handle missing values with appropriate imputation strategies"
    result = await agent.invoke(query)
    
    assert "Missing Value" in result
    assert "imputation" in result.lower()
    assert any(word in result.lower() for word in ["mean", "mode", "median"])
    print("✅ Missing value handling test passed")

@pytest.mark.asyncio
async def test_outlier_detection():
    """Test outlier detection and removal functionality."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Detect and remove extreme outliers using statistical methods"
    result = await agent.invoke(query)
    
    assert "Outlier Detection" in result
    assert "outlier" in result.lower()
    assert any(word in result.lower() for word in ["iqr", "interquartile", "statistical"])
    print("✅ Outlier detection test passed")

@pytest.mark.asyncio
async def test_duplicate_removal():
    """Test duplicate detection and removal functionality."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Remove duplicate rows from the dataset"
    result = await agent.invoke(query)
    
    assert "Duplicate" in result
    assert "duplicate" in result.lower()
    assert any(word in result.lower() for word in ["unique", "removal", "detection"])
    print("✅ Duplicate removal test passed")

@pytest.mark.asyncio
async def test_data_type_optimization():
    """Test data type optimization functionality."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Optimize data types for memory efficiency"
    result = await agent.invoke(query)
    
    assert "Data Type Optimization" in result
    assert "optimization" in result.lower()
    assert any(word in result.lower() for word in ["memory", "efficiency", "reduced"])
    print("✅ Data type optimization test passed")

@pytest.mark.asyncio
async def test_column_removal():
    """Test column removal for high missing values."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Remove columns with excessive missing values"
    result = await agent.invoke(query)
    
    assert "Column Removal" in result
    assert "missing" in result.lower()
    assert any(word in result.lower() for word in ["40%", "excessive", "removed"])
    print("✅ Column removal test passed")

@pytest.mark.asyncio
async def test_data_cleaning_executor_initialization():
    """Test Data Cleaning Executor initialization."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningExecutor
    
    executor = DataCleaningExecutor()
    assert executor is not None
    assert hasattr(executor, 'agent')
    assert executor.agent is not None
    print("✅ Data Cleaning Executor initialization test passed")

@pytest.mark.asyncio
async def test_executor_execute_method():
    """Test executor execute method."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningExecutor
    
    executor = DataCleaningExecutor()
    
    # Mock RequestContext and EventQueue
    mock_context = Mock()
    mock_context.get_user_input.return_value = "Clean dataset removing outliers"
    
    mock_event_queue = AsyncMock()
    
    # Test execute method
    await executor.execute(mock_context, mock_event_queue)
    
    # Verify event_queue.enqueue_event was called
    mock_event_queue.enqueue_event.assert_called_once()
    print("✅ Executor execute method test passed")

@pytest.mark.asyncio
async def test_executor_cancel_method():
    """Test executor cancel method."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningExecutor
    
    executor = DataCleaningExecutor()
    
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
async def test_data_cleaning_error_handling():
    """Test error handling in data cleaning."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    # Test with potentially problematic input
    with patch.object(agent, 'invoke', side_effect=Exception("Test error")):
        # The actual invoke method should handle errors gracefully
        agent_fresh = DataCleaningAgent()
        result = await agent_fresh.invoke("This might cause an error")
        
        # Should still return a valid response (mock mode)
        assert result is not None
        assert isinstance(result, str)
    
    print("✅ Error handling test passed")

def test_data_cleaning_response_content():
    """Test that data cleaning responses contain expected content."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    # Test the mock response content
    query = "Comprehensive data cleaning pipeline"
    result = asyncio.run(agent.invoke(query))
    
    # Check for key data cleaning components
    expected_components = [
        "Data Quality Assessment",
        "Data Cleaning Pipeline",
        "Missing Value Analysis",
        "Column Removal",
        "Missing Value Imputation",
        "Data Type Optimization",
        "Duplicate Row Removal",
        "Outlier Detection",
        "Data Quality Improvements",
        "Generated Function"
    ]
    
    for component in expected_components:
        assert component in result, f"Missing component: {component}"
    
    print("✅ Data cleaning response content test passed")

def test_data_cleaning_techniques():
    """Test that data cleaning mentions appropriate techniques."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Apply all data cleaning techniques"
    result = asyncio.run(agent.invoke(query))
    
    # Check for data cleaning techniques
    expected_techniques = [
        "simpleimputer",
        "mean imputation",
        "mode imputation",
        "iqr",
        "duplicates",
        "outliers"
    ]
    
    result_lower = result.lower()
    for technique in expected_techniques:
        assert technique in result_lower, f"Missing technique mention: {technique}"
    
    print("✅ Data cleaning techniques test passed")

def test_sklearn_integration():
    """Test that response mentions scikit-learn components."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Use scikit-learn for data preprocessing"
    result = asyncio.run(agent.invoke(query))
    
    # Check for sklearn mentions
    sklearn_components = [
        "SimpleImputer",
        "sklearn",
        "fit_transform"
    ]
    
    found_components = [comp for comp in sklearn_components if comp in result]
    assert len(found_components) >= 2, f"Expected sklearn components, found: {found_components}"
    
    print("✅ Sklearn integration test passed")

@pytest.mark.asyncio
async def test_data_quality_metrics():
    """Test data quality metrics and assessment."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Assess data quality with comprehensive metrics"
    result = await agent.invoke(query)
    
    assert "Data Quality" in result
    assert any(word in result.lower() for word in ["completeness", "uniqueness", "consistency"])
    print("✅ Data quality metrics test passed")

@pytest.mark.asyncio
async def test_memory_optimization():
    """Test memory optimization features."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Optimize memory usage through data type conversion"
    result = await agent.invoke(query)
    
    assert "memory" in result.lower()
    assert any(word in result.lower() for word in ["efficiency", "reduced", "optimization"])
    print("✅ Memory optimization test passed")

@pytest.mark.asyncio
async def test_statistical_methods():
    """Test statistical methods for cleaning."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Apply statistical methods for outlier detection"
    result = await agent.invoke(query)
    
    assert any(word in result.lower() for word in ["statistical", "quantile", "iqr", "percentile"])
    print("✅ Statistical methods test passed")

@pytest.mark.asyncio
async def test_real_llm_integration_mock():
    """Test real LLM integration (mocked)."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        with patch('core.llm_factory.create_llm_instance') as mock_llm:
            with patch('ai_data_science_team.agents.DataCleaningAgent') as mock_agent_class:
                # Setup mocks
                mock_llm.return_value = Mock()
                mock_agent = Mock()
                mock_agent.invoke_agent.return_value = None
                mock_agent.response = {
                    'data_cleaned': Mock(),
                    'cleaner_function': 'def data_cleaner(): pass',
                    'recommended_steps': 'Test steps'
                }
                
                # Mock the data_cleaned to have shape attribute
                mock_data = Mock()
                mock_data.shape = (100, 5)
                mock_agent.get_data_cleaned.return_value = mock_data
                mock_agent.get_data_cleaner_function.return_value = 'def data_cleaner(): pass'
                mock_agent.get_recommended_cleaning_steps.return_value = 'Test steps'
                mock_agent_class.return_value = mock_agent
                
                agent = DataCleaningAgent()
                assert agent.use_real_llm == True
                
                result = await agent.invoke("Test data cleaning query")
                assert "Data Cleaning Complete" in result
                assert "Test steps" in result
    
    print("✅ Real LLM integration mock test passed")

def test_imputation_strategies():
    """Test various imputation strategies."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Apply different imputation strategies for numerical and categorical data"
    result = asyncio.run(agent.invoke(query))
    
    # Check for imputation strategies
    imputation_keywords = ["mean imputation", "mode imputation", "numerical", "categorical"]
    result_lower = result.lower()
    
    found_keywords = [word for word in imputation_keywords if word in result_lower]
    assert len(found_keywords) >= 3, f"Expected imputation strategies, found: {found_keywords}"
    
    print("✅ Imputation strategies test passed")

def test_data_validation():
    """Test data validation features."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Validate data quality after cleaning operations"
    result = asyncio.run(agent.invoke(query))
    
    # Check for validation content
    validation_keywords = ["validation", "complete", "unique", "optimized"]
    result_lower = result.lower()
    
    found_keywords = [word for word in validation_keywords if word in result_lower]
    assert len(found_keywords) >= 2, f"Expected validation keywords, found: {found_keywords}"
    
    print("✅ Data validation test passed")

def test_cleaning_pipeline():
    """Test comprehensive cleaning pipeline description."""
    from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
    
    agent = DataCleaningAgent()
    
    query = "Create complete data cleaning pipeline"
    result = asyncio.run(agent.invoke(query))
    
    # Check for pipeline components
    pipeline_steps = [
        "Missing Value Analysis",
        "Column Removal",
        "Missing Value Imputation",
        "Data Type Optimization",
        "Duplicate Row Removal",
        "Outlier Detection"
    ]
    
    for step in pipeline_steps:
        assert step in result, f"Missing pipeline step: {step}"
    
    print("✅ Cleaning pipeline test passed")

def run_all_tests():
    """Run all unit tests."""
    print("🧪 Starting Data Cleaning Agent Unit Tests")
    print("=" * 50)
    
    # Test functions
    test_functions = [
        test_imports,
        test_data_cleaning_agent_initialization,
        test_data_cleaning_response_content,
        test_data_cleaning_techniques,
        test_sklearn_integration,
        test_imputation_strategies,
        test_data_validation,
        test_cleaning_pipeline
    ]
    
    # Async test functions  
    async_test_functions = [
        test_data_cleaning_mock_invoke,
        test_missing_value_handling,
        test_outlier_detection,
        test_duplicate_removal,
        test_data_type_optimization,
        test_column_removal,
        test_data_cleaning_executor_initialization,
        test_executor_execute_method,
        test_executor_cancel_method,
        test_data_cleaning_error_handling,
        test_data_quality_metrics,
        test_memory_optimization,
        test_statistical_methods,
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
    print("🎉 All Data Cleaning Agent unit tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 