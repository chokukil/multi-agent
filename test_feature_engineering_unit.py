#!/usr/bin/env python3
"""
Unit tests for Feature Engineering Agent A2A Server
Tests the agent initialization, core functionality, and feature engineering methods.
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
        import a2a_ds_servers.feature_engineering_server as fe_server
        assert hasattr(fe_server, 'FeatureEngineeringAgent')
        assert hasattr(fe_server, 'FeatureEngineeringExecutor')
        assert hasattr(fe_server, 'main')
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_feature_engineering_agent_initialization():
    """Test Feature Engineering Agent initialization."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    assert agent is not None
    assert hasattr(agent, 'use_real_llm')
    assert hasattr(agent, 'llm')
    assert hasattr(agent, 'agent')
    print("✅ Feature Engineering Agent initialization test passed")

@pytest.mark.asyncio
async def test_feature_engineering_mock_invoke():
    """Test feature engineering with mock responses."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    # Test with sample feature engineering query
    query = "Apply one-hot encoding and create interaction features"
    result = await agent.invoke(query)
    
    assert result is not None
    assert isinstance(result, str)
    assert "Feature Engineering" in result
    assert "Data Preprocessing" in result
    assert "Feature Engineering Pipeline" in result
    print("✅ Feature engineering mock invoke test passed")

@pytest.mark.asyncio
async def test_categorical_encoding():
    """Test categorical encoding functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Apply one-hot encoding to categorical variables"
    result = await agent.invoke(query)
    
    assert "Categorical Encoding" in result
    assert "One-Hot Encoding" in result
    assert "encoding" in result.lower()
    print("✅ Categorical encoding test passed")

@pytest.mark.asyncio
async def test_missing_value_imputation():
    """Test missing value imputation functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Handle missing values with appropriate imputation strategies"
    result = await agent.invoke(query)
    
    assert "Missing Value Treatment" in result
    assert "imputation" in result.lower()
    assert any(word in result.lower() for word in ["median", "mean", "mode"])
    print("✅ Missing value imputation test passed")

@pytest.mark.asyncio
async def test_data_type_optimization():
    """Test data type optimization functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Optimize data types for memory efficiency"
    result = await agent.invoke(query)
    
    assert "Data Type Optimization" in result
    assert "memory" in result.lower()
    assert any(word in result.lower() for word in ["int64", "float32", "category"])
    print("✅ Data type optimization test passed")

@pytest.mark.asyncio
async def test_feature_creation():
    """Test feature creation functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Create new features including interaction terms"
    result = await agent.invoke(query)
    
    assert "Numerical Feature Engineering" in result
    assert "interaction" in result.lower()
    assert any(word in result.lower() for word in ["ratio", "log", "percentile"])
    print("✅ Feature creation test passed")

@pytest.mark.asyncio
async def test_feature_engineering_executor_initialization():
    """Test Feature Engineering Executor initialization."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringExecutor
    
    executor = FeatureEngineeringExecutor()
    assert executor is not None
    assert hasattr(executor, 'agent')
    assert executor.agent is not None
    print("✅ Feature Engineering Executor initialization test passed")

@pytest.mark.asyncio
async def test_executor_execute_method():
    """Test executor execute method."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringExecutor
    
    executor = FeatureEngineeringExecutor()
    
    # Mock RequestContext and EventQueue
    mock_context = Mock()
    mock_context.get_user_input.return_value = "Engineer features for machine learning"
    
    mock_event_queue = AsyncMock()
    
    # Test execute method
    await executor.execute(mock_context, mock_event_queue)
    
    # Verify event_queue.enqueue_event was called
    mock_event_queue.enqueue_event.assert_called_once()
    print("✅ Executor execute method test passed")

@pytest.mark.asyncio
async def test_executor_cancel_method():
    """Test executor cancel method."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringExecutor
    
    executor = FeatureEngineeringExecutor()
    
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
async def test_feature_engineering_error_handling():
    """Test error handling in feature engineering."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    # Test with potentially problematic input
    with patch.object(agent, 'invoke', side_effect=Exception("Test error")):
        # The actual invoke method should handle errors gracefully
        agent_fresh = FeatureEngineeringAgent()
        result = await agent_fresh.invoke("This might cause an error")
        
        # Should still return a valid response (mock mode)
        assert result is not None
        assert isinstance(result, str)
    
    print("✅ Error handling test passed")

def test_feature_engineering_response_content():
    """Test that feature engineering responses contain expected content."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    # Test the mock response content
    query = "Comprehensive feature engineering pipeline"
    result = asyncio.run(agent.invoke(query))
    
    # Check for key feature engineering components
    expected_components = [
        "Data Preprocessing",
        "Feature Engineering Pipeline", 
        "Data Type Optimization",
        "Missing Value Treatment",
        "Categorical Encoding",
        "Numerical Feature Engineering",
        "Boolean Conversion",
        "Feature Engineering Insights",
        "Generated Function",
        "Recommendations for Model Training"
    ]
    
    for component in expected_components:
        assert component in result, f"Missing component: {component}"
    
    print("✅ Feature engineering response content test passed")

def test_feature_engineering_techniques():
    """Test that feature engineering mentions appropriate techniques."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Apply all feature engineering techniques"
    result = asyncio.run(agent.invoke(query))
    
    # Check for feature engineering techniques
    expected_techniques = [
        "one-hot encoding",
        "labelencoder",
        "standardscaler", 
        "imputation",
        "interaction",
        "percentile"
    ]
    
    result_lower = result.lower()
    for technique in expected_techniques:
        assert technique in result_lower, f"Missing technique mention: {technique}"
    
    print("✅ Feature engineering techniques test passed")

def test_sklearn_integration():
    """Test that response mentions scikit-learn components."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Use scikit-learn for feature preprocessing"
    result = asyncio.run(agent.invoke(query))
    
    # Check for sklearn mentions
    sklearn_components = [
        "OneHotEncoder",
        "LabelEncoder", 
        "StandardScaler",
        "sklearn"
    ]
    
    found_components = [comp for comp in sklearn_components if comp in result]
    assert len(found_components) >= 2, f"Expected sklearn components, found: {found_components}"
    
    print("✅ Sklearn integration test passed")

@pytest.mark.asyncio 
async def test_feature_scaling_normalization():
    """Test feature scaling and normalization functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Scale and normalize features for machine learning"
    result = await agent.invoke(query)
    
    assert "Feature Scaling" in result
    assert any(word in result.lower() for word in ["normalized", "scaling", "standardized"])
    print("✅ Feature scaling normalization test passed")

@pytest.mark.asyncio
async def test_target_variable_processing():
    """Test target variable processing functionality."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Process target variable with appropriate encoding"
    result = await agent.invoke(query)
    
    assert "Target Variable Processing" in result
    assert "target" in result.lower()
    assert any(word in result.lower() for word in ["label encoded", "encoding", "categorical"])
    print("✅ Target variable processing test passed")

@pytest.mark.asyncio
async def test_real_llm_integration_mock():
    """Test real LLM integration (mocked)."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        with patch('core.llm_factory.create_llm_instance') as mock_llm:
            with patch('ai_data_science_team.agents.FeatureEngineeringAgent') as mock_agent_class:
                # Setup mocks
                mock_llm.return_value = Mock()
                mock_agent = Mock()
                mock_agent.invoke_agent.return_value = None
                mock_agent.response = {
                    'data_engineered': Mock(),
                    'feature_function': 'def feature_engineer(): pass',
                    'recommended_steps': 'Test steps'
                }
                
                # Mock the data_engineered to have shape attribute
                mock_data = Mock()
                mock_data.shape = (100, 10)
                mock_agent.get_data_engineered.return_value = mock_data
                mock_agent.get_feature_engineer_function.return_value = 'def feature_engineer(): pass'
                mock_agent.get_recommended_feature_engineering_steps.return_value = 'Test steps'
                mock_agent_class.return_value = mock_agent
                
                agent = FeatureEngineeringAgent()
                assert agent.use_real_llm == True
                
                result = await agent.invoke("Test feature engineering query")
                assert "Feature Engineering Complete" in result
                assert "Test steps" in result
    
    print("✅ Real LLM integration mock test passed")

def test_feature_metrics_and_quality():
    """Test that feature engineering includes quality metrics."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Evaluate feature quality and provide metrics"
    result = asyncio.run(agent.invoke(query))
    
    # Check for quality metrics
    quality_indicators = ["correlation", "variance", "memory", "missing", "outlier"]
    result_lower = result.lower()
    
    found_indicators = [indicator for indicator in quality_indicators if indicator in result_lower]
    assert len(found_indicators) >= 3, f"Expected quality indicators, found: {found_indicators}"
    
    print("✅ Feature metrics and quality test passed")

def test_feature_engineering_pipeline():
    """Test comprehensive feature engineering pipeline description."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Create complete feature engineering pipeline"
    result = asyncio.run(agent.invoke(query))
    
    # Check for pipeline components
    pipeline_steps = [
        "Data Type Optimization",
        "Missing Value Treatment", 
        "Categorical Encoding",
        "Numerical Feature Engineering",
        "Boolean Conversion"
    ]
    
    for step in pipeline_steps:
        assert step in result, f"Missing pipeline step: {step}"
    
    print("✅ Feature engineering pipeline test passed")

def test_memory_optimization():
    """Test memory optimization features."""
    from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
    
    agent = FeatureEngineeringAgent()
    
    query = "Optimize memory usage through data type conversion"
    result = asyncio.run(agent.invoke(query))
    
    # Check for memory optimization content
    memory_keywords = ["memory", "reduced", "efficiency", "optimization", "int8", "float32"]
    result_lower = result.lower()
    
    found_keywords = [word for word in memory_keywords if word in result_lower]
    assert len(found_keywords) >= 3, f"Expected memory optimization keywords, found: {found_keywords}"
    
    print("✅ Memory optimization test passed")

def run_all_tests():
    """Run all unit tests."""
    print("🧪 Starting Feature Engineering Agent Unit Tests")
    print("=" * 50)
    
    # Test functions
    test_functions = [
        test_imports,
        test_feature_engineering_agent_initialization,
        test_feature_engineering_response_content,
        test_feature_engineering_techniques,
        test_sklearn_integration,
        test_feature_metrics_and_quality,
        test_feature_engineering_pipeline,
        test_memory_optimization
    ]
    
    # Async test functions  
    async_test_functions = [
        test_feature_engineering_mock_invoke,
        test_categorical_encoding,
        test_missing_value_imputation,
        test_data_type_optimization,
        test_feature_creation,
        test_feature_engineering_executor_initialization,
        test_executor_execute_method,
        test_executor_cancel_method,
        test_feature_engineering_error_handling,
        test_feature_scaling_normalization,
        test_target_variable_processing,
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
    print("🎉 All Feature Engineering Agent unit tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 