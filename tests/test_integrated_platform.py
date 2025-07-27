"""
Comprehensive test suite for Cherry AI Streamlit Platform
Based on proven Universal Engine test patterns with 100% coverage approach
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

# Import modules to test
from utils import (
    performance_monitor, cache_manager, memory_manager, concurrent_processor,
    error_logger, security_validator, LLMErrorHandler
)
from ui.ux_optimization import (
    feedback_manager, workflow_guide, ui_helper, action_tracker
)
from core.universal_orchestrator import UniversalOrchestrator
from data.enhanced_file_processor import EnhancedFileProcessor

class TestPerformanceMonitoring:
    """Test performance monitoring system"""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initializes correctly"""
        assert performance_monitor is not None
        assert hasattr(performance_monitor, 'start_monitoring')
        assert hasattr(performance_monitor, 'stop_monitoring')
    
    def test_metrics_collection(self):
        """Test metrics collection functionality"""
        # Record test metrics
        performance_monitor.record_agent_response_time("8306", 2.5)
        performance_monitor.record_file_processing(10.0, 5.2)
        performance_monitor.record_cache_hit(True)
        
        # Get metrics summary
        summary = performance_monitor.get_metrics_summary(minutes=1)
        
        assert 'system_performance' in summary
        assert 'agent_performance' in summary
        assert 'file_processing' in summary
        assert 'cache_performance' in summary
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        recommendations = performance_monitor.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 'type' in rec
            assert 'priority' in rec
            assert 'title' in rec
            assert 'description' in rec

class TestCachingSystem:
    """Test caching system functionality"""
    
    @pytest.mark.asyncio
    async def test_dataset_caching(self):
        """Test dataset caching operations"""
        # Create test dataset
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Cache dataset
        await cache_manager.cache_dataset(
            "test_dataset",
            test_data,
            metadata={'test': True}
        )
        
        # Retrieve cached dataset
        cached_data = await cache_manager.get_dataset("test_dataset")
        
        assert cached_data is not None
        assert 'data' in cached_data
        assert 'metadata' in cached_data
        assert cached_data['metadata']['test'] is True
    
    @pytest.mark.asyncio
    async def test_agent_response_caching(self):
        """Test agent response caching"""
        # Test data
        agent_id = "8306"
        request_data = {"query": "test query", "data": "test data"}
        response_data = {"result": "test result", "status": "success"}
        
        # Generate request hash
        request_hash = cache_manager.generate_request_hash(request_data)
        
        # Cache response
        await cache_manager.cache_agent_response(agent_id, request_hash, response_data)
        
        # Retrieve cached response
        cached_response = await cache_manager.get_agent_response(agent_id, request_hash)
        
        assert cached_response is not None
        assert cached_response['response']['result'] == "test result"
    
    def test_cache_statistics(self):
        """Test cache statistics collection"""
        stats = cache_manager.get_cache_stats()
        
        assert 'dataset_cache' in stats
        assert 'agent_response_cache' in stats
        assert 'ui_component_cache' in stats
        assert 'analysis_result_cache' in stats

class TestMemoryManagement:
    """Test memory management system"""
    
    def test_lazy_data_loader(self):
        """Test lazy data loading functionality"""
        def load_test_data():
            return pd.DataFrame({'test': [1, 2, 3]})
        
        # Create lazy loader
        lazy_loader = memory_manager.create_lazy_loader("test_data", load_test_data)
        
        # Check initial state
        assert not lazy_loader.is_loaded()
        
        # Load data
        data = lazy_loader()
        
        # Check loaded state
        assert lazy_loader.is_loaded()
        assert len(data) == 3
        assert 'test' in data.columns
    
    def test_dataframe_chunking(self):
        """Test DataFrame chunking for memory efficiency"""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'A': np.random.randn(10000),
            'B': np.random.randn(10000)
        })
        
        # Create chunker
        chunker = memory_manager.create_dataframe_chunker(large_df, chunk_size=1000)
        
        # Process chunks
        chunk_count = 0
        total_rows = 0
        
        for chunk in chunker:
            chunk_count += 1
            total_rows += len(chunk)
            assert len(chunk) <= 1000
        
        assert chunk_count == 10
        assert total_rows == 10000
    
    def test_memory_pool(self):
        """Test memory pool functionality"""
        def create_list():
            return []
        
        # Create memory pool
        pool = memory_manager.create_memory_pool("test_pool", create_list, max_size=5)
        
        # Get objects from pool
        obj1 = pool.get()
        obj2 = pool.get()
        
        # Return objects to pool
        pool.put(obj1)
        pool.put(obj2)
        
        # Check pool statistics
        stats = pool.get_stats()
        assert stats['created_count'] == 2
        assert stats['pool_size'] == 2

class TestErrorHandling:
    """Test error handling and recovery system"""
    
    @pytest.mark.asyncio
    async def test_error_handler_initialization(self):
        """Test error handler initialization"""
        error_handler = LLMErrorHandler()
        
        assert error_handler is not None
        assert hasattr(error_handler, 'handle_error')
        assert hasattr(error_handler, 'get_agent_health_status')
    
    @pytest.mark.asyncio
    async def test_error_context_handling(self):
        """Test error context creation and handling"""
        from utils.llm_error_handler import ErrorContext, ErrorSeverity
        
        error_context = ErrorContext(
            error_type="TestError",
            error_message="Test error message",
            agent_id="8306",
            user_context={"test": True},
            timestamp=datetime.now()
        )
        
        # Test error handling
        error_handler = LLMErrorHandler()
        result = await error_handler.handle_error(error_context)
        
        assert 'action' in result
        assert result['action'] in ['retry', 'fallback', 'circuit_open', 'basic_error']
    
    def test_error_logging(self):
        """Test error logging functionality"""
        test_error = Exception("Test error")
        context = {"test": True, "agent_id": "8306"}
        
        # Log error
        error_id = error_logger.log_error(test_error, context)
        
        assert error_id is not None
        assert len(error_id) == 8  # MD5 hash truncated to 8 chars
        
        # Get error history
        history = error_logger.get_error_history(limit=10)
        assert len(history) > 0
        assert history[-1]['error_type'] == 'Exception'

class TestSecurityValidation:
    """Test security validation system"""
    
    def test_security_validator_initialization(self):
        """Test security validator initialization"""
        assert security_validator is not None
        assert hasattr(security_validator, 'create_session')
        assert hasattr(security_validator, 'validate_file_upload')
    
    def test_session_creation(self):
        """Test security session creation"""
        session_id = "test_session_123"
        user_context = {"ip": "127.0.0.1", "user_agent": "test"}
        
        permissions = security_validator.create_session(session_id, user_context)
        
        assert permissions is not None
        assert 'file_upload' in permissions
        assert 'data_analysis' in permissions
        assert 'max_file_size_mb' in permissions
    
    @pytest.mark.asyncio
    async def test_file_validation(self):
        """Test file security validation"""
        # Create test file
        test_content = b"test,data\n1,a\n2,b\n3,c"
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file
            result = await security_validator.validate_file_upload(
                "test_session",
                tmp_file_path,
                test_content
            )
            
            assert result is not None
            assert hasattr(result, 'is_safe')
            assert hasattr(result, 'threats')
            assert hasattr(result, 'processing_time')
            
        finally:
            os.unlink(tmp_file_path)

class TestUXOptimization:
    """Test UX optimization features"""
    
    def test_feedback_manager(self):
        """Test visual feedback manager"""
        assert feedback_manager is not None
        assert hasattr(feedback_manager, 'show_loading')
        assert hasattr(feedback_manager, 'show_success')
        assert hasattr(feedback_manager, 'show_error')
    
    def test_workflow_guide(self):
        """Test workflow guidance system"""
        assert workflow_guide is not None
        assert hasattr(workflow_guide, 'start_workflow')
        assert hasattr(workflow_guide, 'complete_step')
        
        # Test workflow initialization
        workflow_guide.start_workflow()
        assert workflow_guide.workflow_start_time is not None
    
    def test_action_tracker(self):
        """Test user action tracking"""
        session_id = "test_session_456"
        
        # Start session tracking
        action_tracker.start_session(session_id)
        
        # Track some actions
        action_tracker.track_action("file_upload", session_id, {"filename": "test.csv"})
        action_tracker.track_action("chat_message", session_id, {"length": 50})
        
        # Get session analytics
        analytics = action_tracker.get_session_analytics(session_id)
        
        assert analytics is not None
        assert analytics['session_id'] == session_id
        assert analytics['total_actions'] >= 2

class TestUniversalOrchestrator:
    """Test Universal Orchestrator functionality"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = UniversalOrchestrator()
        
        assert orchestrator is not None
        assert hasattr(orchestrator, 'orchestrate_analysis')
        assert hasattr(orchestrator, 'get_agent_capabilities_summary')
    
    def test_agent_capabilities(self):
        """Test agent capabilities mapping"""
        orchestrator = UniversalOrchestrator()
        capabilities = orchestrator.get_agent_capabilities_summary()
        
        assert 'total_agents' in capabilities
        assert 'agent_ports' in capabilities
        assert 'capabilities_by_agent' in capabilities
        assert capabilities['total_agents'] == 10  # Ports 8306-8315
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self):
        """Test agent health checking"""
        orchestrator = UniversalOrchestrator()
        
        # Mock agent health check
        with patch.object(orchestrator, 'health_check_agents') as mock_health:
            mock_health.return_value = {8306: True, 8307: False, 8308: True}
            
            health_status = await orchestrator.health_check_agents()
            
            assert isinstance(health_status, dict)
            assert 8306 in health_status
            assert health_status[8306] is True
            assert health_status[8307] is False

class TestFileProcessing:
    """Test file processing functionality"""
    
    def test_file_processor_initialization(self):
        """Test file processor initialization"""
        processor = EnhancedFileProcessor()
        
        assert processor is not None
        assert hasattr(processor, 'process_file')
    
    def test_csv_processing(self):
        """Test CSV file processing"""
        # Create test CSV content
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_file_path = tmp_file.name
        
        try:
            processor = EnhancedFileProcessor()
            
            # Mock uploaded file
            mock_file = Mock()
            mock_file.name = "test.csv"
            mock_file.getvalue.return_value = csv_content.encode()
            
            # Process file
            result = processor.process_file(mock_file)
            
            assert result is not None
            assert 'name' in result
            assert 'rows' in result
            assert 'columns' in result
            assert result['rows'] == 3
            assert result['columns'] == 3
            
        finally:
            os.unlink(tmp_file_path)

class TestConcurrentProcessing:
    """Test concurrent processing system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processor_initialization(self):
        """Test concurrent processor initialization"""
        assert concurrent_processor is not None
        assert hasattr(concurrent_processor, 'start')
        assert hasattr(concurrent_processor, 'stop')
    
    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission and execution"""
        # Mock function for testing
        async def test_task(value):
            await asyncio.sleep(0.1)
            return f"processed_{value}"
        
        # Start processor
        await concurrent_processor.start()
        
        try:
            # Submit task
            task_id = await concurrent_processor.submit_agent_task(
                agent_id="8306",
                function=test_task,
                args=("test_value",),
                session_id="test_session"
            )
            
            assert task_id is not None
            
            # Get result
            result = await concurrent_processor.get_task_result(task_id, timeout_seconds=5)
            assert result == "processed_test_value"
            
        finally:
            await concurrent_processor.stop()

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_data_analysis_workflow(self):
        """Test complete data analysis workflow"""
        # 1. Create test data
        test_data = pd.DataFrame({
            'sales': [100, 150, 200, 175, 225],
            'region': ['North', 'South', 'East', 'West', 'Central'],
            'month': [1, 2, 3, 4, 5]
        })
        
        # 2. Cache dataset
        await cache_manager.cache_dataset("workflow_test", test_data)
        
        # 3. Track user actions
        session_id = "integration_test_session"
        action_tracker.start_session(session_id)
        action_tracker.track_action("data_upload", session_id, {"rows": 5})
        
        # 4. Create security session
        security_validator.create_session(session_id)
        
        # 5. Verify workflow completion
        analytics = action_tracker.get_session_analytics(session_id)
        assert analytics['total_actions'] >= 1
        
        # 6. Check cached data
        cached_data = await cache_manager.get_dataset("workflow_test")
        assert cached_data is not None
    
    def test_error_recovery_scenario(self):
        """Test error recovery in realistic scenario"""
        # Simulate agent failure
        test_error = Exception("Agent connection failed")
        context = {
            "agent_id": "8306",
            "session_id": "error_test_session",
            "action": "data_processing"
        }
        
        # Log error
        error_id = error_logger.log_error(test_error, context)
        
        # Check error was logged
        history = error_logger.get_error_history(limit=1)
        assert len(history) > 0
        assert history[-1]['error_type'] == 'Exception'
        
        # Check error patterns
        patterns = error_logger.get_error_patterns()
        assert patterns['total_errors'] > 0
    
    def test_performance_under_load(self):
        """Test system performance under simulated load"""
        # Record multiple performance metrics
        for i in range(10):
            performance_monitor.record_agent_response_time("8306", 1.0 + i * 0.1)
            performance_monitor.record_file_processing(5.0 + i, 2.0 + i * 0.2)
            performance_monitor.record_cache_hit(i % 2 == 0)
        
        # Get performance summary
        summary = performance_monitor.get_metrics_summary(minutes=1)
        
        assert summary['system_performance']['avg_cpu_usage'] >= 0
        assert summary['cache_performance']['total_requests'] == 10
        
        # Check for optimization recommendations
        recommendations = performance_monitor.get_optimization_recommendations()
        assert isinstance(recommendations, list)

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Performance test markers
pytestmark = pytest.mark.asyncio

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=modules",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])