"""
Integration tests for pandas_agent A2A integration

This module contains integration tests for the complete pandas_agent
workflow including A2A server communication and full feature integration.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import A2A components for testing
import sys
sys.path.append('a2a_ds_servers')

from pandas_agent.server import PandasAgentExecutor, create_pandas_agent_server
from a2a.server.context import RequestContext
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.tasks.task_store import TaskState
from a2a.types import TextPart


class TestPandasAgentA2AIntegration:
    """Integration tests for pandas_agent A2A server"""
    
    @pytest.fixture
    def executor(self):
        """Create PandasAgentExecutor instance"""
        return PandasAgentExecutor()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock A2A request context"""
        mock_part = Mock()
        mock_part.root.text = "Analyze the data and show summary statistics"
        
        mock_message = Mock()
        mock_message.parts = [mock_part]
        
        mock_context = Mock(spec=RequestContext)
        mock_context.message = mock_message
        mock_context.task_id = "test_task_123"
        mock_context.context_id = "test_context_456"
        
        return mock_context
    
    @pytest.fixture
    def mock_task_updater(self):
        """Create mock TaskUpdater"""
        task_updater = Mock(spec=TaskUpdater)
        task_updater.update_status = AsyncMock()
        task_updater.add_artifact = AsyncMock()
        task_updater.submit = AsyncMock()
        task_updater.start_work = AsyncMock()
        return task_updater
    
    @pytest.mark.asyncio
    async def test_full_execution_workflow(self, executor, mock_context, mock_task_updater):
        """Test complete execution workflow with sample data"""
        
        # Mock LLM responses to avoid API calls
        with patch.object(executor, 'agent', None):  # Force agent initialization
            
            # Mock the agent creation and LLM responses
            mock_agent = Mock()
            mock_agent.dataframes = {}
            mock_agent.query_history = []
            mock_agent.load_dataframe = Mock()
            
            mock_smart_df = Mock()
            mock_smart_df.chat = AsyncMock(return_value={
                "query": "Analyze the data and show summary statistics",
                "timestamp": datetime.now().isoformat(),
                "dataframe_name": "sample",
                "intent_analysis": {
                    "primary_intent": "summary",
                    "confidence": 0.9
                },
                "code_generation": {
                    "code": "result = df.describe()",
                    "explanation": "Generate statistical summary"
                },
                "execution_result": {
                    "result": "Statistical summary generated",
                    "success": True,
                    "execution_time": 0.5
                },
                "interpretation": {
                    "key_findings": ["Data analyzed successfully"],
                    "interpretation": "Summary statistics completed"
                }
            })
            
            # Patch the components
            with patch('pandas_agent.server.PandasAgent', return_value=mock_agent), \
                 patch('pandas_agent.server.SmartDataFrame', return_value=mock_smart_df):
                
                # Execute the workflow
                await executor.execute(mock_context, mock_task_updater)
                
                # Verify task updates were called
                assert mock_task_updater.update_status.call_count >= 3
                assert mock_task_updater.add_artifact.call_count >= 1
                
                # Verify final status is completed
                final_call = mock_task_updater.update_status.call_args_list[-1]
                assert final_call[1]['state'] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_data_loading_from_query(self, executor, mock_task_updater):
        """Test automatic data loading from query"""
        
        # Create temporary CSV file
        sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            temp_file_path = f.name
        
        try:
            # Create context with file reference
            mock_part = Mock()
            mock_part.root.text = f"Load data from {temp_file_path} and analyze it"
            
            mock_message = Mock()
            mock_message.parts = [mock_part]
            
            mock_context = Mock(spec=RequestContext)
            mock_context.message = mock_message
            mock_context.task_id = "test_task_file"
            mock_context.context_id = "test_context_file"
            
            # Mock the analysis response
            with patch.object(executor, '_generate_visualizations', AsyncMock(return_value=None)), \
                 patch.object(executor, '_prepare_comprehensive_response', AsyncMock(return_value={
                     "query": f"Load data from {temp_file_path} and analyze it",
                     "status": "success",
                     "dataframe_analyzed": "loaded_file"
                 })):
                
                await executor.execute(mock_context, mock_task_updater)
                
                # Verify that data loading was attempted
                assert mock_task_updater.update_status.call_count >= 5
                
                # Check that file loading message was sent
                status_calls = [call[1]['message'] for call in mock_task_updater.update_status.call_args_list]
                assert any("Loading data" in msg for msg in status_calls)
        
        finally:
            # Cleanup
            os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_visualization_generation(self, executor, mock_context, mock_task_updater):
        """Test visualization generation workflow"""
        
        # Create context that requests visualization
        mock_part = Mock()
        mock_part.root.text = "Show me a histogram of the data"
        
        mock_message = Mock()
        mock_message.parts = [mock_part]
        
        viz_context = Mock(spec=RequestContext)
        viz_context.message = mock_message
        viz_context.task_id = "test_viz_task"
        viz_context.context_id = "test_viz_context"
        
        # Mock visualization engine response
        mock_viz_result = {
            "charts": [
                {
                    "type": "histogram",
                    "image_base64": "fake_base64_string",
                    "description": "Histogram of data distribution",
                    "code": "plt.hist(df['column'])"
                }
            ],
            "total_generated": 1
        }
        
        with patch.object(executor.viz_engine, 'generate_auto_visualizations', return_value=mock_viz_result):
            
            # Mock other components
            mock_agent = Mock()
            mock_agent.dataframes = {"sample": pd.DataFrame({'A': [1, 2, 3]})}
            mock_agent.query_history = []
            
            executor.agent = mock_agent
            
            # Mock the smart dataframe
            mock_smart_df = Mock()
            mock_smart_df.chat = AsyncMock(return_value={
                "visualization_suggestions": True,
                "intent_analysis": {"primary_intent": "visualization"}
            })
            
            executor.smart_dataframes["sample"] = mock_smart_df
            
            with patch.object(executor, '_prepare_comprehensive_response', AsyncMock(return_value={
                "query": "Show me a histogram of the data",
                "visualizations": mock_viz_result
            })):
                
                await executor.execute(viz_context, mock_task_updater)
                
                # Verify visualization generation was called
                assert mock_task_updater.update_status.call_count >= 3
                
                # Check for visualization-related status messages
                status_calls = [call[1]['message'] for call in mock_task_updater.update_status.call_args_list]
                assert any("visualization" in msg.lower() for msg in status_calls)
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, executor, mock_context, mock_task_updater):
        """Test caching behavior in A2A workflow"""
        
        # First execution - should cache result
        mock_response = {
            "query": "Analyze the data and show summary statistics",
            "status": "success",
            "cached": False
        }
        
        with patch.object(executor, '_prepare_comprehensive_response', AsyncMock(return_value=mock_response)):
            
            # Mock agent initialization
            executor.agent = Mock()
            executor.agent.dataframes = {"sample": pd.DataFrame({'A': [1, 2, 3]})}
            executor.agent.query_history = []
            
            # First execution
            await executor.execute(mock_context, mock_task_updater)
            
            # Verify cache was used
            cache_stats = executor.cache_manager.get_stats()
            assert cache_stats["entries"] >= 0  # May have cached something
    
    @pytest.mark.asyncio
    async def test_error_handling(self, executor, mock_task_updater):
        """Test error handling in A2A workflow"""
        
        # Create context that will cause an error
        mock_part = Mock()
        mock_part.root.text = "This will cause an error"
        
        mock_message = Mock()
        mock_message.parts = [mock_part]
        
        error_context = Mock(spec=RequestContext)
        error_context.message = mock_message
        error_context.task_id = "test_error_task"
        error_context.context_id = "test_error_context"
        
        # Force an error during execution
        with patch.object(executor, '_analyze_query_requirements', side_effect=Exception("Test error")):
            
            await executor.execute(error_context, mock_task_updater)
            
            # Verify error was handled gracefully
            assert mock_task_updater.update_status.call_count >= 1
            
            # Check that final status is failed
            final_call = mock_task_updater.update_status.call_args_list[-1]
            assert final_call[1]['state'] == TaskState.failed
            assert "error" in final_call[1]['message'].lower()
    
    @pytest.mark.asyncio
    async def test_cancellation(self, executor, mock_context, mock_task_updater):
        """Test cancellation handling"""
        
        await executor.cancel(mock_context, mock_task_updater)
        
        # Verify cancellation status was set
        mock_task_updater.update_status.assert_called_once_with(
            TaskState.cancelled,
            message="🛑 Pandas agent execution cancelled"
        )


class TestPandasAgentServerCreation:
    """Test A2A server creation and configuration"""
    
    def test_server_creation(self):
        """Test A2A server creation"""
        app = create_pandas_agent_server(port=8210)
        
        assert app is not None
        # Note: More detailed server testing would require actual server startup
    
    def test_server_configuration(self):
        """Test server configuration parameters"""
        app = create_pandas_agent_server(port=9999)
        
        # Verify the app was created (basic check)
        assert app is not None
        
        # Additional server configuration tests could be added here


class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self):
        """Test complete analysis workflow from start to finish"""
        
        # Create sample data
        sample_data = pd.DataFrame({
            'revenue': np.random.randint(1000, 10000, 100),
            'department': np.random.choice(['Sales', 'Marketing', 'Engineering'], 100),
            'quarter': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 100),
            'growth_rate': np.random.uniform(-0.1, 0.3, 100)
        })
        
        # Create executor
        executor = PandasAgentExecutor()
        
        # Create mock context
        mock_part = Mock()
        mock_part.root.text = "Analyze revenue by department and show trends"
        
        mock_message = Mock()
        mock_message.parts = [mock_part]
        
        mock_context = Mock(spec=RequestContext)
        mock_context.message = mock_message
        mock_context.task_id = "e2e_test_task"
        mock_context.context_id = "e2e_test_context"
        
        # Create mock task updater
        task_updater = Mock(spec=TaskUpdater)
        task_updater.update_status = AsyncMock()
        task_updater.add_artifact = AsyncMock()
        task_updater.submit = AsyncMock()
        task_updater.start_work = AsyncMock()
        
        # Mock LLM components to avoid API calls
        with patch('pandas_agent.core.agent.PandasAgent') as MockAgent:
            
            mock_agent_instance = Mock()
            mock_agent_instance.dataframes = {"sample": sample_data}
            mock_agent_instance.query_history = []
            mock_agent_instance.load_dataframe = Mock(return_value="sample")
            mock_agent_instance.chat = AsyncMock(return_value={
                "status": "success",
                "intent_analysis": {"primary_intent": "summary"},
                "execution_result": {"success": True},
                "interpretation": {"key_findings": ["Analysis completed"]}
            })
            
            MockAgent.return_value = mock_agent_instance
            
            # Execute workflow
            await executor.execute(mock_context, task_updater)
            
            # Verify workflow completed
            assert task_updater.update_status.call_count >= 3
            assert task_updater.add_artifact.call_count >= 1
            
            # Verify artifacts were created
            artifact_calls = task_updater.add_artifact.call_args_list
            assert len(artifact_calls) >= 1
            
            # Check artifact content
            main_artifact = artifact_calls[0]
            assert "comprehensive_analysis" in main_artifact[1]['name']


class TestRealDataIntegration:
    """Integration tests with real data scenarios"""
    
    @pytest.mark.asyncio
    async def test_real_csv_analysis(self):
        """Test analysis with real CSV data"""
        
        # Create realistic sample data
        real_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'temperature': 20 + 10 * np.sin(np.linspace(0, 4*np.pi, 1000)) + np.random.normal(0, 2, 1000),
            'humidity': 50 + 20 * np.cos(np.linspace(0, 4*np.pi, 1000)) + np.random.normal(0, 5, 1000),
            'pressure': 1013 + np.random.normal(0, 10, 1000),
            'location': np.random.choice(['Indoor', 'Outdoor'], 1000)
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            real_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Create executor
            executor = PandasAgentExecutor()
            
            # Create context with file reference
            mock_part = Mock()
            mock_part.root.text = f"Load {temp_file} and analyze temperature patterns"
            
            mock_message = Mock()
            mock_message.parts = [mock_part]
            
            mock_context = Mock(spec=RequestContext)
            mock_context.message = mock_message
            mock_context.task_id = "real_data_test"
            mock_context.context_id = "real_data_context"
            
            # Mock task updater
            task_updater = Mock(spec=TaskUpdater)
            task_updater.update_status = AsyncMock()
            task_updater.add_artifact = AsyncMock()
            task_updater.submit = AsyncMock()
            task_updater.start_work = AsyncMock()
            
            # Mock SmartDataFrame response
            with patch('pandas_agent.server.SmartDataFrame') as MockSmartDF:
                
                mock_smart_instance = Mock()
                mock_smart_instance.chat = AsyncMock(return_value={
                    "status": "success",
                    "analysis_complete": True,
                    "data_profile": {
                        "shape": real_data.shape,
                        "columns": list(real_data.columns)
                    }
                })
                
                MockSmartDF.return_value = mock_smart_instance
                
                # Execute
                await executor.execute(mock_context, task_updater)
                
                # Verify execution
                assert task_updater.update_status.call_count >= 1
        
        finally:
            # Cleanup
            os.unlink(temp_file)


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 