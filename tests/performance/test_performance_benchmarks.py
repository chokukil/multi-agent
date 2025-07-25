"""
Performance benchmarks for Cherry AI Platform
"""

import pytest
import time
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import psutil
import gc


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    @pytest.mark.slow
    def test_large_dataframe_processing(self):
        """Test processing of large DataFrames."""
        # Create large dataset (100k rows, 20 columns)
        n_rows = 100000
        n_cols = 20
        
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # Generate large DataFrame
        data = {}
        for i in range(n_cols):
            if i % 4 == 0:
                data[f'col_{i}'] = np.random.randint(0, 1000, n_rows)
            elif i % 4 == 1:
                data[f'col_{i}'] = np.random.randn(n_rows)
            elif i % 4 == 2:
                data[f'col_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
            else:
                data[f'col_{i}'] = pd.date_range('2020-01-01', periods=n_rows, freq='1H')[:n_rows]
        
        df = pd.DataFrame(data)
        
        creation_time = time.time() - start_time
        peak_memory = self.get_memory_usage()
        
        # Performance assertions
        assert creation_time < 30.0  # Should create within 30 seconds
        assert (peak_memory - start_memory) < 500  # Should use less than 500MB
        
        # Test basic operations
        start_time = time.time()
        
        # Basic statistics
        stats = df.describe()
        
        # Memory usage calculation
        memory_usage = df.memory_usage(deep=True).sum()
        
        # Data types info
        dtypes = df.dtypes
        
        operations_time = time.time() - start_time
        
        assert operations_time < 10.0  # Operations should complete within 10 seconds
        assert stats is not None
        assert memory_usage > 0
        assert len(dtypes) == n_cols
        
        # Cleanup
        del df, data, stats
        gc.collect()
    
    @pytest.mark.slow
    def test_concurrent_data_processing(self):
        """Test concurrent data processing performance."""
        async def process_data_async(data_size):
            """Async data processing function."""
            df = pd.DataFrame({
                'id': range(data_size),
                'value': np.random.randn(data_size),
                'category': np.random.choice(['A', 'B', 'C'], data_size)
            })
            
            # Simulate processing
            result = df.groupby('category').agg({
                'value': ['mean', 'std', 'count']
            })
            
            return result
        
        async def run_concurrent_processing():
            # Create 10 concurrent processing tasks
            tasks = [process_data_async(10000) for _ in range(10)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            return results, end_time - start_time
        
        # Run the concurrent processing test
        results, total_time = asyncio.run(run_concurrent_processing())
        
        # Performance assertions
        assert total_time < 15.0  # Should complete within 15 seconds
        assert len(results) == 10
        
        # Verify all results are valid
        for result in results:
            assert result is not None
            assert len(result) > 0
    
    def test_memory_efficiency_dataframe_operations(self):
        """Test memory efficiency of DataFrame operations."""
        # Create test DataFrame
        df = pd.DataFrame({
            'id': range(50000),
            'value': np.random.randn(50000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 50000),
            'date': pd.date_range('2020-01-01', periods=50000, freq='1H')
        })
        
        initial_memory = self.get_memory_usage()
        
        # Perform various operations
        operations = [
            lambda: df.groupby('category').sum(),
            lambda: df.sort_values('value'),
            lambda: df.query('value > 0'),
            lambda: df.drop_duplicates(),
            lambda: df.fillna(0),
            lambda: df.describe(),
            lambda: df.corr(),
            lambda: df.pivot_table(values='value', index='category', aggfunc='mean')
        ]
        
        max_memory_increase = 0
        
        for operation in operations:
            operation_start_memory = self.get_memory_usage()
            
            result = operation()
            
            operation_end_memory = self.get_memory_usage()
            memory_increase = operation_end_memory - operation_start_memory
            max_memory_increase = max(max_memory_increase, memory_increase)
            
            # Clean up result
            del result
            gc.collect()
        
        # Memory increase should be reasonable (less than 200MB per operation)
        assert max_memory_increase < 200
        
        final_memory = self.get_memory_usage()
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be minimal after cleanup
        assert total_memory_increase < 100
    
    def test_file_processing_performance(self, temp_dir):
        """Test file processing performance."""
        # Create test files of various sizes
        file_sizes = [1000, 5000, 10000, 25000]  # Number of rows
        test_files = []
        
        for size in file_sizes:
            file_path = f"{temp_dir}/test_{size}.csv"
            
            # Generate test data
            df = pd.DataFrame({
                'id': range(size),
                'name': [f'user_{i}' for i in range(size)],
                'value': np.random.randn(size),
                'category': np.random.choice(['A', 'B', 'C'], size),
                'timestamp': pd.date_range('2020-01-01', periods=size, freq='1H')
            })
            
            df.to_csv(file_path, index=False)
            test_files.append((file_path, size))
        
        # Test file reading performance
        read_times = []
        
        for file_path, expected_size in test_files:
            start_time = time.time()
            
            df = pd.read_csv(file_path)
            
            read_time = time.time() - start_time
            read_times.append(read_time)
            
            # Verify file was read correctly
            assert len(df) == expected_size
            assert len(df.columns) == 5
            
            del df
        
        # Performance assertions
        # Reading should scale reasonably with file size
        for i, read_time in enumerate(read_times):
            expected_max_time = (file_sizes[i] / 1000) * 2  # 2 seconds per 1000 rows max
            assert read_time < expected_max_time
    
    @pytest.mark.asyncio
    async def test_async_operations_performance(self):
        """Test performance of async operations."""
        async def async_data_operation(data_id, size):
            """Simulate async data operation."""
            await asyncio.sleep(0.1)  # Simulate I/O wait
            
            df = pd.DataFrame({
                'id': range(size),
                'data_id': [data_id] * size,
                'value': np.random.randn(size)
            })
            
            # Simulate processing
            result = df.groupby('data_id').agg({
                'value': ['mean', 'std', 'count']
            })
            
            return result
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            tasks = [
                async_data_operation(i, 1000) 
                for i in range(concurrency)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions
            # Higher concurrency should not significantly increase total time
            expected_max_time = 0.2 + (concurrency * 0.05)  # Base time + small overhead per task
            assert total_time < expected_max_time
            assert len(results) == concurrency
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        initial_memory = self.get_memory_usage()
        
        # Perform operations multiple times
        for iteration in range(50):
            # Create and process data
            df = pd.DataFrame({
                'id': range(1000),
                'value': np.random.randn(1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000)
            })
            
            # Perform operations
            result1 = df.groupby('category').mean()
            result2 = df.describe()
            result3 = df.corr()
            
            # Clean up
            del df, result1, result2, result3
            
            # Force garbage collection every 10 iterations
            if iteration % 10 == 0:
                gc.collect()
                current_memory = self.get_memory_usage()
                memory_increase = current_memory - initial_memory
                
                # Memory increase should be minimal and stable
                assert memory_increase < 50  # Less than 50MB increase
        
        # Final cleanup and check
        gc.collect()
        final_memory = self.get_memory_usage()
        total_memory_increase = final_memory - initial_memory
        
        # After all operations and cleanup, memory increase should be minimal
        assert total_memory_increase < 30  # Less than 30MB total increase
    
    def test_cpu_efficiency(self):
        """Test CPU efficiency of operations."""
        # Create test data
        df = pd.DataFrame({
            'id': range(100000),
            'value': np.random.randn(100000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100000)
        })
        
        cpu_intensive_operations = [
            lambda: df.sort_values('value'),
            lambda: df.groupby('category').agg({'value': ['mean', 'std', 'min', 'max']}),
            lambda: df.query('value > 0').groupby('category').size(),
            lambda: df.rolling(window=100).mean(),
            lambda: df.corr()
        ]
        
        cpu_usages = []
        
        for operation in cpu_intensive_operations:
            # Measure CPU usage during operation
            start_cpu = self.get_cpu_usage()
            
            start_time = time.time()
            result = operation()
            end_time = time.time()
            
            end_cpu = self.get_cpu_usage()
            
            cpu_usage = max(start_cpu, end_cpu)
            cpu_usages.append(cpu_usage)
            
            operation_time = end_time - start_time
            
            # Performance assertions
            assert operation_time < 10.0  # Should complete within 10 seconds
            assert cpu_usage < 90.0  # Should not max out CPU
            
            del result
        
        # Average CPU usage should be reasonable
        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        assert avg_cpu_usage < 70.0  # Average should be under 70%


@pytest.mark.performance
class TestStreamlitPerformance:
    """Test Streamlit-specific performance considerations."""
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.dataframe')
    @patch('streamlit.plotly_chart')
    def test_streamlit_rendering_performance(self, mock_plotly, mock_dataframe):
        """Test performance of Streamlit rendering operations."""
        # Simulate large dataset rendering
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        start_time = time.time()
        
        # Simulate multiple Streamlit operations
        for _ in range(10):
            mock_dataframe(large_df.head(100))  # Only show first 100 rows
            
            # Simulate chart creation
            mock_plotly(large_df.sample(1000))  # Sample for visualization
        
        end_time = time.time()
        rendering_time = end_time - start_time
        
        # Rendering should be fast since we're mocking actual Streamlit
        assert rendering_time < 1.0
        
        # Verify functions were called
        assert mock_dataframe.call_count == 10
        assert mock_plotly.call_count == 10
    
    def test_session_state_performance(self):
        """Test session state operations performance."""
        with patch('streamlit.session_state', {}) as mock_session_state:
            start_time = time.time()
            
            # Simulate many session state operations
            for i in range(1000):
                mock_session_state[f'key_{i}'] = f'value_{i}'
                _ = mock_session_state.get(f'key_{i}', 'default')
            
            end_time = time.time()
            operation_time = end_time - start_time
            
            # Session state operations should be fast
            assert operation_time < 0.1  # Should complete in less than 100ms
            
            # Verify data was stored
            assert len(mock_session_state) == 1000


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Test system scalability under load."""
    
    def test_concurrent_user_simulation(self):
        """Simulate multiple concurrent users."""
        from modules.core.security_validation_system import LLMSecurityValidationSystem
        
        security_system = LLMSecurityValidationSystem()
        
        def simulate_user_session(user_id):
            """Simulate a user session."""
            # Create security context
            context = security_system.create_security_context(
                user_id=f"user_{user_id}",
                session_id=f"session_{user_id}",
                ip_address="127.0.0.1",
                user_agent="Load Test Agent"
            )
            
            # Simulate user activities
            for _ in range(10):
                security_system.update_security_context(
                    context.session_id,
                    request_count=context.request_count + 1
                )
            
            return context
        
        start_time = time.time()
        start_memory = self.get_memory_usage() if hasattr(self, 'get_memory_usage') else 0
        
        # Simulate 100 concurrent users
        contexts = []
        for user_id in range(100):
            context = simulate_user_session(user_id)
            contexts.append(context)
        
        end_time = time.time()
        end_memory = self.get_memory_usage() if hasattr(self, 'get_memory_usage') else 0
        
        total_time = end_time - start_time
        memory_increase = end_memory - start_memory if start_memory > 0 else 0
        
        # Performance assertions
        assert total_time < 5.0  # Should handle 100 users within 5 seconds
        if memory_increase > 0:
            assert memory_increase < 100  # Should use less than 100MB for 100 users
        
        # Verify all sessions were created
        assert len(security_system.security_contexts) == 100
        
        # Cleanup
        security_system.clear_expired_sessions(expire_hours=0)
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0