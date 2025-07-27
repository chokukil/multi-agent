"""
Test script for performance optimization and monitoring system.
Tests performance monitoring, caching, memory management, and concurrent processing.
"""

import asyncio
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from utils.performance_monitor import performance_monitor
from utils.caching_system import cache_manager
from utils.memory_manager import memory_manager
from utils.concurrent_processor import concurrent_processor, TaskPriority

async def test_performance_monitoring():
    """Test performance monitoring system"""
    print("🔍 Testing Performance Monitoring System")
    print("=" * 50)
    
    # Start monitoring
    performance_monitor.start_monitoring()
    
    # Simulate some activity
    print("1. Recording file processing metrics...")
    performance_monitor.record_file_processing(5.2, 8.5)  # 5.2MB file, 8.5s processing
    performance_monitor.record_file_processing(10.1, 12.3)  # 10.1MB file, 12.3s processing
    
    print("2. Recording agent response times...")
    performance_monitor.record_agent_response_time("8306", 2.1)
    performance_monitor.record_agent_response_time("8307", 1.8)
    performance_monitor.record_agent_response_time("8308", 3.2)
    
    print("3. Recording cache hits/misses...")
    performance_monitor.record_cache_hit(True)
    performance_monitor.record_cache_hit(True)
    performance_monitor.record_cache_hit(False)
    performance_monitor.record_cache_hit(True)
    
    # Wait a bit for metrics collection
    await asyncio.sleep(2)
    
    # Get current metrics
    current_metrics = performance_monitor.get_current_metrics()
    if current_metrics:
        print(f"   Current CPU usage: {current_metrics.cpu_usage:.1f}%")
        print(f"   Current memory usage: {current_metrics.memory_usage:.1f}%")
        print(f"   Cache hit rate: {current_metrics.cache_hit_rate:.1%}")
    
    # Get metrics summary
    summary = performance_monitor.get_metrics_summary(minutes=5)
    print(f"   Metrics collected: {summary['metrics_count']}")
    print(f"   Average CPU usage: {summary['system_performance']['avg_cpu_usage']:.1f}%")
    
    # Get optimization recommendations
    recommendations = performance_monitor.get_optimization_recommendations()
    print(f"   Optimization recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"     - {rec['title']}: {rec['description']}")
    
    performance_monitor.stop_monitoring()
    print("✅ Performance monitoring test completed!")

async def test_caching_system():
    """Test caching system"""
    print("\n💾 Testing Caching System")
    print("=" * 30)
    
    # Test dataset caching
    print("1. Testing dataset caching...")
    sample_data = pd.DataFrame({
        'A': np.random.randn(1000),
        'B': np.random.randn(1000),
        'C': np.random.randn(1000)
    })
    
    await cache_manager.cache_dataset(
        "test_dataset_1", 
        sample_data, 
        metadata={'file_type': 'csv', 'size_category': 'medium'}
    )
    
    # Retrieve cached dataset
    cached_data = await cache_manager.get_dataset("test_dataset_1")
    if cached_data:
        print(f"   Cached dataset shape: {cached_data['data'].shape}")
        print(f"   Metadata: {cached_data['metadata']}")
    
    # Test agent response caching
    print("2. Testing agent response caching...")
    request_data = {"query": "analyze data", "dataset": "test_data"}
    request_hash = cache_manager.generate_request_hash(request_data)
    
    await cache_manager.cache_agent_response(
        "8306", 
        request_hash, 
        {"result": "analysis complete", "artifacts": ["chart.png", "summary.txt"]}
    )
    
    # Retrieve cached response
    cached_response = await cache_manager.get_agent_response("8306", request_hash)
    if cached_response:
        print(f"   Cached response: {cached_response['response']['result']}")
    
    # Test UI component caching
    print("3. Testing UI component caching...")
    cache_manager.cache_ui_component("data_card_123", {
        "title": "Sales Data",
        "rows": 1000,
        "columns": 5,
        "preview": "Sample preview data"
    })
    
    ui_component = cache_manager.get_ui_component("data_card_123")
    if ui_component:
        print(f"   Cached UI component: {ui_component['title']}")
    
    # Get cache statistics
    cache_stats = cache_manager.get_cache_stats()
    print("4. Cache statistics:")
    for cache_type, stats in cache_stats.items():
        if isinstance(stats, dict) and 'memory_cache' in stats:
            print(f"   {cache_type}: {stats['memory_cache']['size']} items, {stats['memory_cache']['hit_rate']:.1%} hit rate")
        elif isinstance(stats, dict) and 'size' in stats:
            print(f"   {cache_type}: {stats['size']} items, {stats['hit_rate']:.1%} hit rate")
    
    print("✅ Caching system test completed!")

async def test_memory_management():
    """Test memory management system"""
    print("\n🧠 Testing Memory Management System")
    print("=" * 35)
    
    # Start memory monitoring
    memory_manager.start_monitoring(interval_seconds=2)
    
    # Test lazy loading
    print("1. Testing lazy data loading...")
    def load_large_data():
        return pd.DataFrame(np.random.randn(10000, 50))
    
    lazy_loader = memory_manager.create_lazy_loader("large_dataset", load_large_data)
    print(f"   Lazy loader created, loaded: {lazy_loader.is_loaded()}")
    
    # Load data
    data = lazy_loader()
    print(f"   Data loaded, shape: {data.shape}, loaded: {lazy_loader.is_loaded()}")
    
    # Test DataFrame chunking
    print("2. Testing DataFrame chunking...")
    large_df = pd.DataFrame(np.random.randn(50000, 10))
    chunker = memory_manager.create_dataframe_chunker(large_df, chunk_size=10000)
    
    chunk_count = 0
    for chunk in chunker:
        chunk_count += 1
        # Process chunk (just count rows)
        rows_processed = len(chunk)
    
    print(f"   Processed {chunk_count} chunks, total rows: {len(large_df)}")
    
    # Test memory pool
    print("3. Testing memory pool...")
    def create_list():
        return []
    
    list_pool = memory_manager.create_memory_pool("list_pool", create_list, max_size=10)
    
    # Get and return objects
    obj1 = list_pool.get()
    obj2 = list_pool.get()
    list_pool.put(obj1)
    list_pool.put(obj2)
    
    pool_stats = list_pool.get_stats()
    print(f"   Pool stats: {pool_stats['created_count']} created, {pool_stats['reused_count']} reused")
    
    # Test session tracking
    print("4. Testing session memory tracking...")
    memory_manager.session_tracker.start_session("test_session")
    memory_manager.track_session_object("test_session", large_df, "large_dataframe")
    
    session_info = memory_manager.session_tracker.get_session_info("test_session")
    if session_info:
        print(f"   Session memory usage: {session_info['current_memory_mb']:.2f}MB")
    
    # Get memory statistics
    memory_stats = memory_manager.get_memory_stats()
    print("5. Memory statistics:")
    print(f"   Current usage: {memory_stats['current_usage']['percent']:.1f}%")
    print(f"   Process memory: {memory_stats['current_usage']['process_mb']:.1f}MB")
    print(f"   Active sessions: {memory_stats['session_count']}")
    
    # Get optimization recommendations
    recommendations = memory_manager.get_optimization_recommendations()
    print(f"   Memory optimization recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"     - {rec['title']}: {rec['description']}")
    
    memory_manager.stop_monitoring()
    print("✅ Memory management test completed!")

async def test_concurrent_processing():
    """Test concurrent processing system"""
    print("\n⚡ Testing Concurrent Processing System")
    print("=" * 40)
    
    # Start concurrent processor
    await concurrent_processor.start()
    
    # Test task submission
    print("1. Testing task submission...")
    
    async def sample_task(task_id: str, duration: float):
        await asyncio.sleep(duration)
        return f"Task {task_id} completed after {duration}s"
    
    def sync_task(task_id: str, duration: float):
        time.sleep(duration)
        return f"Sync task {task_id} completed after {duration}s"
    
    # Submit multiple tasks
    task_ids = []
    for i in range(5):
        if i % 2 == 0:
            task_id = await concurrent_processor.submit_agent_task(
                agent_id="8306",
                function=sample_task,
                args=(f"async_{i}", 1.0),
                session_id="test_session",
                priority=TaskPriority.NORMAL
            )
        else:
            task_id = await concurrent_processor.submit_agent_task(
                agent_id="8307",
                function=sync_task,
                args=(f"sync_{i}", 0.5),
                session_id="test_session",
                priority=TaskPriority.HIGH
            )
        
        task_ids.append(task_id)
        print(f"   Submitted task {task_id}")
    
    # Wait for results
    print("2. Waiting for task results...")
    results = []
    for task_id in task_ids:
        try:
            result = await concurrent_processor.get_task_result(task_id, timeout_seconds=10)
            results.append(result)
            print(f"   Result: {result}")
        except Exception as e:
            print(f"   Task {task_id} failed: {str(e)}")
    
    # Get system statistics
    print("3. System statistics:")
    stats = concurrent_processor.get_system_stats()
    
    print(f"   Active sessions: {stats['concurrent_processor']['active_sessions']}")
    print(f"   Worker utilization: {stats['task_scheduler']['worker_pool_stats']['active_workers']}/{stats['task_scheduler']['worker_pool_stats']['max_workers']}")
    print(f"   Queue size: {stats['task_scheduler']['worker_pool_stats']['queue_size']}")
    print(f"   Load balancer utilization: {stats['load_balancer']['overall_utilization']:.1%}")
    
    # Get performance recommendations
    recommendations = concurrent_processor.get_performance_recommendations()
    print(f"   Performance recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"     - {rec['title']}: {rec['description']}")
    
    await concurrent_processor.stop()
    print("✅ Concurrent processing test completed!")

async def test_integrated_performance():
    """Test integrated performance optimization"""
    print("\n🚀 Testing Integrated Performance System")
    print("=" * 45)
    
    # Start all systems
    performance_monitor.start_monitoring()
    memory_manager.start_monitoring(interval_seconds=5)
    await concurrent_processor.start()
    
    print("1. Running integrated performance test...")
    
    # Simulate realistic workload
    async def data_processing_task(dataset_size: int):
        # Create dataset
        data = pd.DataFrame(np.random.randn(dataset_size, 20))
        
        # Track memory usage
        memory_manager.track_session_object("perf_test", data, f"dataset_{dataset_size}")
        
        # Cache dataset
        await cache_manager.cache_dataset(f"perf_dataset_{dataset_size}", data)
        
        # Process data in chunks
        chunker = memory_manager.create_dataframe_chunker(data, chunk_size=1000)
        results = []
        
        for chunk in chunker:
            # Simulate processing
            result = chunk.mean().to_dict()
            results.append(result)
            await asyncio.sleep(0.1)  # Simulate processing time
        
        return {"processed_chunks": len(results), "dataset_size": dataset_size}
    
    # Submit multiple concurrent tasks
    task_ids = []
    for i in range(3):
        task_id = await concurrent_processor.submit_agent_task(
            agent_id=f"830{6+i}",
            function=data_processing_task,
            args=(5000 + i * 1000,),
            session_id="perf_test_session",
            priority=TaskPriority.NORMAL,
            timeout_seconds=30
        )
        task_ids.append(task_id)
    
    # Wait for completion
    results = []
    for task_id in task_ids:
        try:
            result = await concurrent_processor.get_task_result(task_id, timeout_seconds=35)
            results.append(result)
        except Exception as e:
            print(f"   Task failed: {str(e)}")
    
    print(f"   Completed {len(results)} data processing tasks")
    
    # Get comprehensive statistics
    print("2. Performance summary:")
    
    # Performance metrics
    perf_summary = performance_monitor.get_metrics_summary(minutes=5)
    print(f"   Average CPU usage: {perf_summary['system_performance']['avg_cpu_usage']:.1f}%")
    print(f"   Average memory usage: {perf_summary['system_performance']['avg_memory_usage']:.1f}%")
    
    # Cache performance
    cache_stats = cache_manager.get_cache_stats()
    dataset_cache = cache_stats.get('dataset_cache', {})
    if 'memory_cache' in dataset_cache:
        print(f"   Dataset cache hit rate: {dataset_cache['memory_cache']['hit_rate']:.1%}")
    
    # Memory usage
    memory_stats = memory_manager.get_memory_stats()
    print(f"   Process memory usage: {memory_stats['current_usage']['process_mb']:.1f}MB")
    
    # Concurrent processing
    concurrent_stats = concurrent_processor.get_system_stats()
    print(f"   Tasks processed: {concurrent_stats['task_scheduler']['worker_pool_stats']['tasks_processed']}")
    
    # Cleanup
    performance_monitor.stop_monitoring()
    memory_manager.stop_monitoring()
    await concurrent_processor.stop()
    
    print("✅ Integrated performance test completed!")

async def main():
    """Run all performance tests"""
    print("🚀 Starting Performance System Tests")
    print("=" * 60)
    
    try:
        await test_performance_monitoring()
        await test_caching_system()
        await test_memory_management()
        await test_concurrent_processing()
        await test_integrated_performance()
        
        print("\n🎉 All performance tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())