"""
Memory management system for Cherry AI Streamlit Platform.
Implements lazy loading, garbage collection optimization, and memory monitoring.
"""

import gc
import sys
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import psutil
import pandas as pd
import numpy as np
from collections import defaultdict

@dataclass
class MemoryUsage:
    """Memory usage information"""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    process_mb: float
    timestamp: datetime

class LazyDataLoader:
    """Lazy loading system for large datasets"""
    
    def __init__(self, load_function: Callable, *args, **kwargs):
        self.load_function = load_function
        self.args = args
        self.kwargs = kwargs
        self._data = None
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()
        self.load_time: Optional[datetime] = None
        self.access_count = 0
    
    def __call__(self) -> Any:
        """Load and return data"""
        if self._loaded:
            self.access_count += 1
            return self._data
        
        with self._lock:
            if self._loaded:
                self.access_count += 1
                return self._data
            
            if self._loading:
                # Wait for loading to complete
                while self._loading:
                    time.sleep(0.1)
                self.access_count += 1
                return self._data
            
            self._loading = True
            try:
                self._data = self.load_function(*self.args, **self.kwargs)
                self._loaded = True
                self.load_time = datetime.now()
                self.access_count += 1
                return self._data
            finally:
                self._loading = False
    
    def is_loaded(self) -> bool:
        """Check if data is loaded"""
        return self._loaded
    
    def unload(self):
        """Unload data to free memory"""
        with self._lock:
            self._data = None
            self._loaded = False
            self.load_time = None
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information"""
        return {
            'loaded': self._loaded,
            'loading': self._loading,
            'load_time': self.load_time,
            'access_count': self.access_count
        }

class DataFrameChunker:
    """Chunked processing for large DataFrames"""
    
    def __init__(self, df: pd.DataFrame, chunk_size: int = 10000):
        self.df = df
        self.chunk_size = chunk_size
        self.total_rows = len(df)
        self.num_chunks = (self.total_rows + chunk_size - 1) // chunk_size
    
    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over DataFrame chunks"""
        for i in range(0, self.total_rows, self.chunk_size):
            yield self.df.iloc[i:i + self.chunk_size]
    
    def process_chunks(self, process_func: Callable[[pd.DataFrame], Any]) -> List[Any]:
        """Process DataFrame in chunks"""
        results = []
        for chunk in self:
            result = process_func(chunk)
            results.append(result)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        return results
    
    def get_chunk(self, chunk_index: int) -> pd.DataFrame:
        """Get specific chunk by index"""
        if chunk_index >= self.num_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range (0-{self.num_chunks-1})")
        
        start_idx = chunk_index * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_rows)
        return self.df.iloc[start_idx:end_idx]

class MemoryPool:
    """Memory pool for reusing objects"""
    
    def __init__(self, object_factory: Callable, max_size: int = 100):
        self.object_factory = object_factory
        self.max_size = max_size
        self.pool: List[Any] = []
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0
    
    def get(self) -> Any:
        """Get object from pool or create new one"""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.reused_count += 1
                return obj
            else:
                obj = self.object_factory()
                self.created_count += 1
                return obj
    
    def put(self, obj: Any):
        """Return object to pool"""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def clear(self):
        """Clear the pool"""
        with self.lock:
            self.pool.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': self.reused_count / (self.created_count + self.reused_count) if (self.created_count + self.reused_count) > 0 else 0
            }

class SessionMemoryTracker:
    """Track memory usage per session"""
    
    def __init__(self):
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, session_id: str):
        """Start tracking a session"""
        with self.lock:
            self.session_data[session_id] = {
                'start_time': datetime.now(),
                'objects': weakref.WeakSet(),
                'peak_memory_mb': 0.0,
                'current_memory_mb': 0.0,
                'allocations': []
            }
    
    def track_object(self, session_id: str, obj: Any, name: str = None):
        """Track an object for a session"""
        with self.lock:
            if session_id not in self.session_data:
                self.start_session(session_id)
            
            session = self.session_data[session_id]
            
            # Try to add to weak set, but handle unhashable types
            try:
                session['objects'].add(obj)
            except TypeError:
                # For unhashable types, just track the metadata
                pass
            
            # Estimate object size
            size_mb = self._estimate_object_size(obj)
            session['current_memory_mb'] += size_mb
            session['peak_memory_mb'] = max(session['peak_memory_mb'], session['current_memory_mb'])
            
            session['allocations'].append({
                'timestamp': datetime.now(),
                'object_name': name or type(obj).__name__,
                'size_mb': size_mb
            })
    
    def end_session(self, session_id: str):
        """End session tracking"""
        with self.lock:
            if session_id in self.session_data:
                session = self.session_data[session_id]
                session['end_time'] = datetime.now()
                session['duration'] = session['end_time'] - session['start_time']
                
                # Clear weak references
                session['objects'].clear()
                
                self.logger.info(f"Session {session_id} ended. Peak memory: {session['peak_memory_mb']:.2f}MB")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session memory information"""
        with self.lock:
            if session_id not in self.session_data:
                return None
            
            session = self.session_data[session_id].copy()
            session['active_objects'] = len(session['objects'])
            return session
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.lock:
            sessions_to_remove = []
            for session_id, session_data in self.session_data.items():
                if session_data.get('end_time', datetime.now()) < cutoff_time:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.session_data[session_id]
    
    def _estimate_object_size(self, obj: Any) -> float:
        """Estimate object size in MB"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes / (1024 * 1024)
            else:
                return sys.getsizeof(obj) / (1024 * 1024)
        except:
            return 0.1  # Default estimate

class GarbageCollectionOptimizer:
    """Optimize garbage collection for better performance"""
    
    def __init__(self):
        self.gc_stats = {
            'collections': defaultdict(int),
            'collected': defaultdict(int),
            'uncollectable': defaultdict(int)
        }
        self.last_gc_time = time.time()
        self.gc_threshold_multiplier = 1.0
        self.logger = logging.getLogger(__name__)
        
        # Set initial thresholds
        self._optimize_thresholds()
    
    def _optimize_thresholds(self):
        """Optimize GC thresholds based on usage patterns"""
        # Get current thresholds
        thresholds = gc.get_threshold()
        
        # Adjust thresholds based on memory pressure
        memory = psutil.virtual_memory()
        
        if memory.percent > 80:
            # High memory usage - more aggressive GC
            multiplier = 0.7
        elif memory.percent < 50:
            # Low memory usage - less aggressive GC
            multiplier = 1.5
        else:
            # Normal memory usage
            multiplier = 1.0
        
        new_thresholds = tuple(int(t * multiplier) for t in thresholds)
        gc.set_threshold(*new_thresholds)
        
        self.gc_threshold_multiplier = multiplier
        self.logger.debug(f"GC thresholds adjusted: {thresholds} -> {new_thresholds}")
    
    def force_collection(self, generation: Optional[int] = None) -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        start_time = time.time()
        
        if generation is not None:
            collected = gc.collect(generation)
            gen_stats = {generation: collected}
        else:
            gen_stats = {}
            for gen in range(3):
                collected = gc.collect(gen)
                gen_stats[gen] = collected
        
        collection_time = time.time() - start_time
        
        # Update statistics
        for gen, collected in gen_stats.items():
            self.gc_stats['collections'][gen] += 1
            self.gc_stats['collected'][gen] += collected
        
        self.last_gc_time = time.time()
        
        self.logger.debug(f"GC completed in {collection_time:.3f}s, collected: {gen_stats}")
        
        return {
            'collected_objects': gen_stats,
            'collection_time': collection_time,
            'total_objects': len(gc.get_objects())
        }
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            'collections': dict(self.gc_stats['collections']),
            'collected': dict(self.gc_stats['collected']),
            'uncollectable': dict(self.gc_stats['uncollectable']),
            'threshold_multiplier': self.gc_threshold_multiplier,
            'current_thresholds': gc.get_threshold(),
            'total_objects': len(gc.get_objects()),
            'time_since_last_gc': time.time() - self.last_gc_time
        }
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize GC for specific workload types"""
        if workload_type == 'batch_processing':
            # Less frequent GC for batch processing
            gc.set_threshold(1000, 15, 15)
        elif workload_type == 'interactive':
            # More frequent GC for interactive use
            gc.set_threshold(500, 8, 8)
        elif workload_type == 'memory_intensive':
            # Very aggressive GC for memory-intensive tasks
            gc.set_threshold(300, 5, 5)
        else:
            # Default settings
            gc.set_threshold(700, 10, 10)

class MemoryManager:
    """Central memory management system"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.session_tracker = SessionMemoryTracker()
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.lazy_loaders: Dict[str, LazyDataLoader] = {}
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.memory_history: List[MemoryUsage] = []
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                usage = self.get_memory_usage()
                self.memory_history.append(usage)
                
                # Keep only recent history
                if len(self.memory_history) > 1000:
                    self.memory_history.pop(0)
                
                # Check for memory pressure
                if usage.percent > 85:
                    self._handle_memory_pressure(usage)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {str(e)}")
                time.sleep(interval_seconds)
    
    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)
        
        return MemoryUsage(
            total_mb=memory.total / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            percent=memory.percent,
            process_mb=process_memory,
            timestamp=datetime.now()
        )
    
    def _handle_memory_pressure(self, usage: MemoryUsage):
        """Handle high memory usage"""
        self.logger.warning(f"High memory usage detected: {usage.percent:.1f}%")
        
        # Force garbage collection
        gc_stats = self.gc_optimizer.force_collection()
        self.logger.info(f"Emergency GC collected {sum(gc_stats['collected_objects'].values())} objects")
        
        # Unload lazy loaders that haven't been accessed recently
        self._unload_stale_lazy_loaders()
        
        # Clear memory pools
        for pool in self.memory_pools.values():
            pool.clear()
        
        # Clean up old session data
        self.session_tracker.cleanup_old_sessions(max_age_hours=1)
    
    def _unload_stale_lazy_loaders(self, max_age_minutes: int = 30):
        """Unload lazy loaders that haven't been accessed recently"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        loaders_to_unload = []
        for name, loader in self.lazy_loaders.items():
            if loader.is_loaded() and loader.load_time and loader.load_time < cutoff_time:
                loaders_to_unload.append(name)
        
        for name in loaders_to_unload:
            self.lazy_loaders[name].unload()
            self.logger.debug(f"Unloaded stale lazy loader: {name}")
    
    def create_lazy_loader(self, name: str, load_function: Callable, *args, **kwargs) -> LazyDataLoader:
        """Create a lazy data loader"""
        loader = LazyDataLoader(load_function, *args, **kwargs)
        self.lazy_loaders[name] = loader
        return loader
    
    def create_memory_pool(self, name: str, object_factory: Callable, max_size: int = 100) -> MemoryPool:
        """Create a memory pool"""
        pool = MemoryPool(object_factory, max_size)
        self.memory_pools[name] = pool
        return pool
    
    def create_dataframe_chunker(self, df: pd.DataFrame, chunk_size: int = 10000) -> DataFrameChunker:
        """Create a DataFrame chunker for memory-efficient processing"""
        return DataFrameChunker(df, chunk_size)
    
    def track_session_object(self, session_id: str, obj: Any, name: str = None):
        """Track an object for a session"""
        self.session_tracker.track_object(session_id, obj, name)
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize memory management for specific workload"""
        self.gc_optimizer.optimize_for_workload(workload_type)
        self.logger.info(f"Memory management optimized for {workload_type} workload")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_usage = self.get_memory_usage()
        
        # Calculate memory trends
        if len(self.memory_history) > 1:
            recent_usage = [u.percent for u in self.memory_history[-10:]]
            memory_trend = 'increasing' if recent_usage[-1] > recent_usage[0] else 'decreasing'
            avg_usage = sum(recent_usage) / len(recent_usage)
        else:
            memory_trend = 'stable'
            avg_usage = current_usage.percent
        
        # Pool statistics
        pool_stats = {}
        for name, pool in self.memory_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Lazy loader statistics
        loader_stats = {}
        for name, loader in self.lazy_loaders.items():
            loader_stats[name] = loader.get_info()
        
        return {
            'current_usage': {
                'total_mb': current_usage.total_mb,
                'used_mb': current_usage.used_mb,
                'available_mb': current_usage.available_mb,
                'percent': current_usage.percent,
                'process_mb': current_usage.process_mb
            },
            'trends': {
                'memory_trend': memory_trend,
                'avg_usage_percent': avg_usage,
                'history_points': len(self.memory_history)
            },
            'gc_stats': self.gc_optimizer.get_gc_stats(),
            'memory_pools': pool_stats,
            'lazy_loaders': loader_stats,
            'session_count': len(self.session_tracker.session_data),
            'monitoring_active': self.monitoring_active
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get memory optimization recommendations"""
        recommendations = []
        current_usage = self.get_memory_usage()
        
        if current_usage.percent > 80:
            recommendations.append({
                'type': 'high_memory_usage',
                'priority': 'high',
                'title': 'High Memory Usage',
                'description': f'Memory usage is at {current_usage.percent:.1f}%',
                'actions': [
                    'Enable more aggressive garbage collection',
                    'Implement data streaming for large files',
                    'Increase lazy loading usage',
                    'Clear unused caches'
                ]
            })
        
        # Check for memory leaks
        if len(self.memory_history) > 10:
            recent_trend = [u.percent for u in self.memory_history[-10:]]
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                recommendations.append({
                    'type': 'potential_memory_leak',
                    'priority': 'medium',
                    'title': 'Potential Memory Leak',
                    'description': 'Memory usage is consistently increasing',
                    'actions': [
                        'Review session cleanup procedures',
                        'Check for unreleased object references',
                        'Monitor garbage collection effectiveness'
                    ]
                })
        
        # Check pool efficiency
        for name, pool in self.memory_pools.items():
            stats = pool.get_stats()
            if stats['reuse_rate'] < 0.5 and stats['created_count'] > 50:
                recommendations.append({
                    'type': 'low_pool_efficiency',
                    'priority': 'low',
                    'title': f'Low Pool Efficiency: {name}',
                    'description': f'Pool reuse rate is {stats["reuse_rate"]:.1%}',
                    'actions': [
                        'Increase pool size',
                        'Review object lifecycle',
                        'Consider different pooling strategy'
                    ]
                })
        
        return recommendations
    
    def cleanup(self):
        """Perform comprehensive cleanup"""
        self.logger.info("Starting memory cleanup")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Force garbage collection
        self.gc_optimizer.force_collection()
        
        # Clear all pools
        for pool in self.memory_pools.values():
            pool.clear()
        
        # Unload all lazy loaders
        for loader in self.lazy_loaders.values():
            loader.unload()
        
        # Clean up session data
        self.session_tracker.cleanup_old_sessions(max_age_hours=0)
        
        self.logger.info("Memory cleanup completed")

# Global memory manager instance
memory_manager = MemoryManager()