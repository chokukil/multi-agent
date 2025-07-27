"""
Intelligent caching system for Cherry AI Streamlit Platform.
Implements multi-level caching for datasets, agent responses, and UI components.
"""

import asyncio
import hashlib
import pickle
import time
import threading
from typing import Any, Dict, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import json
import logging
import os
import tempfile
import weakref

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL expiration
            if self._is_expired(entry):
                del self.cache[key]
                self.stats['expired'] += 1
                self.stats['misses'] += 1
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None):
        """Put value in cache"""
        with self.lock:
            now = datetime.now()
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.ttl_seconds,
                tags=tags or []
            )
            
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict if necessary
            self._evict_if_necessary()
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expired': 0
            }
    
    def clear_by_tags(self, tags: List[str]):
        """Clear entries with specific tags"""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl_seconds
    
    def _evict_if_necessary(self):
        """Evict entries if cache is full"""
        while len(self.cache) > self.max_size:
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'total_size_bytes': total_size,
                **self.stats
            }

class PersistentCache:
    """Persistent cache using file system"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 500):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'cherry_ai_cache')
        self.max_size_mb = max_size_mb
        self.index_file = os.path.join(self.cache_dir, 'cache_index.json')
        self.lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load index
        self.index = self._load_index()
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        with self.lock:
            if key not in self.index:
                return None
            
            entry_info = self.index[key]
            file_path = os.path.join(self.cache_dir, entry_info['filename'])
            
            # Check if file exists
            if not os.path.exists(file_path):
                del self.index[key]
                self._save_index()
                return None
            
            # Check TTL
            if entry_info.get('ttl_seconds'):
                age = time.time() - entry_info['created_at']
                if age > entry_info['ttl_seconds']:
                    self._delete_entry(key)
                    return None
            
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access info
                entry_info['last_accessed'] = time.time()
                entry_info['access_count'] = entry_info.get('access_count', 0) + 1
                self._save_index()
                
                return value
                
            except Exception as e:
                self.logger.error(f"Error loading cache entry {key}: {str(e)}")
                self._delete_entry(key)
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: List[str] = None):
        """Put value in persistent cache"""
        with self.lock:
            try:
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
                file_path = os.path.join(self.cache_dir, filename)
                
                # Save value to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                
                # Update index
                self.index[key] = {
                    'filename': filename,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'access_count': 0,
                    'size_bytes': file_size,
                    'ttl_seconds': ttl_seconds,
                    'tags': tags or []
                }
                
                self._save_index()
                self._cleanup_if_necessary()
                
            except Exception as e:
                self.logger.error(f"Error saving cache entry {key}: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from persistent cache"""
        with self.lock:
            return self._delete_entry(key)
    
    def clear(self):
        """Clear all persistent cache entries"""
        with self.lock:
            for key in list(self.index.keys()):
                self._delete_entry(key)
    
    def clear_by_tags(self, tags: List[str]):
        """Clear entries with specific tags"""
        with self.lock:
            keys_to_remove = []
            for key, entry_info in self.index.items():
                if any(tag in entry_info.get('tags', []) for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._delete_entry(key)
    
    def _delete_entry(self, key: str) -> bool:
        """Delete a single cache entry"""
        if key not in self.index:
            return False
        
        entry_info = self.index[key]
        file_path = os.path.join(self.cache_dir, entry_info['filename'])
        
        # Remove file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            self.logger.error(f"Error removing cache file {file_path}: {str(e)}")
        
        # Remove from index
        del self.index[key]
        self._save_index()
        return True
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from file"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache index: {str(e)}")
        
        return {}
    
    def _save_index(self):
        """Save cache index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache index: {str(e)}")
    
    def _cleanup_if_necessary(self):
        """Clean up cache if it exceeds size limit"""
        total_size = sum(entry['size_bytes'] for entry in self.index.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries until under limit
        for key, entry_info in sorted_entries:
            if total_size <= max_size_bytes:
                break
            
            total_size -= entry_info['size_bytes']
            self._delete_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistent cache statistics"""
        with self.lock:
            total_size = sum(entry['size_bytes'] for entry in self.index.values())
            total_access = sum(entry.get('access_count', 0) for entry in self.index.values())
            
            return {
                'entries': len(self.index),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_mb,
                'total_accesses': total_access,
                'cache_dir': self.cache_dir
            }

class MultiLevelCache:
    """Multi-level caching system combining memory and persistent caches"""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 memory_ttl_seconds: int = 3600,  # 1 hour
                 persistent_cache_size_mb: int = 500,
                 persistent_ttl_seconds: int = 86400):  # 24 hours
        
        self.memory_cache = LRUCache(memory_cache_size, memory_ttl_seconds)
        self.persistent_cache = PersistentCache(max_size_mb=persistent_cache_size_mb)
        self.persistent_ttl = persistent_ttl_seconds
        
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try persistent cache
        value = self.persistent_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value
        
        return None
    
    async def put(self, key: str, value: Any, tags: List[str] = None, memory_only: bool = False):
        """Put value in multi-level cache"""
        # Always put in memory cache
        self.memory_cache.put(key, value, tags=tags)
        
        # Put in persistent cache unless memory_only is True
        if not memory_only:
            self.persistent_cache.put(key, value, ttl_seconds=self.persistent_ttl, tags=tags)
    
    async def delete(self, key: str):
        """Delete from both cache levels"""
        self.memory_cache.delete(key)
        self.persistent_cache.delete(key)
    
    async def clear(self):
        """Clear both cache levels"""
        self.memory_cache.clear()
        self.persistent_cache.clear()
    
    async def clear_by_tags(self, tags: List[str]):
        """Clear entries with specific tags from both levels"""
        self.memory_cache.clear_by_tags(tags)
        self.persistent_cache.clear_by_tags(tags)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        persistent_stats = self.persistent_cache.get_stats()
        
        return {
            'memory_cache': memory_stats,
            'persistent_cache': persistent_stats,
            'combined_hit_rate': memory_stats['hit_rate']  # Memory cache hit rate is most relevant
        }

class CacheManager:
    """Central cache manager for different data types"""
    
    def __init__(self):
        # Different caches for different data types
        self.dataset_cache = MultiLevelCache(
            memory_cache_size=100,  # Smaller for large datasets
            persistent_cache_size_mb=1000  # Larger persistent cache
        )
        
        self.agent_response_cache = MultiLevelCache(
            memory_cache_size=500,
            memory_ttl_seconds=1800,  # 30 minutes
            persistent_ttl_seconds=7200  # 2 hours
        )
        
        self.ui_component_cache = LRUCache(
            max_size=1000,
            ttl_seconds=300  # 5 minutes for UI components
        )
        
        self.analysis_result_cache = MultiLevelCache(
            memory_cache_size=200,
            memory_ttl_seconds=3600,  # 1 hour
            persistent_ttl_seconds=86400  # 24 hours
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def cache_dataset(self, dataset_id: str, dataset: Any, metadata: Dict[str, Any] = None):
        """Cache dataset with metadata"""
        tags = ['dataset']
        if metadata:
            if metadata.get('file_type'):
                tags.append(f"type_{metadata['file_type']}")
            if metadata.get('size_category'):
                tags.append(f"size_{metadata['size_category']}")
        
        await self.dataset_cache.put(dataset_id, {
            'data': dataset,
            'metadata': metadata or {},
            'cached_at': datetime.now()
        }, tags=tags)
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset"""
        return await self.dataset_cache.get(dataset_id)
    
    async def cache_agent_response(self, agent_id: str, request_hash: str, response: Any):
        """Cache agent response"""
        cache_key = f"{agent_id}:{request_hash}"
        tags = ['agent_response', f'agent_{agent_id}']
        
        await self.agent_response_cache.put(cache_key, {
            'response': response,
            'agent_id': agent_id,
            'cached_at': datetime.now()
        }, tags=tags)
    
    async def get_agent_response(self, agent_id: str, request_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached agent response"""
        cache_key = f"{agent_id}:{request_hash}"
        return await self.agent_response_cache.get(cache_key)
    
    def cache_ui_component(self, component_id: str, component_data: Any):
        """Cache UI component"""
        self.ui_component_cache.put(component_id, component_data, tags=['ui_component'])
    
    def get_ui_component(self, component_id: str) -> Optional[Any]:
        """Get cached UI component"""
        return self.ui_component_cache.get(component_id)
    
    async def cache_analysis_result(self, analysis_id: str, result: Any, metadata: Dict[str, Any] = None):
        """Cache analysis result"""
        tags = ['analysis_result']
        if metadata:
            if metadata.get('analysis_type'):
                tags.append(f"type_{metadata['analysis_type']}")
        
        await self.analysis_result_cache.put(analysis_id, {
            'result': result,
            'metadata': metadata or {},
            'cached_at': datetime.now()
        }, tags=tags)
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        return await self.analysis_result_cache.get(analysis_id)
    
    def generate_request_hash(self, request_data: Dict[str, Any]) -> str:
        """Generate hash for request data"""
        # Create a stable hash from request data
        request_str = json.dumps(request_data, sort_keys=True, default=str)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    async def invalidate_agent_cache(self, agent_id: str):
        """Invalidate all cached responses for specific agent"""
        await self.agent_response_cache.clear_by_tags([f'agent_{agent_id}'])
    
    async def invalidate_dataset_cache(self, dataset_id: str = None):
        """Invalidate dataset cache"""
        if dataset_id:
            await self.dataset_cache.delete(dataset_id)
        else:
            await self.dataset_cache.clear_by_tags(['dataset'])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'dataset_cache': self.dataset_cache.get_stats(),
            'agent_response_cache': self.agent_response_cache.get_stats(),
            'ui_component_cache': self.ui_component_cache.get_stats(),
            'analysis_result_cache': self.analysis_result_cache.get_stats()
        }
    
    async def cleanup_expired_entries(self):
        """Clean up expired entries from all caches"""
        # This is handled automatically by TTL, but we can force cleanup
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = self.get_cache_stats()
        
        total_memory_bytes = (
            stats['dataset_cache']['memory_cache']['total_size_bytes'] +
            stats['agent_response_cache']['memory_cache']['total_size_bytes'] +
            stats['ui_component_cache']['total_size_bytes'] +
            stats['analysis_result_cache']['memory_cache']['total_size_bytes']
        )
        
        return {
            'total_memory_bytes': total_memory_bytes,
            'total_memory_mb': total_memory_bytes / (1024 * 1024),
            'breakdown': {
                'datasets': stats['dataset_cache']['memory_cache']['total_size_bytes'],
                'agent_responses': stats['agent_response_cache']['memory_cache']['total_size_bytes'],
                'ui_components': stats['ui_component_cache']['total_size_bytes'],
                'analysis_results': stats['analysis_result_cache']['memory_cache']['total_size_bytes']
            }
        }

# Global cache manager instance
cache_manager = CacheManager()