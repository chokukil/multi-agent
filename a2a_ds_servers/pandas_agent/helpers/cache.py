"""
Cache management for pandas_agent

This module provides intelligent caching capabilities for query results,
data analysis, and visualizations to improve performance.
"""

import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import os

from .logger import get_logger


@dataclass
class CacheEntry:
    """
    Cache entry with metadata
    """
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "ttl_seconds": self.ttl_seconds,
            "tags": self.tags
        }


class CacheManager:
    """
    Intelligent cache manager for pandas_agent
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) expiration
    - Size-based eviction
    - Tag-based cache invalidation
    - Persistent storage option
    """
    
    def __init__(self,
                 max_size_mb: int = 100,
                 max_entries: int = 1000,
                 default_ttl_seconds: Optional[int] = 3600,
                 persistent: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize cache manager
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of cache entries
            default_ttl_seconds: Default TTL for entries (None = no expiration)
            persistent: Whether to persist cache to disk
            cache_dir: Directory for persistent cache files
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.persistent = persistent
        
        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_data: Dict[str, Any] = {}
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_bytes": 0
        }
        
        # Persistent storage
        self.cache_dir = None
        if persistent and cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger()
        self.logger.info(f"CacheManager initialized: max_size={max_size_mb}MB, max_entries={max_entries}")
    
    def _generate_key(self, key_data: Union[str, Dict[str, Any]]) -> str:
        """Generate cache key from data"""
        if isinstance(key_data, str):
            return hashlib.md5(key_data.encode()).hexdigest()
        else:
            # For dict or other objects, serialize to JSON first
            json_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
    
    def get(self, key: Union[str, Dict[str, Any]], default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key or key data
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_key = self._generate_key(key)
        
        if cache_key not in self._cache:
            self._stats["misses"] += 1
            return default
        
        entry = self._cache[cache_key]
        
        # Check if expired
        if entry.is_expired():
            self._remove_entry(cache_key)
            self._stats["misses"] += 1
            return default
        
        # Update access info
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        self._stats["hits"] += 1
        
        return self._cache_data[cache_key]
    
    def set(self, 
            key: Union[str, Dict[str, Any]], 
            value: Any,
            ttl_seconds: Optional[int] = None,
            tags: Optional[List[str]] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key or key data
            value: Value to cache
            ttl_seconds: TTL for this entry (overrides default)
            tags: Tags for this entry
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._generate_key(key)
        
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_size_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                tags=tags or []
            )
            
            # Check if we need to evict entries
            self._ensure_capacity(size_bytes)
            
            # Store entry
            self._cache[cache_key] = entry
            self._cache_data[cache_key] = value
            self._stats["total_size_bytes"] += size_bytes
            
            # Persist if enabled
            if self.persistent and self.cache_dir:
                self._persist_entry(cache_key, entry, value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching value: {e}")
            return False
    
    def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key or key data
            
        Returns:
            True if entry was deleted, False if not found
        """
        cache_key = self._generate_key(key)
        
        if cache_key in self._cache:
            self._remove_entry(cache_key)
            return True
        
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._cache_data.clear()
        self._stats["total_size_bytes"] = 0
        
        # Clear persistent storage
        if self.persistent and self.cache_dir:
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    file_path.unlink()
                except:
                    pass
        
        self.logger.info("Cache cleared")
    
    def clear_by_tags(self, tags: List[str]):
        """
        Clear cache entries by tags
        
        Args:
            tags: List of tags to match
        """
        to_remove = []
        
        for cache_key, entry in self._cache.items():
            if any(tag in entry.tags for tag in tags):
                to_remove.append(cache_key)
        
        for cache_key in to_remove:
            self._remove_entry(cache_key)
        
        self.logger.info(f"Cleared {len(to_remove)} entries with tags: {tags}")
    
    def _ensure_capacity(self, new_size_bytes: int):
        """Ensure cache has capacity for new entry"""
        
        # Clean up expired entries first
        self._cleanup_expired()
        
        # Check size constraint
        while (self._stats["total_size_bytes"] + new_size_bytes > self.max_size_bytes or
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
            
            # Find LRU entry
            lru_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].last_accessed)
            
            self._remove_entry(lru_key)
            self._stats["evictions"] += 1
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self._stats["total_size_bytes"] -= entry.size_bytes
            
            del self._cache[cache_key]
            del self._cache_data[cache_key]
            
            # Remove persistent file
            if self.persistent and self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except:
                        pass
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            # Use pickle to estimate size
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return 1024  # Default estimate
    
    def _persist_entry(self, cache_key: str, entry: CacheEntry, value: Any):
        """Persist cache entry to disk"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            cache_data = {
                "metadata": entry.to_dict(),
                "value": value
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to persist cache entry: {e}")
    
    def load_persistent_cache(self):
        """Load cache from persistent storage"""
        if not self.persistent or not self.cache_dir:
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    metadata = cache_data["metadata"]
                    value = cache_data["value"]
                    
                    # Recreate cache entry
                    entry = CacheEntry(
                        key=metadata["key"],
                        value=value,
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                        last_accessed=datetime.fromisoformat(metadata["last_accessed"]),
                        access_count=metadata["access_count"],
                        size_bytes=metadata["size_bytes"],
                        ttl_seconds=metadata["ttl_seconds"],
                        tags=metadata["tags"]
                    )
                    
                    # Skip if expired
                    if entry.is_expired():
                        cache_file.unlink()
                        continue
                    
                    # Add to cache
                    cache_key = metadata["key"]
                    self._cache[cache_key] = entry
                    self._cache_data[cache_key] = value
                    self._stats["total_size_bytes"] += entry.size_bytes
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
                    try:
                        cache_file.unlink()
                    except:
                        pass
            
            self.logger.info(f"Loaded {len(self._cache)} entries from persistent cache")
            
        except Exception as e:
            self.logger.error(f"Error loading persistent cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self._cache),
            "size_mb": self._stats["total_size_bytes"] / (1024 * 1024),
            "hit_rate": hit_rate,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "max_entries": self.max_entries,
            "max_size_mb": self.max_size_bytes / (1024 * 1024)
        }
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """Get information about cached entries"""
        return [
            {
                "key": entry.key[:16] + "..." if len(entry.key) > 16 else entry.key,
                "size_bytes": entry.size_bytes,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "tags": entry.tags,
                "expired": entry.is_expired()
            }
            for entry in self._cache.values()
        ]


# Global cache instance
_global_cache = None


def get_cache_manager(
    max_size_mb: int = 100,
    max_entries: int = 1000,
    default_ttl_seconds: Optional[int] = 3600,
    persistent: bool = False,
    cache_dir: Optional[str] = None
) -> CacheManager:
    """
    Get global cache manager instance
    
    Args:
        max_size_mb: Maximum cache size in MB
        max_entries: Maximum number of entries
        default_ttl_seconds: Default TTL in seconds
        persistent: Enable persistent storage
        cache_dir: Directory for persistent cache
        
    Returns:
        Global cache manager instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = CacheManager(
            max_size_mb=max_size_mb,
            max_entries=max_entries,
            default_ttl_seconds=default_ttl_seconds,
            persistent=persistent,
            cache_dir=cache_dir
        )
        
        # Load persistent cache if enabled
        if persistent:
            _global_cache.load_persistent_cache()
    
    return _global_cache 