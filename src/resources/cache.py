"""
Generic LRU cache implementation for resource management.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar, OrderedDict
import time
import logging
from pathlib import Path

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    value: T
    access_time: float
    creation_time: float
    size_bytes: Optional[int] = None
    
    def __post_init__(self):
        """Set creation time if not provided."""
        if self.creation_time == 0:
            self.creation_time = time.time()


class LRUCache(Generic[T]):
    """Least Recently Used cache with size limits and TTL support."""
    
    def __init__(self, 
                 max_size: int = 100,
                 max_memory_mb: Optional[int] = None,
                 ttl_seconds: Optional[int] = None):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB (approximate)
            ttl_seconds: Time-to-live for entries in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._total_size_bytes = 0
        
        logger.debug(f"LRU Cache initialized: max_size={max_size}, "
                    f"max_memory_mb={max_memory_mb}, ttl_seconds={ttl_seconds}")
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache, updating access time."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if self.ttl_seconds and time.time() - entry.creation_time > self.ttl_seconds:
            self.remove(key)
            return None
        
        # Update access time and move to end (most recent)
        entry.access_time = time.time()
        self._cache.move_to_end(key)
        
        return entry.value
    
    def put(self, key: str, value: T, size_bytes: Optional[int] = None) -> None:
        """Put value into cache."""
        current_time = time.time()
        
        # If key exists, update it
        if key in self._cache:
            old_entry = self._cache[key]
            if old_entry.size_bytes:
                self._total_size_bytes -= old_entry.size_bytes
        
        # Create new entry
        entry = CacheEntry(
            value=value,
            access_time=current_time,
            creation_time=current_time,
            size_bytes=size_bytes
        )
        
        self._cache[key] = entry
        self._cache.move_to_end(key)
        
        if size_bytes:
            self._total_size_bytes += size_bytes
        
        # Evict if necessary
        self._evict_if_needed()
        
        logger.debug(f"Cache put: {key}, size: {len(self._cache)}/{self.max_size}")
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        if key not in self._cache:
            return False
        
        entry = self._cache.pop(key)
        if entry.size_bytes:
            self._total_size_bytes -= entry.size_bytes
        
        return True
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
        self._total_size_bytes = 0
        logger.debug("Cache cleared")
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds limits."""
        # Evict by count
        while len(self._cache) > self.max_size:
            self._evict_oldest()
        
        # Evict by memory size
        if self.max_memory_bytes:
            while self._total_size_bytes > self.max_memory_bytes and self._cache:
                self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        
        key, entry = self._cache.popitem(last=False)  # Remove from beginning (oldest)
        if entry.size_bytes:
            self._total_size_bytes -= entry.size_bytes
        
        logger.debug(f"Cache evicted: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'memory_usage_bytes': self._total_size_bytes,
            'max_memory_bytes': self.max_memory_bytes,
            'hit_ratio': 0.0,  # Would need to track hits/misses for this
            'keys': list(self._cache.keys())
        }
    
    def __len__(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (without updating access time)."""
        return key in self._cache