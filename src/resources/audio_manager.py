"""
Centralized audio resource manager with caching.
"""

import hashlib
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .cache import LRUCache
from src.utils.audio_utils import load_audio

logger = logging.getLogger(__name__)


class AudioResourceManager:
    """Centralized audio loading with LRU caching and optimization."""
    
    def __init__(self, 
                 cache_size: int = 5,
                 max_memory_mb: int = 500,
                 ttl_seconds: int = 3600):
        """Initialize audio resource manager.
        
        Args:
            cache_size: Maximum number of audio files to cache
            max_memory_mb: Maximum memory usage for audio cache
            ttl_seconds: Time-to-live for cached audio (1 hour default)
        """
        self.cache = LRUCache[Tuple[np.ndarray, int]](
            max_size=cache_size,
            max_memory_mb=max_memory_mb, 
            ttl_seconds=ttl_seconds
        )
        
        # Track statistics
        self._stats = {
            'loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_bytes_loaded': 0
        }
        
        logger.info(f"AudioResourceManager initialized: cache_size={cache_size}, "
                   f"max_memory_mb={max_memory_mb}MB")
    
    def load_audio(self, 
                   audio_path: Path, 
                   sample_rate: int = 16000,
                   force_reload: bool = False) -> Tuple[np.ndarray, int]:
        """Load audio with caching support.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (audio_data, actual_sample_rate)
        """
        audio_path = Path(audio_path)
        cache_key = self._generate_cache_key(audio_path, sample_rate)
        
        self._stats['loads'] += 1
        
        # Try to get from cache first
        if not force_reload:
            cached_audio = self.cache.get(cache_key)
            if cached_audio is not None:
                self._stats['cache_hits'] += 1
                logger.debug(f"Audio cache hit: {audio_path.name}")
                return cached_audio
        
        # Load from disk
        self._stats['cache_misses'] += 1
        logger.debug(f"Loading audio from disk: {audio_path}")
        
        try:
            audio_data, actual_sample_rate = load_audio(audio_path, sample_rate=sample_rate)
            
            # Calculate size for cache management
            size_bytes = audio_data.nbytes if hasattr(audio_data, 'nbytes') else 0
            self._stats['total_bytes_loaded'] += size_bytes
            
            # Cache the result
            self.cache.put(cache_key, (audio_data, actual_sample_rate), size_bytes)
            
            logger.debug(f"Audio loaded and cached: {audio_path.name}, "
                        f"duration={len(audio_data)/actual_sample_rate:.2f}s, "
                        f"size={size_bytes/1024/1024:.1f}MB")
            
            return audio_data, actual_sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    def preload_audio(self, audio_paths: list[Path], sample_rate: int = 16000) -> None:
        """Preload multiple audio files into cache.
        
        Args:
            audio_paths: List of audio file paths to preload
            sample_rate: Target sample rate
        """
        logger.info(f"Preloading {len(audio_paths)} audio files...")
        
        for audio_path in audio_paths:
            try:
                self.load_audio(audio_path, sample_rate)
            except Exception as e:
                logger.warning(f"Failed to preload {audio_path}: {e}")
    
    def get_cached_paths(self) -> list[str]:
        """Get list of currently cached audio file paths."""
        stats = self.cache.get_stats()
        return [key.split('|')[0] for key in stats['keys']]  # Extract path from cache key
    
    def clear_cache(self) -> None:
        """Clear audio cache."""
        self.cache.clear()
        logger.info("Audio cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio manager statistics."""
        cache_stats = self.cache.get_stats()
        
        hit_ratio = 0.0
        if self._stats['loads'] > 0:
            hit_ratio = self._stats['cache_hits'] / self._stats['loads']
        
        return {
            'cache': cache_stats,
            'loads': self._stats['loads'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_ratio': hit_ratio,
            'total_bytes_loaded': self._stats['total_bytes_loaded'],
            'avg_bytes_per_load': (
                self._stats['total_bytes_loaded'] / self._stats['loads'] 
                if self._stats['loads'] > 0 else 0
            )
        }
    
    def _generate_cache_key(self, audio_path: Path, sample_rate: int) -> str:
        """Generate cache key for audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Sample rate
            
        Returns:
            Cache key string
        """
        try:
            # Use file path, size, and modification time for cache key
            stat = audio_path.stat()
            key_data = f"{audio_path}|{sample_rate}|{stat.st_size}|{stat.st_mtime}"
            
            # Hash for consistent key length
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            
            return f"{audio_path.name}_{sample_rate}_{key_hash[:8]}"
            
        except Exception as e:
            logger.warning(f"Could not generate cache key for {audio_path}: {e}")
            # Fallback to simple key
            return f"{audio_path}|{sample_rate}"
    
    def estimate_memory_usage(self, duration_seconds: float, sample_rate: int = 16000) -> int:
        """Estimate memory usage for audio file.
        
        Args:
            duration_seconds: Audio duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Estimated memory usage in bytes
        """
        # Assume 32-bit float samples
        bytes_per_sample = 4
        total_samples = int(duration_seconds * sample_rate)
        return total_samples * bytes_per_sample
    
    def can_cache_audio(self, duration_seconds: float, sample_rate: int = 16000) -> bool:
        """Check if audio file can fit in cache.
        
        Args:
            duration_seconds: Audio duration in seconds
            sample_rate: Sample rate
            
        Returns:
            True if audio can fit in cache
        """
        estimated_size = self.estimate_memory_usage(duration_seconds, sample_rate)
        cache_stats = self.cache.get_stats()
        
        available_memory = (
            cache_stats.get('max_memory_bytes', float('inf')) - 
            cache_stats.get('memory_usage_bytes', 0)
        )
        
        return estimated_size <= available_memory
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.clear_cache()
        except Exception:
            pass  # Ignore errors during cleanup