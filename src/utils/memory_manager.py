"""
Memory optimization utilities for the audio analysis pipeline.
"""

import gc
import psutil
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class MemoryStats:
    """Memory usage statistics."""
    
    total_memory_gb: float
    available_memory_gb: float
    process_memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


class MemoryManager:
    """Manages memory usage and optimization for the pipeline."""
    
    def __init__(self):
        """Initialize memory manager."""
        self._memory_pool: Dict[str, List[np.ndarray]] = {}
        self._tensor_pool: Dict[str, List[torch.Tensor]] = {}
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics.
        
        Returns:
            MemoryStats object with current usage
        """
        # System memory
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        stats = MemoryStats(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3), 
            process_memory_mb=process_memory.rss / (1024**2),
            memory_percent=memory.percent
        )
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_used = torch.cuda.memory_allocated(0)
                
                stats.gpu_memory_mb = gpu_memory_used / (1024**2)
                stats.gpu_memory_percent = (gpu_memory_used / gpu_memory) * 100
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
        
        return stats
    
    def get_optimal_batch_size(self, 
                             sample_size: int,
                             max_memory_mb: Optional[float] = None,
                             safety_factor: float = 0.7) -> int:
        """Calculate optimal batch size based on available memory.
        
        Args:
            sample_size: Memory size of a single sample in bytes
            max_memory_mb: Maximum memory to use (MB). If None, uses available memory
            safety_factor: Safety factor to prevent OOM (0.7 = 70% of available)
            
        Returns:
            Optimal batch size
        """
        stats = self.get_memory_stats()
        
        if max_memory_mb is None:
            # Use available system memory
            available_mb = stats.available_memory_gb * 1024 * safety_factor
        else:
            available_mb = max_memory_mb * safety_factor
        
        # Calculate batch size
        sample_mb = sample_size / (1024**2)
        batch_size = max(1, int(available_mb / sample_mb))
        
        logger.debug(f"Calculated optimal batch size: {batch_size} "
                    f"(available: {available_mb:.1f}MB, sample: {sample_mb:.2f}MB)")
        
        return batch_size
    
    def get_optimal_chunk_size(self, 
                              audio_duration: float,
                              target_memory_mb: float = 500) -> float:
        """Calculate optimal audio chunk size to stay within memory limits.
        
        Args:
            audio_duration: Total audio duration in seconds
            target_memory_mb: Target memory usage per chunk in MB
            
        Returns:
            Optimal chunk duration in seconds
        """
        # Estimate memory per second (rough approximation)
        # Assumes 16kHz, 32-bit float, mono = ~64KB per second base
        # Add overhead for processing: ~4x multiplier
        memory_per_second_kb = 64 * 4
        memory_per_second_mb = memory_per_second_kb / 1024
        
        chunk_duration = min(
            audio_duration,
            target_memory_mb / memory_per_second_mb
        )
        
        # Ensure minimum chunk size for quality
        chunk_duration = max(chunk_duration, 0.5)
        
        logger.debug(f"Calculated optimal chunk size: {chunk_duration:.2f}s "
                    f"for {audio_duration:.1f}s audio")
        
        return chunk_duration
    
    def clear_memory(self, force_gc: bool = True) -> None:
        """Clear memory pools and force garbage collection.
        
        Args:
            force_gc: Whether to force garbage collection
        """
        # Clear memory pools
        self._memory_pool.clear()
        self._tensor_pool.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if force_gc:
            gc.collect()
        
        logger.debug("Memory cleared and garbage collection performed")
    
    def get_memory_pool(self, key: str, 
                       shape: tuple, 
                       dtype: np.dtype = np.float32) -> np.ndarray:
        """Get numpy array from memory pool or create new one.
        
        Args:
            key: Pool key for array type
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Numpy array from pool or newly created
        """
        pool_key = f"{key}_{shape}_{dtype}"
        
        if pool_key in self._memory_pool and self._memory_pool[pool_key]:
            array = self._memory_pool[pool_key].pop()
            array.fill(0)  # Clear previous data
            return array
        else:
            return np.zeros(shape, dtype=dtype)
    
    def return_to_pool(self, key: str, array: np.ndarray) -> None:
        """Return numpy array to memory pool for reuse.
        
        Args:
            key: Pool key for array type  
            array: Array to return to pool
        """
        pool_key = f"{key}_{array.shape}_{array.dtype}"
        
        if pool_key not in self._memory_pool:
            self._memory_pool[pool_key] = []
        
        # Limit pool size to prevent memory leak
        if len(self._memory_pool[pool_key]) < 10:
            self._memory_pool[pool_key].append(array)
    
    def get_tensor_pool(self, key: str, 
                       shape: tuple,
                       dtype: torch.dtype = torch.float32,
                       device: str = "cpu") -> torch.Tensor:
        """Get tensor from pool or create new one.
        
        Args:
            key: Pool key for tensor type
            shape: Tensor shape
            dtype: Tensor data type
            device: Device for tensor
            
        Returns:
            Tensor from pool or newly created
        """
        pool_key = f"{key}_{shape}_{dtype}_{device}"
        
        if pool_key in self._tensor_pool and self._tensor_pool[pool_key]:
            tensor = self._tensor_pool[pool_key].pop()
            tensor.zero_()  # Clear previous data
            return tensor
        else:
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor_to_pool(self, key: str, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse.
        
        Args:
            key: Pool key for tensor type
            tensor: Tensor to return to pool
        """
        pool_key = f"{key}_{tensor.shape}_{tensor.dtype}_{tensor.device}"
        
        if pool_key not in self._tensor_pool:
            self._tensor_pool[pool_key] = []
        
        # Limit pool size
        if len(self._tensor_pool[pool_key]) < 5:  # Smaller limit for GPU tensors
            self._tensor_pool[pool_key].append(tensor)
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage.
        
        Args:
            context: Context description for the log
        """
        stats = self.get_memory_stats()
        
        log_msg = (
            f"Memory usage{' (' + context + ')' if context else ''}: "
            f"Process: {stats.process_memory_mb:.1f}MB, "
            f"System: {stats.memory_percent:.1f}% "
            f"({stats.available_memory_gb:.1f}GB available)"
        )
        
        if stats.gpu_memory_mb is not None:
            log_msg += f", GPU: {stats.gpu_memory_mb:.1f}MB ({stats.gpu_memory_percent:.1f}%)"
        
        logger.info(log_msg)
    
    def check_memory_pressure(self, threshold_percent: float = 85.0) -> bool:
        """Check if system is under memory pressure.
        
        Args:
            threshold_percent: Memory usage threshold to consider as pressure
            
        Returns:
            True if under memory pressure
        """
        stats = self.get_memory_stats()
        return stats.memory_percent > threshold_percent


# Global memory manager instance
memory_manager = MemoryManager()