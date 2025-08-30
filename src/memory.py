"""
Unified Memory Management System

This module provides comprehensive memory optimization, lazy loading, and resource cleanup
functionality for the audio analysis pipeline.

Key Components:
- MemoryManager: Memory pooling, batch sizing, and monitoring
- LazyModelLoader: Deferred model loading with thread safety
- CleanupManager: Resource tracking and explicit cleanup
- Memory utilities and optimization helpers
"""

import gc
import logging
import threading
import weakref
import psutil
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Generic
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ================================
# MEMORY STATISTICS
# ================================

@dataclass 
class MemoryStats:
    """Memory usage statistics."""
    
    total_memory_gb: float
    available_memory_gb: float
    process_memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


# ================================
# MEMORY MANAGER
# ================================

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


# ================================
# LAZY MODEL LOADER
# ================================

class LazyModelLoader(Generic[T]):
    """Generic lazy loader for AI models with thread safety."""
    
    def __init__(self, loader_func: Callable[[], T], model_name: str = "model"):
        """Initialize lazy model loader.
        
        Args:
            loader_func: Function that loads and returns the model
            model_name: Name of the model for logging
        """
        self._loader_func = loader_func
        self._model_name = model_name
        self._model: Optional[T] = None
        self._lock = threading.Lock()
        self._loading = False
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def load(self) -> T:
        """Load the model if not already loaded.
        
        Returns:
            The loaded model instance
        """
        if self._model is not None:
            return self._model
        
        with self._lock:
            # Double-check locking pattern
            if self._model is not None:
                return self._model
            
            if self._loading:
                # Another thread is currently loading, wait for it
                logger.debug(f"Waiting for {self._model_name} to finish loading...")
                while self._loading and self._model is None:
                    threading.Event().wait(0.1)
                
                if self._model is not None:
                    return self._model
            
            try:
                self._loading = True
                logger.info(f"Lazy loading {self._model_name}...")
                self._model = self._loader_func()
                logger.info(f"Successfully loaded {self._model_name}")
                return self._model
            
            except Exception as e:
                logger.error(f"Failed to load {self._model_name}: {e}")
                raise
            
            finally:
                self._loading = False
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self._model is not None:
                logger.info(f"Unloading {self._model_name}")
                # Try to explicitly delete model-specific resources
                if hasattr(self._model, 'cpu'):
                    try:
                        self._model.cpu()  # Move to CPU to free GPU memory
                    except:
                        pass
                
                self._model = None
                
                # Force garbage collection for models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def __call__(self) -> T:
        """Make the loader callable."""
        return self.load()


class LazyModelRegistry:
    """Registry for managing multiple lazy-loaded models."""
    
    def __init__(self):
        self._models: Dict[str, LazyModelLoader] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, loader_func: Callable[[], Any]) -> LazyModelLoader:
        """Register a new lazy model.
        
        Args:
            name: Model name/identifier
            loader_func: Function to load the model
            
        Returns:
            LazyModelLoader instance
        """
        with self._lock:
            if name in self._models:
                logger.warning(f"Model {name} already registered, overwriting")
            
            lazy_loader = LazyModelLoader(loader_func, name)
            self._models[name] = lazy_loader
            logger.debug(f"Registered lazy model: {name}")
            return lazy_loader
    
    def get(self, name: str) -> Optional[LazyModelLoader]:
        """Get a registered lazy model."""
        return self._models.get(name)
    
    def load(self, name: str) -> Any:
        """Load a specific model by name."""
        if name not in self._models:
            raise KeyError(f"Model {name} not registered")
        return self._models[name].load()
    
    def unload(self, name: str) -> None:
        """Unload a specific model by name."""
        if name in self._models:
            self._models[name].unload()
    
    def unload_all(self) -> None:
        """Unload all registered models."""
        logger.info("Unloading all lazy models")
        with self._lock:
            for name, loader in self._models.items():
                try:
                    loader.unload()
                except Exception as e:
                    logger.warning(f"Failed to unload {name}: {e}")
    
    def get_memory_status(self) -> Dict[str, bool]:
        """Get loading status of all models."""
        return {name: loader.is_loaded for name, loader in self._models.items()}
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return [name for name, loader in self._models.items() if loader.is_loaded]


# Global registry instance
model_registry = LazyModelRegistry()


# ================================
# RESOURCE TRACKER AND CLEANUP
# ================================

class ResourceTracker:
    """Tracks and manages system resources for cleanup."""
    
    def __init__(self):
        self._tracked_objects: Set[weakref.ref] = set()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._temp_files: Set[Path] = set()
        self._lock = threading.Lock()
    
    def register_object(self, obj: Any, cleanup_callback: Optional[Callable] = None) -> None:
        """Register an object for cleanup tracking."""
        with self._lock:
            def cleanup_weakref(ref):
                self._tracked_objects.discard(ref)
                if cleanup_callback:
                    try:
                        cleanup_callback()
                    except Exception as e:
                        logger.warning(f"Cleanup callback failed: {e}")
            
            weak_ref = weakref.ref(obj, cleanup_weakref)
            self._tracked_objects.add(weak_ref)
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def register_temp_file(self, file_path: Path) -> None:
        """Register a temporary file for cleanup."""
        with self._lock:
            self._temp_files.add(file_path)
    
    def cleanup_temp_files(self) -> None:
        """Clean up all registered temporary files."""
        with self._lock:
            for file_path in self._temp_files.copy():
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Cleaned up temp file: {file_path}")
                    self._temp_files.discard(file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    def cleanup_all(self) -> None:
        """Execute all registered cleanup callbacks."""
        with self._lock:
            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Clear callbacks after execution
            self._cleanup_callbacks.clear()
            
            # Clean up temp files
            self.cleanup_temp_files()
    
    def get_stats(self) -> dict:
        """Get resource tracking statistics."""
        with self._lock:
            return {
                'tracked_objects': len(self._tracked_objects),
                'cleanup_callbacks': len(self._cleanup_callbacks),
                'temp_files': len(self._temp_files)
            }


class CleanupManager:
    """Manages explicit memory cleanup and resource management."""
    
    def __init__(self):
        self.resource_tracker = ResourceTracker()
        self._cleanup_thresholds = {
            'memory_percent': 85.0,  # Clean up when memory usage > 85%
            'gpu_memory_percent': 90.0,  # Clean up when GPU memory > 90%
            'process_memory_mb': 2000.0  # Clean up when process memory > 2GB
        }
    
    def set_cleanup_thresholds(self, **thresholds) -> None:
        """Set memory cleanup thresholds."""
        self._cleanup_thresholds.update(thresholds)
        logger.info(f"Updated cleanup thresholds: {self._cleanup_thresholds}")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = memory_manager.get_memory_stats()
        
        # Check system memory
        if stats.memory_percent > self._cleanup_thresholds['memory_percent']:
            logger.warning(f"High system memory usage: {stats.memory_percent:.1f}%")
            return True
        
        # Check process memory
        if stats.process_memory_mb > self._cleanup_thresholds['process_memory_mb']:
            logger.warning(f"High process memory usage: {stats.process_memory_mb:.1f}MB")
            return True
        
        # Check GPU memory if available
        if (stats.gpu_memory_percent is not None and 
            stats.gpu_memory_percent > self._cleanup_thresholds['gpu_memory_percent']):
            logger.warning(f"High GPU memory usage: {stats.gpu_memory_percent:.1f}%")
            return True
        
        return False
    
    def cleanup_models(self, force: bool = False) -> None:
        """Clean up all loaded models."""
        if not force and not self.check_memory_pressure():
            return
        
        logger.info("Cleaning up loaded models...")
        
        # Unload all models from registry
        model_registry.unload_all()
        
        # Clear model-related memory pools
        memory_manager.clear_memory()
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Model cleanup completed")
    
    def cleanup_system_resources(self) -> None:
        """Clean up system-level resources."""
        logger.info("Cleaning up system resources...")
        
        # Clean up tracked resources
        self.resource_tracker.cleanup_all()
        
        # Clean up memory pools
        memory_manager.clear_memory()
        
        # Force garbage collection multiple times for thoroughness
        for _ in range(3):
            gc.collect()
        
        # Clear GPU resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset GPU memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
        
        logger.info("System resource cleanup completed")
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup based on memory pressure."""
        if self.check_memory_pressure():
            logger.info("Memory pressure detected, performing cleanup...")
            
            # Clean up memory pools first (less disruptive)
            memory_manager.clear_memory()
            
            # If still under pressure, unload models
            if self.check_memory_pressure():
                self.cleanup_models(force=True)
            
            # If still critical, do emergency cleanup
            if self.check_memory_pressure():
                self.cleanup_system_resources()
    
    def get_memory_report(self) -> dict:
        """Get detailed memory usage report."""
        stats = memory_manager.get_memory_stats()
        resource_stats = self.resource_tracker.get_stats()
        model_status = model_registry.get_memory_status()
        loaded_models = model_registry.list_loaded_models()
        
        return {
            'system_memory': {
                'total_gb': stats.total_memory_gb,
                'available_gb': stats.available_memory_gb,
                'usage_percent': stats.memory_percent,
                'process_memory_mb': stats.process_memory_mb
            },
            'gpu_memory': {
                'usage_mb': stats.gpu_memory_mb,
                'usage_percent': stats.gpu_memory_percent
            } if stats.gpu_memory_mb is not None else None,
            'models': {
                'loaded_count': len(loaded_models),
                'loaded_models': loaded_models,
                'model_status': model_status
            },
            'resources': resource_stats,
            'cleanup_thresholds': self._cleanup_thresholds,
            'memory_pressure': self.check_memory_pressure()
        }
    
    @contextmanager
    def cleanup_context(self, cleanup_on_exit: bool = True):
        """Context manager that ensures cleanup on exit."""
        try:
            yield self
        finally:
            if cleanup_on_exit:
                self.cleanup_system_resources()


# Global cleanup manager instance
cleanup_manager = CleanupManager()


# ================================
# UTILITY FUNCTIONS AND DECORATORS
# ================================

def register_for_cleanup(obj: Any, cleanup_callback: Optional[Callable] = None) -> None:
    """Register an object for cleanup tracking."""
    cleanup_manager.resource_tracker.register_object(obj, cleanup_callback)


def register_temp_file(file_path: Path) -> None:
    """Register a temporary file for cleanup."""
    cleanup_manager.resource_tracker.register_temp_file(file_path)


def cleanup_on_memory_pressure() -> None:
    """Perform cleanup if system is under memory pressure."""
    cleanup_manager.periodic_cleanup()


@contextmanager
def memory_cleanup_context():
    """Context manager for automatic memory cleanup."""
    with cleanup_manager.cleanup_context():
        yield


def require_model(model_attr: str):
    """Decorator to ensure a model is loaded before method execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            model_loader = getattr(self, model_attr)
            if isinstance(model_loader, LazyModelLoader):
                model_loader.load()  # Ensure model is loaded
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def lazy_property(loader_func: Callable[[], T]) -> property:
    """Decorator to create a lazy-loaded property."""
    attr_name = f'_lazy_{loader_func.__name__}'
    
    def getter(self):
        if not hasattr(self, attr_name):
            logger.debug(f"Lazy loading property: {loader_func.__name__}")
            setattr(self, attr_name, loader_func(self))
        return getattr(self, attr_name)
    
    def deleter(self):
        if hasattr(self, attr_name):
            logger.debug(f"Clearing lazy property: {loader_func.__name__}")
            delattr(self, attr_name)
    
    return property(getter, fdel=deleter)