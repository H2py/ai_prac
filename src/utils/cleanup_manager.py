"""
Explicit memory cleanup and resource management utilities.
"""

import gc
import logging
import threading
import weakref
from typing import List, Set, Optional, Any, Callable
from contextlib import contextmanager
from pathlib import Path
import psutil
import torch

from src.utils.memory_manager import memory_manager
from src.utils.lazy_loader import model_registry

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Tracks and manages system resources for cleanup."""
    
    def __init__(self):
        self._tracked_objects: Set[weakref.ref] = set()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._temp_files: Set[Path] = set()
        self._lock = threading.Lock()
    
    def register_object(self, obj: Any, cleanup_callback: Optional[Callable] = None) -> None:
        """Register an object for cleanup tracking.
        
        Args:
            obj: Object to track
            cleanup_callback: Optional cleanup function to call
        """
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
        """Register a cleanup callback.
        
        Args:
            callback: Cleanup function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def register_temp_file(self, file_path: Path) -> None:
        """Register a temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file
        """
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
        """Get resource tracking statistics.
        
        Returns:
            Dictionary with tracking stats
        """
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
        """Set memory cleanup thresholds.
        
        Args:
            **thresholds: Threshold values (memory_percent, gpu_memory_percent, process_memory_mb)
        """
        self._cleanup_thresholds.update(thresholds)
        logger.info(f"Updated cleanup thresholds: {self._cleanup_thresholds}")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.
        
        Returns:
            True if cleanup should be triggered
        """
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
        """Clean up all loaded models.
        
        Args:
            force: Force cleanup even if not under memory pressure
        """
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
    
    def cleanup_memory_pools(self) -> None:
        """Clean up memory pools and caches."""
        logger.info("Cleaning up memory pools...")
        memory_manager.clear_memory()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory pool cleanup completed")
    
    def cleanup_system_resources(self) -> None:
        """Clean up system-level resources."""
        logger.info("Cleaning up system resources...")
        
        # Clean up tracked resources
        self.resource_tracker.cleanup_all()
        
        # Clean up memory pools
        self.cleanup_memory_pools()
        
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
    
    def emergency_cleanup(self) -> None:
        """Perform emergency cleanup when memory is critically low."""
        logger.warning("Performing emergency memory cleanup...")
        
        # Unload all models immediately
        self.cleanup_models(force=True)
        
        # Clean up all resources
        self.cleanup_system_resources()
        
        # Log memory stats after cleanup
        memory_manager.log_memory_usage("after emergency cleanup")
        
        logger.warning("Emergency cleanup completed")
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup based on memory pressure."""
        if self.check_memory_pressure():
            logger.info("Memory pressure detected, performing cleanup...")
            
            # Clean up memory pools first (less disruptive)
            self.cleanup_memory_pools()
            
            # If still under pressure, unload models
            if self.check_memory_pressure():
                self.cleanup_models(force=True)
            
            # If still critical, do emergency cleanup
            if self.check_memory_pressure():
                self.emergency_cleanup()
    
    @contextmanager
    def cleanup_context(self, cleanup_on_exit: bool = True):
        """Context manager that ensures cleanup on exit.
        
        Args:
            cleanup_on_exit: Whether to perform cleanup on context exit
        """
        try:
            yield self
        finally:
            if cleanup_on_exit:
                self.cleanup_system_resources()
    
    def get_memory_report(self) -> dict:
        """Get detailed memory usage report.
        
        Returns:
            Dictionary with memory usage details
        """
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


# Global cleanup manager instance
cleanup_manager = CleanupManager()


def register_for_cleanup(obj: Any, cleanup_callback: Optional[Callable] = None) -> None:
    """Register an object for cleanup tracking.
    
    Args:
        obj: Object to track
        cleanup_callback: Optional cleanup function
    """
    cleanup_manager.resource_tracker.register_object(obj, cleanup_callback)


def register_temp_file(file_path: Path) -> None:
    """Register a temporary file for cleanup.
    
    Args:
        file_path: Path to temporary file
    """
    cleanup_manager.resource_tracker.register_temp_file(file_path)


def cleanup_on_memory_pressure() -> None:
    """Perform cleanup if system is under memory pressure."""
    cleanup_manager.periodic_cleanup()


@contextmanager
def memory_cleanup_context():
    """Context manager for automatic memory cleanup."""
    with cleanup_manager.cleanup_context():
        yield