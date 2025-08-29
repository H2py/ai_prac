"""
Base processor class with dependency injection support.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from src.models import BaseSegment, ProcessingContext, ProcessingResult
from src.resources import AudioResourceManager, ModelResourceManager
from src.utils.logger import PerformanceLogger


class BaseProcessor(ABC):
    """Abstract base processor with dependency injection and error handling."""
    
    def __init__(self,
                 audio_manager: AudioResourceManager,
                 model_manager: ModelResourceManager,
                 logger: Optional[logging.Logger] = None,
                 processor_name: Optional[str] = None):
        """Initialize base processor.
        
        Args:
            audio_manager: Shared audio resource manager
            model_manager: Shared model resource manager
            logger: Logger instance
            processor_name: Name for logging (auto-generated if None)
        """
        self.audio_manager = audio_manager
        self.model_manager = model_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.processor_name = processor_name or self.__class__.__name__
        
        # Performance logging
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Statistics tracking
        self._stats = {
            'processed_files': 0,
            'total_processing_time': 0.0,
            'successful_processes': 0,
            'failed_processes': 0,
            'average_processing_time': 0.0
        }
        
        self.logger.debug(f"{self.processor_name} initialized with dependency injection")
    
    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process audio and return results.
        
        Args:
            context: Processing context with audio path and configuration
            
        Returns:
            Processing result with segments and metadata
        """
        pass
    
    def _start_processing(self, context: ProcessingContext) -> None:
        """Start processing timer and log."""
        self.perf_logger.start_timer(f"{self.processor_name.lower()}_processing")
        self.logger.info(f"Starting {self.processor_name} processing: {context.audio_path}")
        self._stats['processed_files'] += 1
    
    def _end_processing(self, context: ProcessingContext, success: bool = True) -> float:
        """End processing timer and update stats."""
        duration = self.perf_logger.stop_timer(f"{self.processor_name.lower()}_processing")
        
        self._stats['total_processing_time'] += duration
        if success:
            self._stats['successful_processes'] += 1
            self.logger.info(f"{self.processor_name} completed in {duration:.2f}s")
        else:
            self._stats['failed_processes'] += 1
            self.logger.error(f"{self.processor_name} failed after {duration:.2f}s")
        
        # Update average
        if self._stats['processed_files'] > 0:
            self._stats['average_processing_time'] = (
                self._stats['total_processing_time'] / self._stats['processed_files']
            )
        
        return duration
    
    def _handle_processing_error(self, 
                                context: ProcessingContext, 
                                error: Exception, 
                                operation: str = "processing") -> ProcessingResult:
        """Handle processing errors according to context policy.
        
        Args:
            context: Processing context
            error: The error that occurred
            operation: Description of the operation that failed
            
        Returns:
            ProcessingResult with error information
        """
        error_action = context.handle_error(error, self.processor_name, operation)
        
        if error_action.name == "RAISE":
            self._end_processing(context, success=False)
            raise error
        
        # Return empty result for graceful degradation
        return ProcessingResult(
            segments=[],
            errors=context.errors.copy(),
            warnings=context.warnings.copy(),
            metadata={'processor': self.processor_name, 'failed_operation': operation}
        )
    
    def _add_warning(self, context: ProcessingContext, message: str, operation: str = "") -> None:
        """Add warning to processing context."""
        context.add_warning(self.processor_name, message, operation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'processor_name': self.processor_name,
            **self._stats,
            'success_rate': (
                self._stats['successful_processes'] / self._stats['processed_files']
                if self._stats['processed_files'] > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset processor statistics."""
        self._stats = {
            'processed_files': 0,
            'total_processing_time': 0.0,
            'successful_processes': 0,
            'failed_processes': 0,
            'average_processing_time': 0.0
        }
        self.logger.debug(f"{self.processor_name} statistics reset")
    
    def __str__(self) -> str:
        """String representation of processor."""
        return f"{self.processor_name}(processed={self._stats['processed_files']}, " \
               f"success_rate={self._stats['successful_processes'] / max(1, self._stats['processed_files']):.2%})"