"""
Base classes and interfaces for unified data models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

from .errors import ProcessingError, ProcessingWarning, ErrorPolicy, ErrorAction


@dataclass
class BaseSegment(ABC):
    """Base interface for all segment types with unified timing and confidence."""
    
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start
    
    @abstractmethod
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for export."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Legacy compatibility method."""
        return self.to_export_dict()
    
    def overlaps_with(self, other: 'BaseSegment', threshold: float = 0.0) -> bool:
        """Check if this segment overlaps with another segment."""
        return (
            self.start <= other.end - threshold and 
            self.end >= other.start + threshold
        )
    
    def merge_with(self, other: 'BaseSegment') -> 'BaseSegment':
        """Merge this segment with another segment of the same type."""
        if type(self) != type(other):
            raise ValueError("Cannot merge segments of different types")
        
        new_start = min(self.start, other.start)
        new_end = max(self.end, other.end)
        new_confidence = (
            self.confidence * self.duration + 
            other.confidence * other.duration
        ) / (self.duration + other.duration)
        
        # Create new instance of the same type with merged properties
        merged = type(self)(
            start=new_start,
            end=new_end,
            confidence=new_confidence
        )
        
        # Copy over type-specific properties
        self._merge_specific_properties(merged, other)
        
        return merged
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Override in subclasses to merge type-specific properties."""
        pass


@dataclass
class ProcessingContext:
    """Context object passed through processing pipeline with error handling."""
    
    audio_path: Path
    config: Any  # Will be typed more specifically later
    error_policy: ErrorPolicy = ErrorPolicy.GRACEFUL_DEGRADATION
    
    # Runtime state
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[ProcessingWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[logging.Logger] = None
    
    def handle_error(self, 
                    error: Exception, 
                    component: str, 
                    context: str = "") -> ErrorAction:
        """Centralized error handling logic."""
        processing_error = ProcessingError(
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            context=context,
            recoverable=self._is_recoverable_error(error)
        )
        
        self.errors.append(processing_error)
        
        if self.logger:
            self.logger.error(f"Error in {component}: {error}")
        
        # Determine action based on policy
        if self.error_policy == ErrorPolicy.FAIL_FAST:
            return ErrorAction.RAISE
        elif self.error_policy == ErrorPolicy.GRACEFUL_DEGRADATION and not processing_error.recoverable:
            return ErrorAction.RAISE
        else:
            return ErrorAction.CONTINUE
    
    def add_warning(self, 
                   component: str, 
                   message: str, 
                   context: str = "") -> None:
        """Add a warning to the context."""
        warning = ProcessingWarning(
            component=component,
            message=message,
            context=context
        )
        self.warnings.append(warning)
        
        if self.logger:
            self.logger.warning(f"Warning in {component}: {message}")
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        recoverable_types = (
            FileNotFoundError,
            ConnectionError,
            TimeoutError,
            ImportError
        )
        return isinstance(error, recoverable_types)


@dataclass
class ProcessingResult:
    """Result container with segments, errors, and metadata."""
    
    segments: List[BaseSegment]
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[ProcessingWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if processing was successful (no critical errors)."""
        return not any(not error.recoverable for error in self.errors)
    
    @property
    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return len(self.warnings) > 0
    
    def get_segments_by_type(self, segment_type: type) -> List[BaseSegment]:
        """Get all segments of a specific type."""
        return [seg for seg in self.segments if isinstance(seg, segment_type)]