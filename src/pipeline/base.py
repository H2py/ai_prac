"""
Base classes and interfaces for the audio analysis pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime

from config.settings import Config


class PipelineContext:
    """Shared context for passing data between pipeline steps."""
    
    def __init__(self, input_source: str, config: Config):
        self.input_source = input_source
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage for each analysis step
        self.audio_file: Optional[Path] = None
        self.audio_info: Optional[Dict[str, Any]] = None
        self.video_metadata: Optional[Dict[str, Any]] = None
        self.speaker_results: Optional[Dict[str, Any]] = None
        self.emotion_results: Optional[List] = None
        self.acoustic_results: Optional[List] = None
        self.transcription_results: Optional[List] = None
        
        # Pipeline state
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.completed_steps: List[str] = []
        self.step_durations: Dict[str, float] = {}
        
    def get_output_path(self, file_type: str, extension: str = None) -> Path:
        """Generate output path for a specific file type."""
        if extension:
            filename = f"{file_type}_{self.timestamp}.{extension}"
        else:
            filename = f"{file_type}_{self.timestamp}"
        
        return self.config.output.output_dir / filename
    
    def has_audio_file(self) -> bool:
        """Check if audio file has been extracted."""
        return self.audio_file is not None and self.audio_file.exists()
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all analysis results for merging."""
        return {
            'speaker_results': self.speaker_results,
            'emotion_results': self.emotion_results,
            'acoustic_results': self.acoustic_results,
            'transcription_results': self.transcription_results,
            'metadata': self._get_metadata()
        }
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Generate metadata dictionary."""
        metadata = {
            "input_source": self.input_source,
            "processing_timestamp": datetime.now().isoformat(),
            "configuration": {
                "sample_rate": self.config.audio.sample_rate,
                "gpu_enabled": self.config.model.use_gpu,
                "segment_duration": self.config.audio.chunk_duration,
                "emotion_threshold": self.config.model.emotion_threshold
            }
        }
        
        if self.audio_file:
            metadata["audio_file"] = str(self.audio_file)
            
        if self.audio_info:
            metadata.update({
                "audio_duration": self.audio_info['duration_seconds'],
                "sample_rate": self.audio_info['sample_rate'],
                "file_size_mb": self.audio_info['file_size_mb']
            })
        
        return metadata
    
    def add_error(self, step_name: str, error: Exception, critical: bool = False):
        """Add an error to the context."""
        self.errors.append({
            'step': step_name,
            'error': str(error),
            'type': type(error).__name__,
            'critical': critical,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, message: str):
        """Add a warning to the context."""
        self.warnings.append(message)
    
    def mark_step_completed(self, step_name: str, duration: float):
        """Mark a step as completed."""
        self.completed_steps.append(step_name)
        self.step_durations[step_name] = duration
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pipeline."""
        total_time = sum(self.step_durations.values())
        return {
            'total_processing_time': total_time,
            'step_durations': self.step_durations.copy(),
            'completed_steps': len(self.completed_steps),
            'total_errors': len(self.errors),
            'critical_errors': len([e for e in self.errors if e.get('critical', False)]),
            'warnings': len(self.warnings)
        }


class PipelineStep(ABC):
    """Abstract base class for all pipeline steps."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.name = self.__class__.__name__.replace('Step', '').lower()
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> bool:
        """
        Execute the pipeline step.
        
        Args:
            context: Pipeline context containing shared data
            
        Returns:
            bool: True if step completed successfully, False otherwise
        """
        pass
    
    def should_execute(self, context: PipelineContext) -> bool:
        """
        Determine if this step should be executed.
        
        Args:
            context: Pipeline context
            
        Returns:
            bool: True if step should execute, False to skip
        """
        return True
    
    def validate_prerequisites(self, context: PipelineContext) -> bool:
        """
        Validate that all prerequisites for this step are met.
        
        Args:
            context: Pipeline context
            
        Returns:
            bool: True if prerequisites are met
        """
        return True
    
    def get_step_display_name(self) -> str:
        """Get display name for this step."""
        return self.name.replace('_', ' ').title()
    
    def handle_error(self, context: PipelineContext, error: Exception) -> bool:
        """
        Handle errors that occur during step execution.
        
        Args:
            context: Pipeline context
            error: The exception that occurred
            
        Returns:
            bool: True if pipeline should continue, False if it should stop
        """
        # Determine if error is critical based on config
        critical = (
            self.config.processing.require_all or 
            isinstance(error, (ValueError, FileNotFoundError))
        )
        
        context.add_error(self.name, error, critical)
        
        if critical:
            self.logger.error(f"{self.name} failed critically: {error}")
            return False
        else:
            self.logger.warning(f"{self.name} failed, continuing: {error}")
            return True


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ProcessingError(Exception):
    """Raised when processing fails."""
    pass