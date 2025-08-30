"""
Unified Audio Analysis Pipeline

This module contains the complete pipeline architecture for audio analysis,
integrating all components into a single, well-organized file while maintaining SRP.

Key Components:
- PipelineContext: Shared data container between steps
- PipelineStep: Abstract base class for analysis steps
- Concrete Steps: Audio extraction, speaker diarization, emotion analysis, etc.
- Pipeline: Main orchestrator
- Builder: Fluent configuration interface
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime

import click

from config.settings import Config


# ================================
# EXCEPTIONS
# ================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class AuthenticationError(PipelineError):
    """Authentication-related errors."""
    pass


class ModelLoadError(PipelineError):
    """Model loading errors."""
    pass


class ProcessingError(PipelineError):
    """Processing-related errors."""
    pass


# ================================
# CONTEXT CLASS
# ================================

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
        """Get all analysis results."""
        return {
            'audio_info': self.audio_info,
            'video_metadata': self.video_metadata,
            'speaker_results': self.speaker_results,
            'emotion_results': self.emotion_results,
            'acoustic_results': self.acoustic_results,
            'transcription_results': self.transcription_results
        }
    
    def mark_step_completed(self, step_name: str, duration: float) -> None:
        """Mark a step as completed."""
        self.completed_steps.append(step_name)
        self.step_durations[step_name] = duration
    
    def add_error(self, step_name: str, error: str, critical: bool = False) -> None:
        """Add an error to the context."""
        self.errors.append({
            'step': step_name,
            'error': error,
            'critical': critical,
            'timestamp': datetime.now()
        })
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the context."""
        self.warnings.append(warning)


# ================================
# PIPELINE STEP BASE CLASS
# ================================

class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> bool:
        """Execute the step. Return True to continue pipeline, False to stop."""
        pass
    
    def should_execute(self, context: PipelineContext) -> bool:
        """Check if this step should be executed."""
        return True
    
    def validate_prerequisites(self, context: PipelineContext) -> bool:
        """Validate prerequisites for this step."""
        return True
    
    def get_step_display_name(self) -> str:
        """Get display-friendly step name."""
        return self.name.replace('_', ' ').title()


# ================================
# CONCRETE PIPELINE STEPS
# ================================

class AudioExtractionStep(PipelineStep):
    """Extract audio from various sources."""
    
    def __init__(self):
        super().__init__("audio_extraction")
        self.extractor = None
    
    def execute(self, context: PipelineContext) -> bool:
        from .analyzers import AudioExtractor
        
        if self.extractor is None:
            self.extractor = AudioExtractor(context.config.audio)
        
        try:
            result = self.extractor.extract(
                context.input_source,
                context.get_output_path("extracted_audio", "wav")
            )
            
            if result['success']:
                context.audio_file = Path(result['output_path'])
                context.audio_info = result.get('audio_info', {})
                context.video_metadata = result.get('video_metadata', {})
                return True
            else:
                context.add_error(self.name, result.get('error', 'Unknown extraction error'), critical=True)
                return False
                
        except Exception as e:
            context.add_error(self.name, str(e), critical=True)
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.extractor and hasattr(self.extractor, 'cleanup'):
            self.extractor.cleanup()


class SpeakerDiarizationStep(PipelineStep):
    """Perform speaker diarization."""
    
    def __init__(self):
        super().__init__("speaker_diarization")
        self.diarizer = None
    
    def should_execute(self, context: PipelineContext) -> bool:
        return context.has_audio_file()
    
    def validate_prerequisites(self, context: PipelineContext) -> bool:
        return context.audio_file is not None
    
    def execute(self, context: PipelineContext) -> bool:
        from .analyzers import SpeakerDiarizer
        
        if self.diarizer is None:
            self.diarizer = SpeakerDiarizer(context.config.model)
            
            # Set authentication token
            import os
            token = os.getenv('HUGGINGFACE_TOKEN')
            if token:
                self.diarizer.set_auth_token(token)
            else:
                context.add_warning("HuggingFace token not found, using basic speaker analysis")
        
        try:
            segments = self.diarizer.diarize(context.audio_file)
            
            # Process segments into speaker info
            speaker_info = {}
            for segment in segments:
                speaker_id = segment.speaker_id
                if speaker_id not in speaker_info:
                    speaker_info[speaker_id] = {
                        'segments': [],
                        'total_duration': 0.0
                    }
                
                speaker_info[speaker_id]['segments'].append(segment)
                speaker_info[speaker_id]['total_duration'] += segment.end - segment.start
            
            context.speaker_results = {
                'segments': segments,
                'speakers': speaker_info,
                'num_speakers': len(speaker_info)
            }
            
            return True
            
        except Exception as e:
            context.add_error(self.name, str(e))
            return True  # Continue pipeline even if speaker diarization fails
    
    def cleanup(self):
        """Clean up resources."""
        if self.diarizer and hasattr(self.diarizer, 'cleanup'):
            self.diarizer.cleanup()


class EmotionAnalysisStep(PipelineStep):
    """Perform emotion analysis."""
    
    def __init__(self):
        super().__init__("emotion_analysis")
        self.analyzer = None
    
    def should_execute(self, context: PipelineContext) -> bool:
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        from .analyzers import EmotionAnalyzer
        
        if self.analyzer is None:
            self.analyzer = EmotionAnalyzer(context.config.model)
        
        try:
            # Use speaker segments if available, otherwise analyze whole audio
            segments = None
            if context.speaker_results:
                segments = context.speaker_results['segments']
            
            emotions = self.analyzer.analyze_segments(
                context.audio_file,
                segments=segments
            )
            
            context.emotion_results = emotions
            return True
            
        except Exception as e:
            context.add_error(self.name, str(e))
            return True  # Continue pipeline even if emotion analysis fails
    
    def cleanup(self):
        """Clean up resources."""
        if self.analyzer and hasattr(self.analyzer, 'cleanup'):
            self.analyzer.cleanup()


class AcousticAnalysisStep(PipelineStep):
    """Perform acoustic feature analysis."""
    
    def __init__(self):
        super().__init__("acoustic_analysis")
        self.analyzer = None
    
    def should_execute(self, context: PipelineContext) -> bool:
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        from .analyzers import AcousticAnalyzer
        
        if self.analyzer is None:
            self.analyzer = AcousticAnalyzer(context.config.processing)
        
        try:
            # Use speaker segments if available
            segments = None
            if context.speaker_results:
                segments = context.speaker_results['segments']
            
            features = self.analyzer.analyze(
                context.audio_file,
                segments=segments
            )
            
            context.acoustic_results = features
            return True
            
        except Exception as e:
            context.add_error(self.name, str(e))
            return True  # Continue pipeline even if acoustic analysis fails


class SpeechRecognitionStep(PipelineStep):
    """Perform speech recognition."""
    
    def __init__(self, language: Optional[str] = None):
        super().__init__("speech_recognition")
        self.recognizer = None
        self.language = language
    
    def should_execute(self, context: PipelineContext) -> bool:
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        from .analyzers import SpeechRecognizer
        
        if self.recognizer is None:
            self.recognizer = SpeechRecognizer(
                whisper_config=context.config.whisper
            )
        
        try:
            # Use speaker segments if available
            segments = None
            if context.speaker_results:
                segments = [{'start': s.start, 'end': s.end} 
                           for s in context.speaker_results['segments']]
            
            transcriptions = self.recognizer.transcribe(
                context.audio_file,
                segments=segments,
                language=self.language
            )
            
            context.transcription_results = transcriptions
            return True
            
        except Exception as e:
            context.add_error(self.name, str(e))
            return True  # Continue pipeline even if transcription fails
    
    def cleanup(self):
        """Clean up resources."""
        if self.recognizer and hasattr(self.recognizer, 'cleanup'):
            self.recognizer.cleanup()


# ================================
# ERROR HANDLER
# ================================

class ErrorHandler:
    """Centralized error handling for the pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("pipeline.error_handler")
    
    def handle_step_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """Handle errors from pipeline steps."""
        error_msg = f"{step_name} failed: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        # Determine if error is critical
        critical = step_name == "audio_extraction"  # Audio extraction failure is always critical
        
        context.add_error(step_name, str(error), critical=critical)
        
        # If require_all is True, any error stops the pipeline
        if self.config.processing.require_all:
            return False
        
        # Critical errors always stop the pipeline
        return not critical
    
    def handle_keyboard_interrupt(self):
        """Handle keyboard interrupt."""
        click.echo(click.style("\n⚠️  Processing interrupted by user", fg='yellow'))
        return {'success': False, 'error': 'User interrupted'}
    
    def handle_fatal_error(self, error: Exception):
        """Handle fatal errors."""
        self.logger.error(f"Fatal error: {str(error)}", exc_info=True)
        return {'success': False, 'error': str(error)}


# ================================
# MAIN PIPELINE CLASS
# ================================

class AudioAnalysisPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: Config, **options):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        from .utils import PerformanceLogger
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Pipeline components
        self.error_handler = ErrorHandler(config)
        
        # Extract options
        self.enable_stt = options.get('enable_stt', False)
        self.stt_language = options.get('stt_language')
        self.export_ass = options.get('export_ass', False)
        self.use_enhanced_format = options.get('use_enhanced_format', False)
        self.output_formats = options.get('output_formats', ['json'])
        
        # Initialize steps
        self.steps = [
            AudioExtractionStep(),
            SpeakerDiarizationStep(),
            EmotionAnalysisStep(),
            AcousticAnalysisStep()
        ]
        
        # Add speech recognition if enabled
        if self.enable_stt:
            self.steps.append(SpeechRecognitionStep(self.stt_language))
        
        self.processing_start_time = None
    
    def execute(self, input_source: str) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        # Start performance timing
        self.perf_logger.start_timer("total_processing")
        self.processing_start_time = time.time()
        
        # Create pipeline context
        context = PipelineContext(input_source, self.config)
        
        try:
            # Display welcome message
            self._display_welcome_message()
            
            # Execute each pipeline step
            for i, step in enumerate(self.steps, 1):
                if not self._execute_step(step, context, i):
                    break
            
            # Process and export results
            from .memory import cleanup_manager
            
            results = self._process_results(context)
            
            # Display completion message
            self._display_completion_message(context)
            
            # Perform periodic memory cleanup after pipeline execution
            cleanup_manager.periodic_cleanup()
            
            return {
                'context': context,
                'results': results,
                'success': True
            }
            
        except KeyboardInterrupt:
            return self.error_handler.handle_keyboard_interrupt()
            
        except Exception as e:
            return self.error_handler.handle_fatal_error(e)
            
        finally:
            # Stop performance timing
            total_time = self.perf_logger.stop_timer("total_processing")
            self.perf_logger.log_memory_usage()
    
    def _execute_step(self, step: PipelineStep, context: PipelineContext, step_number: int) -> bool:
        """Execute a single pipeline step with error handling."""
        # Check if step should be executed
        if not step.should_execute(context):
            self.logger.debug(f"Skipping {step.name} - prerequisites not met")
            return True
        
        # Validate prerequisites
        if not step.validate_prerequisites(context):
            self.logger.warning(f"Prerequisites not met for {step.name}")
            return True
        
        # Execute the step
        try:
            step_start_time = time.time()
            success = step.execute(context)
            step_duration = time.time() - step_start_time
            
            if success:
                context.mark_step_completed(step.name, step_duration)
                return True
            else:
                return False
                
        except Exception as e:
            return self.error_handler.handle_step_error(step.name, e, context)
    
    def _display_welcome_message(self):
        """Display welcome message."""
        click.echo(click.style("🎵 Audio Analysis Pipeline", fg='cyan', bold=True))
        click.echo(click.style("=" * 50, fg='cyan'))
    
    def _display_completion_message(self, context: PipelineContext):
        """Display completion message with performance stats."""
        total_time = self.perf_logger.stop_timer("total_processing")
        
        click.echo("\n" + click.style("=" * 50, fg='cyan'))
        click.echo(click.style(f"✨ Processing completed in {total_time:.2f} seconds", fg='green', bold=True))
        
        # Show step breakdown if in verbose mode
        if self.config.processing.verbose and context.step_durations:
            click.echo(f"\n{click.style('📊 Step Performance:', fg='cyan')}")
            for step_name, duration in context.step_durations.items():
                step_display = step_name.replace('_', ' ').title()
                click.echo(f"  • {step_display}: {duration:.2f}s")
    
    def _process_results(self, context: PipelineContext) -> Dict[str, Any]:
        """Process and format results."""
        # This would integrate with the result processing system
        # For now, return basic results
        return context.get_analysis_results()
    
    def cleanup(self, debug: bool = False):
        """Clean up temporary files and resources with comprehensive memory management."""
        self.logger.info("Starting comprehensive pipeline cleanup")
        
        if not debug:
            # Clean up temporary files from steps
            for step in self.steps:
                if hasattr(step, 'cleanup_temp_files'):
                    try:
                        step.cleanup_temp_files()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup temp files for {step.__class__.__name__}: {e}")
                
                # Clean up step-specific resources
                if hasattr(step, 'cleanup'):
                    try:
                        step.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup step {step.__class__.__name__}: {e}")
        
        # Comprehensive memory cleanup
        try:
            from .memory import cleanup_manager
            cleanup_manager.cleanup_system_resources()
        except Exception as e:
            self.logger.warning(f"Failed during system resource cleanup: {e}")
        
        # Final memory report
        if self.config.logging.log_memory_usage:
            try:
                from .memory import cleanup_manager
                report = cleanup_manager.get_memory_report()
                self.logger.info(f"Cleanup completed. Memory pressure: {report['memory_pressure']}")
                if report['models']['loaded_count'] > 0:
                    self.logger.info(f"Still loaded models: {report['models']['loaded_models']}")
            except Exception as e:
                self.logger.warning(f"Failed to get memory report: {e}")
        
        self.logger.info("Pipeline cleanup completed")
    
    def get_step_info(self) -> List[Dict[str, str]]:
        """Get information about all pipeline steps."""
        return [
            {
                'name': step.name,
                'display_name': step.get_step_display_name(),
                'class_name': step.__class__.__name__
            }
            for step in self.steps
        ]


# ================================
# PIPELINE BUILDER
# ================================

class PipelineBuilder:
    """Builder class for creating configured pipelines."""
    
    def __init__(self):
        self.config = None
        self.options = {}
    
    def with_config(self, config: Config) -> 'PipelineBuilder':
        """Set the pipeline configuration."""
        self.config = config
        return self
    
    def enable_stt(self, language: Optional[str] = None) -> 'PipelineBuilder':
        """Enable speech-to-text with optional language."""
        self.options['enable_stt'] = True
        self.options['stt_language'] = language
        return self
    
    def enable_ass_export(self) -> 'PipelineBuilder':
        """Enable ASS subtitle export."""
        self.options['export_ass'] = True
        return self
    
    def use_enhanced_format(self) -> 'PipelineBuilder':
        """Use enhanced output format."""
        self.options['use_enhanced_format'] = True
        return self
    
    def with_output_formats(self, formats: List[str]) -> 'PipelineBuilder':
        """Set output formats for results."""
        self.options['output_formats'] = formats
        return self
    
    def with_video_method(self, method: str) -> 'PipelineBuilder':
        """Set video processing method."""
        # This would be used to configure the AudioExtractionStep
        self.options['video_method'] = method
        return self
    
    def build(self) -> AudioAnalysisPipeline:
        """Build the pipeline with current configuration."""
        if not self.config:
            raise ValueError("Configuration must be set before building pipeline")
        
        return AudioAnalysisPipeline(self.config, **self.options)