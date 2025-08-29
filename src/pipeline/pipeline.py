"""
Main pipeline class for orchestrating audio analysis steps.
"""

import time
import logging
from typing import List, Dict, Any, Optional

import click

from .base import PipelineStep, PipelineContext
from .steps import (
    AudioExtractionStep, 
    SpeakerDiarizationStep, 
    EmotionAnalysisStep,
    AcousticAnalysisStep,
    SpeechRecognitionStep
)
from .handlers import ErrorHandler, ResultProcessor
from ..utils.logger import PerformanceLogger
from config.settings import Config


class AudioAnalysisPipeline:
    """
    Main pipeline class that orchestrates all analysis steps.
    
    This class implements the Pipeline pattern to coordinate:
    1. Audio extraction
    2. Speaker diarization  
    3. Emotion analysis
    4. Acoustic analysis
    5. Speech recognition (optional)
    6. Result processing and export
    """
    
    def __init__(self, config: Config, **kwargs):
        """
        Initialize the pipeline with configuration and options.
        
        Args:
            config: Pipeline configuration
            **kwargs: Additional options (enable_stt, stt_language, etc.)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.perf_logger = PerformanceLogger(self.logger)
        
        # Pipeline components
        self.error_handler = ErrorHandler(config)
        self.result_processor = ResultProcessor(config)
        
        # Extract options
        self.enable_stt = kwargs.get('enable_stt', False)
        self.stt_language = kwargs.get('stt_language', None)
        self.export_ass = kwargs.get('export_ass', False)
        self.use_enhanced_format = kwargs.get('use_enhanced_format', False)
        self.output_formats = kwargs.get('output_formats', ['json'])
        self.processing_start_time = None
        
        # Initialize pipeline steps
        self.steps = self._create_pipeline_steps()
        
    def _create_pipeline_steps(self) -> List[PipelineStep]:
        """Create and configure all pipeline steps."""
        steps = [
            AudioExtractionStep(self.config),
            SpeakerDiarizationStep(self.config),
            EmotionAnalysisStep(self.config),
            AcousticAnalysisStep(self.config)
        ]
        
        # Add speech recognition step if enabled
        if self.enable_stt:
            steps.append(SpeechRecognitionStep(
                self.config, 
                enable_stt=True, 
                stt_language=self.stt_language
            ))
        
        return steps
    
    def execute(self, input_source: str) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Args:
            input_source: Path or URL to audio/video source
            
        Returns:
            Dictionary containing results and metadata
        """
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
            
            # Process and export results using enhanced system
            processed_results = self.result_processor.process_results(
                context, 
                use_enhanced_format=self.use_enhanced_format,
                output_formats=self.output_formats,
                processing_start_time=self.processing_start_time
            )
            
            # Generate summary report
            self.result_processor.generate_summary_report(
                processed_results['merger'],
                processed_results['results'],
                context,
                self.use_enhanced_format
            )
            
            # Export ASS subtitles if requested
            if self.export_ass:
                self.result_processor.export_ass_subtitles(
                    processed_results['results'], 
                    context
                )
            
            # Display summary
            self.result_processor.display_summary(processed_results['results'])
            
            # Display completion message
            self._display_completion_message(context)
            
            return {
                'context': context,
                'results': processed_results['results'],
                'exported_files': processed_results['exported_files'],
                'success': True
            }
            
        except KeyboardInterrupt:
            self.error_handler.handle_keyboard_interrupt()
            
        except Exception as e:
            self.error_handler.handle_fatal_error(e)
            
        finally:
            # Stop performance timing
            total_time = self.perf_logger.stop_timer("total_processing")
            self.perf_logger.log_memory_usage()
    
    def _execute_step(self, step: PipelineStep, context: PipelineContext, step_number: int) -> bool:
        """
        Execute a single pipeline step with error handling.
        
        Args:
            step: The pipeline step to execute
            context: Pipeline context
            step_number: Step number for display
            
        Returns:
            bool: True if pipeline should continue, False if it should stop
        """
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
                # Step handled its own error and decided to stop
                return False
                
        except Exception as e:
            # Step threw an unhandled exception
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
        
        # Show any warnings
        if context.warnings:
            click.echo(f"\n{click.style('⚠️  Warnings:', fg='yellow')}")
            for warning in context.warnings:
                click.echo(f"  • {warning}")
        
        # Show error summary
        if context.errors:
            critical_errors = [e for e in context.errors if e['critical']]
            non_critical_errors = [e for e in context.errors if not e['critical']]
            
            if critical_errors:
                click.echo(f"\n{click.style('❌ Critical Errors:', fg='red')}")
                for error in critical_errors:
                    click.echo(f"  • {error['step']}: {error['error']}")
            
            if non_critical_errors:
                click.echo(f"\n{click.style('⚠️  Non-Critical Errors:', fg='yellow')}")
                for error in non_critical_errors:
                    click.echo(f"  • {error['step']}: {error['error']}")
    
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
    
    def cleanup(self, debug: bool = False):
        """Clean up temporary files and resources."""
        if not debug:
            # Find audio extraction step and cleanup
            for step in self.steps:
                if hasattr(step, 'cleanup_temp_files'):
                    step.cleanup_temp_files()
                    break


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
    
    def enable_backend_api(self) -> 'PipelineBuilder':
        """Enable backend API output format."""
        current_formats = self.options.get('output_formats', ['json'])
        if 'backend_api' not in current_formats:
            current_formats.append('backend_api')
        self.options['output_formats'] = current_formats
        return self
    
    def enable_frontend_json(self) -> 'PipelineBuilder':
        """Enable frontend JSON output format."""
        current_formats = self.options.get('output_formats', ['json'])
        if 'frontend_json' not in current_formats:
            current_formats.append('frontend_json')
        self.options['output_formats'] = current_formats
        return self
    
    def with_video_method(self, method: str) -> 'PipelineBuilder':
        """Set video extraction method."""
        valid_methods = ['auto', 'moviepy', 'ffmpeg', 'parallel']
        if method not in valid_methods:
            raise ValueError(f"Invalid video method: {method}. Must be one of {valid_methods}")
        
        # Update config if available
        if self.config:
            self.config.video.extraction_method = method
        else:
            # Store for later
            self.options['video_method'] = method
        return self
    
    def build(self) -> AudioAnalysisPipeline:
        """Build and return the configured pipeline."""
        if self.config is None:
            raise ValueError("Configuration must be set before building pipeline")
        
        return AudioAnalysisPipeline(self.config, **self.options)