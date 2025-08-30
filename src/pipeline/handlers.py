"""
Error handling and result processing for the pipeline.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import click

from .base import PipelineContext, AuthenticationError, ModelLoadError
from ..result_merger import ResultMerger
from ..enhanced_result_merger import EnhancedResultMerger
from ..ass_exporter import ASSExporter
from ..output_manager import OutputFormatManager, MediaInfo, ProcessingMetadata
from ..output_manager import OutputFormat
from config.settings import Config


class ErrorHandler:
    """Centralized error handling for pipeline steps."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def handle_step_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """
        Handle errors from pipeline steps.
        
        Args:
            step_name: Name of the step that failed
            error: The exception that occurred
            context: Pipeline context
            
        Returns:
            bool: True if pipeline should continue, False if it should stop
        """
        if isinstance(error, AuthenticationError):
            return self._handle_auth_error(step_name, error, context)
        elif isinstance(error, ModelLoadError):
            return self._handle_model_error(step_name, error, context)
        elif isinstance(error, (ValueError, FileNotFoundError)):
            return self._handle_critical_error(step_name, error, context)
        else:
            return self._handle_non_critical_error(step_name, error, context)
    
    def _handle_auth_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """Handle authentication errors."""
        click.echo(click.style(f"❌ Authentication failed in {step_name}: {error}", fg='red', bold=True))
        
        if "HuggingFace" in str(error) or "HUGGINGFACE_TOKEN" in str(error):
            click.echo("\n🔑 HuggingFace Token Setup:")
            click.echo("1. Get token: https://huggingface.co/settings/tokens")
            click.echo("2. Add to .env: HUGGINGFACE_TOKEN=hf_xxxx")
            click.echo("3. Accept model agreements (see above URLs)")
        
        context.add_error(step_name, error, critical=True)
        
        if self.config.processing.require_all:
            click.echo(click.style("\n⚠️  Pipeline stopped due to authentication error", fg='red', bold=True))
            return False
        
        click.echo(click.style(f"   Continuing without {step_name}...", fg='yellow'))
        return True
    
    def _handle_model_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """Handle model loading errors."""
        click.echo(click.style(f"❌ Model loading failed in {step_name}: {error}", fg='red'))
        context.add_error(step_name, error, critical=self.config.processing.require_all)
        
        if self.config.processing.require_all:
            click.echo(click.style("\n⚠️  Pipeline stopped due to model error", fg='red', bold=True))
            return False
        
        click.echo(click.style(f"   Continuing without {step_name}...", fg='yellow'))
        return True
    
    def _handle_critical_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """Handle critical errors that always stop the pipeline."""
        click.echo(click.style(f"❌ Critical error in {step_name}: {error}", fg='red', bold=True))
        context.add_error(step_name, error, critical=True)
        
        click.echo(click.style("\n⚠️  Pipeline stopped due to critical error", fg='red', bold=True))
        return False
    
    def _handle_non_critical_error(self, step_name: str, error: Exception, context: PipelineContext) -> bool:
        """Handle non-critical errors."""
        click.echo(click.style(f"⚠️  {step_name} failed: {str(error)[:100]}", fg='yellow'))
        context.add_error(step_name, error, critical=self.config.processing.require_all)
        
        if self.config.processing.require_all:
            click.echo(click.style(f"\n⚠️  Pipeline stopped due to --require-all flag", fg='red', bold=True))
            return False
        
        click.echo(click.style(f"   Continuing without {step_name}...", fg='yellow'))
        return True
    
    def handle_fatal_error(self, error: Exception):
        """Handle fatal errors that crash the entire pipeline."""
        click.echo(click.style(f"❌ Fatal error: {error}", fg='red', bold=True))
        self.logger.error(f"Fatal error: {error}", exc_info=True)
        sys.exit(1)
    
    def handle_keyboard_interrupt(self):
        """Handle Ctrl+C interruption."""
        click.echo("\n" + click.style("⚠️  Processing interrupted by user", fg='yellow'))
        sys.exit(130)


class ResultProcessor:
    """Handle processing and exporting of pipeline results."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_results(
        self, 
        context: PipelineContext, 
        use_enhanced_format: bool = False,
        output_formats: Optional[List[str]] = None,
        processing_start_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process and export all pipeline results using enhanced output system.
        
        Args:
            context: Pipeline context with all results
            use_enhanced_format: Whether to use enhanced output format
            output_formats: List of desired output formats
            processing_start_time: Start time for performance calculation
            
        Returns:
            Dictionary containing processed results and file paths
        """
        click.echo(f"\n{click.style('🔄 Step 6: Processing Results', fg='yellow', bold=True)}")
        
        # Create merger
        if use_enhanced_format:
            merger = EnhancedResultMerger()
            click.echo(click.style("📊 Using enhanced output format with linguistic precision", fg='cyan'))
        else:
            merger = ResultMerger()
        
        # Merge all results
        analysis_results = context.get_analysis_results()
        merged_results = merger.merge_all_results(**analysis_results)
        
        click.echo(click.style("✅ Results merged successfully", fg='green'))
        
        # Use enhanced output system
        exported_files = self._export_enhanced_results(
            merged_results, context, output_formats, processing_start_time
        )
        
        return {
            'results': merged_results,
            'exported_files': exported_files,
            'merger': merger
        }
    
    def _export_enhanced_results(
        self, 
        results: Dict[str, Any], 
        context: PipelineContext, 
        output_formats: Optional[List[str]] = None,
        processing_start_time: Optional[float] = None
    ) -> Dict[str, Path]:
        """Export results using enhanced output system."""
        try:
            # Initialize output manager
            output_manager = OutputFormatManager(self.config, self.config.output.output_dir)
            
            # Determine formats to export
            if not output_formats:
                output_formats = self.config.output.default_formats
            
            # Create media info
            audio_info = context.audio_info or {}
            video_metadata = getattr(context, 'video_metadata', {}) or {}
            
            # Determine source type
            source_type = "video" if video_metadata else "audio"
            if context.input_source and ("youtube.com" in str(context.input_source) or "youtu.be" in str(context.input_source)):
                source_type = "youtube"
            
            media_info = MediaInfo(
                source_type=source_type,
                duration=audio_info.get('duration', 0),
                sample_rate=audio_info.get('sample_rate', 0),
                channels=audio_info.get('channels', 0),
                language=results.get('language', 'auto'),
                video_resolution=video_metadata.get('resolution'),
                codec=video_metadata.get('video_codec') or audio_info.get('codec'),
                file_size=video_metadata.get('size') or audio_info.get('file_size_bytes', 0)
            )
            
            # Create processing metadata
            processing_time = 0
            if processing_start_time:
                processing_time = time.time() - processing_start_time
            
            metadata = ProcessingMetadata(
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                pipeline_version="2.0",
                whisper_enhancements=getattr(context, 'whisper_performance_stats', {}),
                performance_stats=context.get_performance_stats()
            )
            
            # Convert string formats to OutputFormat enums
            format_enums = []
            for fmt in output_formats:
                try:
                    format_enums.append(OutputFormat(fmt))
                except ValueError:
                    # Fallback to JSON if format is not recognized
                    self.logger.warning(f"Unknown format '{fmt}', using JSON instead")
                    format_enums.append(OutputFormat.JSON)
            
            # Export in multiple formats
            exported_files = output_manager.export_results(
                results=results,
                media_info=media_info,
                metadata=metadata,
                formats=format_enums
            )
            
            # Display export summary
            click.echo(click.style("📤 Export Summary:", fg='cyan'))
            for format_type, file_path in exported_files.items():
                click.echo(f"  • {format_type.value}: {file_path}")
            
            # Convert enum keys to strings for return
            return {fmt.value: path for fmt, path in exported_files.items()}
            
        except Exception as e:
            self.logger.error(f"Enhanced export failed: {e}")
            # Fallback to legacy export
            return self._export_results_legacy(results, context, use_enhanced_format=False)
    
    def _export_results_legacy(self, results: Dict[str, Any], context: PipelineContext, use_enhanced_format: bool = False) -> Dict[str, Path]:
        """Export results to files."""
        click.echo(f"\n{click.style('💾 Saving Results', fg='yellow', bold=True)}")
        
        exported_files = {}
        
        # Create merger based on format preference
        if use_enhanced_format:
            merger = EnhancedResultMerger()
            # Export enhanced format
            json_path = context.get_output_path("enhanced", "json")
            merger.export_to_json(results, json_path)
            exported_files['json'] = json_path
            
            # Also export WebVTT for web compatibility
            vtt_path = context.get_output_path("subtitles", "vtt")
            merger.export_to_webvtt(results, vtt_path)
            exported_files['webvtt'] = vtt_path
            
        else:
            # Standard export
            merger = ResultMerger()
            format_type = 'both' if 'both' in self.config.output.output_format else self.config.output.output_format[0]
            exported_files = merger.export_timeline_segments(
                results,
                self.config.output.output_dir,
                format=format_type
            )
        
        # Display exported files
        for format_name, file_path in exported_files.items():
            click.echo(click.style(f"✅ {format_name.upper()} results saved: {file_path}", fg='green'))
        
        return exported_files
    
    def generate_summary_report(self, merger, results: Dict[str, Any], context: PipelineContext, use_enhanced_format: bool):
        """Generate and save summary report."""
        if not use_enhanced_format:
            report_path = context.get_output_path("report", "txt")
            merger.generate_summary_report(results, report_path)
            click.echo(click.style(f"✅ Summary report saved: {report_path}", fg='green'))
        else:
            click.echo(click.style("✅ Statistics included in enhanced JSON output", fg='green'))
    
    def export_ass_subtitles(self, results: Dict[str, Any], context: PipelineContext) -> bool:
        """Export ASS subtitle file if requested."""
        if not self.config.output.export_ass:
            return True
        
        click.echo(f"\n{click.style('📝 Exporting ASS Subtitle', fg='yellow', bold=True)}")
        
        try:
            ass_exporter = ASSExporter()
            ass_path = context.get_output_path("subtitles", "ass")
            
            ass_file = ass_exporter.export_to_ass(
                results,
                ass_path,
                use_emotions=True,
                use_speaker_colors=True
            )
            
            click.echo(click.style(f"✅ ASS subtitle saved: {ass_file}", fg='green'))
            
            # If no transcriptions, remind user to enable STT
            if not context.transcription_results:
                click.echo(click.style("💡 Tip: Use --enable-stt to add actual text to subtitles", fg='cyan'))
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"⚠️  ASS export failed: {str(e)[:100]}", fg='yellow'))
            self.logger.warning(f"ASS export failed: {e}")
            return False
    
    
    def display_summary(self, results: Dict[str, Any]):
        """Display brief analysis summary."""
        if not results.get('summary'):
            return
        
        summary = results['summary']
        click.echo(f"\n{click.style('📊 Analysis Summary:', fg='cyan', bold=True)}")
        
        if 'total_speakers' in summary:
            click.echo(f"  • Speakers detected: {summary['total_speakers']}")
        if 'dominant_emotion' in summary:
            click.echo(f"  • Dominant emotion: {summary['dominant_emotion']}")
        if 'emotion_changes' in summary:
            click.echo(f"  • Emotion changes: {summary['emotion_changes']}")
        if 'total_segments' in summary:
            click.echo(f"  • Total segments: {summary['total_segments']}")