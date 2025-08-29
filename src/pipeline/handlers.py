"""
Error handling and result processing for the pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import click

from .base import PipelineContext, AuthenticationError, ModelLoadError
from ..result_merger import ResultMerger
from ..enhanced_result_merger import EnhancedResultMerger
from ..ass_exporter import ASSExporter
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
    
    def process_results(self, context: PipelineContext, use_enhanced_format: bool = False) -> Dict[str, Any]:
        """
        Process and export all pipeline results.
        
        Args:
            context: Pipeline context with all results
            use_enhanced_format: Whether to use enhanced output format
            
        Returns:
            Dictionary containing processed results and file paths
        """
        click.echo(f"\n{click.style('🔄 Step 6: Merging Results', fg='yellow', bold=True)}")
        
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
        
        # Export results
        exported_files = self._export_results(merger, merged_results, context, use_enhanced_format)
        
        return {
            'results': merged_results,
            'exported_files': exported_files,
            'merger': merger
        }
    
    def _export_results(self, merger, results: Dict[str, Any], context: PipelineContext, use_enhanced_format: bool) -> Dict[str, Path]:
        """Export results to files."""
        click.echo(f"\n{click.style('💾 Saving Results', fg='yellow', bold=True)}")
        
        exported_files = {}
        
        if use_enhanced_format:
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
    
    def generate_visualizations(self, results: Dict[str, Any], context: PipelineContext) -> bool:
        """Generate visualization plots if requested."""
        if not self.config.output.generate_visualizations:
            return True
        
        click.echo(f"\n{click.style('📈 Generating Visualizations', fg='yellow', bold=True)}")
        
        try:
            from ..utils.visualization import create_visualization_report
            from ..utils.audio_utils import load_audio
            
            # Load audio for visualization
            audio_data, sr = load_audio(context.audio_file, sample_rate=self.config.audio.sample_rate)
            
            viz_dir = self.config.output.output_dir / "visualizations"
            viz_files = create_visualization_report(
                results,
                audio_data=audio_data,
                sample_rate=sr,
                output_dir=viz_dir,
                include_waveform=True,
                include_spectrogram=True
            )
            
            if viz_files:
                click.echo(click.style(f"✅ Generated {len(viz_files)} visualization plots", fg='green'))
                for plot_name, plot_path in viz_files.items():
                    click.echo(f"  • {plot_name}: {plot_path}")
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"⚠️  Visualization generation failed: {str(e)[:100]}", fg='yellow'))
            self.logger.warning(f"Visualization failed: {e}")
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