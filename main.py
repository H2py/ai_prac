#!/usr/bin/env python3
"""
Refactored main CLI interface using the new pipeline architecture.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import click

from src.pipeline import PipelineBuilder
from src.utils.logger import setup_logger
from config.settings import Config


class CLIManager:
    """Manages CLI operations and pipeline execution."""
    
    def __init__(self):
        self.logger = setup_logger(
            "audio_analysis",
            level="INFO",
            use_colors=True
        )
        self.pipeline = None
    
    def create_pipeline(self, config: Config, **options) -> None:
        """Create pipeline with given configuration and options."""
        builder = PipelineBuilder().with_config(config)
        
        # Configure pipeline options
        if options.get('enable_stt'):
            builder = builder.enable_stt(options.get('stt_language'))
        
        if options.get('export_ass'):
            builder = builder.enable_ass_export()
        
        if options.get('use_enhanced_format'):
            builder = builder.use_enhanced_format()
        
        # Set output formats
        if options.get('output_formats'):
            builder = builder.with_output_formats(options['output_formats'])
        
        # Set video processing method if specified
        if hasattr(config.video, 'extraction_method'):
            builder = builder.with_video_method(config.video.extraction_method)
        
        self.pipeline = builder.build()
    
    def execute_pipeline(self, input_source: str, debug: bool = False) -> dict:
        """Execute the pipeline and return results."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
        
        try:
            results = self.pipeline.execute(input_source)
            return results
        
        finally:
            if self.pipeline:
                self.pipeline.cleanup(debug=debug)
    
    def set_logging_level(self, debug: bool, verbose: bool):
        """Set appropriate logging level."""
        if debug:
            self.logger.setLevel(logging.DEBUG)
        elif verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)


@click.command()
@click.argument('input_source', type=str)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='./output',
    help='Output directory for results'
)
@click.option(
    '--format', '-f',
    multiple=True,
    type=click.Choice(['json', 'ass', 'vtt', 'srt', 'backend_api', 'frontend_json']),
    default=['json'],
    help='Output formats for results (can specify multiple)'
)
@click.option(
    '--gpu/--no-gpu',
    default=True,
    help='Use GPU acceleration if available'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file (YAML)'
)
@click.option(
    '--sample-rate',
    type=int,
    default=16000,
    help='Target sample rate for audio processing'
)
@click.option(
    '--segment-duration',
    type=float,
    default=30.0,
    help='Duration of audio segments for processing (seconds)'
)
@click.option(
    '--min-segment-length',
    type=float,
    default=1.0,
    help='Minimum segment length (seconds)'
)
@click.option(
    '--emotion-threshold',
    type=float,
    default=0.6,
    help='Confidence threshold for emotion detection'
)
@click.option(
    '--export-features/--no-export-features',
    default=False,
    help='Export raw acoustic features'
)
@click.option(
    '--visualize/--no-visualize',
    default=False,
    help='Generate visualization plots'
)
@click.option(
    '--enable-stt/--no-stt',
    default=False,
    help='Enable speech-to-text transcription'
)
@click.option(
    '--stt-language',
    type=str,
    default=None,
    help='Language for STT (e.g., ko, en, auto for detect)'
)
@click.option(
    '--export-ass/--no-ass',
    default=False,
    help='Export results as ASS subtitle file'
)
@click.option(
    '--require-all',
    is_flag=True,
    default=False,
    help='Require all analysis steps to succeed (exit on any failure)'
)
@click.option(
    '--use-enhanced-format',
    is_flag=True,
    default=False,
    help='Use enhanced output format with linguistic precision and standard compliance'
)
@click.option(
    '--video-method',
    type=click.Choice(['auto', 'moviepy', 'ffmpeg', 'parallel']),
    default='auto',
    help='Method for video processing (auto selects best)'
)
@click.option(
    '--enable-backend-api',
    is_flag=True,
    default=False,
    help='Generate backend API compatible output'
)
@click.option(
    '--enable-frontend-json',
    is_flag=True,
    default=False,
    help='Generate frontend-optimized JSON output'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Enable debug mode'
)
def main(
    input_source: str,
    output: str,
    format: tuple,
    gpu: bool,
    config: Optional[str],
    sample_rate: int,
    segment_duration: float,
    min_segment_length: float,
    emotion_threshold: float,
    export_features: bool,
    visualize: bool,
    enable_stt: bool,
    stt_language: Optional[str],
    export_ass: bool,
    require_all: bool,
    use_enhanced_format: bool,
    video_method: str,
    enable_backend_api: bool,
    enable_frontend_json: bool,
    verbose: bool,
    debug: bool
):
    """
    Audio Analysis Pipeline - Extract and analyze audio from various sources.
    
    INPUT_SOURCE can be:
    - Path to an audio file (WAV, MP3, FLAC, etc.)
    - Path to a video file (MP4, AVI, MOV, etc.)  
    - YouTube URL
    - Other video platform URL
    
    Examples:
    
        # Process local audio file
        python main.py audio.wav --output ./results
        
        # Extract from video file
        python main.py video.mp4 --format both --gpu
        
        # Download and analyze YouTube video
        python main.py "https://youtube.com/watch?v=xxx" --visualize
        
        # Use enhanced format with linguistic precision
        python main.py audio.wav --use-enhanced-format
        
        # Use custom configuration
        python main.py input.mp4 --config config.yaml
    """
    
    # Load environment variables
    load_dotenv()
    
    # Initialize CLI manager
    cli_manager = CLIManager()
    cli_manager.set_logging_level(debug, verbose)
    
    try:
        # Load configuration
        pipeline_config = _load_configuration(config)
        
        # Override config with CLI parameters
        _override_config_with_cli_params(
            pipeline_config,
            output=output,
            format=format,
            gpu=gpu,
            sample_rate=sample_rate,
            segment_duration=segment_duration,
            min_segment_length=min_segment_length,
            emotion_threshold=emotion_threshold,
            export_features=export_features,
            visualize=visualize,
            verbose=verbose,
            require_all=require_all,
            video_method=video_method,
            enable_backend_api=enable_backend_api,
            enable_frontend_json=enable_frontend_json
        )
        
        # Validate configuration
        pipeline_config.validate()
        
        # Create output directory
        pipeline_config.output.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline with options
        pipeline_options = {
            'enable_stt': enable_stt,
            'stt_language': stt_language,
            'export_ass': export_ass,
            'use_enhanced_format': use_enhanced_format,
            'output_formats': list(format)
        }
        
        cli_manager.create_pipeline(pipeline_config, **pipeline_options)
        
        # Execute pipeline
        results = cli_manager.execute_pipeline(input_source, debug=debug)
        
        if results['success']:
            click.echo(click.style("\n🎉 Pipeline completed successfully!", fg='green', bold=True))
        else:
            click.echo(click.style("\n⚠️ Pipeline completed with some errors", fg='yellow', bold=True))
        
    except KeyboardInterrupt:
        click.echo("\n" + click.style("⚠️  Processing interrupted by user", fg='yellow'))
        sys.exit(130)
    
    except Exception as e:
        click.echo(click.style(f"❌ Fatal error: {e}", fg='red'))
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _load_configuration(config_path: Optional[str]) -> Config:
    """Load configuration from file or environment."""
    if config_path:
        click.echo(f"Loading configuration from: {config_path}")
        return Config.from_yaml(Path(config_path))
    else:
        return Config.from_env()


def _override_config_with_cli_params(config: Config, **params) -> None:
    """Override configuration with CLI parameters."""
    # Audio settings
    config.audio.sample_rate = params['sample_rate']
    config.audio.chunk_duration = params['segment_duration']
    config.audio.min_duration = params['min_segment_length']
    
    # Model settings
    config.model.use_gpu = params['gpu']
    config.model.emotion_threshold = params['emotion_threshold']
    
    # Video settings
    config.video.extraction_method = params['video_method']
    
    # Output settings
    config.output.output_dir = Path(params['output'])
    format_params = params['format']
    
    # Handle new format system
    if format_params:
        config.output.default_formats = list(format_params)
        # Add API formats if requested
        if params.get('enable_backend_api'):
            config.output.default_formats.append('backend_api')
        if params.get('enable_frontend_json'):
            config.output.default_formats.append('frontend_json')
    
    config.output.include_raw_features = params['export_features']
    config.output.generate_visualizations = params['visualize']
    config.output.enable_backend_api = params.get('enable_backend_api', False)
    config.output.enable_frontend_json = params.get('enable_frontend_json', False)
    
    # Processing settings
    config.processing.verbose = params['verbose']
    config.processing.require_all = params['require_all']


@click.group()
def cli():
    """Audio Analysis Pipeline CLI."""
    pass


@cli.command()
def version():
    """Show version information."""
    click.echo("Audio Analysis Pipeline v2.0.0 (Refactored)")
    click.echo("Python " + sys.version)


@cli.command()
def formats():
    """Show supported formats."""
    from src.audio_extractor import AudioExtractor
    from src.output_manager import OutputFormatManager
    
    extractor = AudioExtractor()
    formats = extractor.get_supported_formats()
    
    # Get output formats from OutputFormatManager
    output_manager = OutputFormatManager(Config(), Path('./output'))
    output_formats = output_manager.get_supported_formats()
    
    click.echo(click.style("🎵 Enhanced Video Processing Pipeline", fg='cyan', bold=True))
    click.echo(click.style("=" * 50, fg='cyan'))
    
    click.echo("\n📥 Input Audio Formats:")
    for fmt in formats['input_audio']:
        click.echo(f"  • {fmt}")
    
    click.echo("\n📹 Input Video Formats:")
    for fmt in formats['input_video']:
        click.echo(f"  • {fmt}")
    
    click.echo("\n📤 Output Formats:")
    for fmt in output_formats:
        description = {
            'json': 'Enhanced JSON with metadata',
            'ass': 'ASS subtitles with emotion styling',
            'vtt': 'WebVTT subtitles',
            'srt': 'SRT subtitles',
            'backend_api': 'Backend API compatible JSON',
            'frontend_json': 'Frontend optimized JSON with visualization data'
        }.get(fmt, 'Standard format')
        click.echo(f"  • {fmt}: {description}")
    
    click.echo("\n🌐 Supported Sources:")
    for src in formats['sources']:
        click.echo(f"  • {src}")
    
    click.echo("\n🎬 Video Processing Methods:")
    click.echo("  • auto: Automatically select best method")
    click.echo("  • moviepy: MoviePy-based extraction")
    click.echo("  • ffmpeg: FFmpeg-based extraction")
    click.echo("  • parallel: Try multiple methods simultaneously")
    
    click.echo("\n✨ Enhanced Features:")
    click.echo("  • YouTube embed URL support")
    click.echo("  • Video metadata extraction")
    click.echo("  • Parallel processing optimization")
    click.echo("  • Multiple output format generation")
    click.echo("  • Backend/frontend API compatibility")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml', help='Output path for config file')
def generate_config(output):
    """Generate a sample configuration file."""
    config = Config()
    config.to_yaml(Path(output))
    click.echo(click.style(f"✅ Configuration file generated: {output}", fg='green'))
    click.echo("Edit this file to customize your pipeline settings.")


@cli.command()
def pipeline_info():
    """Show information about the pipeline architecture."""
    click.echo(click.style("🏗️  Pipeline Architecture Information", fg='cyan', bold=True))
    click.echo(click.style("=" * 50, fg='cyan'))
    
    # Create a dummy pipeline to get step information
    config = Config()
    pipeline = PipelineBuilder().with_config(config).build()
    
    click.echo("\n📊 Pipeline Steps:")
    for i, step_info in enumerate(pipeline.get_step_info(), 1):
        click.echo(f"  {i}. {step_info['display_name']} ({step_info['class_name']})")
    
    click.echo(f"\n🔧 Key Features:")
    click.echo("  • Modular architecture with independent steps")
    click.echo("  • Robust error handling and recovery")
    click.echo("  • Configurable via YAML files or environment")
    click.echo("  • Enhanced format with linguistic precision")
    click.echo("  • Comprehensive performance logging")
    click.echo("  • Advanced video processing with multiple extraction methods")
    click.echo("  • Multiple output formats (JSON, ASS, VTT, SRT, API formats)")
    click.echo("  • YouTube embed URL support")
    click.echo("  • Backend/frontend API compatibility")
    
    click.echo(f"\n📚 Usage Examples:")
    click.echo("  # Basic audio processing")
    click.echo("  python main.py audio.wav")
    click.echo("")
    click.echo("  # Video processing with multiple formats")
    click.echo("  python main.py video.mp4 --format json ass vtt")
    click.echo("")
    click.echo("  # YouTube video with enhanced format")
    click.echo("  python main.py 'https://youtube.com/watch?v=xyz' --use-enhanced-format")
    click.echo("")
    click.echo("  # YouTube embed URL support")
    click.echo("  python main.py 'https://youtube.com/embed/xyz' --format json backend_api")
    click.echo("")
    click.echo("  # With speech recognition and video method selection")
    click.echo("  python main.py video.mp4 --enable-stt --video-method parallel")
    click.echo("")
    click.echo("  # Backend/frontend API output")
    click.echo("  python main.py audio.wav --enable-backend-api --enable-frontend-json")


if __name__ == '__main__':
    # Use the main command as default
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # If first argument doesn't start with -, treat as input source
        main()
    else:
        # Otherwise show help or run subcommand
        cli()