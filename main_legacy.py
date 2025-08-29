#!/usr/bin/env python3
"""
Main CLI interface for the audio analysis pipeline.
"""

import sys
import os
import click
from pathlib import Path
from typing import Optional, List
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from src.audio_extractor import AudioExtractor
from src.speaker_diarizer import SpeakerDiarizer
from src.emotion_analyzer import EmotionAnalyzer
from src.acoustic_analyzer import AcousticAnalyzer
from src.speech_recognizer import SpeechRecognizer
from src.ass_exporter import ASSExporter
from src.result_merger import ResultMerger
from src.enhanced_result_merger import EnhancedResultMerger
from src.utils.logger import setup_logger, PerformanceLogger, ProgressLogger
from config.settings import Config, default_config


# Setup logging
logger = setup_logger(
    "audio_analysis",
    level="INFO",
    use_colors=True
)


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
    type=click.Choice(['json', 'csv', 'both']),
    default='json',
    help='Output format for results'
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
    format: str,
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
        
        # Use custom configuration
        python main.py input.mp4 --config config.yaml
    """
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set logging level
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Performance logger
    perf_logger = PerformanceLogger(logger)
    perf_logger.start_timer("total_processing")
    
    try:
        # Welcome message
        click.echo(click.style("🎵 Audio Analysis Pipeline", fg='cyan', bold=True))
        click.echo(click.style("=" * 50, fg='cyan'))
        
        # Load configuration
        if config:
            logger.info(f"Loading configuration from: {config}")
            pipeline_config = Config.from_yaml(Path(config))
        else:
            pipeline_config = Config.from_env()
        
        # Override config with CLI parameters
        pipeline_config.audio.sample_rate = sample_rate
        pipeline_config.audio.chunk_duration = segment_duration
        pipeline_config.audio.min_duration = min_segment_length
        pipeline_config.model.use_gpu = gpu
        pipeline_config.model.emotion_threshold = emotion_threshold
        pipeline_config.output.output_dir = Path(output)
        pipeline_config.output.output_format = [format] if format != 'both' else ['json', 'csv']
        pipeline_config.output.include_raw_features = export_features
        pipeline_config.output.generate_visualizations = visualize
        pipeline_config.processing.verbose = verbose
        
        # Validate configuration
        pipeline_config.validate()
        
        # Create output directory
        pipeline_config.output.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract audio
        click.echo("\n" + click.style("📥 Step 1: Extracting Audio", fg='yellow', bold=True))
        
        extractor = AudioExtractor(pipeline_config.audio)
        
        # Determine output audio path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"extracted_audio_{timestamp}.wav"
        audio_output_path = pipeline_config.output.output_dir / audio_filename
        
        # Extract audio
        try:
            extracted_audio = extractor.extract(
                input_source,
                output_path=audio_output_path
            )
            
            click.echo(click.style(f"✅ Audio extracted successfully: {extracted_audio}", fg='green'))
            
            # Get audio info
            from src.utils.audio_utils import get_audio_info
            audio_info = get_audio_info(extracted_audio)
            
            click.echo("\n" + click.style("📊 Audio Information:", fg='cyan'))
            click.echo(f"  • Duration: {audio_info['duration_formatted']}")
            click.echo(f"  • Sample Rate: {audio_info['sample_rate']} Hz")
            click.echo(f"  • Channels: {audio_info['channels']}")
            click.echo(f"  • File Size: {audio_info['file_size_mb']:.2f} MB")
            
        except Exception as e:
            click.echo(click.style(f"❌ Audio extraction failed: {e}", fg='red'))
            logger.error(f"Extraction error: {e}")
            sys.exit(1)
        
        # Step 2: Speaker Diarization
        click.echo("\n" + click.style("📈 Step 2: Speaker Diarization", fg='yellow', bold=True))
        speaker_results = None
        
        try:
            diarizer = SpeakerDiarizer(pipeline_config.model)
            
            # Get HuggingFace token from environment
            auth_token = os.getenv('HUGGINGFACE_TOKEN')
            if not auth_token:
                click.echo(click.style("❌ HuggingFace 토큰이 필요합니다!", fg='red', bold=True))
                click.echo("\n.env 파일에 HUGGINGFACE_TOKEN이 설정되어 있는지 확인하세요.")
                click.echo("토큰이 없다면:")
                click.echo("1. https://huggingface.co/settings/tokens 에서 토큰 발급")
                click.echo("2. .env 파일에 추가: HUGGINGFACE_TOKEN=hf_xxxx")
                sys.exit(1)
            
            diarizer.initialize(auth_token=auth_token)
            speaker_results = diarizer.analyze(
                extracted_audio,
                extract_embeddings=False
            )
            
            if speaker_results and speaker_results['segments']:
                click.echo(click.style(f"✅ Found {speaker_results['total_speakers']} speakers with {speaker_results['total_segments']} segments", fg='green'))
                
                # Show speaker distribution
                for speaker_id, speaker_info in speaker_results['speakers'].items():
                    percentage = speaker_info.get('speaking_percentage', 0)
                    click.echo(f"  • {speaker_id}: {percentage:.1f}% speaking time")
            else:
                click.echo(click.style("⚠️  No speakers detected or diarization unavailable", fg='yellow'))
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                click.echo(click.style("❌ HuggingFace 인증 실패!", fg='red', bold=True))
                click.echo("\n다음을 확인하세요:")
                click.echo("1. 토큰이 유효한지 확인: https://huggingface.co/settings/tokens")
                click.echo("2. 다음 모델들의 약관에 동의했는지 확인:")
                click.echo("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
                click.echo("   - https://huggingface.co/pyannote/segmentation-3.0")
                click.echo("   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
                click.echo("\n각 페이지에서 'Agree and access repository' 버튼을 클릭하세요.")
                sys.exit(1)
            elif require_all:
                click.echo(click.style(f"❌ Speaker diarization failed: {str(e)[:100]}", fg='red'))
                logger.error(f"Speaker diarization failed: {e}")
                click.echo(click.style("\n⚠️  Pipeline stopped due to critical error (--require-all enabled)", fg='red', bold=True))
                sys.exit(1)
            else:
                click.echo(click.style(f"⚠️  Speaker diarization skipped: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"Speaker diarization failed: {e}")
                click.echo(click.style("   Continuing with other analyses...", fg='yellow'))
        
        # Step 3: Emotion Analysis
        click.echo("\n" + click.style("🎭 Step 3: Emotion Analysis", fg='yellow', bold=True))
        emotion_predictions = None
        
        try:
            emotion_analyzer = EmotionAnalyzer(pipeline_config.model)
            emotion_analyzer.initialize()
            
            # Use speaker segments if available, otherwise analyze whole audio
            if speaker_results and speaker_results.get('segments'):
                emotion_predictions = emotion_analyzer.analyze_segments(
                    extracted_audio,
                    segments=speaker_results['segments']
                )
            else:
                emotion_predictions = emotion_analyzer.analyze_segments(
                    extracted_audio,
                    chunk_duration=pipeline_config.audio.chunk_duration
                )
            
            if emotion_predictions:
                # Get emotion statistics
                emotion_stats = emotion_analyzer.get_emotion_statistics(emotion_predictions)
                click.echo(click.style(f"✅ Analyzed {len(emotion_predictions)} segments for emotions", fg='green'))
                click.echo(f"  • Dominant emotion: {emotion_stats['dominant_emotion']}")
                
                # Show emotion distribution
                if emotion_stats.get('emotion_distribution'):
                    for emotion, percentage in emotion_stats['emotion_distribution'].items():
                        click.echo(f"  • {emotion}: {percentage:.1f}%")
            else:
                click.echo(click.style("⚠️  No emotions detected", fg='yellow'))
                
        except Exception as e:
            if require_all:
                click.echo(click.style(f"❌ Emotion analysis failed: {str(e)[:100]}", fg='red'))
                logger.error(f"Emotion analysis failed: {e}")
                click.echo(click.style("\n⚠️  Pipeline stopped due to critical error (--require-all enabled)", fg='red', bold=True))
                sys.exit(1)
            else:
                click.echo(click.style(f"⚠️  Emotion analysis skipped: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"Emotion analysis failed: {e}")
                click.echo(click.style("   Continuing with other analyses...", fg='yellow'))
        
        # Step 4: Acoustic Analysis
        click.echo("\n" + click.style("🔊 Step 4: Acoustic Analysis", fg='yellow', bold=True))
        acoustic_features = None
        
        try:
            acoustic_analyzer = AcousticAnalyzer()
            
            # Analyze acoustic features
            acoustic_features = acoustic_analyzer.analyze(extracted_audio)
            
            if acoustic_features:
                click.echo(click.style(f"✅ Extracted acoustic features from {len(acoustic_features)} segments", fg='green'))
                
                # Show some sample features
                if acoustic_features[0]:
                    sample = acoustic_features[0]
                    click.echo("  Sample features from first segment:")
                    click.echo(f"    • RMS Energy: {sample.get('rms_energy', 0):.4f}")
                    click.echo(f"    • Spectral Centroid: {sample.get('spectral_centroid', 0):.1f} Hz")
                    if sample.get('pitch_mean'):
                        click.echo(f"    • Mean Pitch: {sample['pitch_mean']:.1f} Hz")
            else:
                click.echo(click.style("⚠️  No acoustic features extracted", fg='yellow'))
                
        except Exception as e:
            if require_all:
                click.echo(click.style(f"❌ Acoustic analysis failed: {str(e)[:100]}", fg='red'))
                logger.error(f"Acoustic analysis failed: {e}")
                click.echo(click.style("\n⚠️  Pipeline stopped due to critical error (--require-all enabled)", fg='red', bold=True))
                sys.exit(1)
            else:
                click.echo(click.style(f"⚠️  Acoustic analysis skipped: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"Acoustic analysis failed: {e}")
                click.echo(click.style("   Continuing with other analyses...", fg='yellow'))
        
        # Step 5: Speech Recognition (if enabled)
        transcription_results = None
        if enable_stt:
            click.echo("\n" + click.style("🎤 Step 5: Speech Recognition", fg='yellow', bold=True))
            
            try:
                recognizer = SpeechRecognizer(
                    model_name="base",
                    whisper_config=pipeline_config.whisper
                )
                recognizer.initialize()
                
                # Use speaker segments if available
                segments_for_stt = None
                if speaker_results and speaker_results.get('segments'):
                    segments_for_stt = speaker_results['segments']
                
                # Detect language if not specified
                if stt_language == 'auto' or stt_language is None:
                    detected_lang = recognizer.detect_language(extracted_audio)
                    if detected_lang:
                        click.echo(f"  • Detected language: {detected_lang}")
                        stt_language = detected_lang
                
                # Transcribe
                transcription_results = recognizer.transcribe(
                    extracted_audio,
                    segments=segments_for_stt,
                    language=stt_language,
                    verbose=verbose
                )
                
                if transcription_results:
                    click.echo(click.style(f"✅ Transcribed {len(transcription_results)} segments", fg='green'))
                    # Show sample
                    if transcription_results[0]:
                        sample = transcription_results[0]
                        sample_dict = sample if isinstance(sample, dict) else sample.to_dict()
                        click.echo(f"  • Sample: \"{sample_dict.get('text', '')[:50]}...\"")
                else:
                    click.echo(click.style("⚠️  No transcriptions generated", fg='yellow'))
                    
            except Exception as e:
                click.echo(click.style(f"⚠️  Speech recognition skipped: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"Speech recognition failed: {e}")
        
        # Step 6: Merge Results
        click.echo("\n" + click.style("🔄 Step 6: Merging Results", fg='yellow', bold=True))
        
        # Use enhanced format if requested
        if use_enhanced_format:
            merger = EnhancedResultMerger()
            click.echo(click.style("📊 Using enhanced output format with linguistic precision", fg='cyan'))
        else:
            merger = ResultMerger()
        
        # Prepare metadata
        metadata = {
            "input_source": str(input_source),
            "processing_timestamp": datetime.now().isoformat(),
            "audio_file": str(extracted_audio),
            "audio_duration": audio_info['duration_seconds'],
            "sample_rate": audio_info['sample_rate'],
            "file_size_mb": audio_info['file_size_mb'],
            "configuration": {
                "sample_rate": pipeline_config.audio.sample_rate,
                "gpu_enabled": pipeline_config.model.use_gpu,
                "segment_duration": pipeline_config.audio.chunk_duration,
                "emotion_threshold": pipeline_config.model.emotion_threshold
            }
        }
        
        # Merge all results
        results = merger.merge_all_results(
            speaker_results=speaker_results,
            emotion_results=emotion_predictions,
            acoustic_results=acoustic_features,
            transcription_results=transcription_results,
            metadata=metadata
        )
        
        click.echo(click.style("✅ Results merged successfully", fg='green'))
        
        # Save Results
        click.echo("\n" + click.style("💾 Saving Results", fg='yellow', bold=True))
        
        # Save results
        if use_enhanced_format:
            # Export enhanced format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = pipeline_config.output.output_dir / f"enhanced_{timestamp}.json"
            merger.export_to_json(results, json_path)
            exported_files = {'json': json_path}
            
            # Also export WebVTT for web compatibility
            vtt_path = pipeline_config.output.output_dir / f"subtitles_{timestamp}.vtt"
            merger.export_to_webvtt(results, vtt_path)
            exported_files['webvtt'] = vtt_path
        else:
            exported_files = merger.export_timeline_segments(
                results,
                pipeline_config.output.output_dir,
                format='both' if 'both' in pipeline_config.output.output_format else pipeline_config.output.output_format[0]
            )
        
        for format_type, file_path in exported_files.items():
            click.echo(click.style(f"✅ {format_type.upper()} results saved: {file_path}", fg='green'))
        
        # Generate summary report
        if not use_enhanced_format:
            report_path = pipeline_config.output.output_dir / f"report_{timestamp}.txt"
            report = merger.generate_summary_report(results, report_path)
            click.echo(click.style(f"✅ Summary report saved: {report_path}", fg='green'))
        else:
            # Enhanced format includes statistics in JSON
            click.echo(click.style("✅ Statistics included in enhanced JSON output", fg='green'))
        
        # Export ASS subtitle file if requested
        if export_ass:
            click.echo("\n" + click.style("📝 Exporting ASS Subtitle", fg='yellow', bold=True))
            try:
                ass_exporter = ASSExporter()
                ass_path = pipeline_config.output.output_dir / f"subtitles_{timestamp}.ass"
                
                ass_file = ass_exporter.export_to_ass(
                    results,
                    ass_path,
                    use_emotions=True,
                    use_speaker_colors=True
                )
                
                click.echo(click.style(f"✅ ASS subtitle saved: {ass_file}", fg='green'))
                
                # If no transcriptions, remind user to enable STT
                if not transcription_results:
                    click.echo(click.style("💡 Tip: Use --enable-stt to add actual text to subtitles", fg='cyan'))
                    
            except Exception as e:
                click.echo(click.style(f"⚠️  ASS export failed: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"ASS export failed: {e}")
        
        # Generate visualizations if requested
        if visualize:
            click.echo("\n" + click.style("📈 Generating Visualizations", fg='yellow', bold=True))
            try:
                from src.utils.visualization import create_visualization_report
                from src.utils.audio_utils import load_audio
                
                # Load audio for visualization
                audio_data, sr = load_audio(extracted_audio, sample_rate=pipeline_config.audio.sample_rate)
                
                viz_dir = pipeline_config.output.output_dir / "visualizations"
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
            except Exception as e:
                click.echo(click.style(f"⚠️  Visualization generation failed: {str(e)[:100]}", fg='yellow'))
                logger.warning(f"Visualization failed: {e}")
        
        # Show brief summary
        if results.get('summary'):
            summary = results['summary']
            click.echo("\n" + click.style("📊 Analysis Summary:", fg='cyan', bold=True))
            if 'total_speakers' in summary:
                click.echo(f"  • Speakers detected: {summary['total_speakers']}")
            if 'dominant_emotion' in summary:
                click.echo(f"  • Dominant emotion: {summary['dominant_emotion']}")
            if 'emotion_changes' in summary:
                click.echo(f"  • Emotion changes: {summary['emotion_changes']}")
        
        # Performance summary
        total_time = perf_logger.stop_timer("total_processing")
        perf_logger.log_memory_usage()
        
        click.echo("\n" + click.style("=" * 50, fg='cyan'))
        click.echo(click.style(f"✨ Processing completed in {total_time:.2f} seconds", fg='green', bold=True))
        
        # Cleanup
        if not debug:
            extractor.cleanup_temp_files()
        
    except KeyboardInterrupt:
        click.echo("\n" + click.style("⚠️  Processing interrupted by user", fg='yellow'))
        sys.exit(130)
    
    except Exception as e:
        click.echo(click.style(f"❌ Fatal error: {e}", fg='red'))
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


@click.group()
def cli():
    """Audio Analysis Pipeline CLI."""
    pass


@cli.command()
def version():
    """Show version information."""
    click.echo("Audio Analysis Pipeline v1.0.0")
    click.echo("Python " + sys.version)


@cli.command()
def formats():
    """Show supported formats."""
    from src.audio_extractor import AudioExtractor
    
    extractor = AudioExtractor()
    formats = extractor.get_supported_formats()
    
    click.echo(click.style("🎵 Supported Formats", fg='cyan', bold=True))
    click.echo(click.style("=" * 40, fg='cyan'))
    
    click.echo("\n📥 Input Audio Formats:")
    for fmt in formats['input_audio']:
        click.echo(f"  • {fmt}")
    
    click.echo("\n📹 Input Video Formats:")
    for fmt in formats['input_video']:
        click.echo(f"  • {fmt}")
    
    click.echo("\n📤 Output Formats:")
    for fmt in formats['output']:
        click.echo(f"  • {fmt}")
    
    click.echo("\n🌐 Supported Sources:")
    for src in formats['sources']:
        click.echo(f"  • {src}")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml', help='Output path for config file')
def generate_config(output):
    """Generate a sample configuration file."""
    config = Config()
    config.to_yaml(Path(output))
    click.echo(click.style(f"✅ Configuration file generated: {output}", fg='green'))
    click.echo("Edit this file to customize your pipeline settings.")


if __name__ == '__main__':
    # Use the main command as default
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # If first argument doesn't start with -, treat as input source
        main()
    else:
        # Otherwise show help or run subcommand
        cli()