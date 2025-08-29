"""
Individual pipeline step implementations.
"""

import os
import time
from typing import Optional

import click

from .base import PipelineStep, PipelineContext, AuthenticationError
from ..audio_extractor import AudioExtractor
from ..speaker_diarizer import SpeakerDiarizer
from ..emotion_analyzer import EmotionAnalyzer
from ..acoustic_analyzer import AcousticAnalyzer
from ..speech_recognizer import SpeechRecognizer
from ..utils.audio_utils import get_audio_info


class AudioExtractionStep(PipelineStep):
    """Extract audio from various input sources."""
    
    def execute(self, context: PipelineContext) -> bool:
        """Extract audio from input source."""
        click.echo(click.style(f"📥 Step 1: {self.get_step_display_name()}", fg='yellow', bold=True))
        
        start_time = time.time()
        
        try:
            extractor = AudioExtractor(self.config.audio)
            
            # Generate output path
            audio_output_path = context.get_output_path("extracted_audio", "wav")
            
            # Extract audio (enhanced with video metadata)
            if extractor._is_video_file(context.input_source) or extractor._is_url(context.input_source):
                # For video files or URLs, get enhanced metadata
                if extractor._is_video_file(context.input_source):
                    extracted_audio, video_metadata = extractor.extract_from_video(
                        context.input_source,
                        output_path=audio_output_path,
                        method=self.config.video.extraction_method
                    )
                    context.video_metadata = video_metadata
                else:
                    # URL extraction (including YouTube embed support)
                    extracted_audio = extractor.extract(
                        context.input_source,
                        output_path=audio_output_path
                    )
            else:
                # Standard audio extraction
                extracted_audio = extractor.extract(
                    context.input_source,
                    output_path=audio_output_path
                )
            
            # Store results in context
            context.audio_file = extracted_audio
            context.audio_info = get_audio_info(extracted_audio)
            
            # Display success message
            click.echo(click.style(f"✅ Audio extracted successfully: {extracted_audio}", fg='green'))
            
            # Show audio info
            click.echo(click.style("📊 Audio Information:", fg='cyan'))
            click.echo(f"  • Duration: {context.audio_info['duration_formatted']}")
            click.echo(f"  • Sample Rate: {context.audio_info['sample_rate']} Hz")
            click.echo(f"  • Channels: {context.audio_info['channels']}")
            click.echo(f"  • File Size: {context.audio_info['file_size_mb']:.2f} MB")
            
            # Show video info if available
            if hasattr(context, 'video_metadata') and context.video_metadata:
                click.echo(click.style("🎬 Video Information:", fg='cyan'))
                click.echo(f"  • Resolution: {context.video_metadata.get('resolution', 'unknown')}")
                click.echo(f"  • Video Codec: {context.video_metadata.get('video_codec', 'unknown')}")
                click.echo(f"  • Format: {context.video_metadata.get('format', 'unknown')}")
                if context.video_metadata.get('fps', 0) > 0:
                    click.echo(f"  • FPS: {context.video_metadata['fps']:.2f}")
            
            duration = time.time() - start_time
            context.mark_step_completed(self.name, duration)
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"❌ Audio extraction failed: {e}", fg='red'))
            return self.handle_error(context, e)


class SpeakerDiarizationStep(PipelineStep):
    """Perform speaker diarization on extracted audio."""
    
    def should_execute(self, context: PipelineContext) -> bool:
        """Only execute if audio file is available."""
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        """Perform speaker diarization."""
        click.echo(f"\n{click.style(f'📈 Step 2: {self.get_step_display_name()}', fg='yellow', bold=True)}")
        
        start_time = time.time()
        
        try:
            diarizer = SpeakerDiarizer(self.config.model)
            
            # Get HuggingFace token
            auth_token = os.getenv('HUGGINGFACE_TOKEN')
            if not auth_token:
                raise AuthenticationError("HUGGINGFACE_TOKEN not found in environment")
            
            # Initialize and analyze
            diarizer.initialize(auth_token=auth_token)
            
            if not context.audio_file:
                raise ValueError("No audio file available for speaker diarization")
                
            speaker_results = diarizer.analyze(
                context.audio_file,
                extract_embeddings=False
            )
            
            # Store results
            context.speaker_results = speaker_results
            
            # Display results
            if speaker_results and speaker_results['segments']:
                click.echo(click.style(
                    f"✅ Found {speaker_results['total_speakers']} speakers with {speaker_results['total_segments']} segments", 
                    fg='green'
                ))
                
                # Show speaker distribution
                for speaker_id, speaker_info in speaker_results['speakers'].items():
                    percentage = speaker_info.get('speaking_percentage', 0)
                    click.echo(f"  • {speaker_id}: {percentage:.1f}% speaking time")
            else:
                click.echo(click.style("⚠️  No speakers detected or diarization unavailable", fg='yellow'))
            
            duration = time.time() - start_time
            context.mark_step_completed(self.name, duration)
            
            return True
            
        except AuthenticationError as e:
            click.echo(click.style("❌ HuggingFace 인증 실패!", fg='red', bold=True))
            click.echo("\n다음을 확인하세요:")
            click.echo("1. .env 파일에 HUGGINGFACE_TOKEN이 설정되어 있는지 확인")
            click.echo("2. 토큰이 유효한지 확인: https://huggingface.co/settings/tokens")
            click.echo("3. 다음 모델들의 약관에 동의했는지 확인:")
            click.echo("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
            click.echo("   - https://huggingface.co/pyannote/segmentation-3.0")
            click.echo("   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
            return self.handle_error(context, e)
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                click.echo(click.style("❌ HuggingFace 인증 실패!", fg='red', bold=True))
                click.echo("각 모델 페이지에서 'Agree and access repository' 버튼을 클릭하세요.")
            else:
                click.echo(click.style(f"⚠️  Speaker diarization failed: {str(e)[:100]}", fg='yellow'))
            
            return self.handle_error(context, e)


class EmotionAnalysisStep(PipelineStep):
    """Analyze emotions in audio segments."""
    
    def should_execute(self, context: PipelineContext) -> bool:
        """Only execute if audio file is available."""
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        """Perform emotion analysis."""
        click.echo(f"\n{click.style(f'🎭 Step 3: {self.get_step_display_name()}', fg='yellow', bold=True)}")
        
        start_time = time.time()
        
        try:
            emotion_analyzer = EmotionAnalyzer(self.config.model)
            emotion_analyzer.initialize()
            
            if not context.audio_file:
                raise ValueError("No audio file available for emotion analysis")
            
            # Use speaker segments if available, otherwise analyze whole audio
            if context.speaker_results and context.speaker_results.get('segments'):
                emotion_predictions = emotion_analyzer.analyze_segments(
                    context.audio_file,
                    segments=context.speaker_results['segments']
                )
            else:
                emotion_predictions = emotion_analyzer.analyze_segments(
                    context.audio_file,
                    chunk_duration=self.config.audio.chunk_duration
                )
            
            # Store results
            context.emotion_results = emotion_predictions
            
            # Display results
            if emotion_predictions:
                emotion_stats = emotion_analyzer.get_emotion_statistics(emotion_predictions)
                click.echo(click.style(f"✅ Analyzed {len(emotion_predictions)} segments for emotions", fg='green'))
                click.echo(f"  • Dominant emotion: {emotion_stats['dominant_emotion']}")
                
                # Show emotion distribution
                if emotion_stats.get('emotion_distribution'):
                    for emotion, percentage in emotion_stats['emotion_distribution'].items():
                        click.echo(f"  • {emotion}: {percentage:.1f}%")
            else:
                click.echo(click.style("⚠️  No emotions detected", fg='yellow'))
            
            duration = time.time() - start_time
            context.mark_step_completed(self.name, duration)
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"⚠️  Emotion analysis failed: {str(e)[:100]}", fg='yellow'))
            return self.handle_error(context, e)


class AcousticAnalysisStep(PipelineStep):
    """Extract acoustic features from audio."""
    
    def should_execute(self, context: PipelineContext) -> bool:
        """Only execute if audio file is available."""
        return context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        """Perform acoustic analysis."""
        click.echo(f"\n{click.style(f'🔊 Step 4: {self.get_step_display_name()}', fg='yellow', bold=True)}")
        
        start_time = time.time()
        
        try:
            acoustic_analyzer = AcousticAnalyzer()
            
            if not context.audio_file:
                raise ValueError("No audio file available for acoustic analysis")
                
            acoustic_features = acoustic_analyzer.analyze(context.audio_file)
            
            # Store results
            context.acoustic_results = acoustic_features
            
            # Display results
            if acoustic_features:
                click.echo(click.style(f"✅ Extracted acoustic features from {len(acoustic_features)} segments", fg='green'))
                
                # Show sample features
                if acoustic_features[0]:
                    sample = acoustic_features[0]
                    click.echo("  Sample features from first segment:")
                    click.echo(f"    • RMS Energy: {sample.rms_energy or 0:.4f}")
                    click.echo(f"    • Spectral Centroid: {sample.spectral_centroid or 0:.1f} Hz")
                    if sample.pitch_mean:
                        click.echo(f"    • Mean Pitch: {sample.pitch_mean:.1f} Hz")
            else:
                click.echo(click.style("⚠️  No acoustic features extracted", fg='yellow'))
            
            duration = time.time() - start_time
            context.mark_step_completed(self.name, duration)
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"⚠️  Acoustic analysis failed: {str(e)[:100]}", fg='yellow'))
            return self.handle_error(context, e)


class SpeechRecognitionStep(PipelineStep):
    """Convert speech to text."""
    
    def __init__(self, config, enable_stt: bool = False, stt_language: Optional[str] = None):
        super().__init__(config)
        self.enable_stt = enable_stt
        self.stt_language = stt_language
    
    def should_execute(self, context: PipelineContext) -> bool:
        """Only execute if STT is enabled and audio file is available."""
        return self.enable_stt and context.has_audio_file()
    
    def execute(self, context: PipelineContext) -> bool:
        """Perform speech recognition."""
        click.echo(f"\n{click.style(f'🎤 Step 5: {self.get_step_display_name()}', fg='yellow', bold=True)}")
        
        start_time = time.time()
        
        try:
            recognizer = SpeechRecognizer(
                model_name="base",
                whisper_config=self.config.whisper
            )
            recognizer.initialize()
            
            # Use speaker segments if available
            segments_for_stt = None
            if context.speaker_results and context.speaker_results.get('segments'):
                segments_for_stt = context.speaker_results['segments']
            
            if not context.audio_file:
                raise ValueError("No audio file available for speech recognition")
            
            # Detect language if not specified
            if self.stt_language == 'auto' or self.stt_language is None:
                detected_lang = recognizer.detect_language(context.audio_file)
                if detected_lang:
                    click.echo(f"  • Detected language: {detected_lang}")
                    self.stt_language = detected_lang
            
            # Transcribe
            transcription_results = recognizer.transcribe(
                context.audio_file,
                segments=segments_for_stt,
                language=self.stt_language,
                verbose=self.config.processing.verbose
            )
            
            # Store results
            context.transcription_results = transcription_results
            
            # Display results
            if transcription_results:
                click.echo(click.style(f"✅ Transcribed {len(transcription_results)} segments", fg='green'))
                # Show sample
                if transcription_results[0]:
                    sample = transcription_results[0]
                    sample_dict = sample if isinstance(sample, dict) else sample.to_dict()
                    click.echo(f"  • Sample: \"{sample_dict.get('text', '')[:50]}...\"")
            else:
                click.echo(click.style("⚠️  No transcriptions generated", fg='yellow'))
            
            # Display performance stats if enabled
            if hasattr(recognizer, 'get_performance_stats'):
                perf_stats = recognizer.get_performance_stats()
                if perf_stats and self.config.whisper.enable_performance_monitoring:
                    click.echo(f"  📊 Whisper enhancements:")
                    if perf_stats.get('enhanced_language_detection', {}).get('enabled'):
                        lang_stats = perf_stats['enhanced_language_detection']
                        click.echo(f"    • Enhanced language detection: {lang_stats['max_samples']} samples")
                        if lang_stats.get('cached_languages', 0) > 0:
                            click.echo(f"    • Language cache: {lang_stats['cached_languages']} entries")
                    if perf_stats.get('vad', {}).get('vad_enabled'):
                        vad_stats = perf_stats['vad']
                        click.echo(f"    • Voice Activity Detection: {vad_stats.get('vad_mode', 'unknown')} mode")
                    if perf_stats.get('anomaly_detection', {}).get('enabled'):
                        click.echo(f"    • Textual anomaly detection: enabled")
            
            duration = time.time() - start_time
            context.mark_step_completed(self.name, duration)
            
            return True
            
        except Exception as e:
            click.echo(click.style(f"⚠️  Speech recognition failed: {str(e)[:100]}", fg='yellow'))
            return self.handle_error(context, e)