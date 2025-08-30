"""
Speech recognition module using OpenAI Whisper for transcription.
Enhanced with performance-optimized language detection, VAD preprocessing, and anomaly handling.
"""

import logging
import hashlib
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

# Import new models for compatibility  
from src.models.segments import TranscriptionSegment
from collections import defaultdict

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger
from src.utils.lazy_loader import LazyModelLoader, model_registry
from src.utils.cleanup_manager import cleanup_manager, register_for_cleanup
from src.vad_processor import VADProcessor, SpeechSegment
from config.settings import WhisperConfig

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


# Using unified TranscriptionSegment from src.models.segments


class SpeechRecognizer:
    """Enhanced speech recognition using Whisper with performance optimizations."""
    
    def __init__(
        self, 
        model_name: str = "base", 
        device: Optional[str] = None,
        whisper_config: Optional[WhisperConfig] = None
    ):
        """Initialize speech recognizer.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, or None for auto)
            whisper_config: Configuration for performance optimizations
        """
        self.model_name = model_name
        self.device = device
        
        # Enhanced features
        self.config = whisper_config or WhisperConfig()
        self.vad_processor = VADProcessor(self.config) if self.config.enable_vad else None
        self._language_cache = {}
        
        # Set up lazy loader for Whisper model
        self._model_loader = LazyModelLoader(
            loader_func=self._load_whisper_model,
            model_name="whisper_model"
        )
        
        # Register with global registry
        model_registry.register("whisper_model", self._load_whisper_model)
        
        # Register for cleanup tracking
        register_for_cleanup(self, self.cleanup)
        
        self._setup_cache_directory()
        
        logger.info(f"SpeechRecognizer initialized with lazy loading (model: {model_name})")
    
    def _setup_cache_directory(self):
        """Setup language detection cache directory."""
        if not self.config.enable_language_caching:
            return
            
        cache_dir = self.config.language_cache_dir or Path.home() / ".cache" / "whisper_lang_detection"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / "language_cache.json"
        
        # Load existing cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._language_cache = json.load(f)
                logger.debug(f"Loaded {len(self._language_cache)} cached language detections")
            except Exception as e:
                logger.warning(f"Could not load language cache: {e}")
                self._language_cache = {}
    
    def _save_language_cache(self):
        """Save language detection cache to disk."""
        if not self.config.enable_language_caching or not hasattr(self, 'cache_file'):
            return
            
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._language_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save language cache: {e}")
    
    def _get_audio_hash(self, audio_path: Path) -> str:
        """Get hash of audio file for caching.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Hash string
        """
        try:
            # Use file size and modification time for quick hash
            stat = audio_path.stat()
            hash_input = f"{audio_path.name}_{stat.st_size}_{stat.st_mtime}".encode()
            return hashlib.md5(hash_input).hexdigest()[:16]
        except Exception:
            return str(audio_path)
        
    def _load_whisper_model(self):
        """Load the Whisper model.
        
        Returns:
            Loaded Whisper model or None
        """
        perf_logger.start_timer("whisper_model_loading")
        
        try:
            import whisper
            import torch
            
            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper model '{self.model_name}' on {self.device}")
            
            # Load model
            model = whisper.load_model(self.model_name, device=self.device)
            
            # Initialize VAD processor if enabled
            if self.vad_processor:
                self.vad_processor.initialize()
            
            duration = perf_logger.stop_timer("whisper_model_loading")
            logger.info(f"Whisper model loaded in {duration:.2f}s")
            
            return model
            
        except ImportError:
            perf_logger.stop_timer("whisper_model_loading")
            logger.warning("Whisper not installed. Using fallback mode.")
            logger.info("Install with: pip install openai-whisper")
            return None
            
        except Exception as e:
            perf_logger.stop_timer("whisper_model_loading")
            logger.error(f"Failed to load Whisper model: {e}")
            return None
    
    @property
    def model(self):
        """Lazily loaded Whisper model."""
        return self._model_loader.load()
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        logger.info("Unloading Whisper model")
        self._model_loader.unload()
        model_registry.unload("whisper_model")
    
    def cleanup(self) -> None:
        """Comprehensive cleanup of all resources."""
        logger.info("Performing comprehensive cleanup of SpeechRecognizer")
        
        # Unload model
        self.unload_model()
        
        # Save language cache before cleanup
        self._save_language_cache()
        
        # Clean up VAD processor if it exists
        if self.vad_processor:
            try:
                if hasattr(self.vad_processor, 'cleanup'):
                    self.vad_processor.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup VAD processor: {e}")
        
        # Check for memory pressure and trigger cleanup if needed
        if cleanup_manager.check_memory_pressure():
            cleanup_manager.periodic_cleanup()
        
        logger.info("SpeechRecognizer cleanup completed")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        segments: Optional[List[Dict]] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False
    ) -> List[TranscriptionSegment]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            segments: Optional segments to transcribe separately
            language: Language code (e.g., 'ko', 'en') or None for auto-detect
            task: Task to perform ('transcribe' or 'translate')
            verbose: Show progress
            
        Returns:
            List of transcription segments
        """
        perf_logger.start_timer("speech_recognition")
        audio_path = Path(audio_path)
        
        try:
            if self.model is None:
                logger.warning("Using fallback transcription (demo mode)")
                return self._fallback_transcription(audio_path, segments)
            
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Load audio
            audio_data, sample_rate = load_audio(audio_path, sample_rate=16000)
            
            # Use VAD preprocessing if enabled and no segments provided
            if not segments and self.vad_processor and self.config.enable_vad:
                speech_segments = self.vad_processor.detect_speech_segments(audio_path)
                if speech_segments:
                    # Convert VAD segments to transcription format
                    segments = [
                        {
                            'start': seg.start,
                            'end': seg.end,
                            'speaker': f'speaker_vad_{i+1}'
                        }
                        for i, seg in enumerate(speech_segments)
                    ]
                    if verbose:
                        logger.info(f"VAD found {len(segments)} speech segments")
            
            transcriptions = []
            
            if segments:
                # Transcribe each segment separately
                progress = ProgressLogger(logger, total=len(segments))
                
                for i, segment in enumerate(segments):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', len(audio_data) / sample_rate)
                    speaker = segment.get('speaker', f'speaker_{i+1}')
                    
                    # Extract segment audio
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        # Transcribe segment
                        result = self._transcribe_segment(
                            segment_audio,
                            sample_rate,
                            language=language,
                            task=task,
                            verbose=verbose
                        )
                        
                        if result:
                            # Add timing offset
                            for trans in result:
                                trans.start += start_time
                                trans.end += start_time
                                trans.speaker_id = speaker
                                transcriptions.append(trans)
                    
                    progress.update(message=f"Transcribed segment {i+1}/{len(segments)}")
                
                progress.complete()
                
            else:
                # Transcribe entire audio
                result = self._transcribe_segment(
                    audio_data,
                    sample_rate,
                    language=language,
                    task=task,
                    verbose=verbose
                )
                transcriptions = result
            
            duration = perf_logger.stop_timer("speech_recognition")
            logger.info(f"Transcription completed in {duration:.2f}s")
            
            return transcriptions
            
        except Exception as e:
            perf_logger.stop_timer("speech_recognition")
            logger.error(f"Transcription failed: {e}")
            return self._fallback_transcription(audio_path, segments)
    
    def _transcribe_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False
    ) -> List[TranscriptionSegment]:
        """Transcribe a single audio segment.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            language: Language code or None
            task: Task to perform
            verbose: Show progress
            
        Returns:
            List of transcription segments
        """
        import whisper
        
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Transcribe with null check
            if self.model is None:
                raise RuntimeError("Model not initialized")
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                verbose=verbose,
                word_timestamps=True
            )
            
            # Extract segments
            transcriptions = []
            
            if 'segments' in result:
                for seg in result['segments']:
                    # Ensure seg is a dictionary (Whisper returns dict segments)
                    if not isinstance(seg, dict):
                        continue
                    
                    # Safe access with type checking
                    start_val = seg.get('start', 0)
                    start_time = float(start_val) if isinstance(start_val, (int, float)) else 0.0
                    end_val = seg.get('end', 0)
                    end_time = float(end_val) if isinstance(end_val, (int, float)) else 0.0
                    text_content = seg.get('text', '')
                    if isinstance(text_content, list):
                        text_content = ' '.join(str(t) for t in text_content)
                    elif not isinstance(text_content, str):
                        text_content = str(text_content)
                    
                    detected_lang = result.get('language', language)
                    if isinstance(detected_lang, list):
                        detected_lang = detected_lang[0] if detected_lang else language
                    elif not isinstance(detected_lang, str):
                        detected_lang = language or 'en'
                    
                    trans = TranscriptionSegment(
                        start=start_time,
                        end=end_time,
                        text=text_content.strip() if hasattr(text_content, 'strip') else str(text_content),
                        language=detected_lang,
                        confidence=seg.get('confidence', 0.0)
                    )
                    transcriptions.append(trans)
            else:
                # Single segment with safe type handling
                text_content = result.get('text', '')
                if isinstance(text_content, list):
                    text_content = ' '.join(str(t) for t in text_content)
                elif not isinstance(text_content, str):
                    text_content = str(text_content)
                    
                detected_lang = result.get('language', language)
                if isinstance(detected_lang, list):
                    detected_lang = detected_lang[0] if detected_lang else language
                elif not isinstance(detected_lang, str):
                    detected_lang = language or 'en'
                    
                trans = TranscriptionSegment(
                    start=0.0,
                    end=len(audio) / sample_rate,
                    text=text_content.strip() if hasattr(text_content, 'strip') else str(text_content),
                    language=detected_lang
                )
                transcriptions.append(trans)
            
            # Apply anomaly detection and retry if needed
            if self.config.enable_anomaly_detection:
                transcriptions = self._handle_textual_anomalies(
                    transcriptions, audio, sample_rate, language, task, verbose
                )
            
            return transcriptions
            
        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            return []
    
    def _fallback_transcription(
        self,
        audio_path: Path,
        segments: Optional[List[Dict]] = None
    ) -> List[TranscriptionSegment]:
        """Fallback transcription for demo purposes.
        
        Args:
            audio_path: Path to audio file
            segments: Optional segments
            
        Returns:
            Demo transcription segments
        """
        logger.info("Generating demo transcriptions")
        
        transcriptions = []
        
        if segments:
            for i, segment in enumerate(segments):
                # Ensure segment is a dictionary
                if not isinstance(segment, dict):
                    segment = {'start': i * 3.0, 'end': (i + 1) * 3.0, 'speaker': f'speaker_{i+1}'}
                
                trans = TranscriptionSegment(
                    start=segment.get('start', 0),
                    end=segment.get('end', 0),
                    text=f"[화자 {i+1} 발화 내용]",
                    language='ko',
                    confidence=0.95,
                    speaker_id=segment.get('speaker', f'speaker_{i+1}')
                )
                transcriptions.append(trans)
        else:
            # Create demo segments
            duration = 10.0  # Assume 10 seconds
            for i in range(3):
                start = i * 3.0
                end = min((i + 1) * 3.0, duration)
                trans = TranscriptionSegment(
                    start=start,
                    end=end,
                    text=f"[세그먼트 {i+1} 텍스트]",
                    language='ko',
                    confidence=0.95,
                    speaker_id=f'speaker_1'
                )
                transcriptions.append(trans)
        
        return transcriptions
    
    def detect_language(self, audio_path: Union[str, Path]) -> Optional[str]:
        """Enhanced language detection with multiple samples and caching.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language code or None
        """
        if self.model is None:
            return None
        
        audio_path = Path(audio_path)
        
        # Check cache first
        if self.config.enable_language_caching:
            audio_hash = self._get_audio_hash(audio_path)
            if audio_hash in self._language_cache:
                cached_result = self._language_cache[audio_hash]
                logger.debug(f"Using cached language detection: {cached_result['language']}")
                return cached_result['language']
        
        perf_logger.start_timer("enhanced_language_detection")
        
        try:
            import whisper
            
            logger.info(f"Enhanced language detection for: {audio_path}")
            
            # Load full audio to determine sampling strategy
            audio_data, sr = load_audio(audio_path, sample_rate=16000)
            duration = len(audio_data) / sr
            
            # Adaptive sampling based on duration
            if duration <= 30:
                # Short audio: single sample (optimization)
                num_samples = 1
                logger.debug("Short audio: using single sample")
            elif duration <= 120:
                # Medium audio: reduced samples
                num_samples = min(self.config.max_language_samples, 2)
            else:
                # Long audio: full sampling
                num_samples = self.config.max_language_samples
            
            probabilities_map = defaultdict(list)
            
            for i in range(num_samples):
                # Strategic sampling: beginning, middle, end + random
                if num_samples == 1:
                    fragment_audio = self._extract_language_sample(audio_data, sr, 0, duration)
                else:
                    if i == 0:
                        # Beginning
                        start_time = 0
                    elif i == 1 and num_samples > 2:
                        # Middle
                        start_time = duration / 2
                    elif i == num_samples - 1:
                        # End
                        start_time = max(0, duration - 30)
                    else:
                        # Random position
                        start_time = random.uniform(0, max(0, duration - 30))
                    
                    fragment_audio = self._extract_language_sample(audio_data, sr, start_time, duration)
                
                if len(fragment_audio) > 0:
                    # Detect language for this fragment
                    probs = self._detect_language_fragment(fragment_audio)
                    
                    if probs:
                        for lang_key in probs:
                            probabilities_map[lang_key].append(probs[lang_key])
                        
                        # Early exit optimization: if confidence is very high, stop sampling
                        max_prob = max(probs.values())
                        if max_prob >= self.config.language_confidence_threshold and i >= 1:
                            logger.debug(f"Early exit: high confidence {max_prob:.2f}")
                            break
            
            if not probabilities_map:
                logger.warning("No language probabilities detected")
                return None
            
            # Calculate average probabilities
            avg_probabilities = {}
            for lang_key in probabilities_map:
                avg_probabilities[lang_key] = sum(probabilities_map[lang_key]) / len(probabilities_map[lang_key])
            
            # Get most confident language with proper type checking
            if avg_probabilities:
                detected_lang = max(avg_probabilities.keys(), key=lambda k: avg_probabilities.get(k, 0.0))
                confidence = avg_probabilities[detected_lang]
            else:
                logger.warning("No average probabilities available")
                return None
            
            duration_used = perf_logger.stop_timer("enhanced_language_detection")
            
            logger.info(f"Enhanced detection: {detected_lang} (confidence: {confidence:.2f}, "
                       f"samples: {num_samples}, time: {duration_used:.2f}s)")
            
            # Cache result
            if self.config.enable_language_caching and hasattr(self, 'cache_file'):
                self._language_cache[audio_hash] = {
                    'language': detected_lang,
                    'confidence': confidence,
                    'samples': num_samples
                }
                self._save_language_cache()
            
            return detected_lang
            
        except Exception as e:
            perf_logger.stop_timer("enhanced_language_detection")
            logger.error(f"Enhanced language detection failed: {e}")
            return None
    
    def _extract_language_sample(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        start_time: float, 
        total_duration: float
    ) -> np.ndarray:
        """Extract 30-second sample for language detection.
        
        Args:
            audio_data: Full audio data
            sample_rate: Sample rate
            start_time: Start time for sample
            total_duration: Total audio duration
            
        Returns:
            Audio fragment for language detection
        """
        # Ensure we don't go beyond audio bounds
        start_sample = int(start_time * sample_rate)
        end_sample = int(min(start_time + 30, total_duration) * sample_rate)
        end_sample = min(end_sample, len(audio_data))
        
        fragment = audio_data[start_sample:end_sample]
        
        # Pad or trim to 30 seconds as Whisper expects
        import whisper
        padded = whisper.pad_or_trim(fragment)
        # Ensure we return numpy array
        if hasattr(padded, 'cpu'):
            return padded.cpu().numpy()  # type: ignore[attr-defined]
        elif hasattr(padded, 'numpy'):
            return padded.numpy()  # type: ignore[attr-defined]
        else:
            return np.asarray(padded)
    
    def _detect_language_fragment(self, audio_fragment: np.ndarray) -> Optional[Dict[str, float]]:
        """Detect language for a single audio fragment.
        
        Args:
            audio_fragment: Audio fragment
            
        Returns:
            Language probabilities dictionary
        """
        try:
            import whisper
            
            # Create mel spectrogram with null checks
            if self.model is None:
                logger.warning("Model not initialized for language detection")
                return None
                
            mel = whisper.log_mel_spectrogram(audio_fragment).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            
            # Convert probs to dict safely
            if hasattr(probs, 'cpu'):
                probs = probs.cpu()  # type: ignore[attr-defined]
            if hasattr(probs, 'numpy'):
                probs = probs.numpy()  # type: ignore[attr-defined]
            
            # Convert to proper dict
            if hasattr(probs, 'items'):
                return dict(probs)  # type: ignore[call-overload,arg-type]
            else:
                # If it's a tensor/array, create language mapping
                # This is a fallback - in practice whisper returns proper dict
                return {'detected': float(probs) if isinstance(probs, (int, float)) else 0.5}
            
        except Exception as e:
            logger.debug(f"Fragment language detection failed: {e}")
            return None
    
    def _handle_textual_anomalies(
        self,
        transcriptions: List[TranscriptionSegment],
        original_audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        task: str,
        verbose: bool
    ) -> List[TranscriptionSegment]:
        """Handle textual anomalies with limited retries.
        
        Args:
            transcriptions: Original transcriptions
            original_audio: Original audio data
            sample_rate: Sample rate
            language: Language code
            task: Transcription task
            verbose: Verbose flag
            
        Returns:
            Cleaned transcriptions
        """
        if not self.config.enable_anomaly_detection:
            return transcriptions
        
        cleaned_transcriptions = []
        
        for trans in transcriptions:
            if self._detect_textual_anomaly(trans.text):
                logger.warning(f"Textual anomaly detected in segment {trans.start:.2f}-{trans.end:.2f}s: "
                             f"{trans.text[:50]}...")
                
                # Attempt retry with shifted segments
                retry_trans = self._retry_with_shifted_segments(
                    original_audio, sample_rate, trans, language, task, verbose
                )
                
                if retry_trans:
                    cleaned_transcriptions.extend(retry_trans)
                else:
                    # If all retries fail, mark as anomaly but keep
                    trans.text = f"[ANOMALY DETECTED] {trans.text}"
                    cleaned_transcriptions.append(trans)
            else:
                cleaned_transcriptions.append(trans)
        
        return cleaned_transcriptions
    
    def _detect_textual_anomaly(self, text: str) -> bool:
        """Detect textual anomalies using heuristic rules.
        
        Args:
            text: Text to check
            
        Returns:
            True if anomaly detected
        """
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip()
        
        # Check for repetitive character patterns
        if self._has_repetitive_pattern(text):
            return True
        
        # Check for excessive repeated words
        if self._has_repeated_words(text):
            return True
        
        # Check for non-linguistic patterns
        if self._has_non_linguistic_pattern(text):
            return True
        
        return False
    
    def _has_repetitive_pattern(self, text: str) -> bool:
        """Check for repetitive character patterns."""
        # Check for repeated characters (e.g., "AAAAA...")
        char_counts: Dict[str, int] = {}
        for char in text:
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return False
        
        # If any character makes up more than 80% of the text
        max_char_ratio = max(char_counts.values()) / total_chars
        return max_char_ratio > self.config.anomaly_repetition_threshold
    
    def _has_repeated_words(self, text: str) -> bool:
        """Check for excessive repeated words."""
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check if more than 60% of words are the same
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_word_count = max(word_counts.values())
        return max_word_count / len(words) > 0.6
    
    def _has_non_linguistic_pattern(self, text: str) -> bool:
        """Check for non-linguistic patterns."""
        # Check for patterns like "♪♪♪" or "..."
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / len(text) > 0.5:
            return True
        
        # Check for very long words (likely corrupted)
        words = text.split()
        for word in words:
            if len(word) > 30:
                return True
        
        return False
    
    def _retry_with_shifted_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        original_segment: TranscriptionSegment,
        language: Optional[str],
        task: str,
        verbose: bool
    ) -> Optional[List[TranscriptionSegment]]:
        """Retry transcription with shifted audio segments.
        
        Args:
            audio: Original audio data
            sample_rate: Sample rate
            original_segment: Segment with anomaly
            language: Language code
            task: Transcription task
            verbose: Verbose flag
            
        Returns:
            Retried transcriptions or None if all fail
        """
        segment_start = original_segment.start
        segment_end = original_segment.end
        
        for attempt in range(self.config.max_retry_attempts):
            # Shift segment forward slightly
            shift = self.config.retry_shift_duration * (attempt + 1)
            new_start = segment_start + shift
            new_end = segment_end + shift
            
            # Ensure we don't go beyond audio bounds
            if new_end * sample_rate >= len(audio):
                break
            
            try:
                # Extract shifted segment
                start_sample = int(new_start * sample_rate)
                end_sample = int(new_end * sample_rate)
                shifted_audio = audio[start_sample:end_sample]
                
                if len(shifted_audio) == 0:
                    continue
                
                # Retry transcription
                retry_result = self._transcribe_segment_simple(
                    shifted_audio, sample_rate, language, task
                )
                
                if retry_result and not self._detect_textual_anomaly(retry_result.text):
                    logger.info(f"Anomaly resolved after {attempt + 1} retry attempts")
                    # Adjust timing back to original segment
                    retry_result.start = segment_start
                    retry_result.end = segment_end
                    return [retry_result]
                
            except Exception as e:
                logger.debug(f"Retry attempt {attempt + 1} failed: {e}")
                continue
        
        logger.warning("All retry attempts failed for anomaly segment")
        return None
    
    def _transcribe_segment_simple(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        task: str
    ) -> Optional[TranscriptionSegment]:
        """Simple transcription without anomaly handling (for retries).
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            language: Language code
            task: Task type
            
        Returns:
            Transcription segment or None
        """
        try:
            import whisper
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Transcribe without word timestamps for speed
            if self.model is None:
                raise RuntimeError("Model not initialized")
            result = self.model.transcribe(
                audio,
                language=language,
                task=task,
                verbose=False,
                word_timestamps=False
            )
            
            # Safe text handling
            text_content = result.get('text', '')
            if isinstance(text_content, list):
                text_content = ' '.join(str(t) for t in text_content)
            elif not isinstance(text_content, str):
                text_content = str(text_content)
                
            text = text_content.strip() if hasattr(text_content, 'strip') else str(text_content)
            if not text:
                return None
            
            detected_lang = result.get('language', language)
            if isinstance(detected_lang, list):
                detected_lang = detected_lang[0] if detected_lang else language
            elif not isinstance(detected_lang, str):
                detected_lang = language or 'en'
            
            return TranscriptionSegment(
                start=0.0,
                end=len(audio) / sample_rate,
                text=text,
                language=detected_lang
            )
            
        except Exception as e:
            logger.debug(f"Simple transcription failed: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages.
        
        Returns:
            List of language codes
        """
        # Whisper supports many languages
        languages = [
            'en', 'ko', 'ja', 'zh', 'de', 'es', 'ru', 'fr', 'pt', 'tr',
            'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi',
            'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no',
            'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk',
            'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk',
            'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw',
            'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
            'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo',
            'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl',
            'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]
        return languages
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring.
        
        Returns:
            Performance statistics dictionary
        """
        if not self.config.enable_performance_monitoring:
            return {}
        
        stats = {
            'whisper_model': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            
            # Enhanced language detection stats
            'enhanced_language_detection': {
                'enabled': True,
                'max_samples': self.config.max_language_samples,
                'confidence_threshold': self.config.language_confidence_threshold,
                'caching_enabled': self.config.enable_language_caching,
                'cached_languages': len(self._language_cache) if hasattr(self, '_language_cache') else 0
            },
            
            # VAD stats
            'vad': self.vad_processor.get_performance_stats() if self.vad_processor else {'enabled': False},
            
            # Anomaly detection stats
            'anomaly_detection': {
                'enabled': self.config.enable_anomaly_detection,
                'repetition_threshold': self.config.anomaly_repetition_threshold,
                'max_retry_attempts': self.config.max_retry_attempts,
                'retry_shift_duration': self.config.retry_shift_duration
            }
        }
        
        return stats