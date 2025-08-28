"""
Speech recognition module using OpenAI Whisper for transcription.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


@dataclass
class TranscriptionSegment:
    """Transcription segment with timing and text."""
    start: float
    end: float
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    speaker: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'speaker': self.speaker
        }


class SpeechRecognizer:
    """Speech recognition using Whisper."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """Initialize speech recognizer.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    def initialize(self):
        """Initialize the Whisper model."""
        try:
            import whisper
            import torch
            
            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper model '{self.model_name}' on {self.device}")
            
            # Load model
            self.model = whisper.load_model(self.model_name, device=self.device)
            
            logger.info(f"Whisper model loaded successfully")
            
        except ImportError:
            logger.warning("Whisper not installed. Using fallback mode.")
            logger.info("Install with: pip install openai-whisper")
            self.model = None
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.model = None
    
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
                                trans.speaker = speaker
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
            
            # Transcribe
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
                    trans = TranscriptionSegment(
                        start=seg['start'],
                        end=seg['end'],
                        text=seg['text'].strip(),
                        language=result.get('language', language),
                        confidence=seg.get('confidence')
                    )
                    transcriptions.append(trans)
            else:
                # Single segment
                trans = TranscriptionSegment(
                    start=0.0,
                    end=len(audio) / sample_rate,
                    text=result.get('text', '').strip(),
                    language=result.get('language', language)
                )
                transcriptions.append(trans)
            
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
                trans = TranscriptionSegment(
                    start=segment.get('start', 0),
                    end=segment.get('end', 0),
                    text=f"[화자 {i+1} 발화 내용]",
                    language='ko',
                    confidence=0.95,
                    speaker=segment.get('speaker', f'speaker_{i+1}')
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
                    speaker=f'speaker_1'
                )
                transcriptions.append(trans)
        
        return transcriptions
    
    def detect_language(self, audio_path: Union[str, Path]) -> Optional[str]:
        """Detect language of audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language code or None
        """
        if self.model is None:
            return None
        
        try:
            import whisper
            
            audio_path = Path(audio_path)
            
            # Load audio (30 seconds for detection)
            audio, sr = load_audio(audio_path, sample_rate=16000, duration=30)
            
            # Detect language
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            _, probs = self.model.detect_language(mel)
            language = max(probs, key=probs.get)
            
            logger.info(f"Detected language: {language} (confidence: {probs[language]:.2f})")
            
            return language
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
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