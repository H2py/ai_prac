"""
Voice Activity Detection module using Silero VAD for performance-optimized speech segment extraction.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass

# Import new models for compatibility
from ..models import SpeechSegment as NewSpeechSegment

from ..utils import load_audio
from ..utils import PerformanceLogger
from config.settings import WhisperConfig

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


@dataclass
class SpeechSegment:
    """Speech segment with timing information."""
    start: float
    end: float
    confidence: float
    duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'duration': self.duration
        }


class VADProcessor:
    """Lightweight Voice Activity Detection using Silero VAD."""
    
    def __init__(self, config: WhisperConfig):
        """Initialize VAD processor.
        
        Args:
            config: Whisper configuration with VAD settings
        """
        self.config = config
        self.model: Optional[Any] = None  # PyTorch model type varies by implementation
        self.sample_rate = 16000  # Silero VAD requires 16kHz
        
    def initialize(self):
        """Initialize the Silero VAD model."""
        if not self.config.enable_vad:
            logger.info("VAD disabled in configuration")
            return
            
        try:
            import silero_vad  # type: ignore[import-not-found]
            
            logger.info(f"Loading Silero VAD model in {self.config.vad_mode} mode")
            
            # Load model based on mode
            if self.config.vad_mode == "fast":
                model_name = "silero_vad"
            elif self.config.vad_mode == "balanced":
                model_name = "silero_vad_v4"  
            else:  # accurate
                model_name = "silero_vad_v5"
            
            # Load pre-trained model
            model_result = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model=model_name,
                force_reload=False,
                verbose=False
            )
            # Handle both tuple and single return cases
            if isinstance(model_result, tuple):
                self.model, _ = model_result
            else:
                self.model = model_result
            
            # Set to evaluation mode for inference
            self.model.eval()  # type: ignore[attr-defined]
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)  # type: ignore[attr-defined]
            
            logger.info(f"Silero VAD loaded successfully on {device}")
            
        except ImportError:
            logger.warning("Silero VAD not installed. Install with: pip install silero-vad")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            self.model = None
    
    def detect_speech_segments(
        self,
        audio_path: Path,
        min_duration: Optional[float] = None
    ) -> List[SpeechSegment]:
        """Detect speech segments in audio file.
        
        Args:
            audio_path: Path to audio file
            min_duration: Minimum segment duration (uses config if None)
            
        Returns:
            List of speech segments with timing
        """
        if not self.config.enable_vad or self.model is None:
            # Return single segment covering entire audio
            return self._fallback_segments(audio_path)
        
        perf_logger.start_timer("vad_processing")
        
        try:
            min_dur = min_duration or self.config.min_speech_duration
            
            logger.info(f"Detecting speech segments in: {audio_path}")
            
            # Load audio at required sample rate
            audio_data, sr = load_audio(audio_path, sample_rate=self.sample_rate)
            
            if len(audio_data) == 0:
                logger.warning("Empty audio data")
                return []
            
            # Process in chunks for memory efficiency
            segments = self._process_audio_chunks(audio_data, sr, min_dur)
            
            duration = perf_logger.stop_timer("vad_processing")
            logger.info(f"VAD completed in {duration:.2f}s, found {len(segments)} speech segments")
            
            return segments
            
        except Exception as e:
            perf_logger.stop_timer("vad_processing")
            logger.error(f"VAD processing failed: {e}")
            return self._fallback_segments(audio_path)
    
    def _process_audio_chunks(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        min_duration: float
    ) -> List[SpeechSegment]:
        """Process audio in chunks for memory efficiency.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            min_duration: Minimum segment duration
            
        Returns:
            List of speech segments
        """
        segments = []
        chunk_size_samples = int(self.config.vad_chunk_size * sample_rate)
        total_chunks = (len(audio_data) + chunk_size_samples - 1) // chunk_size_samples
        
        logger.debug(f"Processing {total_chunks} chunks of {self.config.vad_chunk_size}s each")
        
        current_segment_start = None
        current_segment_confidences = []
        
        for chunk_idx in range(total_chunks):
            start_sample = chunk_idx * chunk_size_samples
            end_sample = min(start_sample + chunk_size_samples, len(audio_data))
            chunk = audio_data[start_sample:end_sample]
            
            # Pad chunk if too small
            if len(chunk) < chunk_size_samples:
                chunk = np.pad(chunk, (0, chunk_size_samples - len(chunk)))
            
            # Detect speech in chunk
            chunk_time_start = start_sample / sample_rate
            chunk_time_end = end_sample / sample_rate
            
            is_speech, confidence = self._detect_speech_in_chunk(chunk)
            
            if is_speech:
                if current_segment_start is None:
                    # Start new segment
                    current_segment_start = chunk_time_start
                    current_segment_confidences = [confidence]
                else:
                    # Continue current segment
                    current_segment_confidences.append(confidence)
            else:
                if current_segment_start is not None:
                    # End current segment
                    segment_end = chunk_time_start
                    segment_duration = segment_end - current_segment_start
                    
                    if segment_duration >= min_duration:
                        avg_confidence = float(np.mean(current_segment_confidences))
                        segment = SpeechSegment(
                            start=current_segment_start,
                            end=segment_end,
                            confidence=avg_confidence,
                            duration=segment_duration
                        )
                        segments.append(segment)
                    
                    # Reset for next segment
                    current_segment_start = None
                    current_segment_confidences = []
        
        # Handle final segment if still active
        if current_segment_start is not None:
            segment_end = len(audio_data) / sample_rate
            segment_duration = segment_end - current_segment_start
            
            if segment_duration >= min_duration:
                avg_confidence = float(np.mean(current_segment_confidences))
                segment = SpeechSegment(
                    start=current_segment_start,
                    end=segment_end,
                    confidence=avg_confidence,
                    duration=segment_duration
                )
                segments.append(segment)
        
        return self._merge_close_segments(segments)
    
    def _detect_speech_in_chunk(self, chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect speech in a single audio chunk.
        
        Args:
            chunk: Audio chunk
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        try:
            # Convert to tensor and normalize
            chunk_tensor = torch.from_numpy(chunk).float()
            
            # Ensure single dimension
            if len(chunk_tensor.shape) > 1:
                chunk_tensor = chunk_tensor.mean(dim=0)
            
            # Move to same device as model
            if self.model is not None:
                chunk_tensor = chunk_tensor.to(next(self.model.parameters()).device)  # type: ignore[attr-defined]
            
            # Get speech probability
            if self.model is None:
                return False, 0.0
                
            with torch.no_grad():
                speech_prob = self.model(chunk_tensor, self.sample_rate).item()  # type: ignore[misc]
            
            is_speech = speech_prob > self.config.vad_threshold
            
            return is_speech, speech_prob
            
        except Exception as e:
            logger.debug(f"Speech detection error in chunk: {e}")
            return False, 0.0
    
    def _merge_close_segments(
        self,
        segments: List[SpeechSegment],
        max_gap: float = 0.5
    ) -> List[SpeechSegment]:
        """Merge speech segments that are close together.
        
        Args:
            segments: List of speech segments
            max_gap: Maximum gap to merge (seconds)
            
        Returns:
            Merged segments
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            gap = next_segment.start - current.end
            
            if gap <= max_gap:
                # Merge segments
                merged_duration = next_segment.end - current.start
                merged_confidence = (
                    current.confidence * current.duration +
                    next_segment.confidence * next_segment.duration
                ) / (current.duration + next_segment.duration)
                
                current = SpeechSegment(
                    start=current.start,
                    end=next_segment.end,
                    confidence=merged_confidence,
                    duration=merged_duration
                )
            else:
                # Keep current and move to next
                merged.append(current)
                current = next_segment
        
        # Add final segment
        merged.append(current)
        
        return merged
    
    def _fallback_segments(self, audio_path: Path) -> List[SpeechSegment]:
        """Fallback to single segment covering entire audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Single segment covering entire audio
        """
        try:
            # Get audio duration
            audio_data, sr = load_audio(audio_path, sample_rate=16000)
            duration = len(audio_data) / sr
            
            segment = SpeechSegment(
                start=0.0,
                end=duration,
                confidence=1.0,  # Full confidence for fallback
                duration=duration
            )
            
            logger.info(f"Using fallback: single segment of {duration:.2f}s")
            return [segment]
            
        except Exception as e:
            logger.error(f"Fallback segment creation failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance stats dictionary
        """
        if not self.config.enable_performance_monitoring:
            return {}
        
        return {
            'vad_enabled': self.config.enable_vad,
            'vad_mode': self.config.vad_mode,
            'vad_chunk_size': self.config.vad_chunk_size,
            'min_speech_duration': self.config.min_speech_duration,
            'vad_threshold': self.config.vad_threshold,
            'model_loaded': self.model is not None
        }