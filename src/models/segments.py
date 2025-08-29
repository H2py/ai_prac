"""
Specialized segment implementations inheriting from BaseSegment.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from .base import BaseSegment


@dataclass
class SpeakerSegment(BaseSegment):
    """Speaker diarization segment with speaker identity."""
    
    speaker_id: str = ""
    embedding: Optional[np.ndarray] = None
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'speaker': self.speaker_id,
            'confidence': self.confidence
        }
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Merge speaker-specific properties."""
        if isinstance(merged, SpeakerSegment) and isinstance(other, SpeakerSegment):
            # Keep speaker_id from the longer segment
            if self.duration >= other.duration:
                merged.speaker_id = self.speaker_id
                merged.embedding = self.embedding
            else:
                merged.speaker_id = other.speaker_id
                merged.embedding = other.embedding


@dataclass
class TranscriptionSegment(BaseSegment):
    """Speech transcription segment with text content."""
    
    text: str = ""
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    word_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate word count if not provided."""
        if self.word_count is None and self.text:
            self.word_count = len(self.text.split())
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'text': self.text,
            'language': self.language,
            'speaker': self.speaker_id,
            'confidence': self.confidence,
            'word_count': self.word_count
        }
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Merge transcription-specific properties."""
        if isinstance(merged, TranscriptionSegment) and isinstance(other, TranscriptionSegment):
            # Concatenate text with space
            merged.text = f"{self.text} {other.text}".strip()
            
            # Use language from segment with higher confidence
            if self.confidence >= other.confidence:
                merged.language = self.language
                merged.speaker_id = self.speaker_id
            else:
                merged.language = other.language
                merged.speaker_id = other.speaker_id
            
            # Recalculate word count
            merged.word_count = len(merged.text.split()) if merged.text else 0


@dataclass
class EmotionSegment(BaseSegment):
    """Emotion analysis segment with predicted emotion."""
    
    predicted_emotion: str = ""
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    speaker_id: Optional[str] = None
    
    # Legacy compatibility properties
    @property
    def primary_emotion(self) -> str:
        """Legacy property name."""
        return self.predicted_emotion
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'predicted_emotion': self.predicted_emotion,
            'primary_emotion': self.predicted_emotion,  # Legacy compatibility
            'emotion_scores': self.emotion_scores,
            'confidence': self.confidence,
            'speaker': self.speaker_id
        }
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Merge emotion-specific properties."""
        if isinstance(merged, EmotionSegment) and isinstance(other, EmotionSegment):
            # Use emotion from segment with higher confidence
            if self.confidence >= other.confidence:
                merged.predicted_emotion = self.predicted_emotion
                merged.emotion_scores = self.emotion_scores.copy()
                merged.speaker_id = self.speaker_id
            else:
                merged.predicted_emotion = other.predicted_emotion
                merged.emotion_scores = other.emotion_scores.copy()
                merged.speaker_id = other.speaker_id


@dataclass
class AcousticSegment(BaseSegment):
    """Acoustic analysis segment with extracted features."""
    
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    rms_energy: Optional[float] = None
    spectral_centroid: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        data = {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'confidence': self.confidence
        }
        
        # Add individual features
        if self.pitch_mean is not None:
            data['pitch_mean'] = self.pitch_mean
        if self.pitch_std is not None:
            data['pitch_std'] = self.pitch_std
        if self.rms_energy is not None:
            data['rms_energy'] = self.rms_energy
        if self.spectral_centroid is not None:
            data['spectral_centroid'] = self.spectral_centroid
        
        # Add additional features
        data.update(self.features)
        
        return data
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Merge acoustic-specific properties."""
        if isinstance(merged, AcousticSegment) and isinstance(other, AcousticSegment):
            # Average acoustic features weighted by duration
            total_duration = self.duration + other.duration
            self_weight = self.duration / total_duration
            other_weight = other.duration / total_duration
            
            # Merge individual features
            merged.pitch_mean = self._weighted_average(
                self.pitch_mean, other.pitch_mean, self_weight, other_weight
            )
            merged.pitch_std = self._weighted_average(
                self.pitch_std, other.pitch_std, self_weight, other_weight
            )
            merged.rms_energy = self._weighted_average(
                self.rms_energy, other.rms_energy, self_weight, other_weight
            )
            merged.spectral_centroid = self._weighted_average(
                self.spectral_centroid, other.spectral_centroid, self_weight, other_weight
            )
            
            # Merge additional features
            all_feature_keys = set(self.features.keys()) | set(other.features.keys())
            merged.features = {}
            for key in all_feature_keys:
                self_val = self.features.get(key)
                other_val = other.features.get(key)
                merged.features[key] = self._weighted_average(
                    self_val, other_val, self_weight, other_weight
                )
    
    def _weighted_average(self, val1: Optional[float], val2: Optional[float], 
                         weight1: float, weight2: float) -> Optional[float]:
        """Calculate weighted average of two optional values."""
        if val1 is None and val2 is None:
            return None
        elif val1 is None:
            return val2
        elif val2 is None:
            return val1
        else:
            return val1 * weight1 + val2 * weight2


@dataclass
class SpeechSegment(BaseSegment):
    """Voice Activity Detection segment indicating speech presence."""
    
    speech_probability: float = 1.0
    
    def to_export_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'confidence': self.confidence,
            'speech_probability': self.speech_probability
        }
    
    def _merge_specific_properties(self, merged: 'BaseSegment', other: 'BaseSegment') -> None:
        """Merge speech-specific properties."""
        if isinstance(merged, SpeechSegment) and isinstance(other, SpeechSegment):
            # Average speech probability weighted by duration
            total_duration = self.duration + other.duration
            merged.speech_probability = (
                self.speech_probability * self.duration + 
                other.speech_probability * other.duration
            ) / total_duration