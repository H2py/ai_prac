"""
Timeline and statistics models for analysis results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from .base import BaseSegment


class EventType(Enum):
    """Types of timeline events."""
    SPEAKER_SEGMENT = "speaker_segment"
    TRANSCRIPTION = "transcription"
    EMOTION = "emotion"
    ACOUSTIC = "acoustic"
    SPEECH = "speech"


@dataclass
class TimelineEvent:
    """Timeline event representing a segment occurrence."""
    
    timestamp: float
    duration: float
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    @classmethod
    def from_segment(cls, segment: BaseSegment) -> 'TimelineEvent':
        """Create timeline event from a segment."""
        from .segments import (
            SpeakerSegment, TranscriptionSegment, EmotionSegment, 
            AcousticSegment, SpeechSegment
        )
        
        # Determine event type
        if isinstance(segment, SpeakerSegment):
            event_type = EventType.SPEAKER_SEGMENT
            data = {
                'speaker_id': segment.speaker_id,
                'type': 'speaker_segment'
            }
        elif isinstance(segment, TranscriptionSegment):
            event_type = EventType.TRANSCRIPTION
            data = {
                'text': segment.text,
                'language': segment.language,
                'speaker_id': segment.speaker_id,
                'word_count': segment.word_count,
                'type': 'transcription'
            }
        elif isinstance(segment, EmotionSegment):
            event_type = EventType.EMOTION
            data = {
                'predicted_emotion': segment.predicted_emotion,
                'emotion_scores': segment.emotion_scores,
                'speaker_id': segment.speaker_id,
                'type': 'emotion'
            }
        elif isinstance(segment, AcousticSegment):
            event_type = EventType.ACOUSTIC
            data = {
                'pitch_mean': segment.pitch_mean,
                'rms_energy': segment.rms_energy,
                'spectral_centroid': segment.spectral_centroid,
                'features': segment.features,
                'type': 'acoustic'
            }
        elif isinstance(segment, SpeechSegment):
            event_type = EventType.SPEECH
            data = {
                'speech_probability': segment.speech_probability,
                'type': 'speech'
            }
        else:
            event_type = EventType.SPEAKER_SEGMENT  # Default fallback
            data = {'type': 'unknown'}
        
        # Add common data
        data.update({
            'start': segment.start,
            'end': segment.end
        })
        
        return cls(
            timestamp=segment.start,
            duration=segment.duration,
            event_type=event_type,
            data=data,
            confidence=segment.confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'event_type': self.event_type.value,
            'confidence': self.confidence,
            **self.data
        }
    
    def overlaps_with(self, other: 'TimelineEvent', threshold: float = 0.0) -> bool:
        """Check if this event overlaps with another."""
        return (
            self.timestamp <= other.timestamp + other.duration - threshold and
            self.timestamp + self.duration >= other.timestamp + threshold
        )


@dataclass
class AnalysisStatistics:
    """Comprehensive statistics for analysis results."""
    
    total_segments: int
    total_speakers: int
    speaker_distribution: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dominant_emotion: str = 'neutral'
    emotion_distribution: Dict[str, int] = field(default_factory=dict)
    total_words: int = 0
    total_duration: float = 0.0
    segments_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    processing_efficiency: Optional[float] = None  # segments per second
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_duration > 0 and self.total_segments > 0:
            self.processing_efficiency = self.total_segments / self.total_duration
    
    def calculate_emotion_percentages(self) -> Dict[str, float]:
        """Calculate emotion distribution as percentages."""
        if not self.emotion_distribution:
            return {}
        
        total = sum(self.emotion_distribution.values())
        if total == 0:
            return {}
        
        return {
            emotion: (count / total) * 100
            for emotion, count in self.emotion_distribution.items()
        }
    
    def calculate_speaking_percentages(self) -> Dict[str, float]:
        """Calculate speaking time percentages."""
        if not self.speaker_distribution or self.total_duration == 0:
            return {}
        
        return {
            speaker_id: info.get('speaking_percentage', 0.0)
            for speaker_id, info in self.speaker_distribution.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for display."""
        return {
            'total_segments': self.total_segments,
            'total_speakers': self.total_speakers,
            'total_words': self.total_words,
            'total_duration': self.total_duration,
            'dominant_emotion': self.dominant_emotion,
            'processing_efficiency': self.processing_efficiency,
            'segments_by_type': self.segments_by_type
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'total_segments': self.total_segments,
            'total_speakers': self.total_speakers,
            'speaker_distribution': self.speaker_distribution,
            'dominant_emotion': self.dominant_emotion,
            'emotion_distribution': self.emotion_distribution,
            'emotion_percentages': self.calculate_emotion_percentages(),
            'speaking_percentages': self.calculate_speaking_percentages(),
            'total_words': self.total_words,
            'total_duration': self.total_duration,
            'segments_by_type': self.segments_by_type,
            'processing_efficiency': self.processing_efficiency,
            'confidence_metrics': self.confidence_metrics
        }