"""
Unified Data Models for Audio Analysis

This module contains all data structures and models used throughout the audio analysis pipeline.
Organized by functional groups while maintaining clear separation of concerns.

Model Categories:
- Base classes and interfaces
- Segment models for different analysis types
- Container and result models
- Timeline and metadata structures
- Error and exception classes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np


# ================================
# BASE CLASSES AND INTERFACES
# ================================

@dataclass
class BaseSegment:
    """Base class for all audio segment types."""
    
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start
    
    def overlaps_with(self, other: 'BaseSegment', threshold: float = 0.0) -> bool:
        """Check if this segment overlaps with another."""
        return not (self.end <= other.start + threshold or self.start >= other.end - threshold)
    
    def intersection(self, other: 'BaseSegment') -> Optional[Tuple[float, float]]:
        """Get intersection time range with another segment."""
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return (start, end) if start < end else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'confidence': self.confidence
        }


# ================================
# SEGMENT MODELS
# ================================

@dataclass
class SpeakerSegment(BaseSegment):
    """Speaker diarization segment."""
    
    speaker_id: str = "SPEAKER_00"
    speaker_confidence: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'speaker_id': self.speaker_id,
            'speaker_confidence': self.speaker_confidence
        })
        return base_dict


@dataclass  
class EmotionSegment(BaseSegment):
    """Emotion analysis segment."""
    
    predicted_emotion: str = "neutral"
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    
    def get_top_emotions(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N emotions by score."""
        sorted_emotions = sorted(
            self.emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'predicted_emotion': self.predicted_emotion,
            'emotion_scores': self.emotion_scores,
            'top_emotions': self.get_top_emotions()
        })
        return base_dict


@dataclass
class TranscriptionSegment(BaseSegment):
    """Speech recognition transcription segment."""
    
    text: str = ""
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    
    def get_word_count(self) -> int:
        """Get word count in transcription."""
        return len(self.text.split()) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'text': self.text,
            'language': self.language,
            'language_confidence': self.language_confidence,
            'word_count': self.get_word_count(),
            'words': self.words
        })
        return base_dict


@dataclass
class AcousticSegment(BaseSegment):
    """Acoustic features segment."""
    
    # Core acoustic features
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    rms_energy: Optional[float] = None
    spectral_centroid: Optional[float] = None
    
    # Extended feature dictionary
    features: Dict[str, float] = field(default_factory=dict)
    
    def get_feature(self, feature_name: str, default: float = 0.0) -> float:
        """Get a specific acoustic feature."""
        return self.features.get(feature_name, default)
    
    def has_pitch_data(self) -> bool:
        """Check if pitch data is available."""
        return self.pitch_mean is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'pitch_mean': self.pitch_mean,
            'pitch_std': self.pitch_std,
            'rms_energy': self.rms_energy,
            'spectral_centroid': self.spectral_centroid,
            'features': self.features,
            'has_pitch': self.has_pitch_data()
        })
        return base_dict


@dataclass
class SpeechSegment:
    """Voice Activity Detection segment."""
    
    start: float
    end: float
    confidence: float
    speech_probability: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'confidence': self.confidence,
            'speech_probability': self.speech_probability
        }


# ================================
# ANALYSIS RESULTS CONTAINERS
# ================================

@dataclass
class AnalysisResults:
    """Container for comprehensive analysis results."""
    
    # Analysis results by type
    speaker_segments: List[SpeakerSegment] = field(default_factory=list)
    emotion_segments: List[EmotionSegment] = field(default_factory=list)
    transcription_segments: List[TranscriptionSegment] = field(default_factory=list)
    acoustic_segments: List[AcousticSegment] = field(default_factory=list)
    speech_segments: List[SpeechSegment] = field(default_factory=list)
    
    # Metadata
    audio_duration: float = 0.0
    sample_rate: int = 16000
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_total_segments(self) -> int:
        """Get total number of segments across all types."""
        return (len(self.speaker_segments) + len(self.emotion_segments) + 
                len(self.transcription_segments) + len(self.acoustic_segments) +
                len(self.speech_segments))
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results."""
        return {
            'total_segments': self.get_total_segments(),
            'audio_duration': self.audio_duration,
            'speakers': len(set(s.speaker_id for s in self.speaker_segments)),
            'emotions_detected': len(set(e.predicted_emotion for e in self.emotion_segments)),
            'transcribed_segments': len([t for t in self.transcription_segments if t.text]),
            'processing_time': self.processing_time,
            'segments_by_type': {
                'speaker': len(self.speaker_segments),
                'emotion': len(self.emotion_segments),
                'transcription': len(self.transcription_segments),
                'acoustic': len(self.acoustic_segments),
                'speech': len(self.speech_segments)
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all results to dictionary format."""
        return {
            'speaker_segments': [s.to_dict() for s in self.speaker_segments],
            'emotion_segments': [e.to_dict() for e in self.emotion_segments],
            'transcription_segments': [t.to_dict() for t in self.transcription_segments],
            'acoustic_segments': [a.to_dict() for a in self.acoustic_segments],
            'speech_segments': [s.to_dict() for s in self.speech_segments],
            'metadata': {
                'audio_duration': self.audio_duration,
                'sample_rate': self.sample_rate,
                'processing_time': self.processing_time,
                'timestamp': self.timestamp.isoformat(),
                'summary': self.get_analysis_summary()
            }
        }


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline."""
    
    input_source: str
    output_directory: Path
    timestamp: str
    
    # File information
    original_format: Optional[str] = None
    extracted_audio_path: Optional[Path] = None
    file_size_mb: float = 0.0
    
    # Processing configuration
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    enabled_features: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    step_durations: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    processing_quality: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_step_duration(self, step_name: str, duration: float):
        """Add timing for a processing step."""
        self.step_durations[step_name] = duration
    
    def add_warning(self, warning: str):
        """Add a processing warning."""
        self.warnings.append(warning)
    
    def add_error(self, error: str, step: str, critical: bool = False):
        """Add a processing error."""
        self.errors.append({
            'error': error,
            'step': step,
            'critical': critical,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_success_rate(self) -> float:
        """Calculate processing success rate."""
        total_steps = len(self.step_durations) + len(self.errors)
        if total_steps == 0:
            return 1.0
        
        successful_steps = len(self.step_durations)
        return successful_steps / total_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'input_source': self.input_source,
            'output_directory': str(self.output_directory),
            'timestamp': self.timestamp,
            'file_info': {
                'original_format': self.original_format,
                'extracted_audio_path': str(self.extracted_audio_path) if self.extracted_audio_path else None,
                'file_size_mb': self.file_size_mb
            },
            'processing': {
                'total_time': self.total_processing_time,
                'step_durations': self.step_durations,
                'enabled_features': self.enabled_features,
                'success_rate': self.get_success_rate()
            },
            'quality': self.processing_quality,
            'issues': {
                'warnings': self.warnings,
                'errors': self.errors
            },
            'memory': self.memory_usage,
            'config': self.config_snapshot
        }


# ================================
# TIMELINE AND VISUALIZATION
# ================================

@dataclass
class TimelineEvent:
    """Event in the analysis timeline."""
    
    start_time: float
    end_time: float
    event_type: str  # 'speaker', 'emotion', 'transcription', etc.
    data: Dict[str, Any]
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def overlaps_with_time(self, start: float, end: float) -> bool:
        """Check if event overlaps with given time range."""
        return not (self.end_time <= start or self.start_time >= end)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'event_type': self.event_type,
            'confidence': self.confidence,
            'data': self.data
        }


@dataclass
class AnalysisTimeline:
    """Timeline representation of all analysis events."""
    
    events: List[TimelineEvent] = field(default_factory=list)
    total_duration: float = 0.0
    
    def add_event(self, event: TimelineEvent):
        """Add an event to the timeline."""
        self.events.append(event)
        # Update total duration if necessary
        if event.end_time > self.total_duration:
            self.total_duration = event.end_time
    
    def get_events_in_range(self, start: float, end: float) -> List[TimelineEvent]:
        """Get all events that occur in the given time range."""
        return [event for event in self.events 
                if event.overlaps_with_time(start, end)]
    
    def get_events_by_type(self, event_type: str) -> List[TimelineEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def sort_by_time(self):
        """Sort events chronologically."""
        self.events.sort(key=lambda x: x.start_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert timeline to dictionary."""
        self.sort_by_time()
        return {
            'total_duration': self.total_duration,
            'event_count': len(self.events),
            'events_by_type': {
                event_type: len(self.get_events_by_type(event_type))
                for event_type in set(e.event_type for e in self.events)
            },
            'events': [event.to_dict() for event in self.events]
        }


# ================================
# MEDIA INFORMATION
# ================================

@dataclass
class MediaInfo:
    """Information about source media file."""
    
    # Basic file information
    source_path: str
    source_type: str  # 'file', 'url', 'youtube', etc.
    file_size_bytes: Optional[int] = None
    
    # Audio properties
    duration_seconds: float = 0.0
    sample_rate: int = 16000
    channels: int = 1
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None
    
    # Video properties (if applicable)
    has_video: bool = False
    video_codec: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    
    # Metadata
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    creation_date: Optional[datetime] = None
    
    def get_duration_formatted(self) -> str:
        """Get duration in human-readable format."""
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_file_size_formatted(self) -> str:
        """Get file size in human-readable format."""
        if not self.file_size_bytes:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size_bytes < 1024:
                return f"{self.file_size_bytes:.1f} {unit}"
            self.file_size_bytes /= 1024
        return f"{self.file_size_bytes:.1f} TB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert media info to dictionary."""
        return {
            'source': {
                'path': self.source_path,
                'type': self.source_type,
                'file_size': self.get_file_size_formatted()
            },
            'audio': {
                'duration': self.duration_seconds,
                'duration_formatted': self.get_duration_formatted(),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'codec': self.audio_codec,
                'bitrate': self.bitrate
            },
            'video': {
                'has_video': self.has_video,
                'codec': self.video_codec,
                'resolution': self.resolution,
                'fps': self.fps
            } if self.has_video else None,
            'metadata': {
                'title': self.title,
                'artist': self.artist,
                'album': self.album,
                'genre': self.genre,
                'creation_date': self.creation_date.isoformat() if self.creation_date else None
            }
        }


# ================================
# ERROR HANDLING
# ================================

class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    
    def __init__(self, message: str, step: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.step = step
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': str(self),
            'step': self.step,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class AudioExtractionError(PipelineError):
    """Errors related to audio extraction."""
    pass


class AnalysisError(PipelineError):
    """Errors related to audio analysis."""
    pass


class OutputError(PipelineError):
    """Errors related to output generation."""
    pass


# ================================
# UTILITY FUNCTIONS
# ================================

def merge_overlapping_segments(segments: List[BaseSegment], threshold: float = 0.1) -> List[BaseSegment]:
    """Merge segments that overlap beyond the threshold."""
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.start)
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        last_merged = merged[-1]
        
        # Check if segments overlap beyond threshold
        if current.start <= last_merged.end + threshold:
            # Merge segments
            merged_segment = BaseSegment(
                start=last_merged.start,
                end=max(last_merged.end, current.end),
                confidence=min(last_merged.confidence, current.confidence)
            )
            merged[-1] = merged_segment
        else:
            merged.append(current)
    
    return merged


def find_segment_overlaps(segments1: List[BaseSegment], segments2: List[BaseSegment]) -> List[Tuple[int, int, float]]:
    """Find overlapping segments between two lists."""
    overlaps = []
    
    for i, seg1 in enumerate(segments1):
        for j, seg2 in enumerate(segments2):
            intersection = seg1.intersection(seg2)
            if intersection:
                overlap_duration = intersection[1] - intersection[0]
                min_duration = min(seg1.duration, seg2.duration)
                overlap_ratio = overlap_duration / min_duration if min_duration > 0 else 0
                overlaps.append((i, j, overlap_ratio))
    
    return overlaps