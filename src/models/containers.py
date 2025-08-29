"""
Container classes for analysis results and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type, TypeVar, Generic
from pathlib import Path

from .base import BaseSegment
from .timeline import TimelineEvent, AnalysisStatistics

T = TypeVar('T', bound=BaseSegment)


@dataclass
class MediaInfo:
    """Media file information with comprehensive metadata."""
    
    source_type: str  # "video", "audio", "youtube"
    duration: float
    sample_rate: int
    channels: int
    language: Optional[str] = None
    video_resolution: Optional[str] = None
    codec: Optional[str] = None
    file_size: Optional[int] = None
    file_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'source_type': self.source_type,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'language': self.language,
            'video_resolution': self.video_resolution,
            'codec': self.codec,
            'file_size': self.file_size,
            'file_path': str(self.file_path) if self.file_path else None
        }


@dataclass
class ProcessingMetadata:
    """Processing metadata and performance statistics."""
    
    timestamp: str
    processing_time: float
    pipeline_version: str = "2.0"
    whisper_enhancements: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None
    components_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'timestamp': self.timestamp,
            'processing_time': self.processing_time,
            'pipeline_version': self.pipeline_version,
            'whisper_enhancements': self.whisper_enhancements or {},
            'performance_stats': self.performance_stats or {},
            'components_used': self.components_used
        }


@dataclass
class AnalysisResults:
    """Unified container for all analysis results with type-safe access."""
    
    media_info: MediaInfo
    segments: List[BaseSegment] = field(default_factory=list)
    metadata: ProcessingMetadata = field(default_factory=lambda: ProcessingMetadata(
        timestamp="", processing_time=0.0
    ))
    
    # Cached properties for performance
    _timeline_cache: Optional[List[TimelineEvent]] = field(default=None, init=False)
    _statistics_cache: Optional[AnalysisStatistics] = field(default=None, init=False)
    
    def get_segments_by_type(self, segment_type: Type[T]) -> List[T]:
        """Get all segments of a specific type with proper typing."""
        return [seg for seg in self.segments if isinstance(seg, segment_type)]
    
    def get_timeline(self, force_refresh: bool = False) -> List[TimelineEvent]:
        """Get timeline of events, cached for performance."""
        if self._timeline_cache is None or force_refresh:
            self._timeline_cache = self._build_timeline()
        return self._timeline_cache
    
    def get_statistics(self, force_refresh: bool = False) -> AnalysisStatistics:
        """Get analysis statistics, cached for performance."""
        if self._statistics_cache is None or force_refresh:
            self._statistics_cache = self._build_statistics()
        return self._statistics_cache
    
    def add_segments(self, segments: List[BaseSegment]) -> None:
        """Add segments and invalidate caches."""
        self.segments.extend(segments)
        self._invalidate_caches()
    
    def merge_overlapping_segments(self, threshold: float = 0.1) -> None:
        """Merge overlapping segments of the same type."""
        from .segments import SpeakerSegment, TranscriptionSegment, EmotionSegment, AcousticSegment
        
        # Group segments by type
        segment_groups = {}
        for segment in self.segments:
            seg_type = type(segment)
            if seg_type not in segment_groups:
                segment_groups[seg_type] = []
            segment_groups[seg_type].append(segment)
        
        # Merge within each group
        merged_segments = []
        for seg_type, segments in segment_groups.items():
            if not segments:
                continue
                
            # Sort by start time
            segments.sort(key=lambda s: s.start)
            
            merged = []
            current = segments[0]
            
            for next_seg in segments[1:]:
                if current.overlaps_with(next_seg, threshold):
                    # Check if they can be merged (same speaker for some types)
                    if self._can_merge_segments(current, next_seg):
                        current = current.merge_with(next_seg)
                    else:
                        merged.append(current)
                        current = next_seg
                else:
                    merged.append(current)
                    current = next_seg
            
            merged.append(current)
            merged_segments.extend(merged)
        
        self.segments = merged_segments
        self._invalidate_caches()
    
    def _can_merge_segments(self, seg1: BaseSegment, seg2: BaseSegment) -> bool:
        """Check if two segments can be merged based on type-specific rules."""
        from .segments import SpeakerSegment, TranscriptionSegment, EmotionSegment
        
        # Same type is required
        if type(seg1) != type(seg2):
            return False
        
        # Speaker segments can merge if same speaker
        if isinstance(seg1, SpeakerSegment) and isinstance(seg2, SpeakerSegment):
            return seg1.speaker_id == seg2.speaker_id
        
        # Transcription segments can merge if same speaker
        if isinstance(seg1, TranscriptionSegment) and isinstance(seg2, TranscriptionSegment):
            return seg1.speaker_id == seg2.speaker_id
        
        # Emotion segments can merge if same speaker and emotion
        if isinstance(seg1, EmotionSegment) and isinstance(seg2, EmotionSegment):
            return (seg1.speaker_id == seg2.speaker_id and 
                   seg1.predicted_emotion == seg2.predicted_emotion)
        
        # Other types can generally merge
        return True
    
    def _build_timeline(self) -> List[TimelineEvent]:
        """Build timeline from all segments."""
        events = []
        
        for segment in self.segments:
            event = TimelineEvent.from_segment(segment)
            events.append(event)
        
        # Sort by start time
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def _build_statistics(self) -> AnalysisStatistics:
        """Build statistics from all segments."""
        from .segments import SpeakerSegment, TranscriptionSegment, EmotionSegment, AcousticSegment
        
        # Count segments by type
        speaker_segments = self.get_segments_by_type(SpeakerSegment)
        transcription_segments = self.get_segments_by_type(TranscriptionSegment)
        emotion_segments = self.get_segments_by_type(EmotionSegment)
        acoustic_segments = self.get_segments_by_type(AcousticSegment)
        
        # Calculate speaking statistics
        speaker_stats = {}
        total_speaking_time = 0.0
        
        for seg in speaker_segments:
            if seg.speaker_id not in speaker_stats:
                speaker_stats[seg.speaker_id] = {
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'speaking_percentage': 0.0
                }
            
            speaker_stats[seg.speaker_id]['total_duration'] += seg.duration
            speaker_stats[seg.speaker_id]['segment_count'] += 1
            total_speaking_time += seg.duration
        
        # Calculate percentages
        for speaker_id in speaker_stats:
            if total_speaking_time > 0:
                speaker_stats[speaker_id]['speaking_percentage'] = (
                    speaker_stats[speaker_id]['total_duration'] / self.media_info.duration * 100
                )
        
        # Calculate emotion statistics
        emotion_counts = {}
        for seg in emotion_segments:
            emotion = seg.predicted_emotion
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        dominant_emotion = 'neutral'
        if emotion_counts:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate transcription statistics
        total_words = sum(seg.word_count or 0 for seg in transcription_segments)
        
        return AnalysisStatistics(
            total_segments=len(self.segments),
            total_speakers=len(speaker_stats),
            speaker_distribution=speaker_stats,
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_counts,
            total_words=total_words,
            total_duration=self.media_info.duration,
            segments_by_type={
                'speaker': len(speaker_segments),
                'transcription': len(transcription_segments),
                'emotion': len(emotion_segments),
                'acoustic': len(acoustic_segments)
            }
        )
    
    def _invalidate_caches(self) -> None:
        """Invalidate cached properties."""
        self._timeline_cache = None
        self._statistics_cache = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'media_info': self.media_info.to_dict(),
            'segments': [seg.to_export_dict() for seg in self.segments],
            'metadata': self.metadata.to_dict(),
            'statistics': self.get_statistics().to_dict(),
            'timeline': [event.to_dict() for event in self.get_timeline()]
        }