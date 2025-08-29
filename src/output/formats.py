"""
Output format definitions and data structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.models import AnalysisResults, TimelineEvent, AnalysisStatistics


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    ASS = "ass" 
    VTT = "vtt"
    SRT = "srt"
    BACKEND_API = "backend_api"
    FRONTEND_JSON = "frontend_json"


@dataclass
class ExportData:
    """Unified data structure for export operations."""
    
    # Core analysis data
    results: AnalysisResults
    
    # Derived data (computed by DataTransformer)
    timeline: List[TimelineEvent] = field(default_factory=list)
    statistics: Optional[AnalysisStatistics] = None
    
    # Format-specific data
    backend_summary: Optional[Dict[str, Any]] = None
    frontend_visualization: Optional[Dict[str, Any]] = None
    subtitle_data: Optional[Dict[str, Any]] = None
    
    # Export metadata
    export_timestamp: Optional[str] = None
    base_filename: Optional[str] = None
    
    def get_transcription_segments(self) -> List[Dict[str, Any]]:
        """Get transcription segments for subtitle formats."""
        from src.models.segments import TranscriptionSegment
        
        transcription_segments = self.results.get_segments_by_type(TranscriptionSegment)
        return [seg.to_export_dict() for seg in transcription_segments]
    
    def get_speaker_segments(self) -> List[Dict[str, Any]]:
        """Get speaker segments."""
        from src.models.segments import SpeakerSegment
        
        speaker_segments = self.results.get_segments_by_type(SpeakerSegment)
        return [seg.to_export_dict() for seg in speaker_segments]
    
    def get_emotion_segments(self) -> List[Dict[str, Any]]:
        """Get emotion segments."""
        from src.models.segments import EmotionSegment
        
        emotion_segments = self.results.get_segments_by_type(EmotionSegment)
        return [seg.to_export_dict() for seg in emotion_segments]
    
    def has_transcriptions(self) -> bool:
        """Check if export data has transcription segments."""
        from src.models.segments import TranscriptionSegment
        return len(self.results.get_segments_by_type(TranscriptionSegment)) > 0
    
    def has_speakers(self) -> bool:
        """Check if export data has speaker segments.""" 
        from src.models.segments import SpeakerSegment
        return len(self.results.get_segments_by_type(SpeakerSegment)) > 0
    
    def has_emotions(self) -> bool:
        """Check if export data has emotion segments."""
        from src.models.segments import EmotionSegment
        return len(self.results.get_segments_by_type(EmotionSegment)) > 0