"""
Unified data models for audio analysis pipeline.
"""

from .base import BaseSegment, ProcessingContext, ProcessingResult
from .segments import (
    SpeakerSegment, 
    TranscriptionSegment, 
    EmotionSegment, 
    AcousticSegment,
    SpeechSegment
)
from .containers import AnalysisResults, MediaInfo, ProcessingMetadata
from .timeline import TimelineEvent, AnalysisStatistics
from .errors import ProcessingError, ProcessingWarning, ErrorPolicy, ErrorAction

__all__ = [
    'BaseSegment',
    'ProcessingContext', 
    'ProcessingResult',
    'SpeakerSegment',
    'TranscriptionSegment', 
    'EmotionSegment',
    'AcousticSegment',
    'SpeechSegment',
    'AnalysisResults',
    'MediaInfo',
    'ProcessingMetadata',
    'TimelineEvent',
    'AnalysisStatistics', 
    'ProcessingError',
    'ProcessingWarning',
    'ErrorPolicy',
    'ErrorAction'
]