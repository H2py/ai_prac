"""
Refactored processors with dependency injection and unified interfaces.
"""

from .base import BaseProcessor
from .speaker_processor import SpeakerDiarizationProcessor
from .speech_processor import SpeechRecognitionProcessor
from .emotion_processor import EmotionAnalysisProcessor
from .vad_processor import VADProcessor

__all__ = [
    'BaseProcessor',
    'SpeakerDiarizationProcessor', 
    'SpeechRecognitionProcessor',
    'EmotionAnalysisProcessor',
    'VADProcessor'
]