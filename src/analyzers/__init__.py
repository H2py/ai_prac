"""
Audio Analysis Modules

This package contains all the core analysis components for the audio analysis pipeline.
Each analyzer focuses on a single responsibility (SRP compliance).

Available Analyzers:
- AudioExtractor: Extract audio from various sources (files, URLs, videos)
- EmotionAnalyzer: Emotion recognition from speech segments
- SpeakerDiarizer: Speaker identification and diarization
- SpeechRecognizer: Speech-to-text transcription using Whisper
- AcousticAnalyzer: Acoustic feature extraction
- VADProcessor: Voice Activity Detection
"""

from .audio_extractor import AudioExtractor
from .emotion_analyzer import EmotionAnalyzer
from .speaker_diarizer import SpeakerDiarizer, SpeakerInfo
from .speech_recognizer import SpeechRecognizer
from .acoustic_analyzer import AcousticAnalyzer
from .vad_processor import VADProcessor, SpeechSegment

__all__ = [
    'AudioExtractor',
    'EmotionAnalyzer',
    'SpeakerDiarizer',
    'SpeakerInfo',
    'SpeechRecognizer',
    'AcousticAnalyzer', 
    'VADProcessor',
    'SpeechSegment'
]