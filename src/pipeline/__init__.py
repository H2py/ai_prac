"""
Audio Analysis Pipeline Package

This package contains a refactored, modular pipeline architecture for audio analysis.

Key components:
- PipelineStep: Abstract base class for analysis steps
- PipelineContext: Shared data container between steps  
- AudioAnalysisPipeline: Main pipeline orchestrator
- ErrorHandler: Centralized error handling
- ResultProcessor: Result processing and export
- PipelineBuilder: Builder pattern for pipeline configuration

Example usage:

    from src.pipeline import PipelineBuilder
    from config.settings import Config
    
    # Build pipeline
    config = Config.from_env()
    pipeline = (PipelineBuilder()
                .with_config(config)
                .enable_stt('ko')
                .use_enhanced_format()
                .build())
    
    # Execute
    results = pipeline.execute('input.wav')
"""

from .base import PipelineStep, PipelineContext, AuthenticationError, ModelLoadError, ProcessingError
from .steps import (
    AudioExtractionStep,
    SpeakerDiarizationStep, 
    EmotionAnalysisStep,
    AcousticAnalysisStep,
    SpeechRecognitionStep
)
from .handlers import ErrorHandler, ResultProcessor
from .pipeline import AudioAnalysisPipeline, PipelineBuilder

__all__ = [
    # Base classes
    'PipelineStep',
    'PipelineContext',
    'AuthenticationError',
    'ModelLoadError', 
    'ProcessingError',
    
    # Step classes
    'AudioExtractionStep',
    'SpeakerDiarizationStep',
    'EmotionAnalysisStep', 
    'AcousticAnalysisStep',
    'SpeechRecognitionStep',
    
    # Handler classes
    'ErrorHandler',
    'ResultProcessor',
    
    # Main pipeline
    'AudioAnalysisPipeline',
    'PipelineBuilder',
]