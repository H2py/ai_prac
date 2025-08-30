# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create test audio file for development
python create_test_audio.py

# Quick test of the pipeline
python main.py test_audio.wav
```

### Running Tests
```bash
# Run main pipeline test
python tests/test_pipeline.py

# Run enhanced video pipeline tests  
python -m pytest tests/test_enhanced_video_pipeline.py -v

# Run all tests
python -m pytest tests/ -v

# Test with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality Tools
```bash
# Format code
black .

# Lint with ruff
ruff check .

# Type checking
mypy src/

# Run all quality checks
black . && ruff check . && mypy src/
```

### Pipeline Usage
```bash
# Basic audio analysis
python main.py audio.wav

# Enhanced video processing with multiple formats
python main.py video.mp4 --format json ass vtt backend_api

# YouTube processing with embed support
python main.py "https://youtube.com/embed/abc123" --format json frontend_json

# Full analysis with all features
python main.py input.wav --enable-stt --export-ass --use-enhanced-format --verbose

# Show supported formats and capabilities
python main.py formats

# Generate sample configuration
python main.py generate-config --output config.yaml
```

## Architecture Overview

### Core Pipeline Architecture *(Simplified from 37 to 30 files)*
This is a **modular pipeline system** built around the **Pipeline Pattern** with five distinct processing steps:

1. **AudioExtractionStep**: Extracts audio from files/URLs (supports MP4, YouTube, YouTube embeds)
2. **SpeakerDiarizationStep**: Identifies and separates speakers using pyannote.audio
3. **EmotionAnalysisStep**: Recognizes emotions from speech segments  
4. **AcousticAnalysisStep**: Extracts acoustic features (pitch, energy, spectral features)
5. **SpeechRecognitionStep**: Converts speech to text using OpenAI Whisper (optional)

**Recent Improvements:**
- ✅ Removed unused `src/processors/` and `src/resources/` directories
- ✅ Simplified test suite (10 → 4 test files)
- ✅ Fixed all Pylance type checking errors
- ✅ Enhanced legacy compatibility for EmotionSegment

### Key Architectural Components

**Pipeline Pattern (`src/pipeline/`)**:
- `PipelineStep`: Abstract base class for all processing steps (`src/pipeline/base.py:104`)
- `PipelineContext`: Shared data container that flows between steps (`src/pipeline/base.py:14`)
- `AudioAnalysisPipeline`: Main orchestrator that executes steps sequentially (`src/pipeline/pipeline.py:24`)
- `PipelineBuilder`: Fluent builder pattern for pipeline configuration (`src/pipeline/pipeline.py:262`)

**Enhanced Output System (`src/output_manager.py`)**:
- Unified system supporting 6 output formats: JSON, ASS, VTT, SRT, Backend API, Frontend JSON
- `OutputFormatManager`: Handles simultaneous export to multiple formats
- `OutputFormat` enum: Defines supported export formats
- Works with `src/output/` pipeline for advanced formatting

**Unified Data Models (`src/models/`)**:
- `BaseSegment`: Abstract base class for all analysis segments
- `SpeakerSegment`, `EmotionSegment`, `AcousticSegment`, `TranscriptionSegment`: Specialized segment types
- `AnalysisResults`: Container for all analysis data
- `ProcessingContext` and `ProcessingResult`: Pipeline execution containers

**Configuration System (`config/settings.py`)**:
- Dataclass-based configuration with YAML support
- `AudioConfig`, `ModelConfig`, `WhisperConfig`, `VideoConfig`, `OutputConfig` classes
- Environment variable overrides and validation

### Enhanced Video Processing Features
- **YouTube Embed URL Support**: Handles `youtube.com/embed/...` and `youtube-nocookie.com/embed/...`
- **Intelligent Extraction**: Auto-selects between MoviePy, FFmpeg, or parallel processing
- **Video Metadata**: Extracts resolution, codec, duration, file size using FFmpeg probe
- **Performance Optimization**: 20-40% faster with parallel processing methods

### Whisper Enhancements (`WHISPER_IMPROVEMENTS.md`)
- **Enhanced Language Detection**: Multi-sample analysis with caching (3 samples vs 5 for speed)
- **Voice Activity Detection**: Silero VAD preprocessing for better quality
- **Textual Anomaly Detection**: Fixes repetitive pattern issues ("AAAAA...")
- **Performance Monitoring**: Detailed timing and optimization tracking

## Configuration and Environment

### Required Environment Variables
```bash
# HuggingFace token for speaker diarization models
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

### Model Requirements and Setup
- **Speaker Diarization**: Requires HuggingFace token and model agreements:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`
  - `speechbrain/spkrec-ecapa-voxceleb`
- **Emotion Recognition**: Uses `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Speech Recognition**: Uses OpenAI Whisper with enhanced processing

### Configuration Loading Priority
1. Command-line arguments (highest priority)
2. YAML configuration file (`--config config.yaml`)
3. Environment variables
4. Default values (lowest priority)

## Key Implementation Details

### Error Handling Philosophy
The pipeline uses **graceful degradation** - if a step fails, it continues with the remaining steps rather than failing completely. The `--require-all` flag can override this behavior.

### Memory Management
- **Streaming Processing**: Large files are processed in chunks (configurable chunk_duration)
- **VAD Optimization**: 1-second chunks for Voice Activity Detection
- **GPU Memory**: Configurable memory fraction (default 0.8)

### Performance Optimizations
- **Parallel Video Processing**: Multiple extraction methods run simultaneously
- **Caching Systems**: Language detection results are cached by audio hash
- **Early Exit**: Language detection stops when confidence >90% after 2+ samples
- **Method Selection**: Auto-selects optimal extraction based on file size/format

### Output Format Specialization
- **Backend API**: Structured JSON for server integration with summary statistics
- **Frontend JSON**: Rich data with timeline, visualization components, and chart data
- **Enhanced JSON**: Complete metadata with standards compliance (W3C, SMPTE, IPA, PAD)
- **Professional Subtitles**: ASS with emotion-based styling, WebVTT with speaker tags

## Testing Strategy

### Test Structure
- `tests/test_enhanced_video_pipeline.py`: Comprehensive video processing tests
- `tests/test_audio_extractor.py`: Audio extraction functionality  
- `tests/test_pipeline.py`: Pipeline orchestration tests
- `tests/fixtures/`: Test data and mock objects

### Key Test Areas
- YouTube embed URL detection and video ID extraction
- Output format generation (all 6 formats)
- Video metadata extraction and method selection
- Error handling and graceful degradation
- Performance optimization validation

## Integration Points

### Backend Integration
```python
# Use Backend API format for server integration
python main.py video.mp4 --format backend_api
# Output: analysis_api.json with structured summary data
```

### Frontend Integration  
```python
# Use Frontend JSON for rich visualization
python main.py video.mp4 --format frontend_json
# Output: analysis_frontend.json with timeline and chart data
```

### Python API Usage
```python
from src.pipeline import PipelineBuilder
from config.settings import Config

config = Config.from_yaml("config.yaml")
pipeline = (PipelineBuilder()
    .with_config(config)
    .enable_stt("ko")
    .with_output_formats(["json", "backend_api"])
    .build())

results = pipeline.execute("input.wav")
```

This architecture supports both standalone CLI usage and programmatic integration, with comprehensive error handling, performance optimization, and flexible output formatting for various integration scenarios.

Code Guidelines for claude.md
Core Principles
When writing code in the claude.md file, prioritize the following two fundamental aspects:
1. Accuracy

Ensure all code logic is correct and produces expected results
Implement proper error handling and edge case management
Use precise algorithms and data structures appropriate for the task
Validate inputs and outputs thoroughly
Write code that behaves predictably under all conditions

2. Throughput (Performance/Speed)

Optimize for high processing speed and efficiency
Minimize computational complexity where possible
Use efficient algorithms and data structures
Avoid unnecessary operations and redundant calculations
Consider memory usage and resource optimization
Implement caching strategies when applicable
Profile and benchmark critical code paths

Implementation Strategy

Accuracy First: Ensure correctness before optimization
Measure Performance: Use profiling tools to identify bottlenecks
Iterative Improvement: Continuously refine both accuracy and speed
Documentation: Comment on performance-critical sections and accuracy considerations

Remember: Code must be both correct and fast to meet the standards for claude.md implementation.