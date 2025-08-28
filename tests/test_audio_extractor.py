"""
Test cases for the audio extractor module.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from src.audio_extractor import AudioExtractor
from src.utils.audio_utils import create_temp_audio_file, validate_audio_file
from config.settings import AudioConfig


@pytest.fixture
def audio_config():
    """Create test audio configuration."""
    config = AudioConfig(
        sample_rate=16000,
        mono=True,
        normalize=True,
        chunk_duration=10.0,
        overlap=1.0
    )
    return config


@pytest.fixture
def audio_extractor(audio_config):
    """Create audio extractor instance."""
    return AudioExtractor(config=audio_config)


@pytest.fixture
def sample_audio_file():
    """Create a temporary sample audio file for testing."""
    # Generate sample audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Create temporary file
    temp_path = create_temp_audio_file(audio_data, sample_rate, suffix='.wav')
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestAudioExtractor:
    """Test cases for AudioExtractor class."""
    
    def test_initialization(self, audio_config):
        """Test audio extractor initialization."""
        extractor = AudioExtractor(config=audio_config)
        
        assert extractor.config == audio_config
        assert extractor.temp_dir.exists()
        assert 'format' in extractor.ydl_opts
    
    def test_extract_audio_file(self, audio_extractor, sample_audio_file):
        """Test extracting audio from an audio file."""
        output_path = audio_extractor.extract(sample_audio_file)
        
        assert output_path.exists()
        assert validate_audio_file(output_path)
        
        # Check audio properties
        info = sf.info(str(output_path))
        assert info.samplerate == audio_extractor.config.sample_rate
        assert info.channels == 1  # mono
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
    
    def test_extract_with_custom_output(self, audio_extractor, sample_audio_file):
        """Test extracting audio with custom output path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_output.wav"
            
            result = audio_extractor.extract(
                sample_audio_file,
                output_path=output_path
            )
            
            assert result == output_path
            assert output_path.exists()
            assert validate_audio_file(output_path)
    
    def test_is_url(self, audio_extractor):
        """Test URL detection."""
        assert audio_extractor._is_url("https://www.example.com/video.mp4")
        assert audio_extractor._is_url("http://example.com")
        assert not audio_extractor._is_url("/path/to/file.mp4")
        assert not audio_extractor._is_url("file.mp4")
    
    def test_is_youtube_url(self, audio_extractor):
        """Test YouTube URL detection."""
        assert audio_extractor._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert audio_extractor._is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert audio_extractor._is_youtube_url("http://m.youtube.com/watch?v=test")
        assert not audio_extractor._is_youtube_url("https://vimeo.com/123456")
    
    def test_extract_video_id(self, audio_extractor):
        """Test YouTube video ID extraction."""
        # Standard YouTube URL
        video_id = audio_extractor._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert video_id == "dQw4w9WgXcQ"
        
        # Short YouTube URL
        video_id = audio_extractor._extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        assert video_id == "dQw4w9WgXcQ"
        
        # URL with additional parameters
        video_id = audio_extractor._extract_video_id("https://www.youtube.com/watch?v=test123&t=10s")
        assert video_id == "test123"
    
    def test_is_video_file(self, audio_extractor):
        """Test video file detection."""
        # Create temporary video file paths (don't need actual files for this test)
        assert audio_extractor._is_video_file(sample_audio_file) == False
        
        # Test with video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        for ext in video_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext) as temp_file:
                assert audio_extractor._is_video_file(temp_file.name) == True
    
    def test_invalid_source(self, audio_extractor):
        """Test extraction with invalid source."""
        with pytest.raises(ValueError, match="Invalid source"):
            audio_extractor.extract("nonexistent_file.xyz")
    
    def test_cleanup_temp_files(self, audio_extractor):
        """Test temporary file cleanup."""
        # Create some temp files
        temp_file = audio_extractor.temp_dir / "test_temp.wav"
        temp_file.write_text("test")
        
        assert temp_file.exists()
        
        audio_extractor.cleanup_temp_files()
        
        assert not temp_file.exists()
    
    def test_get_supported_formats(self, audio_extractor):
        """Test getting supported formats."""
        formats = audio_extractor.get_supported_formats()
        
        assert 'input_audio' in formats
        assert 'input_video' in formats
        assert 'output' in formats
        assert 'sources' in formats
        
        assert '.wav' in formats['input_audio']
        assert '.mp4' in formats['input_video']
        assert '.wav' in formats['output']
        assert 'YouTube URLs' in formats['sources']


class TestAudioExtractorIntegration:
    """Integration tests for audio extraction pipeline."""
    
    def test_full_extraction_pipeline(self, audio_extractor, sample_audio_file):
        """Test complete extraction pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_path = output_dir / "extracted.wav"
            
            # Extract audio
            result = audio_extractor.extract(
                sample_audio_file,
                output_path=output_path
            )
            
            # Verify result
            assert result.exists()
            assert result == output_path
            
            # Verify audio properties
            info = sf.info(str(result))
            assert info.samplerate == 16000
            assert info.channels == 1
            
            # Cleanup
            audio_extractor.cleanup_temp_files()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])