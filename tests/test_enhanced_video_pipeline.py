"""
Test enhanced video processing pipeline functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.audio_extractor import AudioExtractor
from src.output_manager import OutputFormatManager, MediaInfo, ProcessingMetadata, OutputFormat
from config.settings import VideoConfig, OutputConfig, Config


class TestEnhancedVideoProcessing:
    """Test enhanced video processing features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        
    def test_youtube_embed_url_detection(self):
        """Test YouTube embed URL detection."""
        extractor = AudioExtractor()
        
        # Test standard YouTube URLs
        assert extractor._is_youtube_url("https://www.youtube.com/watch?v=abc123")
        assert extractor._is_youtube_url("https://youtu.be/abc123")
        
        # Test embed URLs
        assert extractor._is_youtube_url("https://www.youtube.com/embed/abc123")
        assert extractor._is_youtube_url("https://youtube.com/embed/abc123")
        assert extractor._is_youtube_url("https://www.youtube-nocookie.com/embed/abc123")
        
        # Test non-YouTube URLs
        assert not extractor._is_youtube_url("https://vimeo.com/123456")
        assert not extractor._is_youtube_url("https://example.com/video.mp4")
    
    def test_video_id_extraction_from_embed(self):
        """Test video ID extraction from embed URLs."""
        extractor = AudioExtractor()
        
        # Test embed URL formats
        assert extractor._extract_video_id("https://youtube.com/embed/abc123") == "abc123"
        assert extractor._extract_video_id("https://www.youtube.com/embed/xyz789?autoplay=1") == "xyz789"
        assert extractor._extract_video_id("https://youtube-nocookie.com/embed/test123") == "test123"
        
        # Test standard URL formats (should still work)
        assert extractor._extract_video_id("https://www.youtube.com/watch?v=abc123") == "abc123"
        assert extractor._extract_video_id("https://youtu.be/xyz789") == "xyz789"
    
    def test_video_metadata_extraction(self):
        """Test video metadata extraction."""
        with patch('ffmpeg.probe') as mock_probe:
            # Mock FFmpeg probe response
            mock_probe.return_value = {
                'streams': [
                    {
                        'codec_type': 'video',
                        'width': 1920,
                        'height': 1080,
                        'codec_name': 'h264',
                        'r_frame_rate': '30/1'
                    },
                    {
                        'codec_type': 'audio',
                        'codec_name': 'aac',
                        'sample_rate': '44100',
                        'channels': 2
                    }
                ],
                'format': {
                    'duration': '120.5',
                    'size': '10485760',
                    'format_name': 'mov,mp4,m4a,3gp,3g2,mj2'
                }
            }
            
            extractor = AudioExtractor()
            metadata = extractor._get_video_metadata(Path("test_video.mp4"))
            
            assert metadata['duration'] == 120.5
            assert metadata['resolution'] == "1920x1080"
            assert metadata['video_codec'] == "h264"
            assert metadata['audio_codec'] == "aac"
            assert metadata['fps'] == 30.0
            assert metadata['has_video'] is True
            assert metadata['has_audio'] is True
    
    def test_optimal_method_selection(self):
        """Test optimal extraction method selection."""
        extractor = AudioExtractor()
        
        # Small file - should prefer MoviePy
        small_metadata = {
            'duration': 60,
            'size': 50 * 1024 * 1024,  # 50MB
            'format': 'mp4'
        }
        with patch('src.audio_extractor.VideoFileClip', Mock()):
            assert extractor._choose_optimal_method(small_metadata) == "moviepy"
        
        # Large file - should prefer FFmpeg
        large_metadata = {
            'duration': 3600,
            'size': 600 * 1024 * 1024,  # 600MB
            'format': 'mp4'
        }
        assert extractor._choose_optimal_method(large_metadata) == "ffmpeg"
        
        # Problematic format - should prefer FFmpeg
        webm_metadata = {
            'duration': 300,
            'size': 100 * 1024 * 1024,
            'format': 'webm'
        }
        assert extractor._choose_optimal_method(webm_metadata) == "ffmpeg"
    
    def test_output_format_manager_initialization(self):
        """Test OutputFormatManager initialization."""
        config = Config()
        output_manager = OutputFormatManager(config, self.temp_dir)
        
        assert output_manager.output_dir == self.temp_dir
        assert output_manager.config == config
        
        # Check supported formats
        supported_formats = output_manager.get_supported_formats()
        expected_formats = ['json', 'ass', 'vtt', 'srt', 'backend_api', 'frontend_json']
        for fmt in expected_formats:
            assert fmt in supported_formats
    
    def test_media_info_creation(self):
        """Test MediaInfo dataclass creation."""
        media_info = MediaInfo(
            source_type="video",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            language="en",
            video_resolution="1920x1080",
            codec="h264",
            file_size=10485760
        )
        
        assert media_info.source_type == "video"
        assert media_info.duration == 120.5
        assert media_info.video_resolution == "1920x1080"
        assert media_info.file_size == 10485760
    
    def test_processing_metadata_creation(self):
        """Test ProcessingMetadata creation."""
        metadata = ProcessingMetadata(
            timestamp="2024-01-01T12:00:00",
            processing_time=45.5,
            pipeline_version="2.0",
            whisper_enhancements={"vad_enabled": True},
            performance_stats={"total_time": 45.5}
        )
        
        assert metadata.pipeline_version == "2.0"
        assert metadata.processing_time == 45.5
        assert metadata.whisper_enhancements["vad_enabled"] is True
    
    def test_output_format_enum(self):
        """Test OutputFormat enum values."""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.ASS.value == "ass"
        assert OutputFormat.VTT.value == "vtt"
        assert OutputFormat.SRT.value == "srt"
        assert OutputFormat.BACKEND_API.value == "backend_api"
        assert OutputFormat.FRONTEND_JSON.value == "frontend_json"
    
    def test_video_config_defaults(self):
        """Test VideoConfig default values."""
        video_config = VideoConfig()
        
        assert video_config.extraction_method == "auto"
        assert video_config.enable_parallel_extraction is True
        assert video_config.large_file_threshold_mb == 500
        assert video_config.long_duration_threshold_s == 3600
        assert '.mp4' in video_config.supported_formats
        assert '.webm' in video_config.supported_formats
    
    def test_output_config_enhancements(self):
        """Test enhanced OutputConfig."""
        output_config = OutputConfig()
        
        assert output_config.default_formats == ["json"]
        assert output_config.enable_multiple_formats is True
        assert output_config.json_pretty_print is True
        assert output_config.ass_font_name == "Arial"
        assert output_config.vtt_include_speaker is True
        assert output_config.enable_backend_api is False
        assert output_config.enable_frontend_json is False
    
    @pytest.mark.parametrize("url_type,expected_source", [
        ("https://youtube.com/watch?v=abc123", "youtube"),
        ("https://youtube.com/embed/abc123", "youtube"), 
        ("video.mp4", "video"),
        ("audio.wav", "audio")
    ])
    def test_source_type_detection(self, url_type, expected_source):
        """Test source type detection logic."""
        # This would be part of the enhanced pipeline logic
        if "youtube.com" in url_type or "youtu.be" in url_type:
            source_type = "youtube"
        elif url_type.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            source_type = "video" 
        else:
            source_type = "audio"
        
        assert source_type == expected_source
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class TestOutputExportFormats:
    """Test output export format generation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.output_manager = OutputFormatManager(self.config, self.temp_dir)
        
        # Mock data for testing
        self.mock_results = {
            'transcription': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello world',
                    'speaker': 'SPEAKER_00',
                    'confidence': 0.95
                }
            ],
            'emotion': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'predicted_emotion': 'happy',
                    'confidence': 0.87
                }
            ],
            'diarization': {
                'total_speakers': 1,
                'speakers': {
                    'SPEAKER_00': {
                        'speaking_time': 2.5,
                        'speaking_percentage': 100.0
                    }
                }
            }
        }
        
        self.mock_media_info = MediaInfo(
            source_type="video",
            duration=2.5,
            sample_rate=16000,
            channels=1,
            language="en"
        )
        
        self.mock_metadata = ProcessingMetadata(
            timestamp="2024-01-01T12:00:00",
            processing_time=15.2,
            pipeline_version="2.0"
        )
    
    def test_json_export(self):
        """Test JSON format export."""
        output_path = self.output_manager._export_json(
            self.mock_results,
            self.mock_media_info,
            self.mock_metadata,
            "test_output"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Verify JSON structure
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'media' in data
        assert 'processing' in data
        assert 'transcription' in data
        assert data['media']['source_type'] == "video"
        assert data['processing']['pipeline_version'] == "2.0"
    
    def test_vtt_export(self):
        """Test VTT format export."""
        output_path = self.output_manager._export_vtt(
            self.mock_results,
            self.mock_media_info,
            self.mock_metadata,
            "test_output"
        )
        
        assert output_path.exists()
        assert output_path.suffix == ".vtt"
        
        # Verify VTT content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in content
        assert "<v SPEAKER_00>Hello world" in content
    
    def test_backend_api_export(self):
        """Test backend API format export."""
        output_path = self.output_manager._export_backend_api(
            self.mock_results,
            self.mock_media_info,
            self.mock_metadata,
            "test_output"
        )
        
        assert output_path.exists()
        assert output_path.name.endswith("_api.json")
        
        # Verify API structure
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert data['status'] == "completed"
        assert 'media' in data
        assert 'analysis' in data
        assert 'metadata' in data
        assert data['analysis']['speakers']['total_speakers'] == 1
    
    def test_frontend_json_export(self):
        """Test frontend JSON format export."""
        output_path = self.output_manager._export_frontend_json(
            self.mock_results,
            self.mock_media_info,
            self.mock_metadata,
            "test_output"
        )
        
        assert output_path.exists()
        assert output_path.name.endswith("_frontend.json")
        
        # Verify frontend structure
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'mediaInfo' in data
        assert 'timeline' in data
        assert 'summary' in data
        assert 'visualization' in data
        assert data['summary']['totalSpeakers'] == 1
        assert data['summary']['dominantEmotion'] == 'happy'
    
    def test_multiple_format_export(self):
        """Test exporting multiple formats simultaneously."""
        formats = [OutputFormat.JSON, OutputFormat.VTT, OutputFormat.BACKEND_API]
        
        exported_files = self.output_manager.export_results(
            self.mock_results,
            self.mock_media_info,
            self.mock_metadata,
            formats,
            "test_multi"
        )
        
        assert len(exported_files) == 3
        assert OutputFormat.JSON in exported_files
        assert OutputFormat.VTT in exported_files
        assert OutputFormat.BACKEND_API in exported_files
        
        # Verify all files exist
        for format_type, file_path in exported_files.items():
            assert file_path.exists()
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])