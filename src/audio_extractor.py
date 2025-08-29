"""
Audio extraction module for processing MP4 files and YouTube URLs.
"""

import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple, cast
import logging
from urllib.parse import urlparse, parse_qs

import yt_dlp
try:
    from moviepy.editor import VideoFileClip  # type: ignore[import-untyped]
except ImportError:
    VideoFileClip = None
import ffmpeg

from src.utils.audio_utils import (
    load_audio, save_audio, convert_audio_format,
    get_audio_info, validate_audio_file, apply_preprocessing
)
from src.utils.logger import PerformanceLogger, log_exception
from config.settings import AudioConfig


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class AudioExtractor:
    """Extract and process audio from various sources."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio extractor.
        
        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()
        self.temp_dir = Path(tempfile.gettempdir()) / "audio_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure yt-dlp
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def extract(
        self,
        source: Union[str, Path],
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """Extract audio from source (file or URL).
        
        Args:
            source: Input source (file path or URL)
            output_path: Optional output path
            **kwargs: Additional extraction parameters
            
        Returns:
            Path to extracted audio file
            
        Raises:
            ValueError: If source is invalid
            IOError: If extraction fails
        """
        perf_logger.start_timer("audio_extraction")
        
        try:
            # Determine source type
            if self._is_url(str(source)):
                logger.info(f"Extracting audio from URL: {source}")
                extracted_path = self._extract_from_url(str(source), **kwargs)
            elif self._is_video_file(source):
                logger.info(f"Extracting audio from video file: {source}")
                extracted_path = self._extract_from_video(Path(source), **kwargs)
            elif validate_audio_file(source):
                logger.info(f"Processing audio file: {source}")
                extracted_path = self._process_audio_file(Path(source), **kwargs)
            else:
                raise ValueError(f"Invalid source: {source}")
            
            # Convert to target format if needed
            if output_path:
                output_path = Path(output_path)
                if extracted_path.suffix != output_path.suffix:
                    extracted_path = convert_audio_format(
                        extracted_path,
                        output_path,
                        output_format=output_path.suffix[1:],  # Remove dot
                        sample_rate=self.config.sample_rate,
                        mono=self.config.mono
                    )
                else:
                    # Move to output path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted_path.rename(output_path)
                    extracted_path = output_path
            
            # Apply standard processing
            extracted_path = self._apply_standard_processing(extracted_path)
            
            duration = perf_logger.stop_timer("audio_extraction")
            logger.info(f"Audio extraction completed in {duration:.2f}s")
            
            return extracted_path
            
        except Exception as e:
            perf_logger.stop_timer("audio_extraction")
            log_exception(logger, e, "Audio extraction failed")
            raise
    
    def extract_from_youtube(
        self,
        url: str,
        output_path: Optional[Path] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> Tuple[Path, Dict[str, Any]]:
        """Extract audio from YouTube URL with metadata.
        
        Args:
            url: YouTube URL
            output_path: Optional output path
            start_time: Start time in seconds
            end_time: End time in seconds
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (audio_path, metadata)
            
        Raises:
            ValueError: If URL is invalid
            IOError: If download fails
        """
        perf_logger.start_timer("youtube_extraction")
        
        try:
            # Validate YouTube URL
            if not self._is_youtube_url(url):
                raise ValueError(f"Invalid YouTube URL: {url}")
            
            # Get video info
            logger.info("Fetching video information...")
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                # Handle case where info might be None or not a dict
                if not info or not isinstance(info, dict):
                    raise IOError(f"Failed to extract video information from: {url}")
                    
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'url': url,
                    'video_id': self._extract_video_id(url),
                }
            
            logger.info(f"Downloading: {metadata['title']}")
            
            # Download audio
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
            
            # Find downloaded file
            downloaded_files = list(self.temp_dir.glob("*.wav"))
            if not downloaded_files:
                raise IOError("Failed to download audio from YouTube")
            
            audio_path = max(downloaded_files, key=lambda p: p.stat().st_mtime)
            
            # Trim if time range specified
            if start_time is not None or end_time is not None:
                audio_path = self._trim_audio(audio_path, start_time, end_time)
            
            # Move to output path if specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.rename(output_path)
                audio_path = output_path
            
            # Apply processing
            audio_path = self._apply_standard_processing(audio_path)
            
            duration = perf_logger.stop_timer("youtube_extraction")
            logger.info(f"YouTube extraction completed in {duration:.2f}s")
            
            return audio_path, metadata
            
        except Exception as e:
            perf_logger.stop_timer("youtube_extraction")
            log_exception(logger, e, "YouTube extraction failed")
            raise
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Path] = None,
        method: str = "auto",
        **kwargs
    ) -> Tuple[Path, Dict[str, Any]]:
        """Extract audio from video file with optimization and metadata.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path
            method: Extraction method ("auto", "moviepy", "ffmpeg", "parallel")
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (audio_path, video_metadata)
            
        Raises:
            IOError: If extraction fails
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise IOError(f"Video file not found: {video_path}")
        
        perf_logger.start_timer("video_extraction")
        
        try:
            # Get video metadata first
            video_metadata = self._get_video_metadata(video_path)
            logger.info(f"Video metadata: {video_metadata['duration']:.2f}s, {video_metadata.get('resolution', 'unknown')}")
            
            # Choose optimal extraction method
            if method == "auto":
                method = self._choose_optimal_method(video_metadata)
                logger.info(f"Auto-selected extraction method: {method}")
            
            if method == "parallel":
                audio_path = self._extract_with_parallel_methods(video_path, output_path, **kwargs)
            elif method == "moviepy":
                audio_path = self._extract_with_moviepy(video_path, output_path, **kwargs)
            elif method == "ffmpeg":
                audio_path = self._extract_with_ffmpeg(video_path, output_path, **kwargs)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
            
            duration = perf_logger.stop_timer("video_extraction")
            logger.info(f"Video extraction completed in {duration:.2f}s")
            
            return audio_path, video_metadata
            
        except Exception as e:
            perf_logger.stop_timer("video_extraction")
            logger.error(f"Video extraction failed: {e}")
            raise
    
    def _extract_from_url(self, url: str, **kwargs) -> Path:
        """Extract audio from URL.
        
        Args:
            url: Source URL
            **kwargs: Additional parameters
            
        Returns:
            Path to extracted audio
        """
        if self._is_youtube_url(url):
            audio_path, _ = self.extract_from_youtube(url, **kwargs)
            return audio_path
        else:
            # Try generic extraction with yt-dlp
            logger.info(f"Attempting generic extraction from: {url}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
            
            # Find downloaded file
            downloaded_files = list(self.temp_dir.glob("*.wav"))
            if not downloaded_files:
                raise IOError(f"Failed to extract audio from URL: {url}")
            
            return max(downloaded_files, key=lambda p: p.stat().st_mtime)
    
    def _extract_from_video(self, video_path: Path, **kwargs) -> Path:
        """Extract audio from video file.
        
        Args:
            video_path: Path to video file
            **kwargs: Additional parameters
            
        Returns:
            Path to extracted audio
        """
        audio_path, _ = self.extract_from_video(video_path, **kwargs)
        return audio_path
    
    def _process_audio_file(self, audio_path: Union[str, Path], **kwargs) -> Path:  # pylint: disable=unused-argument
        """Process existing audio file.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            Path to processed audio
        """
        audio_path = Path(audio_path)  # Ensure it's a Path object
        # Create temp copy for processing
        temp_path = self.temp_dir / f"processed_{audio_path.name}"
        
        # Load and process
        audio_data, sr = load_audio(
            audio_path,
            sample_rate=self.config.sample_rate,
            mono=self.config.mono,
            normalize=self.config.normalize
        )
        
        # Save processed audio
        save_audio(audio_data, temp_path, sr)
        
        return temp_path
    
    def _extract_with_moviepy(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        **kwargs  # pylint: disable=unused-argument
    ) -> Path:
        """Extract audio using MoviePy.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path
            **kwargs: Additional parameters
            
        Returns:
            Path to extracted audio
        """
        if VideoFileClip is None:
            raise ImportError("MoviePy is not installed. Please install with: pip install moviepy")
        
        try:
            # Load video
            video = VideoFileClip(str(video_path))
            
            if not video.audio:
                raise IOError(f"No audio stream found in video: {video_path}")
            
            # Set output path
            if not output_path:
                output_path = self.temp_dir / f"{video_path.stem}.wav"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            video.audio.write_audiofile(
                str(output_path),
                fps=self.config.sample_rate,
                logger=None  # Suppress MoviePy's verbose output
            )
            
            # Clean up
            video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"MoviePy extraction failed: {e}")
            raise IOError(f"Failed to extract audio with MoviePy: {e}")
    
    def _extract_with_ffmpeg(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        **kwargs  # pylint: disable=unused-argument
    ) -> Path:
        """Extract audio using FFmpeg.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path
            **kwargs: Additional parameters
            
        Returns:
            Path to extracted audio
        """
        try:
            # Set output path
            if not output_path:
                output_path = self.temp_dir / f"{video_path.stem}.wav"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg command
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec=self.config.output_codec,
                ar=self.config.sample_rate,
                ac=1 if self.config.mono else 2
            )
            
            # Run FFmpeg
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            return output_path
            
        except Exception as e:
            logger.error(f"FFmpeg extraction failed: {e}")
            raise IOError(f"Failed to extract audio with FFmpeg: {e}")
    
    def _apply_standard_processing(self, audio_path: Path) -> Path:
        """Apply standard audio processing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to processed audio
        """
        try:
            # Load audio
            audio_data, sr = load_audio(
                audio_path,
                sample_rate=self.config.sample_rate,
                mono=self.config.mono
            )
            
            # Apply preprocessing
            audio_data = apply_preprocessing(
                audio_data,
                sr,
                normalize=self.config.normalize,
                trim_silence_flag=True,
                pre_emphasis=False
            )
            
            # Save processed audio
            processed_path = audio_path.parent / f"processed_{audio_path.name}"
            save_audio(audio_data, processed_path, sr)
            
            # Replace original with processed
            processed_path.replace(audio_path)
            
            return audio_path
            
        except Exception as e:
            logger.warning(f"Standard processing failed, using original: {e}")
            return audio_path
    
    def _trim_audio(
        self,
        audio_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Path:
        """Trim audio to specified time range.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path to trimmed audio
        """
        # Load audio
        audio_data, sr = load_audio(audio_path)
        
        # Calculate sample indices
        start_sample = int(start_time * sr) if start_time else 0
        end_sample = int(end_time * sr) if end_time else len(audio_data)
        
        # Trim
        trimmed = audio_data[start_sample:end_sample]
        
        # Save trimmed audio
        trimmed_path = audio_path.parent / f"trimmed_{audio_path.name}"
        save_audio(trimmed, trimmed_path, sr)
        
        # Replace original
        trimmed_path.replace(audio_path)
        
        return audio_path
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL.
        
        Args:
            source: Input source
            
        Returns:
            True if URL, False otherwise
        """
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is from YouTube.
        
        Args:
            url: Input URL
            
        Returns:
            True if YouTube URL, False otherwise
        """
        youtube_patterns = [
            r'(https?://)?(www\.)?(youtube\.com|youtu\.be)',
            r'(https?://)?(www\.)?(m\.youtube\.com)',
            r'(https?://)?(www\.)?youtube\.com/embed',  # YouTube embed URLs
            r'(https?://)?(www\.)?youtube-nocookie\.com/embed',  # Privacy-enhanced embed
        ]
        
        return any(re.search(pattern, url) for pattern in youtube_patterns)
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None
        """
        # Handle youtu.be format
        if 'youtu.be' in url:
            parts = url.split('/')
            if parts:
                return parts[-1].split('?')[0]
        
        # Handle embed format (youtube.com/embed/VIDEO_ID)
        if '/embed/' in url:
            embed_match = re.search(r'/embed/([a-zA-Z0-9_-]+)', url)
            if embed_match:
                return embed_match.group(1)
        
        # Handle youtube.com format
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query)
            if 'v' in params:
                return params['v'][0]
        
        return None
    
    def _is_video_file(self, source: Union[str, Path]) -> bool:
        """Check if source is a video file.
        
        Args:
            source: Input source
            
        Returns:
            True if video file, False otherwise
        """
        video_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.flv',
            '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'
        }
        
        try:
            path = Path(source)
            return path.exists() and path.suffix.lower() in video_extensions
        except Exception:
            return False
    
    def _get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata using FFmpeg probe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video metadata dictionary
        """
        try:
            # Use ffmpeg.probe to get metadata
            probe = ffmpeg.probe(str(video_path))
            
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0)),
                'format': probe['format'].get('format_name', ''),
                'has_audio': audio_stream is not None,
                'has_video': video_stream is not None
            }
            
            if video_stream:
                metadata.update({
                    'resolution': f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                    'video_codec': video_stream.get('codec_name', ''),
                    'fps': self._parse_fps(video_stream.get('r_frame_rate', ''))
                })
            
            if audio_stream:
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name', ''),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0))
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to get video metadata: {e}")
            return {
                'duration': 0,
                'size': 0,
                'format': 'unknown',
                'has_audio': True,
                'has_video': True
            }
    
    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS from FFmpeg r_frame_rate format."""
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den)
            return float(fps_str)
        except Exception:
            return 0.0
    
    def _choose_optimal_method(self, metadata: Dict[str, Any]) -> str:
        """Choose optimal extraction method based on video metadata.
        
        Args:
            metadata: Video metadata
            
        Returns:
            Optimal extraction method
        """
        duration = metadata.get('duration', 0)
        size = metadata.get('size', 0)
        format_name = metadata.get('format', '').lower()
        
        # Use FFmpeg for large files or specific formats
        if size > 500 * 1024 * 1024 or duration > 3600:  # > 500MB or > 1 hour
            return "ffmpeg"
        
        # Use FFmpeg for problematic formats
        if any(fmt in format_name for fmt in ['webm', 'mkv', 'flv']):
            return "ffmpeg"
        
        # Use MoviePy for smaller files (better error handling)
        if VideoFileClip is not None:
            return "moviepy"
        
        # Fallback to FFmpeg
        return "ffmpeg"
    
    def _extract_with_parallel_methods(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """Try multiple extraction methods in parallel and use the fastest.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path
            **kwargs: Additional parameters
            
        Returns:
            Path to extracted audio
        """
        import concurrent.futures
        
        # Create temporary paths for each method
        temp_paths = {
            'moviepy': self.temp_dir / f"parallel_moviepy_{video_path.stem}.wav",
            'ffmpeg': self.temp_dir / f"parallel_ffmpeg_{video_path.stem}.wav"
        }
        
        methods = {}
        if VideoFileClip is not None:
            methods['moviepy'] = lambda: self._extract_with_moviepy(video_path, temp_paths['moviepy'], **kwargs)
        methods['ffmpeg'] = lambda: self._extract_with_ffmpeg(video_path, temp_paths['ffmpeg'], **kwargs)
        
        # Run methods in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(method): name 
                for name, method in methods.items()
            }
            
            # Get the first successful result
            for future in concurrent.futures.as_completed(futures):
                method_name = futures[future]
                try:
                    result_path = future.result()
                    logger.info(f"Parallel extraction: {method_name} completed first")
                    
                    # Clean up other temp files
                    for name, temp_path in temp_paths.items():
                        if name != method_name and temp_path.exists():
                            try:
                                temp_path.unlink()
                            except:
                                pass
                    
                    # Move to final output path if specified
                    if output_path:
                        output_path = Path(output_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        result_path.rename(output_path)
                        return output_path
                    
                    return result_path
                    
                except Exception as e:
                    logger.warning(f"Parallel extraction method {method_name} failed: {e}")
                    continue
        
        # If all methods failed
        raise IOError("All parallel extraction methods failed")

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")
    
    def get_supported_formats(self) -> Dict[str, list]:
        """Get supported input and output formats.
        
        Returns:
            Dictionary with supported formats
        """
        return {
            'input_audio': ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'],
            'input_video': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v'],
            'output': ['.wav', '.mp3', '.flac', '.ogg'],
            'sources': ['Local files', 'YouTube URLs', 'Other video URLs']
        }