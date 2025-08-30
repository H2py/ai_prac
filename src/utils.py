"""
Utilities for the audio analysis pipeline.

This module consolidates audio processing utilities and logging functionality
to provide a comprehensive set of tools for audio analysis and monitoring.
"""

import gc
import logging
import sys
import time
import numpy as np
import librosa
import soundfile as sf
import psutil
import colorama
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
from pydub import AudioSegment
from datetime import datetime
from colorama import Fore, Style
import tempfile
import os
from logging.handlers import RotatingFileHandler

# Initialize colorama
colorama.init(autoreset=True)


# =====================================
# Audio Processing Utilities
# =====================================

def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file with specified parameters.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        offset: Start reading after this time (in seconds)
        duration: Only load this duration (in seconds)
        normalize: Normalize audio to [-1, 1] range
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        IOError: If file cannot be loaded
    """
    try:
        load_kwargs = {
            'sr': sample_rate,
            'mono': mono
        }
        if offset is not None:
            load_kwargs['offset'] = offset
        if duration is not None:
            load_kwargs['duration'] = duration
            
        audio_data, sr = librosa.load(file_path, **load_kwargs)
        sr = int(sr) if sr is not None else sample_rate
        
        if normalize and len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        return audio_data, sr
        
    except Exception as e:
        raise IOError(f"Cannot load audio file: {e}")


def save_audio(
    audio_data: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    subtype: str = 'PCM_16',
    normalize: bool = True
) -> None:
    """Save audio data to file.
    
    Args:
        audio_data: Audio data array
        file_path: Output file path
        sample_rate: Sample rate
        subtype: Audio subtype for encoding
        normalize: Normalize before saving
        
    Raises:
        IOError: If file cannot be saved
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if normalize and len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        sf.write(file_path, audio_data, sample_rate, subtype=subtype)
        
    except Exception as e:
        raise IOError(f"Cannot save audio file: {e}")


def convert_audio_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: str = 'wav',
    sample_rate: int = 16000,
    mono: bool = True,
    bitrate: Optional[str] = None
) -> Path:
    """Convert audio file to different format.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        output_format: Output format (wav, mp3, flac, etc.)
        sample_rate: Target sample rate
        mono: Convert to mono if True
        bitrate: Output bitrate (for compressed formats)
        
    Returns:
        Path to converted file
        
    Raises:
        IOError: If conversion fails
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        audio = AudioSegment.from_file(str(input_path))
        
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)
        
        audio = audio.set_frame_rate(sample_rate)
        
        export_params = {'format': output_format}
        if bitrate:
            export_params['bitrate'] = bitrate
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), **export_params)
        
        return output_path
        
    except Exception as e:
        raise IOError(f"Audio conversion failed: {e}")


def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
        
    Raises:
        IOError: If file cannot be read
    """
    try:
        file_path = Path(file_path)
        info = sf.info(str(file_path))
        audio_data, sr = librosa.load(file_path, sr=None, duration=1.0)
        
        return {
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'duration_seconds': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'frames': info.frames,
            'sections': getattr(info, 'sections', 1),
            'seekable': getattr(info, 'seekable', True),
            'duration_formatted': format_duration(info.duration),
            'bit_depth': get_bit_depth(info.subtype),
            'is_mono': info.channels == 1,
            'is_stereo': info.channels == 2,
        }
        
    except Exception as e:
        raise IOError(f"Cannot read audio file info: {e}")


def split_audio_chunks(
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_duration: float = 30.0,
    overlap: float = 2.0,
    min_chunk_duration: float = 1.0
) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
    """Split audio into chunks with optional overlap.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
        min_chunk_duration: Minimum chunk duration in seconds
        
    Returns:
        List of (chunk_data, (start_time, end_time)) tuples
    """
    chunks = []
    total_duration = len(audio_data) / sample_rate
    
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    stride_samples = chunk_samples - overlap_samples
    
    if stride_samples <= 0:
        stride_samples = chunk_samples
        overlap_samples = 0
    
    position = 0
    while position < len(audio_data):
        end_position = min(position + chunk_samples, len(audio_data))
        chunk = audio_data[position:end_position]
        
        start_time = position / sample_rate
        end_time = end_position / sample_rate
        chunk_duration_actual = end_time - start_time
        
        if chunk_duration_actual >= min_chunk_duration:
            chunks.append((chunk, (start_time, end_time)))
        
        position += stride_samples
        
        if position + int(min_chunk_duration * sample_rate) > len(audio_data):
            break
    
    return chunks


def resample_audio(
    audio_data: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio_data: Audio data array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio_data
    
    return librosa.resample(
        audio_data,
        orig_sr=orig_sr,
        target_sr=target_sr
    )


def normalize_audio(
    audio_data: np.ndarray,
    method: str = 'peak',
    target_level: float = -3.0
) -> np.ndarray:
    """Normalize audio data.
    
    Args:
        audio_data: Audio data array
        method: Normalization method ('peak' or 'rms')
        target_level: Target level in dB
        
    Returns:
        Normalized audio data
    """
    if len(audio_data) == 0:
        return audio_data
    
    if method == 'peak':
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            target_linear = 10 ** (target_level / 20)
            normalized = audio_data * (target_linear / max_val)
        else:
            normalized = audio_data
    
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms > 0:
            target_linear = 10 ** (target_level / 20)
            normalized = audio_data * (target_linear / rms)
        else:
            normalized = audio_data
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized


def trim_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Trim silence from beginning and end of audio.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation
        
    Returns:
        Tuple of (trimmed_audio, (start_time, end_time))
    """
    trimmed, index = librosa.effects.trim(
        audio_data,
        top_db=-threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    start_time = index[0] / sample_rate
    end_time = index[1] / sample_rate
    
    return trimmed, (start_time, end_time)


def apply_preprocessing(
    audio_data: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
    trim_silence_flag: bool = True,
    pre_emphasis: bool = False,
    pre_emphasis_coef: float = 0.97
) -> np.ndarray:
    """Apply standard preprocessing to audio.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        normalize: Apply normalization
        trim_silence_flag: Trim silence from ends
        pre_emphasis: Apply pre-emphasis filter
        pre_emphasis_coef: Pre-emphasis coefficient
        
    Returns:
        Preprocessed audio data
    """
    processed = audio_data.copy()
    
    if trim_silence_flag:
        processed, _ = trim_silence(processed, sample_rate)
    
    if pre_emphasis:
        processed = librosa.effects.preemphasis(processed, coef=pre_emphasis_coef)
    
    if normalize:
        processed = normalize_audio(processed, method='peak')
    
    return processed


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def get_bit_depth(subtype: str) -> Optional[int]:
    """Get bit depth from audio subtype.
    
    Args:
        subtype: Audio subtype string
        
    Returns:
        Bit depth or None if unknown
    """
    bit_depth_map = {
        'PCM_16': 16,
        'PCM_24': 24,
        'PCM_32': 32,
        'FLOAT': 32,
        'DOUBLE': 64,
        'PCM_S8': 8,
        'PCM_U8': 8,
    }
    
    for key, depth in bit_depth_map.items():
        if key in subtype:
            return depth
    
    return None


def create_temp_audio_file(
    audio_data: np.ndarray,
    sample_rate: int,
    suffix: str = '.wav'
) -> Path:
    """Create a temporary audio file.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        suffix: File suffix
        
    Returns:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        save_audio(audio_data, temp_path, sample_rate)
        return temp_path


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists() or not file_path.is_file():
            return False
        
        info = sf.info(str(file_path))
        
        if info.duration <= 0 or info.samplerate <= 0:
            return False
        
        return True
        
    except:
        return False


# =====================================
# Logging Utilities
# =====================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        message = super().format(record)
        record.levelname = levelname
        
        return message


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, list] = {}
        
    def start_timer(self, name: str) -> None:
        """Start a timer for a named operation.
        
        Args:
            name: Name of the operation
        """
        self.start_times[name] = time.time()
        self.logger.debug(f"Started timer for: {name}")
        
    def stop_timer(self, name: str) -> float:
        """Stop a timer and log the duration.
        
        Args:
            name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        del self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        self.logger.info(f"Completed {name} in {duration:.2f} seconds")
        return duration
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent
        
        self.logger.info(
            f"Memory usage: {memory_mb:.1f} MB "
            f"(System: {system_percent:.1f}%)"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        for name, times in self.metrics.items():
            summary[name] = {
                'count': len(times),
                'total': sum(times),
                'average': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        return summary
    
    def log_summary(self) -> None:
        """Log performance summary."""
        summary = self.get_summary()
        if not summary:
            return
        
        self.logger.info("Performance Summary:")
        self.logger.info("-" * 50)
        
        for name, stats in summary.items():
            self.logger.info(
                f"{name}: "
                f"Count={stats['count']}, "
                f"Total={stats['total']:.2f}s, "
                f"Avg={stats['average']:.2f}s, "
                f"Min={stats['min']:.2f}s, "
                f"Max={stats['max']:.2f}s"
            )


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, total: Optional[int] = None):
        """Initialize progress logger.
        
        Args:
            logger: Base logger instance
            total: Total number of items to process
        """
        self.logger = logger
        self.total = total
        self.current = 0
        self.start_time = time.time()
        
    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """Update progress.
        
        Args:
            increment: Number of items completed
            message: Optional progress message
        """
        self.current += increment
        
        if self.total:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
            else:
                eta = 0
            
            progress_msg = (
                f"Progress: {self.current}/{self.total} "
                f"({percentage:.1f}%) - "
                f"Elapsed: {elapsed:.1f}s - "
                f"ETA: {eta:.1f}s"
            )
        else:
            elapsed = time.time() - self.start_time
            progress_msg = f"Progress: {self.current} items - Elapsed: {elapsed:.1f}s"
        
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark operation as complete.
        
        Args:
            message: Optional completion message
        """
        elapsed = time.time() - self.start_time
        complete_msg = f"Completed {self.current} items in {elapsed:.1f}s"
        
        if message:
            complete_msg += f" - {message}"
        
        self.logger.info(complete_msg)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        log_to_console: Whether to log to console
        log_format: Log message format
        date_format: Date format for log messages
        use_colors: Whether to use colored console output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if use_colors:
            console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[str] = None,
        suppress: bool = False
    ):
        """Initialize log context.
        
        Args:
            logger: Logger to configure
            level: Temporary log level
            suppress: Whether to suppress all logging
        """
        self.logger = logger
        self.original_level = logger.level
        self.original_disabled = logger.disabled
        
        if suppress:
            self.new_level = logging.CRITICAL + 1
            self.disabled = True
        elif level:
            self.new_level = getattr(logging, level.upper())
            self.disabled = False
        else:
            self.new_level = self.original_level
            self.disabled = self.original_disabled
    
    def __enter__(self):
        """Enter context."""
        self.logger.setLevel(self.new_level)
        self.logger.disabled = self.disabled
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.original_level)
        self.logger.disabled = self.original_disabled


def log_exception(logger: logging.Logger, exception: Exception, context: Optional[str] = None) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Optional context information
    """
    error_msg = f"Exception occurred: {type(exception).__name__}: {str(exception)}"
    
    if context:
        error_msg = f"{context} - {error_msg}"
    
    logger.error(error_msg, exc_info=True)


def create_file_logger(
    name: str,
    log_file: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """Create a rotating file logger.
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger