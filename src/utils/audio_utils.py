"""
Audio processing utilities for the audio analysis pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import logging
from pydub import AudioSegment
import tempfile
import os


logger = logging.getLogger(__name__)


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
        # Load audio using librosa - handle optional parameters
        load_kwargs = {
            'sr': sample_rate,
            'mono': mono
        }
        if offset is not None:
            load_kwargs['offset'] = offset
        if duration is not None:
            load_kwargs['duration'] = duration
            
        audio_data, sr = librosa.load(file_path, **load_kwargs)
        
        # Ensure sr is int (librosa sometimes returns float)
        sr = int(sr) if sr is not None else sample_rate
        
        # Normalize if requested
        if normalize and len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        logger.debug(f"Loaded audio: shape={audio_data.shape}, sr={sr}")
        return audio_data, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
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
        
        # Normalize if requested
        if normalize and len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        # Save using soundfile
        sf.write(file_path, audio_data, sample_rate, subtype=subtype)
        logger.debug(f"Saved audio to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")
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
        
        # Load audio using pydub
        audio = AudioSegment.from_file(str(input_path))
        
        # Convert to mono if requested
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Prepare export parameters
        export_params = {'format': output_format}
        if bitrate:
            export_params['bitrate'] = bitrate
        
        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(output_path), **export_params)
        
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to convert audio format: {e}")
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
        
        # Get basic info using soundfile
        info = sf.info(str(file_path))
        
        # Load a small portion to get additional info
        audio_data, sr = librosa.load(file_path, sr=None, duration=1.0)
        
        audio_info = {
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
            # Additional computed properties
            'duration_formatted': format_duration(info.duration),
            'bit_depth': get_bit_depth(info.subtype),
            'is_mono': info.channels == 1,
            'is_stereo': info.channels == 2,
        }
        
        return audio_info
        
    except Exception as e:
        logger.error(f"Failed to get audio info for {file_path}: {e}")
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
        logger.warning("Overlap is too large, disabling overlap")
    
    position = 0
    while position < len(audio_data):
        # Get chunk
        end_position = min(position + chunk_samples, len(audio_data))
        chunk = audio_data[position:end_position]
        
        # Calculate times
        start_time = position / sample_rate
        end_time = end_position / sample_rate
        chunk_duration_actual = end_time - start_time
        
        # Only include chunk if it meets minimum duration
        if chunk_duration_actual >= min_chunk_duration:
            chunks.append((chunk, (start_time, end_time)))
        
        # Move to next position
        position += stride_samples
        
        # Break if we've reached the end
        if position + int(min_chunk_duration * sample_rate) > len(audio_data):
            break
    
    logger.debug(f"Split audio into {len(chunks)} chunks")
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
    
    resampled = librosa.resample(
        audio_data,
        orig_sr=orig_sr,
        target_sr=target_sr
    )
    
    logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
    return resampled


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
        # Peak normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Convert dB to linear
            target_linear = 10 ** (target_level / 20)
            normalized = audio_data * (target_linear / max_val)
        else:
            normalized = audio_data
    
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms > 0:
            target_linear = 10 ** (target_level / 20)
            normalized = audio_data * (target_linear / rms)
        else:
            normalized = audio_data
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Prevent clipping
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
    # Convert threshold to amplitude
    threshold = librosa.db_to_amplitude(threshold_db)
    
    # Trim silence
    trimmed, index = librosa.effects.trim(
        audio_data,
        top_db=-threshold_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Calculate trimmed times
    start_time = index[0] / sample_rate
    end_time = index[1] / sample_rate
    
    logger.debug(f"Trimmed silence: {start_time:.2f}s - {end_time:.2f}s")
    
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
    
    # Trim silence
    if trim_silence_flag:
        processed, _ = trim_silence(processed, sample_rate)
    
    # Apply pre-emphasis
    if pre_emphasis:
        processed = librosa.effects.preemphasis(processed, coef=pre_emphasis_coef)
    
    # Normalize
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
        
        if not file_path.exists():
            return False
        
        if not file_path.is_file():
            return False
        
        # Try to get info
        info = sf.info(str(file_path))
        
        # Check basic properties
        if info.duration <= 0 or info.samplerate <= 0:
            return False
        
        return True
        
    except:
        return False