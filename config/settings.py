"""
Configuration settings for the audio analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import os


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    
    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    min_duration: float = 0.5  # Minimum segment duration in seconds
    max_duration: Optional[float] = None  # Maximum duration to process
    chunk_duration: float = 30.0  # Duration for chunked processing
    overlap: float = 2.0  # Overlap between chunks in seconds
    
    # Format conversion settings
    output_format: str = "wav"
    output_codec: str = "pcm_s16le"
    output_bitrate: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration for AI components."""
    
    # Speaker diarization
    speaker_model: str = "pyannote/speaker-diarization-3.1"
    speaker_min_speakers: Optional[int] = None
    speaker_max_speakers: Optional[int] = None
    
    # Emotion recognition
    emotion_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    emotion_batch_size: int = 8
    emotion_threshold: float = 0.6
    
    # Device settings
    use_gpu: bool = True
    device: Optional[str] = None  # Will be auto-detected if None
    gpu_memory_fraction: float = 0.8
    
    # Model cache
    cache_dir: Optional[Path] = None


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    
    # Parallel processing
    num_workers: int = 4
    batch_mode: bool = False
    
    # Memory management
    max_memory_gb: float = 4.0
    clear_cache_frequency: int = 10  # Clear cache every N segments
    
    # Progress tracking
    show_progress: bool = True
    verbose: bool = False
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    require_all: bool = False  # Exit on any failure if True


@dataclass
class OutputConfig:
    """Output configuration."""
    
    output_dir: Path = Path("./output")
    output_format: List[str] = field(default_factory=lambda: ["json"])  # json, csv, both
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    include_timestamp: bool = True
    
    # Result content
    include_raw_features: bool = False
    include_confidence_scores: bool = True
    include_acoustic_features: bool = True
    
    # Visualization
    generate_visualizations: bool = False
    visualization_format: str = "png"
    
    # Export options
    pretty_print_json: bool = True
    csv_delimiter: str = ","
    csv_include_header: bool = True
    export_ass: bool = False  # Export ASS subtitle files


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_to_console: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Performance logging
    log_performance_metrics: bool = True
    log_memory_usage: bool = True


@dataclass
class Config:
    """Main configuration class."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Update audio config
        if 'audio' in data:
            config.audio = AudioConfig(**data['audio'])
        
        # Update model config
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        
        # Update processing config
        if 'processing' in data:
            config.processing = ProcessingConfig(**data['processing'])
        
        # Update output config
        if 'output' in data:
            output_data = data['output'].copy()
            if 'output_dir' in output_data:
                output_data['output_dir'] = Path(output_data['output_dir'])
            config.output = OutputConfig(**output_data)
        
        # Update logging config
        if 'logging' in data:
            logging_data = data['logging'].copy()
            if 'log_file' in logging_data and logging_data['log_file']:
                logging_data['log_file'] = Path(logging_data['log_file'])
            config.logging = LoggingConfig(**logging_data)
        
        return config
    
    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        data = {
            'audio': self.audio.__dict__,
            'model': self.model.__dict__,
            'processing': self.processing.__dict__,
            'output': {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.output.__dict__.items()},
            'logging': {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.logging.__dict__.items()}
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.
        
        Returns:
            Config instance with values from environment
        """
        config = cls()
        
        # Override with environment variables
        audio_sr = os.getenv('AUDIO_SAMPLE_RATE')
        if audio_sr:
            config.audio.sample_rate = int(audio_sr)
        
        use_gpu = os.getenv('USE_GPU')
        if use_gpu:
            config.model.use_gpu = use_gpu.lower() == 'true'
        
        output_dir = os.getenv('OUTPUT_DIR')
        if output_dir:
            config.output.output_dir = Path(output_dir)
        
        log_level = os.getenv('LOG_LEVEL')
        if log_level:
            config.logging.log_level = log_level
        
        return config
    
    def validate(self) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate audio settings
        if self.audio.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.audio.chunk_duration <= 0:
            raise ValueError("Chunk duration must be positive")
        
        if self.audio.overlap < 0:
            raise ValueError("Overlap cannot be negative")
        
        if self.audio.overlap >= self.audio.chunk_duration:
            raise ValueError("Overlap must be less than chunk duration")
        
        # Validate model settings
        if self.model.emotion_batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if not 0 <= self.model.emotion_threshold <= 1:
            raise ValueError("Emotion threshold must be between 0 and 1")
        
        if not 0 < self.model.gpu_memory_fraction <= 1:
            raise ValueError("GPU memory fraction must be between 0 and 1")
        
        # Validate processing settings
        if self.processing.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        
        if self.processing.max_memory_gb <= 0:
            raise ValueError("Max memory must be positive")
        
        # Validate output settings
        valid_formats = {'json', 'csv', 'both'}
        for fmt in self.output.output_format:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid output format: {fmt}. Must be one of {valid_formats}")


# Default configuration instance
default_config = Config()