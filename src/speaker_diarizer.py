"""
Speaker diarization module for identifying and segmenting different speakers.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass, field

from src.models.segments import SpeakerSegment

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    PretrainedSpeakerEmbedding = None

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger, log_exception
from config.settings import ModelConfig


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


# Using unified SpeakerSegment from src.models.segments


@dataclass
class SpeakerInfo:
    """Information about a speaker."""
    
    speaker_id: str
    total_duration: float = 0.0
    segment_count: int = 0
    segments: List[SpeakerSegment] = field(default_factory=list)
    average_confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'speaker_id': self.speaker_id,
            'total_duration': self.total_duration,
            'segment_count': self.segment_count,
            'average_confidence': self.average_confidence,
            'speaking_percentage': 0.0  # Will be calculated later
        }


class SpeakerDiarizer:
    """Speaker diarization using pyannote-audio."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize speaker diarizer.
        
        Args:
            config: Model configuration
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote-audio is not installed. "
                "Please install with: pip install pyannote.audio"
            )
        
        self.config = config or ModelConfig()
        self.device = self._setup_device()
        self.pipeline = None
        self.embedding_model = None
        self._initialized = False
        
        logger.info(f"SpeakerDiarizer initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device.
        
        Returns:
            torch.device instance
        """
        if self.config.device:
            return torch.device(self.config.device)
        
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Set memory fraction
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.gpu_memory_fraction
                )
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for processing")
        
        return device
    
    def initialize(self, auth_token: str) -> None:
        """Initialize the diarization pipeline.
        
        Args:
            auth_token: Hugging Face authentication token for model access (REQUIRED)
        """
        if self._initialized:
            return
        
        # Token is mandatory
        if not auth_token:
            raise ValueError(
                "❌ HuggingFace 토큰이 필요합니다!\n"
                ".env 파일에 HUGGINGFACE_TOKEN을 설정하세요."
            )
        
        perf_logger.start_timer("model_initialization")
        
        try:
            # Initialize diarization pipeline
            logger.info("Loading speaker diarization pipeline...")
            
            # Use the specified model or default
            model_name = self.config.speaker_model
            
            # Token is mandatory for pyannote models
            # User must accept model agreements at:
            # https://huggingface.co/pyannote/speaker-diarization-3.1
            if Pipeline is not None:
                self.pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=auth_token
                )
            else:
                raise ImportError("Pipeline class is not available")
            
            if self.pipeline:
                # Move pipeline to device
                self.pipeline.to(self.device)
            
            # Initialize embedding model for speaker representations
            logger.info("Loading speaker embedding model...")
            if PretrainedSpeakerEmbedding is not None:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=self.device
                )
            else:
                logger.warning("PretrainedSpeakerEmbedding not available")
                self.embedding_model = None
            
            self._initialized = True
            duration = perf_logger.stop_timer("model_initialization")
            logger.info(f"Models initialized in {duration:.2f}s")
            
        except Exception as e:
            perf_logger.stop_timer("model_initialization")
            log_exception(logger, e, "Failed to initialize diarization pipeline")
            raise
    
    def _setup_offline_pipeline(self) -> None:
        """Setup a basic offline pipeline for testing."""
        # This is a fallback for when the model cannot be loaded
        # In production, you should always use the proper model with auth token
        logger.warning("Using offline/mock pipeline for demonstration")
        self.pipeline = None
    
    def diarize(
        self,
        audio_path: Union[str, Path],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs
    ) -> List[SpeakerSegment]:
        """Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            **kwargs: Additional pipeline parameters
            
        Returns:
            List of speaker segments
        """
        if not self._initialized:
            raise RuntimeError(
                "SpeakerDiarizer가 초기화되지 않았습니다. "
                "initialize() 메서드를 먼저 호출하세요."
            )
        
        perf_logger.start_timer("speaker_diarization")
        audio_path = Path(audio_path)
        
        try:
            logger.info(f"Performing speaker diarization on: {audio_path}")
            
            # Pipeline must be available (no fallback)
            if self.pipeline is None:
                raise RuntimeError(
                    "Pipeline이 로드되지 않았습니다. "
                    "HuggingFace 토큰과 모델 동의를 확인하세요."
                )
            
            # Configure pipeline parameters
            params = {
                "min_speakers": min_speakers or self.config.speaker_min_speakers,
                "max_speakers": max_speakers or self.config.speaker_max_speakers
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Run diarization
            diarization = self.pipeline(str(audio_path), **params)
            
            # Convert to segments
            segments = self._annotation_to_segments(diarization)
            
            duration = perf_logger.stop_timer("speaker_diarization")
            logger.info(
                f"Diarization completed: {len(segments)} segments found "
                f"in {duration:.2f}s"
            )
            
            return segments
            
        except Exception as e:
            perf_logger.stop_timer("speaker_diarization")
            log_exception(logger, e, "Speaker diarization failed")
            # Return empty segments on failure
            return []
    
    def _alternative_diarization(
        self,
        audio_path: Path
    ) -> List[SpeakerSegment]:
        """Alternative diarization using energy-based segmentation.
        
        This is a fallback method when pyannote models are not available.
        It uses simple energy-based voice activity detection.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments
        """
        logger.info("Using energy-based segmentation (demo mode)")
        
        # Load audio
        audio_data, sample_rate = load_audio(audio_path)
        duration = len(audio_data) / sample_rate
        
        # Simple energy-based voice activity detection
        segments = []
        chunk_duration = 2.0  # 2-second chunks
        energy_threshold = 0.01
        
        # Split into chunks and detect speech
        chunks = split_audio_chunks(
            audio_data,
            sample_rate,
            chunk_duration=chunk_duration,
            overlap=0.5
        )
        
        current_speaker = 0
        last_active = False
        segment_start = None
        
        for chunk, (start_time, end_time) in chunks:
            # Calculate energy
            energy = np.sqrt(np.mean(chunk ** 2))
            is_active = energy > energy_threshold
            
            if is_active and not last_active:
                # Start new segment
                segment_start = start_time
                current_speaker = (current_speaker % 2) + 1  # Alternate speakers
            elif not is_active and last_active and segment_start is not None:
                # End segment
                segments.append(
                    SpeakerSegment(
                        start=segment_start,
                        end=end_time,
                        speaker_id=f"speaker_{current_speaker}",
                        confidence=0.5  # Low confidence for demo
                    )
                )
                segment_start = None
            
            last_active = is_active
        
        # Close last segment if needed
        if segment_start is not None:
            segments.append(
                SpeakerSegment(
                    start=segment_start,
                    end=duration,
                    speaker_id=f"speaker_{current_speaker}",
                    confidence=0.5
                )
            )
        
        # Merge nearby segments from same speaker
        segments = self._merge_segments(segments)
        
        return segments
    
    def _annotation_to_segments(
        self,
        annotation: Any  # Use Any to avoid type issues when pyannote not available
    ) -> List[SpeakerSegment]:
        """Convert pyannote annotation to speaker segments.
        
        Args:
            annotation: Pyannote annotation object
            
        Returns:
            List of speaker segments
        """
        segments = []
        
        # Handle both forms of itertracks output
        try:
            for track_info in annotation.itertracks(yield_label=True):
                if len(track_info) == 3:
                    turn, _, speaker = track_info
                else:
                    # Fallback for different pyannote versions
                    turn, speaker = track_info[:2]
                
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_id=str(speaker),  # Use speaker_id attribute
                    confidence=1.0
                )
                segments.append(segment)
        except Exception as e:
            logger.warning(f"Error processing annotation tracks: {e}")
            # Fallback: create single segment
            segments = [SpeakerSegment(
                start=0.0,
                end=annotation.get_timeline().duration,
                speaker_id="speaker_1",
                confidence=0.5
            )]
        
        # Sort by start time
        segments.sort(key=lambda s: s.start)
        
        return segments
    
    def _merge_segments(
        self,
        segments: List[SpeakerSegment],
        gap_threshold: float = 0.5
    ) -> List[SpeakerSegment]:
        """Merge nearby segments from the same speaker.
        
        Args:
            segments: List of segments
            gap_threshold: Maximum gap to merge (seconds)
            
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for segment in segments[1:]:
            # Check if same speaker and close enough
            if (segment.speaker_id == current.speaker_id and
                segment.start - current.end < gap_threshold):
                # Merge segments
                current = SpeakerSegment(
                    start=current.start,
                    end=segment.end,
                    speaker_id=current.speaker_id,
                    confidence=min(current.confidence, segment.confidence)
                )
            else:
                merged.append(current)
                current = segment
        
        merged.append(current)
        
        return merged
    
    def get_speaker_embeddings(
        self,
        audio_path: Union[str, Path],
        segments: List[SpeakerSegment]
    ) -> Dict[str, np.ndarray]:
        """Extract speaker embeddings for each speaker.
        
        Args:
            audio_path: Path to audio file
            segments: Speaker segments
            
        Returns:
            Dictionary mapping speaker IDs to embeddings
        """
        if not self.embedding_model:
            logger.warning("Embedding model not initialized")
            return {}
        
        perf_logger.start_timer("embedding_extraction")
        
        try:
            # Load audio
            audio_data, sample_rate = load_audio(audio_path, sample_rate=16000)
            
            # Group segments by speaker
            speakers = {}
            for segment in segments:
                if segment.speaker_id not in speakers:
                    speakers[segment.speaker_id] = []
                speakers[segment.speaker_id].append(segment)
            
            # Extract embeddings for each speaker
            embeddings = {}
            
            for speaker_id, speaker_segments in speakers.items():
                # Collect audio chunks for this speaker
                speaker_audio = []
                
                for seg in speaker_segments[:5]:  # Use up to 5 segments
                    start_sample = int(seg.start * sample_rate)
                    end_sample = int(seg.end * sample_rate)
                    chunk = audio_data[start_sample:end_sample]
                    
                    if len(chunk) > 0:
                        speaker_audio.append(chunk)
                
                if speaker_audio:
                    # Concatenate chunks
                    speaker_audio = np.concatenate(speaker_audio)
                    
                    # Get embedding
                    with torch.no_grad():
                        # Convert audio to proper format for embedding model
                        audio_tensor = torch.tensor(speaker_audio).float().unsqueeze(0)
                        
                        # Different API depending on embedding model type
                        try:
                            # Try speechbrain format
                            embedding = self.embedding_model(audio_tensor)
                            if hasattr(embedding, 'cpu'):
                                embeddings[speaker_id] = embedding.squeeze().cpu().numpy()
                            else:
                                embeddings[speaker_id] = np.array(embedding).squeeze()
                        except Exception as e:
                            logger.warning(f"Failed to extract embedding for {speaker_id}: {e}")
                            # Create dummy embedding
                            embeddings[speaker_id] = np.random.rand(192).astype(np.float32)
            
            duration = perf_logger.stop_timer("embedding_extraction")
            logger.info(f"Extracted embeddings for {len(embeddings)} speakers in {duration:.2f}s")
            
            return embeddings
            
        except Exception as e:
            perf_logger.stop_timer("embedding_extraction")
            log_exception(logger, e, "Failed to extract speaker embeddings")
            return {}
    
    def analyze(
        self,
        audio_path: Union[str, Path],
        extract_embeddings: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete speaker analysis of audio file.
        
        Args:
            audio_path: Path to audio file
            extract_embeddings: Whether to extract speaker embeddings
            **kwargs: Additional parameters for diarization
            
        Returns:
            Dictionary with complete speaker analysis
        """
        audio_path = Path(audio_path)
        
        # Perform diarization
        segments = self.diarize(audio_path, **kwargs)
        
        # Extract embeddings if requested
        embeddings = {}
        if extract_embeddings and segments:
            embeddings = self.get_speaker_embeddings(audio_path, segments)
        
        # Calculate speaker statistics
        speakers = self._calculate_speaker_stats(segments)
        
        # Get audio duration
        audio_data, sample_rate = load_audio(audio_path)
        total_duration = len(audio_data) / sample_rate
        
        # Calculate speaking percentages
        for speaker_info in speakers.values():
            speaker_info['speaking_percentage'] = (
                speaker_info['total_duration'] / total_duration * 100
            )
        
        return {
            'segments': [seg.to_export_dict() for seg in segments],
            'speakers': speakers,
            'embeddings': embeddings,
            'total_duration': total_duration,
            'total_speakers': len(speakers),
            'total_segments': len(segments)
        }
    
    def _calculate_speaker_stats(
        self,
        segments: List[SpeakerSegment]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each speaker.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary with speaker statistics
        """
        speakers = {}
        
        for segment in segments:
            if segment.speaker_id not in speakers:
                speakers[segment.speaker_id] = {
                    'speaker_id': segment.speaker_id,
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'segments': [],
                    'average_confidence': 0.0
                }
            
            speaker = speakers[segment.speaker_id]
            speaker['total_duration'] += segment.duration
            speaker['segment_count'] += 1
            speaker['segments'].append(segment.to_export_dict())
            speaker['average_confidence'] += segment.confidence
        
        # Calculate averages
        for speaker in speakers.values():
            if speaker['segment_count'] > 0:
                speaker['average_confidence'] /= speaker['segment_count']
        
        return speakers
    
    def visualize_diarization(
        self,
        segments: List[SpeakerSegment],
        audio_duration: float,
        output_path: Optional[Path] = None
    ) -> None:
        """Create visualization of speaker diarization.
        
        Args:
            segments: Speaker segments
            audio_duration: Total audio duration
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(15, 4))
            
            # Get unique speakers
            speakers = list(set(seg.speaker_id for seg in segments))
            # Use a colormap that's always available
            # Use matplotlib colormaps safely without extra import
            # Use matplotlib colormaps safely
            if len(speakers) <= 10:
                colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(speakers)))
            else:
                colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(speakers)))
            speaker_colors = dict(zip(speakers, colors))
            
            # Plot segments
            for segment in segments:
                rect = patches.Rectangle(
                    (segment.start, speakers.index(segment.speaker_id)),
                    segment.duration,
                    0.8,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=speaker_colors[segment.speaker_id],
                    alpha=0.7
                )
                ax.add_patch(rect)
            
            # Configure plot
            ax.set_xlim(0, audio_duration)
            ax.set_ylim(-0.5, len(speakers) - 0.5)
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Speaker', fontsize=12)
            ax.set_yticks(range(len(speakers)))
            ax.set_yticklabels(speakers)
            ax.set_title('Speaker Diarization Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add legend
            handles = [
                patches.Patch(
                    color=speaker_colors[speaker],
                    label=f'{speaker} ({sum(1 for s in segments if s.speaker_id == speaker)} segments)'
                )
                for speaker in speakers
            ]
            ax.legend(handles=handles, loc='upper right')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                logger.info(f"Diarization visualization saved to: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            log_exception(logger, e, "Failed to create visualization")