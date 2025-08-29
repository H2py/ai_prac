"""
Refactored speaker diarization processor with dependency injection.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import numpy as np

from .base import BaseProcessor
from src.models import ProcessingContext, ProcessingResult, SpeakerSegment, BaseSegment
from src.resources import AudioResourceManager, ModelResourceManager


class SpeakerDiarizationProcessor(BaseProcessor):
    """Speaker diarization processor using pyannote-audio with resource sharing."""
    
    def __init__(self,
                 audio_manager: AudioResourceManager,
                 model_manager: ModelResourceManager,
                 auth_token: str,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None,
                 **kwargs):
        """Initialize speaker diarization processor.
        
        Args:
            audio_manager: Shared audio resource manager
            model_manager: Shared model resource manager
            auth_token: HuggingFace authentication token
            model_name: Speaker diarization model name
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            **kwargs: Additional arguments for BaseProcessor
        """
        super().__init__(audio_manager, model_manager, **kwargs)
        
        self.auth_token = auth_token
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Pipeline will be loaded lazily
        self._pipeline = None
        self._embedding_model = None
        
        if not auth_token:
            raise ValueError("HuggingFace authentication token is required for speaker diarization")
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process audio for speaker diarization.
        
        Args:
            context: Processing context with audio path and configuration
            
        Returns:
            Processing result with speaker segments
        """
        self._start_processing(context)
        
        try:
            # Get pipeline (lazy loading)
            pipeline = self._get_pipeline()
            if pipeline is None:
                raise RuntimeError("Failed to load speaker diarization pipeline")
            
            # Load audio (cached if available)
            audio_data, sample_rate = self.audio_manager.load_audio(
                context.audio_path, sample_rate=16000
            )
            
            # Configure pipeline parameters
            params = {}
            if self.min_speakers:
                params['min_speakers'] = self.min_speakers
            if self.max_speakers:
                params['max_speakers'] = self.max_speakers
            
            # Override with context config if available
            if hasattr(context.config, 'speaker_min_speakers') and context.config.speaker_min_speakers:
                params['min_speakers'] = context.config.speaker_min_speakers
            if hasattr(context.config, 'speaker_max_speakers') and context.config.speaker_max_speakers:
                params['max_speakers'] = context.config.speaker_max_speakers
            
            self.logger.debug(f"Diarization parameters: {params}")
            
            # Run diarization
            diarization = pipeline(str(context.audio_path), **params)
            
            # Convert to our segment format
            segments = self._convert_diarization_to_segments(diarization)
            
            # Extract embeddings if requested
            embeddings = {}
            if hasattr(context.config, 'extract_speaker_embeddings') and context.config.extract_speaker_embeddings:
                embeddings = self._extract_embeddings(context.audio_path, segments)
            
            # Calculate speaker statistics
            total_duration = len(audio_data) / sample_rate
            speaker_stats = self._calculate_speaker_stats(segments, total_duration)
            
            self._end_processing(context, success=True)
            
            return ProcessingResult(
                segments=self._convert_to_base_segments(segments),
                metadata={
                    'processor': self.processor_name,
                    'model_name': self.model_name,
                    'parameters': params,
                    'speaker_stats': speaker_stats,
                    'embeddings': embeddings,
                    'total_speakers': len(set(seg.speaker_id for seg in segments)),
                    'total_segments': len(segments)
                }
            )
            
        except Exception as e:
            return self._handle_processing_error(context, e, "speaker diarization")
    
    def _get_pipeline(self):
        """Get speaker diarization pipeline (lazy loading)."""
        if self._pipeline is None:
            self.logger.debug("Loading speaker diarization pipeline...")
            self._pipeline = self.model_manager.get_speaker_pipeline(
                self.auth_token, self.model_name
            )
        return self._pipeline
    
    def _get_embedding_model(self):
        """Get speaker embedding model (lazy loading)."""
        if self._embedding_model is None:
            self.logger.debug("Loading speaker embedding model...")
            try:
                from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
                
                device_str = "cuda" if hasattr(self._get_pipeline(), 'device') else "cpu"
                device = torch.device(device_str)
                self._embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=device
                )
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self._embedding_model = None
        
        return self._embedding_model
    
    def _convert_diarization_to_segments(self, diarization) -> List[SpeakerSegment]:
        """Convert pyannote diarization to our segment format.
        
        Args:
            diarization: Pyannote annotation object
            
        Returns:
            List of SpeakerSegment objects
        """
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                start=turn.start,
                end=turn.end,
                confidence=1.0,  # Pyannote doesn't provide confidence scores
                speaker_id=speaker
            )
            segments.append(segment)
        
        # Sort by start time
        segments.sort(key=lambda s: s.start)
        
        self.logger.debug(f"Converted {len(segments)} diarization segments")
        
        return segments
    
    def _extract_embeddings(self, audio_path: Path, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """Extract speaker embeddings for identification.
        
        Args:
            audio_path: Path to audio file
            segments: List of speaker segments
            
        Returns:
            Dictionary with speaker embeddings
        """
        embedding_model = self._get_embedding_model()
        if not embedding_model:
            self._add_warning(
                ProcessingContext(audio_path, None),  # Temporary context for warning
                "Could not load embedding model, skipping embeddings"
            )
            return {}
        
        try:
            # Load audio at correct sample rate
            audio_data, sample_rate = self.audio_manager.load_audio(
                audio_path, sample_rate=16000
            )
            
            # Group segments by speaker
            speakers = {}
            for segment in segments:
                if segment.speaker_id not in speakers:
                    speakers[segment.speaker_id] = []
                speakers[segment.speaker_id].append(segment)
            
            # Extract embeddings for each speaker
            embeddings = {}
            
            for speaker_id, speaker_segments in speakers.items():
                # Use up to 5 segments for embedding extraction
                selected_segments = sorted(speaker_segments, key=lambda s: s.duration, reverse=True)[:5]
                
                speaker_audio_chunks = []
                for seg in selected_segments:
                    start_sample = int(seg.start * sample_rate)
                    end_sample = int(seg.end * sample_rate)
                    chunk = audio_data[start_sample:end_sample]
                    
                    if len(chunk) > 0:
                        speaker_audio_chunks.append(chunk)
                
                if speaker_audio_chunks:
                    # Concatenate audio chunks
                    speaker_audio = np.concatenate(speaker_audio_chunks)
                    
                    # Extract embedding
                    with torch.no_grad():
                        waveform = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)
                        embedding = embedding_model(waveform)
                        
                        # Convert to numpy - handle both tensor and numpy array cases  
                        try:
                            # Check if it's a PyTorch tensor and convert accordingly
                            if hasattr(embedding, 'detach') and hasattr(embedding, 'cpu'):
                                # Cast to torch.Tensor to satisfy type checker
                                torch_embedding = torch.as_tensor(embedding)
                                embeddings[speaker_id] = torch_embedding.squeeze().detach().cpu().numpy()
                            elif isinstance(embedding, np.ndarray):
                                # It's already a numpy array
                                embeddings[speaker_id] = embedding.squeeze() if embedding.ndim > 1 else embedding
                            else:
                                # Fallback conversion for other types
                                embeddings[speaker_id] = np.array(embedding).squeeze()
                        except Exception as conv_error:
                            self.logger.warning(f"Failed to convert embedding for {speaker_id}: {conv_error}")
                            # Skip this speaker's embedding
                            continue
            
            self.logger.debug(f"Extracted embeddings for {len(embeddings)} speakers")
            return embeddings
            
        except Exception as e:
            self.logger.warning(f"Failed to extract speaker embeddings: {e}")
            return {}
    
    def _calculate_speaker_stats(self, 
                               segments: List[SpeakerSegment], 
                               total_duration: float) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each speaker.
        
        Args:
            segments: List of speaker segments
            total_duration: Total audio duration
            
        Returns:
            Dictionary with speaker statistics
        """
        speaker_stats = {}
        
        for segment in segments:
            if segment.speaker_id not in speaker_stats:
                speaker_stats[segment.speaker_id] = {
                    'speaker_id': segment.speaker_id,
                    'total_duration': 0.0,
                    'segment_count': 0,
                    'average_confidence': 0.0,
                    'segments': []
                }
            
            stats = speaker_stats[segment.speaker_id]
            stats['total_duration'] += segment.duration
            stats['segment_count'] += 1
            stats['segments'].append(segment.to_export_dict())
            stats['average_confidence'] += segment.confidence
        
        # Calculate averages and percentages
        for stats in speaker_stats.values():
            if stats['segment_count'] > 0:
                stats['average_confidence'] /= stats['segment_count']
            
            if total_duration > 0:
                stats['speaking_percentage'] = (stats['total_duration'] / total_duration) * 100
            else:
                stats['speaking_percentage'] = 0.0
        
        return speaker_stats
    
    def _convert_to_base_segments(self, segments: List[SpeakerSegment]) -> List[BaseSegment]:
        """Convert SpeakerSegment list to BaseSegment list for type compatibility."""
        return [segment for segment in segments]
    
    def can_process(self, context: ProcessingContext) -> bool:
        """Check if processor can handle the given context.
        
        Args:
            context: Processing context
            
        Returns:
            True if processor can handle the context
        """
        try:
            # Check if pyannote is available
            try:
                import pyannote.audio  # noqa: F401
            except ImportError:
                return False
            
            # Check if audio file exists
            if not context.audio_path.exists():
                return False
            
            # Check if token is valid
            if not self.auth_token:
                return False
            
            return True
            
        except ImportError:
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'mp4']