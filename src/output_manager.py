"""
Refactored unified output format management using new architecture.
Backward-compatible wrapper around the new OutputPipeline system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from src.utils.logger import PerformanceLogger
from config.settings import Config
from src.models import AnalysisResults, MediaInfo, ProcessingMetadata
from src.models.segments import SpeakerSegment, TranscriptionSegment, EmotionSegment, AcousticSegment


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    ASS = "ass" 
    VTT = "vtt"
    SRT = "srt"
    BACKEND_API = "backend_api"
    FRONTEND_JSON = "frontend_json"

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class OutputFormatManager:
    """Unified manager for all output formats - refactored to use new architecture."""
    
    def __init__(self, config: Config, output_dir: Path):
        """Initialize output format manager.
        
        Args:
            config: Application configuration
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize new output pipeline
        self.pipeline = OutputPipeline(
            transformer=DataTransformer(),
            output_dir=self.output_dir
        )
        
        # Backward compatibility
        self._importers = {}
    
    def export_results(
        self,
        results: Union[Dict[str, Any], AnalysisResults],
        media_info: Optional[MediaInfo] = None,
        metadata: Optional[ProcessingMetadata] = None,
        formats: List[Union[str, OutputFormat]] = None,
        base_filename: Optional[str] = None
    ) -> Dict[OutputFormat, Path]:
        """Export results in multiple formats using new architecture.
        
        Args:
            results: Analysis results (dict for backward compatibility or AnalysisResults)
            media_info: Media file information (for backward compatibility)
            metadata: Processing metadata (for backward compatibility)
            formats: List of output formats to generate
            base_filename: Base filename (without extension)
            
        Returns:
            Dictionary mapping formats to output file paths
        """
        perf_logger.start_timer("output_export")
        
        try:
            # Convert to new AnalysisResults format if needed
            if isinstance(results, dict):
                analysis_results = self._convert_legacy_results(
                    results, media_info, metadata
                )
            else:
                analysis_results = results
            
            # Use new pipeline for export
            exported_files = self.pipeline.export_results(
                analysis_results, 
                formats or ["json"], 
                base_filename
            )
            
            duration = perf_logger.stop_timer("output_export")
            logger.info(f"Export completed in {duration:.2f}s, generated {len(exported_files)} files")
            
            return exported_files
            
        except Exception as e:
            perf_logger.stop_timer("output_export")
            logger.error(f"Export failed: {e}")
            raise
    
    def _convert_legacy_results(self, 
                               results: Dict[str, Any],
                               media_info: Optional[MediaInfo],
                               metadata: Optional[ProcessingMetadata]) -> AnalysisResults:
        """Convert legacy results format to new AnalysisResults format.
        
        Args:
            results: Legacy results dictionary
            media_info: Media information
            metadata: Processing metadata
            
        Returns:
            AnalysisResults object
        """
        # Create default media info if not provided
        if media_info is None:
            media_info = MediaInfo(
                source_type="unknown",
                duration=results.get('duration', 0.0),
                sample_rate=16000,
                channels=1
            )
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = ProcessingMetadata(
                timestamp=datetime.now().isoformat(),
                processing_time=0.0
            )
        
        # Convert legacy segments to new format
        segments = []
        
        # Convert transcription segments
        if 'transcription' in results:
            for trans_data in results['transcription']:
                if hasattr(trans_data, 'to_dict'):
                    trans_dict = trans_data.to_dict()
                else:
                    trans_dict = trans_data
                
                segment = TranscriptionSegment(
                    start=trans_dict.get('start', 0.0),
                    end=trans_dict.get('end', 0.0),
                    confidence=trans_dict.get('confidence', 1.0),
                    text=trans_dict.get('text', ''),
                    language=trans_dict.get('language'),
                    speaker_id=trans_dict.get('speaker')
                )
                segments.append(segment)
        
        # Convert speaker segments
        if 'diarization' in results and 'segments' in results['diarization']:
            for speaker_data in results['diarization']['segments']:
                segment = SpeakerSegment(
                    start=speaker_data.get('start', 0.0),
                    end=speaker_data.get('end', 0.0),
                    confidence=speaker_data.get('confidence', 1.0),
                    speaker_id=speaker_data.get('speaker', 'unknown')
                )
                segments.append(segment)
        
        # Convert emotion segments
        if 'emotion' in results:
            for emotion_data in results['emotion']:
                if hasattr(emotion_data, 'to_dict'):
                    emotion_dict = emotion_data.to_dict()
                else:
                    emotion_dict = emotion_data
                
                segment = EmotionSegment(
                    start=emotion_dict.get('start', 0.0),
                    end=emotion_dict.get('end', 0.0),
                    confidence=emotion_dict.get('confidence', 1.0),
                    predicted_emotion=emotion_dict.get('predicted_emotion', 'neutral'),
                    emotion_scores=emotion_dict.get('emotion_scores', {}),
                    speaker_id=emotion_dict.get('speaker')
                )
                segments.append(segment)
        
        # Convert acoustic segments
        if 'acoustic' in results:
            for acoustic_data in results['acoustic']:
                segment = AcousticSegment(
                    start=acoustic_data.get('start', 0.0),
                    end=acoustic_data.get('end', 0.0),
                    confidence=acoustic_data.get('confidence', 1.0),
                    pitch_mean=acoustic_data.get('pitch_mean'),
                    rms_energy=acoustic_data.get('rms_energy'),
                    spectral_centroid=acoustic_data.get('spectral_centroid'),
                    features=acoustic_data.get('features', {})
                )
                segments.append(segment)
        
        return AnalysisResults(
            media_info=media_info,
            segments=segments,
            metadata=metadata
        )
    
    # Legacy methods - now delegate to new pipeline
    def _export_json(self,
                    results: Dict[str, Any],
                    media_info: MediaInfo,
                    metadata: ProcessingMetadata,
                    base_filename: str) -> Path:
        """Legacy JSON export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.JSON, base_filename
        )
    
    def _export_ass(self,
                   results: Dict[str, Any],
                   media_info: MediaInfo,
                   metadata: ProcessingMetadata,
                   base_filename: str) -> Path:
        """Legacy ASS export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.ASS, base_filename
        )
    
    def _export_vtt(self,
                   results: Dict[str, Any],
                   media_info: MediaInfo,
                   metadata: ProcessingMetadata,
                   base_filename: str) -> Path:
        """Legacy VTT export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.VTT, base_filename
        )
    
    def _export_srt(self,
                   results: Dict[str, Any],
                   media_info: MediaInfo,
                   metadata: ProcessingMetadata,
                   base_filename: str) -> Path:
        """Legacy SRT export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.SRT, base_filename
        )
    
    def _export_backend_api(self,
                           results: Dict[str, Any],
                           media_info: MediaInfo,
                           metadata: ProcessingMetadata,
                           base_filename: str) -> Path:
        """Legacy backend API export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.BACKEND_API, base_filename
        )
    
    def _export_frontend_json(self,
                             results: Dict[str, Any],
                             media_info: MediaInfo,
                             metadata: ProcessingMetadata,
                             base_filename: str) -> Path:
        """Legacy frontend JSON export - delegates to new pipeline."""
        analysis_results = self._convert_legacy_results(results, media_info, metadata)
        return self.pipeline.export_single_format(
            analysis_results, OutputFormat.FRONTEND_JSON, base_filename
        )
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return self.pipeline.get_supported_formats()
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary output files."""
        self.pipeline.cleanup_temp_files()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get output manager statistics."""
        return self.pipeline.get_stats()
    
    def __str__(self) -> str:
        """String representation."""
        return f"OutputFormatManager(pipeline={self.pipeline})"