"""
Unified output format management for audio/video analysis results.
Supports JSON, ASS, VTT, and backend/frontend communication formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from src.utils.logger import PerformanceLogger
from config.settings import Config

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    ASS = "ass" 
    VTT = "vtt"
    SRT = "srt"
    BACKEND_API = "backend_api"
    FRONTEND_JSON = "frontend_json"


@dataclass 
class MediaInfo:
    """Media file information."""
    source_type: str  # "video", "audio", "youtube"
    duration: float
    sample_rate: int
    channels: int
    language: Optional[str] = None
    video_resolution: Optional[str] = None
    codec: Optional[str] = None
    file_size: Optional[int] = None


@dataclass
class ProcessingMetadata:
    """Processing metadata and performance stats."""
    timestamp: str
    processing_time: float
    pipeline_version: str = "2.0"
    whisper_enhancements: Dict[str, Any] = None
    performance_stats: Dict[str, Any] = None


class OutputFormatManager:
    """Unified manager for all output formats and backend communication."""
    
    def __init__(self, config: Config, output_dir: Path):
        """Initialize output format manager.
        
        Args:
            config: Application configuration
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import exporters as needed
        self._importers = {}
        
    def export_results(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        formats: List[Union[str, OutputFormat]],
        base_filename: Optional[str] = None
    ) -> Dict[OutputFormat, Path]:
        """Export results in multiple formats.
        
        Args:
            results: Analysis results dictionary
            media_info: Media file information
            metadata: Processing metadata
            formats: List of output formats to generate
            base_filename: Base filename (without extension)
            
        Returns:
            Dictionary mapping formats to output file paths
        """
        perf_logger.start_timer("output_export")
        
        if not base_filename:
            base_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert string formats to OutputFormat enums
        output_formats = []
        for fmt in formats:
            if isinstance(fmt, str):
                try:
                    output_formats.append(OutputFormat(fmt.lower()))
                except ValueError:
                    logger.warning(f"Unknown output format: {fmt}")
                    continue
            else:
                output_formats.append(fmt)
        
        exported_files = {}
        
        # Export each format
        for fmt in output_formats:
            try:
                if fmt == OutputFormat.JSON:
                    exported_files[fmt] = self._export_json(
                        results, media_info, metadata, base_filename
                    )
                elif fmt == OutputFormat.ASS:
                    exported_files[fmt] = self._export_ass(
                        results, media_info, metadata, base_filename
                    )
                elif fmt == OutputFormat.VTT:
                    exported_files[fmt] = self._export_vtt(
                        results, media_info, metadata, base_filename
                    )
                elif fmt == OutputFormat.SRT:
                    exported_files[fmt] = self._export_srt(
                        results, media_info, metadata, base_filename
                    )
                elif fmt == OutputFormat.BACKEND_API:
                    exported_files[fmt] = self._export_backend_api(
                        results, media_info, metadata, base_filename
                    )
                elif fmt == OutputFormat.FRONTEND_JSON:
                    exported_files[fmt] = self._export_frontend_json(
                        results, media_info, metadata, base_filename
                    )
                
                logger.info(f"Successfully exported {fmt.value} format")
                
            except Exception as e:
                logger.error(f"Failed to export {fmt.value} format: {e}")
                continue
        
        duration = perf_logger.stop_timer("output_export")
        logger.info(f"Export completed in {duration:.2f}s, generated {len(exported_files)} files")
        
        return exported_files
    
    def _export_json(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export enhanced JSON format."""
        output_path = self.output_dir / f"{base_filename}.json"
        
        # Build comprehensive JSON structure
        json_data = {
            "media": {
                **asdict(media_info),
                "timestamp": metadata.timestamp,
                "standards": {
                    "speech_api": "W3C Web Speech API 1.0",
                    "timed_text": "SMPTE ST 2052-1:2013",
                    "phonetic": "IPA (International Phonetic Alphabet)",
                    "emotion": "PAD (Pleasure-Arousal-Dominance) model"
                }
            },
            "processing": {
                "pipeline_version": metadata.pipeline_version,
                "processing_time": metadata.processing_time,
                "whisper_enhancements": metadata.whisper_enhancements or {},
                "performance_stats": metadata.performance_stats or {}
            }
        }
        
        # Add analysis results
        json_data.update(results)
        
        # Write JSON with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _export_ass(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export ASS subtitle format with emotion styling."""
        output_path = self.output_dir / f"{base_filename}.ass"
        
        # Import ASS exporter
        if 'ass_exporter' not in self._importers:
            try:
                from src.ass_exporter import ASSExporter
                self._importers['ass_exporter'] = ASSExporter
            except ImportError:
                raise ImportError("ASSExporter not available")
        
        exporter = self._importers['ass_exporter'](self.config)
        
        # Build ASS-compatible data
        ass_data = {
            'transcription_results': results.get('transcription', []),
            'emotion_results': results.get('emotion', []),
            'speaker_results': results.get('diarization', {}),
            'media_info': asdict(media_info)
        }
        
        exporter.export_to_ass(ass_data, output_path)
        return output_path
    
    def _export_vtt(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export WebVTT subtitle format."""
        output_path = self.output_dir / f"{base_filename}.vtt"
        
        lines = ["WEBVTT", ""]
        
        # Add metadata
        lines.append(f"NOTE Created: {metadata.timestamp}")
        lines.append(f"NOTE Duration: {media_info.duration:.2f}s")
        lines.append(f"NOTE Language: {media_info.language or 'auto'}")
        lines.append("")
        
        # Add transcription segments
        transcription = results.get('transcription', [])
        for i, segment in enumerate(transcription):
            if hasattr(segment, 'to_dict'):
                seg_dict = segment.to_dict()
            else:
                seg_dict = segment
            
            start_time = self._format_vtt_time(seg_dict.get('start', 0))
            end_time = self._format_vtt_time(seg_dict.get('end', 0))
            text = seg_dict.get('text', '').strip()
            speaker = seg_dict.get('speaker', '')
            
            lines.append(f"{i + 1}")
            lines.append(f"{start_time} --> {end_time}")
            
            if speaker:
                lines.append(f"<v {speaker}>{text}")
            else:
                lines.append(text)
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def _export_srt(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export SRT subtitle format."""
        output_path = self.output_dir / f"{base_filename}.srt"
        
        lines = []
        transcription = results.get('transcription', [])
        
        for i, segment in enumerate(transcription):
            if hasattr(segment, 'to_dict'):
                seg_dict = segment.to_dict()
            else:
                seg_dict = segment
            
            start_time = self._format_srt_time(seg_dict.get('start', 0))
            end_time = self._format_srt_time(seg_dict.get('end', 0))
            text = seg_dict.get('text', '').strip()
            speaker = seg_dict.get('speaker', '')
            
            lines.append(str(i + 1))
            lines.append(f"{start_time} --> {end_time}")
            
            if speaker:
                lines.append(f"[{speaker}] {text}")
            else:
                lines.append(text)
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def _export_backend_api(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export backend API compatible format."""
        output_path = self.output_dir / f"{base_filename}_api.json"
        
        # Simplified structure for backend consumption
        api_data = {
            "status": "completed",
            "media": {
                "type": media_info.source_type,
                "duration": media_info.duration,
                "language": media_info.language
            },
            "analysis": {
                "speakers": self._extract_speaker_summary(results.get('diarization', {})),
                "emotions": self._extract_emotion_summary(results.get('emotion', [])),
                "transcription": self._extract_transcription_summary(results.get('transcription', [])),
                "acoustic": self._extract_acoustic_summary(results.get('acoustic', []))
            },
            "metadata": {
                "processed_at": metadata.timestamp,
                "processing_time": metadata.processing_time,
                "version": metadata.pipeline_version
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _export_frontend_json(
        self,
        results: Dict[str, Any],
        media_info: MediaInfo,
        metadata: ProcessingMetadata,
        base_filename: str
    ) -> Path:
        """Export frontend-optimized JSON format."""
        output_path = self.output_dir / f"{base_filename}_frontend.json"
        
        # Frontend-optimized structure
        frontend_data = {
            "mediaInfo": asdict(media_info),
            "timeline": self._build_timeline_data(results),
            "summary": {
                "totalSpeakers": len(results.get('diarization', {}).get('speakers', {})),
                "dominantEmotion": self._get_dominant_emotion(results.get('emotion', [])),
                "transcriptionWordCount": self._count_words(results.get('transcription', [])),
                "processingTime": metadata.processing_time
            },
            "visualization": {
                "emotionTimeline": self._build_emotion_timeline(results.get('emotion', [])),
                "speakerDistribution": self._build_speaker_distribution(results.get('diarization', {})),
                "acousticFeatures": self._build_acoustic_visualization(results.get('acoustic', []))
            },
            "metadata": {
                "timestamp": metadata.timestamp,
                "version": metadata.pipeline_version
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for WebVTT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = int(seconds % 60)
        millisec = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{sec:02d},{millisec:03d}"
    
    def _extract_speaker_summary(self, diarization: Dict[str, Any]) -> Dict[str, Any]:
        """Extract speaker summary for API."""
        if not diarization:
            return {}
        
        return {
            "total_speakers": diarization.get('total_speakers', 0),
            "speakers": {
                speaker_id: {
                    "speaking_time": speaker_info.get('speaking_time', 0),
                    "speaking_percentage": speaker_info.get('speaking_percentage', 0)
                }
                for speaker_id, speaker_info in diarization.get('speakers', {}).items()
            }
        }
    
    def _extract_emotion_summary(self, emotions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract emotion summary for API."""
        if not emotions:
            return {}
        
        # Calculate dominant emotion
        emotion_counts = {}
        for emotion_data in emotions:
            if hasattr(emotion_data, 'to_dict'):
                emotion_dict = emotion_data.to_dict()
            else:
                emotion_dict = emotion_data
            
            predicted = emotion_dict.get('predicted_emotion', 'neutral')
            emotion_counts[predicted] = emotion_counts.get(predicted, 0) + 1
        
        dominant = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        return {
            "dominant_emotion": dominant,
            "emotion_distribution": emotion_counts,
            "total_segments": len(emotions)
        }
    
    def _extract_transcription_summary(self, transcription: List[Any]) -> Dict[str, Any]:
        """Extract transcription summary for API."""
        if not transcription:
            return {}
        
        total_words = 0
        total_chars = 0
        
        for segment in transcription:
            if hasattr(segment, 'to_dict'):
                seg_dict = segment.to_dict()
            else:
                seg_dict = segment
            
            text = seg_dict.get('text', '')
            total_words += len(text.split())
            total_chars += len(text)
        
        return {
            "total_segments": len(transcription),
            "total_words": total_words,
            "total_characters": total_chars,
            "average_words_per_segment": total_words / len(transcription) if transcription else 0
        }
    
    def _extract_acoustic_summary(self, acoustic: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract acoustic summary for API."""
        if not acoustic:
            return {}
        
        return {
            "total_segments": len(acoustic),
            "has_pitch_data": any('pitch_mean' in seg for seg in acoustic),
            "has_energy_data": any('rms_energy' in seg for seg in acoustic)
        }
    
    def _build_timeline_data(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build timeline data combining all analysis results."""
        timeline = []
        
        # Get all segments and sort by time
        transcription = results.get('transcription', [])
        emotion = results.get('emotion', [])
        
        # Combine data by timestamp
        for segment in transcription:
            if hasattr(segment, 'to_dict'):
                seg_dict = segment.to_dict()
            else:
                seg_dict = segment
            
            timeline_item = {
                "start": seg_dict.get('start', 0),
                "end": seg_dict.get('end', 0),
                "text": seg_dict.get('text', ''),
                "speaker": seg_dict.get('speaker', ''),
                "confidence": seg_dict.get('confidence', 1.0)
            }
            
            # Find corresponding emotion
            segment_start = seg_dict.get('start', 0)
            for emotion_data in emotion:
                if hasattr(emotion_data, 'to_dict'):
                    emo_dict = emotion_data.to_dict()
                else:
                    emo_dict = emotion_data
                
                if (emo_dict.get('start', 0) <= segment_start <= emo_dict.get('end', float('inf'))):
                    timeline_item['emotion'] = emo_dict.get('predicted_emotion', 'neutral')
                    timeline_item['emotion_confidence'] = emo_dict.get('confidence', 0.0)
                    break
            
            timeline.append(timeline_item)
        
        return sorted(timeline, key=lambda x: x['start'])
    
    def _get_dominant_emotion(self, emotions: List[Any]) -> str:
        """Get dominant emotion from emotion analysis results."""
        if not emotions:
            return 'neutral'
        
        emotion_counts = {}
        for emotion_data in emotions:
            if hasattr(emotion_data, 'to_dict'):
                emotion_dict = emotion_data.to_dict()
            else:
                emotion_dict = emotion_data
            
            predicted = emotion_dict.get('predicted_emotion', 'neutral')
            emotion_counts[predicted] = emotion_counts.get(predicted, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
    
    def _count_words(self, transcription: List[Any]) -> int:
        """Count total words in transcription."""
        total_words = 0
        for segment in transcription:
            if hasattr(segment, 'to_dict'):
                seg_dict = segment.to_dict()
            else:
                seg_dict = segment
            
            text = seg_dict.get('text', '')
            total_words += len(text.split())
        
        return total_words
    
    def _build_emotion_timeline(self, emotions: List[Any]) -> List[Dict[str, Any]]:
        """Build emotion timeline for visualization."""
        timeline = []
        for emotion_data in emotions:
            if hasattr(emotion_data, 'to_dict'):
                emotion_dict = emotion_data.to_dict()
            else:
                emotion_dict = emotion_data
            
            timeline.append({
                "timestamp": emotion_dict.get('start', 0),
                "emotion": emotion_dict.get('predicted_emotion', 'neutral'),
                "confidence": emotion_dict.get('confidence', 0.0)
            })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def _build_speaker_distribution(self, diarization: Dict[str, Any]) -> Dict[str, float]:
        """Build speaker distribution for visualization."""
        if not diarization or 'speakers' not in diarization:
            return {}
        
        return {
            speaker_id: speaker_info.get('speaking_percentage', 0)
            for speaker_id, speaker_info in diarization['speakers'].items()
        }
    
    def _build_acoustic_visualization(self, acoustic: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Build acoustic features for visualization."""
        if not acoustic:
            return {}
        
        features = {
            "rms_energy": [],
            "spectral_centroid": [],
            "pitch": []
        }
        
        for segment in acoustic:
            features["rms_energy"].append(segment.get('rms_energy', 0.0))
            features["spectral_centroid"].append(segment.get('spectral_centroid', 0.0))
            features["pitch"].append(segment.get('pitch_mean', 0.0) if segment.get('pitch_mean') else 0.0)
        
        return features
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return [fmt.value for fmt in OutputFormat]
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary output files."""
        try:
            temp_files = list(self.output_dir.glob("temp_*"))
            for file in temp_files:
                file.unlink()
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")