"""
Refactored result merger with performance optimizations and unified data models.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
from src.models import AnalysisResults, MediaInfo, ProcessingMetadata
from src.models.segments import SpeakerSegment, TranscriptionSegment, EmotionSegment, AcousticSegment
from src.models.timeline import TimelineEvent, AnalysisStatistics
from src.utils.logger import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class ResultMerger:
    """Optimized result merger using new unified data models."""
    
    def __init__(self):
        """Initialize result merger."""
        self.results = None  # Will be AnalysisResults
        
    def merge_all_results(
        self,
        speaker_results: Optional[Dict[str, Any]] = None,
        emotion_results: Optional[List[Any]] = None,
        acoustic_results: Optional[List[Dict]] = None,
        transcription_results: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        media_info: Optional[MediaInfo] = None
    ) -> Dict[str, Any]:
        """Merge all analysis results with optimized processing.
        
        Args:
            speaker_results: Speaker diarization results
            emotion_results: Emotion analysis results
            acoustic_results: Acoustic analysis results
            transcription_results: Speech recognition results
            metadata: Additional metadata
            media_info: Media information
            
        Returns:
            Merged results dictionary (backward compatible)
        """
        perf_logger.start_timer("result_merging")
        
        try:
            # Create media info if not provided
            if media_info is None:
                media_info = MediaInfo(
                    source_type="unknown",
                    duration=metadata.get('duration', 0.0) if metadata else 0.0,
                    sample_rate=16000,
                    channels=1
                )
            
            # Create processing metadata
            processing_metadata = ProcessingMetadata(
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                **metadata if metadata else {}
            )
            
            # Convert all results to unified segments (optimized - no object recreation)
            segments = []
            
            # Process speaker segments
            if speaker_results and 'segments' in speaker_results:
                segments.extend(self._convert_speaker_segments(speaker_results['segments']))
            
            # Process emotion segments (optimized - direct conversion)
            if emotion_results:
                segments.extend(self._convert_emotion_segments(emotion_results))
            
            # Process acoustic segments
            if acoustic_results:
                segments.extend(self._convert_acoustic_segments(acoustic_results))
            
            # Process transcription segments
            if transcription_results:
                segments.extend(self._convert_transcription_segments(transcription_results))
            
            # Create AnalysisResults object
            analysis_results = AnalysisResults(
                media_info=media_info,
                segments=segments,
                metadata=processing_metadata
            )
            
            # Merge overlapping segments efficiently
            analysis_results.merge_overlapping_segments(threshold=0.1)
            
            # Generate statistics (cached in AnalysisResults)
            statistics = analysis_results.get_statistics()
            timeline = analysis_results.get_timeline()
            
            duration = perf_logger.stop_timer("result_merging")
            logger.info(f"Results merged in {duration:.2f}s with {len(segments)} total segments")
            
            # Return backward-compatible format
            return self._convert_to_legacy_format(analysis_results, statistics, timeline)
            
        except Exception as e:
            perf_logger.stop_timer("result_merging")
            logger.error(f"Failed to merge results: {e}")
            raise
    
    def _convert_speaker_segments(self, speaker_segments: List[Dict]) -> List[SpeakerSegment]:
        """Convert speaker segments with optimized processing."""
        segments = []
        
        for seg_data in speaker_segments:
            segment = SpeakerSegment(
                start=seg_data.get('start', 0.0),
                end=seg_data.get('end', 0.0),
                confidence=seg_data.get('confidence', 1.0),
                speaker_id=seg_data.get('speaker', 'unknown')
            )
            segments.append(segment)
        
        return segments
    
    def _convert_emotion_segments(self, emotion_results: List[Any]) -> List[EmotionSegment]:
        """Convert emotion segments with optimized processing - no object recreation."""
        segments = []
        
        for emotion_data in emotion_results:
            # Optimize: handle both dict and object types efficiently
            if hasattr(emotion_data, 'to_dict'):
                # Use direct attribute access instead of to_dict() for better performance
                segment = EmotionSegment(
                    start=emotion_data.start,
                    end=emotion_data.end,
                    confidence=emotion_data.confidence,
                    predicted_emotion=emotion_data.predicted_emotion,
                    emotion_scores=emotion_data.emotion_scores if hasattr(emotion_data, 'emotion_scores') else {},
                    speaker_id=getattr(emotion_data, 'speaker', None)
                )
            else:
                # Direct dictionary access
                segment = EmotionSegment(
                    start=emotion_data.get('start', 0.0),
                    end=emotion_data.get('end', 0.0),
                    confidence=emotion_data.get('confidence', 1.0),
                    predicted_emotion=emotion_data.get('predicted_emotion', 'neutral'),
                    emotion_scores=emotion_data.get('emotion_scores', {}),
                    speaker_id=emotion_data.get('speaker')
                )
            
            segments.append(segment)
        
        return segments
    
    def _convert_acoustic_segments(self, acoustic_results: List[Dict]) -> List[AcousticSegment]:
        """Convert acoustic segments with optimized processing."""
        segments = []
        
        for acoustic_data in acoustic_results:
            segment = AcousticSegment(
                start=acoustic_data.get('start', 0.0),
                end=acoustic_data.get('end', 0.0),
                confidence=acoustic_data.get('confidence', 1.0),
                pitch_mean=acoustic_data.get('pitch'),
                rms_energy=acoustic_data.get('energy'),
                spectral_centroid=acoustic_data.get('spectral_centroid'),
                features=acoustic_data.get('features', {})
            )
            segments.append(segment)
        
        return segments
    
    def _convert_transcription_segments(self, transcription_results: List[Dict]) -> List[TranscriptionSegment]:
        """Convert transcription segments with optimized processing."""
        segments = []
        
        for trans_data in transcription_results:
            # Handle both dict and object types
            if hasattr(trans_data, 'to_dict') and callable(getattr(trans_data, 'to_dict')):
                trans_dict = trans_data.to_dict()
            else:
                trans_dict = trans_data if isinstance(trans_data, dict) else trans_data.__dict__
            
            segment = TranscriptionSegment(
                start=trans_dict.get('start', 0.0),
                end=trans_dict.get('end', 0.0),
                confidence=trans_dict.get('confidence', 1.0),
                text=trans_dict.get('text', ''),
                language=trans_dict.get('language'),
                speaker_id=trans_dict.get('speaker')
            )
            segments.append(segment)
        
        return segments
    
    def _convert_to_legacy_format(self, 
                                 results: AnalysisResults,
                                 statistics: AnalysisStatistics,
                                 timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Convert new format back to legacy format for backward compatibility."""
        # Get segments by type
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        transcription_segments = results.get_segments_by_type(TranscriptionSegment)
        acoustic_segments = results.get_segments_by_type(AcousticSegment)
        
        return {
            'metadata': results.metadata.to_dict(),
            'summary': statistics.get_summary(),
            'timeline': [event.to_dict() for event in timeline],
            'speakers': statistics.speaker_distribution,
            'emotions': {
                'dominant_emotion': statistics.dominant_emotion,
                'emotion_distribution': statistics.emotion_distribution,
                'total_segments': len(emotion_segments)
            },
            'acoustics': {
                'features': [seg.to_export_dict() for seg in acoustic_segments]
            },
            'transcriptions': {
                'segments': [seg.to_export_dict() for seg in transcription_segments],
                'total_segments': len(transcription_segments),
                'languages': list(set(seg.language for seg in transcription_segments if seg.language))
            }
        }
    
    def export_to_json(self,
                      results: Union[Dict[str, Any], AnalysisResults],
                      output_path: Union[str, Path],
                      pretty: bool = True) -> Path:
        """Export results to JSON format with optimized serialization.
        
        Args:
            results: Results to export
            output_path: Output file path
            pretty: Whether to pretty-print JSON
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use new AnalysisResults format if available
            if isinstance(results, AnalysisResults):
                data = results.to_dict()
            else:
                data = results
            
            # Optimized JSON writing with better performance
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str, separators=(',', ': '))
                else:
                    json.dump(data, f, ensure_ascii=False, default=str, separators=(',', ':'))
            
            logger.info(f"Results exported to JSON: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_to_csv(self,
                     results: Union[Dict[str, Any], AnalysisResults],
                     output_path: Union[str, Path],
                     include_header: bool = True) -> Path:
        """Export results to CSV format with streaming for large datasets.
        
        Args:
            results: Results to export
            output_path: Output file path
            include_header: Whether to include header row
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get timeline data
            if isinstance(results, AnalysisResults):
                timeline = results.get_timeline()
                timeline_data = [event.to_dict() for event in timeline]
            else:
                timeline_data = results.get('timeline', [])
            
            if not timeline_data:
                logger.warning("No timeline data to export")
                # Create empty CSV with headers
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if include_header:
                        writer.writerow(['start', 'end', 'duration', 'type', 'data'])
                return output_path
            
            # Optimize: Use streaming CSV writing for large datasets
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                if include_header and timeline_data:
                    # Determine headers from first row
                    first_row = timeline_data[0]
                    headers = list(first_row.keys())
                    writer.writerow(headers)
                
                # Write data rows with flattened structure
                for event_data in timeline_data:
                    flattened = self._flatten_dict(event_data)
                    if include_header:
                        # Ensure consistent column order
                        row = [flattened.get(header, '') for header in headers]
                        writer.writerow(row)
                    else:
                        writer.writerow(list(flattened.values()))
            
            logger.info(f"Results exported to CSV: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def _flatten_dict(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_dict(value, new_key))
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                flattened[new_key] = ', '.join(str(v) for v in value)
            else:
                flattened[new_key] = value
        
        return flattened
    
    def export_timeline_segments(self,
                               results: Union[Dict[str, Any], AnalysisResults],
                               output_dir: Union[str, Path],
                               format: str = 'both') -> Dict[str, Path]:
        """Export timeline segments with optimized processing.
        
        Args:
            results: Results dictionary or AnalysisResults
            output_dir: Output directory
            format: Export format ('json', 'csv', or 'both')
            
        Returns:
            Dictionary with paths to exported files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ['json', 'both']:
            json_path = output_dir / f"timeline_{timestamp}.json"
            self.export_to_json(results, json_path)
            exported['json'] = json_path
        
        if format in ['csv', 'both']:
            csv_path = output_dir / f"timeline_{timestamp}.csv"
            self.export_to_csv(results, csv_path)
            exported['csv'] = csv_path
        
        return exported
    
    def generate_summary_report(self,
                              results: Union[Dict[str, Any], AnalysisResults],
                              output_path: Optional[Path] = None) -> str:
        """Generate optimized human-readable summary report.
        
        Args:
            results: Analysis results
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        # Use AnalysisResults format for optimized access
        if isinstance(results, AnalysisResults):
            statistics = results.get_statistics()
            media_info = results.media_info
            metadata = results.metadata
        else:
            # Fallback to legacy format
            statistics = None
            media_info = None
            metadata = results.get('metadata', {})
        
        # Build report with optimized string operations
        report_parts = []
        report_parts.append("=" * 60)
        report_parts.append("AUDIO ANALYSIS REPORT")
        report_parts.append("=" * 60)
        report_parts.append("")
        
        # Media information
        if media_info:
            report_parts.extend([
                "FILE INFORMATION",
                "-" * 40,
                f"  Source Type: {media_info.source_type}",
                f"  Duration: {media_info.duration:.2f}s",
                f"  Sample Rate: {media_info.sample_rate}Hz",
                f"  Channels: {media_info.channels}",
                f"  Language: {media_info.language or 'auto-detected'}",
                ""
            ])
        
        # Analysis summary
        if statistics:
            report_parts.extend([
                "ANALYSIS SUMMARY",
                "-" * 40,
                f"  Total Speakers: {statistics.total_speakers}",
                f"  Total Segments: {statistics.total_segments}",
                f"  Total Words: {statistics.total_words}",
                f"  Dominant Emotion: {statistics.dominant_emotion}",
                ""
            ])
            
            # Speaker distribution
            if statistics.speaker_distribution:
                report_parts.extend(["SPEAKER DISTRIBUTION", "-" * 40])
                for speaker, info in statistics.speaker_distribution.items():
                    percentage = info.get('speaking_percentage', 0)
                    report_parts.append(f"  {speaker}: {percentage:.1f}%")
                report_parts.append("")
            
            # Emotion distribution
            emotion_percentages = statistics.calculate_emotion_percentages()
            if emotion_percentages:
                report_parts.extend(["EMOTION DISTRIBUTION", "-" * 40])
                for emotion, percentage in emotion_percentages.items():
                    report_parts.append(f"  {emotion}: {percentage:.1f}%")
                report_parts.append("")
        
        # Processing metadata
        if metadata:
            if isinstance(metadata, ProcessingMetadata):
                timestamp = metadata.timestamp
                processing_time = metadata.processing_time
                pipeline_version = metadata.pipeline_version
            else:
                timestamp = metadata.get('timestamp', 'unknown')
                processing_time = metadata.get('processing_time', 0)
                pipeline_version = metadata.get('pipeline_version', 'unknown')
            
            report_parts.extend([
                "PROCESSING INFORMATION", 
                "-" * 40,
                f"  Processed At: {timestamp}",
                f"  Processing Time: {processing_time:.2f}s",
                f"  Pipeline Version: {pipeline_version}",
                ""
            ])
        
        report_parts.append("=" * 60)
        
        # Optimized string joining
        report_text = '\n'.join(report_parts)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text