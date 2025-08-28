"""
Result merger module for combining and exporting analysis results.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import pandas as pd

from src.speaker_diarizer import SpeakerSegment
from src.emotion_analyzer import EmotionPrediction
from src.utils.logger import PerformanceLogger


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class ResultMerger:
    """Merge and export analysis results from different modules."""
    
    def __init__(self):
        """Initialize result merger."""
        self.results = {}
        
    def merge_all_results(
        self,
        speaker_results: Optional[Dict[str, Any]] = None,
        emotion_results: Optional[List[EmotionPrediction]] = None,
        acoustic_results: Optional[List[Dict]] = None,
        transcription_results: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge all analysis results with proper timestamp alignment.
        
        Args:
            speaker_results: Speaker diarization results
            emotion_results: Emotion analysis results
            acoustic_results: Acoustic analysis results
            transcription_results: Speech recognition results
            metadata: Additional metadata
            
        Returns:
            Merged results dictionary
        """
        perf_logger.start_timer("result_merging")
        
        try:
            # Initialize merged results
            merged = {
                'metadata': metadata or {},
                'summary': {},
                'timeline': [],
                'speakers': {},
                'emotions': {},
                'acoustics': {},
                'transcriptions': {}
            }
            
            # Process speaker results
            if speaker_results:
                merged['speakers'] = speaker_results.get('speakers', {})
                segments = speaker_results.get('segments', [])
                
                # Add speaker info to timeline
                for segment in segments:
                    timeline_entry = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'speaker': segment['speaker'],
                        'type': 'speaker_segment'
                    }
                    merged['timeline'].append(timeline_entry)
            
            # Process emotion results
            if emotion_results:
                emotion_list = []
                for pred in emotion_results:
                    emotion_dict = pred.to_dict() if hasattr(pred, 'to_dict') else pred
                    emotion_list.append(emotion_dict)
                    
                    # Add to timeline
                    timeline_entry = {
                        'start': emotion_dict['start'],
                        'end': emotion_dict['end'],
                        'emotion': emotion_dict['primary_emotion'],
                        'confidence': emotion_dict['confidence'],
                        'type': 'emotion_segment'
                    }
                    if 'speaker' in emotion_dict and emotion_dict['speaker']:
                        timeline_entry['speaker'] = emotion_dict['speaker']
                    merged['timeline'].append(timeline_entry)
                
                # Calculate emotion statistics
                if emotion_list:
                    from src.emotion_analyzer import EmotionAnalyzer, EmotionPrediction
                    analyzer = EmotionAnalyzer()
                    # Convert dictionaries to EmotionPrediction objects properly
                    emotion_objects = []
                    for e in emotion_list:
                        # Remove 'duration' if it exists (it's a computed property)
                        e_copy = e.copy()
                        e_copy.pop('duration', None)
                        emotion_objects.append(EmotionPrediction(**e_copy))
                    emotion_stats = analyzer.get_emotion_statistics(emotion_objects)
                    merged['emotions'] = emotion_stats
            
            # Process acoustic results
            if acoustic_results:
                merged['acoustics']['features'] = acoustic_results
                
                # Add to timeline
                for acoustic in acoustic_results:
                    timeline_entry = {
                        'start': acoustic.get('start', 0),
                        'end': acoustic.get('end', 0),
                        'type': 'acoustic_segment',
                        'features': {
                            'pitch': acoustic.get('pitch'),
                            'energy': acoustic.get('energy')
                        }
                    }
                    merged['timeline'].append(timeline_entry)
            
            # Process transcription results
            if transcription_results:
                transcription_list = []
                for trans in transcription_results:
                    # Handle both dict and object types
                    trans_dict = trans if isinstance(trans, dict) else trans.to_dict()
                    transcription_list.append(trans_dict)
                    
                    # Add to timeline with text
                    timeline_entry = {
                        'start': trans_dict['start'],
                        'end': trans_dict['end'],
                        'text': trans_dict.get('text', ''),
                        'language': trans_dict.get('language', 'unknown'),
                        'type': 'transcription',
                        'speaker': trans_dict.get('speaker', 'unknown')
                    }
                    merged['timeline'].append(timeline_entry)
                
                # Store transcriptions
                merged['transcriptions'] = {
                    'segments': transcription_list,
                    'total_segments': len(transcription_list),
                    'languages': list(set(t.get('language', 'unknown') for t in transcription_list))
                }
            
            # Sort timeline by start time
            merged['timeline'].sort(key=lambda x: x['start'])
            
            # Merge overlapping segments
            merged['timeline'] = self._merge_timeline_segments(merged['timeline'])
            
            # Generate summary
            merged['summary'] = self._generate_summary(merged)
            
            duration = perf_logger.stop_timer("result_merging")
            logger.info(f"Results merged in {duration:.2f}s")
            
            return merged
            
        except Exception as e:
            perf_logger.stop_timer("result_merging")
            logger.error(f"Failed to merge results: {e}")
            raise
    
    def _merge_timeline_segments(
        self,
        timeline: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Merge overlapping timeline segments.
        
        Args:
            timeline: List of timeline segments
            
        Returns:
            Merged timeline
        """
        if not timeline:
            return timeline
        
        merged = []
        current_segment = {}
        
        for segment in timeline:
            if not current_segment:
                current_segment = segment.copy()
            elif (segment['start'] <= current_segment['end'] and
                  segment.get('speaker') == current_segment.get('speaker')):
                # Merge segments
                current_segment['end'] = max(current_segment['end'], segment['end'])
                
                # Merge other properties
                if 'emotion' in segment and 'emotion' not in current_segment:
                    current_segment['emotion'] = segment['emotion']
                    current_segment['confidence'] = segment.get('confidence', 1.0)
                
                if 'features' in segment:
                    if 'features' not in current_segment:
                        current_segment['features'] = {}
                    current_segment['features'].update(segment['features'])
            else:
                # Save current and start new
                merged.append(current_segment)
                current_segment = segment.copy()
        
        if current_segment:
            merged.append(current_segment)
        
        return merged
    
    def _generate_summary(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics.
        
        Args:
            merged: Merged results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_speakers': len(merged.get('speakers', {})),
            'total_segments': len(merged.get('timeline', [])),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Add speaker summary
        if merged.get('speakers'):
            speakers = merged['speakers']
            summary['speaker_distribution'] = {
                speaker: info.get('speaking_percentage', 0)
                for speaker, info in speakers.items()
            }
        
        # Add emotion summary
        if merged.get('emotions'):
            emotions = merged['emotions']
            summary['dominant_emotion'] = emotions.get('dominant_emotion', 'unknown')
            summary['emotion_distribution'] = emotions.get('emotion_distribution', {})
            summary['emotion_changes'] = emotions.get('emotion_changes', 0)
        
        return summary
    
    def export_to_json(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> Path:
        """Export results to JSON format.
        
        Args:
            results: Results dictionary
            output_path: Output file path
            pretty: Whether to pretty-print JSON
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                if pretty:
                    json.dump(results, f, indent=2, default=str)
                else:
                    json.dump(results, f, default=str)
            
            logger.info(f"Results exported to JSON: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def export_to_csv(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_header: bool = True
    ) -> Path:
        """Export results to CSV format.
        
        Args:
            results: Results dictionary
            output_path: Output file path
            include_header: Whether to include header row
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert timeline to DataFrame
            timeline = results.get('timeline', [])
            
            if not timeline:
                logger.warning("No timeline data to export")
                return output_path
            
            # Flatten nested dictionaries
            flattened = []
            for segment in timeline:
                flat_segment = {
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'duration': segment.get('end', 0) - segment.get('start', 0),
                    'speaker': segment.get('speaker', ''),
                    'emotion': segment.get('emotion', ''),
                    'confidence': segment.get('confidence', ''),
                    'type': segment.get('type', '')
                }
                
                # Add features if present
                if 'features' in segment:
                    for key, value in segment['features'].items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                flat_segment[f'{key}_{subkey}'] = subvalue
                        else:
                            flat_segment[key] = value
                
                flattened.append(flat_segment)
            
            # Create DataFrame
            df = pd.DataFrame(flattened)
            
            # Export to CSV
            df.to_csv(output_path, index=False, header=include_header)
            
            logger.info(f"Results exported to CSV: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def export_timeline_segments(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        format: str = 'both'
    ) -> Dict[str, Path]:
        """Export timeline segments in various formats.
        
        Args:
            results: Results dictionary
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
    
    def generate_summary_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a human-readable summary report.
        
        Args:
            results: Analysis results
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("AUDIO ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Metadata
        metadata = results.get('metadata', {})
        if metadata:
            report.append("FILE INFORMATION")
            report.append("-" * 40)
            for key, value in metadata.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # Summary
        summary = results.get('summary', {})
        if summary:
            report.append("ANALYSIS SUMMARY")
            report.append("-" * 40)
            report.append(f"  Total Speakers: {summary.get('total_speakers', 0)}")
            report.append(f"  Total Segments: {summary.get('total_segments', 0)}")
            
            if 'dominant_emotion' in summary:
                report.append(f"  Dominant Emotion: {summary['dominant_emotion']}")
            
            if 'emotion_changes' in summary:
                report.append(f"  Emotion Changes: {summary['emotion_changes']}")
            report.append("")
        
        # Speaker Distribution
        if 'speaker_distribution' in summary:
            report.append("SPEAKER DISTRIBUTION")
            report.append("-" * 40)
            for speaker, percentage in summary['speaker_distribution'].items():
                report.append(f"  {speaker}: {percentage:.1f}%")
            report.append("")
        
        # Emotion Distribution
        if 'emotion_distribution' in summary:
            report.append("EMOTION DISTRIBUTION")
            report.append("-" * 40)
            for emotion, percentage in summary['emotion_distribution'].items():
                report.append(f"  {emotion}: {percentage:.1f}%")
            report.append("")
        
        # Timeline Summary
        timeline = results.get('timeline', [])
        if timeline:
            report.append("TIMELINE SUMMARY")
            report.append("-" * 40)
            report.append(f"  First Segment: {timeline[0]['start']:.2f}s")
            report.append(f"  Last Segment: {timeline[-1]['end']:.2f}s")
            report.append(f"  Total Duration: {timeline[-1]['end']:.2f}s")
            report.append("")
        
        report.append("=" * 60)
        report_text = "\n".join(report)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            logger.info(f"Report saved to: {output_path}")
        
        return report_text