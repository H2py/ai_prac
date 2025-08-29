"""
Pure data transformation logic separated from serialization.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from src.models import AnalysisResults, TimelineEvent, AnalysisStatistics
from src.models.segments import SpeakerSegment, TranscriptionSegment, EmotionSegment, AcousticSegment
from .formats import ExportData

logger = logging.getLogger(__name__)


class DataTransformer:
    """Pure data transformation logic for export operations."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.logger = logging.getLogger(__name__)
    
    def prepare_export_data(self, 
                           results: AnalysisResults, 
                           base_filename: Optional[str] = None) -> ExportData:
        """Prepare unified export data from analysis results.
        
        Args:
            results: Analysis results to transform
            base_filename: Base filename for export
            
        Returns:
            Prepared export data
        """
        if not base_filename:
            base_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get timeline and statistics (cached)
        timeline = results.get_timeline()
        statistics = results.get_statistics()
        
        # Prepare export data
        export_data = ExportData(
            results=results,
            timeline=timeline,
            statistics=statistics,
            export_timestamp=datetime.now().isoformat(),
            base_filename=base_filename
        )
        
        # Prepare format-specific data
        export_data.backend_summary = self.build_backend_summary(results, statistics)
        export_data.frontend_visualization = self.build_frontend_visualization(results, timeline, statistics)
        export_data.subtitle_data = self.build_subtitle_data(results)
        
        return export_data
    
    def build_timeline_data(self, results: AnalysisResults) -> List[Dict[str, Any]]:
        """Build timeline data combining all analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            Timeline data as list of dictionaries
        """
        timeline_events = results.get_timeline()
        
        # Convert to dictionary format
        timeline_data = []
        for event in timeline_events:
            event_data = event.to_dict()
            
            # Add cross-references between segments
            self._add_cross_references(event_data, results, event)
            
            timeline_data.append(event_data)
        
        return timeline_data
    
    def build_backend_summary(self, 
                            results: AnalysisResults, 
                            statistics: AnalysisStatistics) -> Dict[str, Any]:
        """Build backend API compatible summary.
        
        Args:
            results: Analysis results
            statistics: Analysis statistics
            
        Returns:
            Backend summary data
        """
        return {
            "status": "completed",
            "media": {
                "type": results.media_info.source_type,
                "duration": results.media_info.duration,
                "language": results.media_info.language,
                "sample_rate": results.media_info.sample_rate,
                "channels": results.media_info.channels
            },
            "analysis": {
                "speakers": self._extract_speaker_summary(results),
                "emotions": self._extract_emotion_summary(results),
                "transcription": self._extract_transcription_summary(results),
                "acoustic": self._extract_acoustic_summary(results)
            },
            "summary": statistics.get_summary(),
            "metadata": {
                "processed_at": results.metadata.timestamp,
                "processing_time": results.metadata.processing_time,
                "version": results.metadata.pipeline_version,
                "components_used": results.metadata.components_used
            }
        }
    
    def build_frontend_visualization(self, 
                                   results: AnalysisResults,
                                   timeline: List[TimelineEvent],
                                   statistics: AnalysisStatistics) -> Dict[str, Any]:
        """Build frontend-optimized data with visualization components.
        
        Args:
            results: Analysis results
            timeline: Timeline events
            statistics: Analysis statistics
            
        Returns:
            Frontend visualization data
        """
        return {
            "mediaInfo": results.media_info.to_dict(),
            "timeline": [event.to_dict() for event in timeline],
            "summary": {
                "totalSpeakers": statistics.total_speakers,
                "dominantEmotion": statistics.dominant_emotion,
                "transcriptionWordCount": statistics.total_words,
                "processingTime": results.metadata.processing_time,
                "totalSegments": statistics.total_segments
            },
            "visualization": {
                "emotionTimeline": self._build_emotion_timeline(results),
                "speakerDistribution": self._build_speaker_distribution(statistics),
                "acousticFeatures": self._build_acoustic_visualization(results),
                "wordCloud": self._build_word_cloud_data(results)
            },
            "charts": {
                "speakingTime": self._build_speaking_time_chart(statistics),
                "emotionFlow": self._build_emotion_flow_chart(results),
                "acousticTrends": self._build_acoustic_trends_chart(results)
            }
        }
    
    def build_subtitle_data(self, results: AnalysisResults) -> Dict[str, Any]:
        """Build subtitle-specific data for ASS/VTT/SRT formats.
        
        Args:
            results: Analysis results
            
        Returns:
            Subtitle data
        """
        transcription_segments = results.get_segments_by_type(TranscriptionSegment)
        speaker_segments = results.get_segments_by_type(SpeakerSegment)
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        
        # Build combined subtitle segments
        subtitle_segments = []
        
        for trans_seg in transcription_segments:
            subtitle_segment = {
                'start': trans_seg.start,
                'end': trans_seg.end,
                'text': trans_seg.text,
                'speaker': trans_seg.speaker_id or 'unknown',
                'language': trans_seg.language,
                'confidence': trans_seg.confidence
            }
            
            # Find corresponding emotion
            emotion = self._find_corresponding_emotion(trans_seg, emotion_segments)
            if emotion:
                subtitle_segment['emotion'] = emotion.predicted_emotion
                subtitle_segment['emotion_confidence'] = emotion.confidence
            
            subtitle_segments.append(subtitle_segment)
        
        return {
            'segments': subtitle_segments,
            'metadata': {
                'total_segments': len(subtitle_segments),
                'languages': list(set(seg.get('language') for seg in subtitle_segments if seg.get('language'))),
                'speakers': list(set(seg.get('speaker') for seg in subtitle_segments if seg.get('speaker')))
            },
            'styles': self._build_subtitle_styles(results)
        }
    
    def merge_segments_by_timeline(self, 
                                 results: AnalysisResults, 
                                 merge_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Merge overlapping segments into unified timeline entries.
        
        Args:
            results: Analysis results
            merge_threshold: Time threshold for merging (seconds)
            
        Returns:
            Merged timeline segments
        """
        timeline_events = results.get_timeline()
        
        if not timeline_events:
            return []
        
        # Sort by timestamp
        timeline_events.sort(key=lambda e: e.timestamp)
        
        merged_segments = []
        current_segment = None
        
        for event in timeline_events:
            if current_segment is None:
                current_segment = {
                    'start': event.timestamp,
                    'end': event.timestamp + event.duration,
                    'data': [event.data]
                }
            elif event.timestamp <= current_segment['end'] + merge_threshold:
                # Merge with current segment
                current_segment['end'] = max(
                    current_segment['end'], 
                    event.timestamp + event.duration
                )
                current_segment['data'].append(event.data)
            else:
                # Start new segment
                merged_segments.append(current_segment)
                current_segment = {
                    'start': event.timestamp,
                    'end': event.timestamp + event.duration,
                    'data': [event.data]
                }
        
        if current_segment:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def _add_cross_references(self, 
                            event_data: Dict[str, Any], 
                            results: AnalysisResults, 
                            event: TimelineEvent) -> None:
        """Add cross-references between different segment types."""
        event_start = event.timestamp
        event_end = event.timestamp + event.duration
        
        # Find overlapping segments of other types
        cross_refs = {}
        
        for segment in results.segments:
            if (segment.start <= event_end and segment.end >= event_start and
                type(segment).__name__ != event.event_type.name):
                
                segment_type = type(segment).__name__.lower()
                if segment_type not in cross_refs:
                    cross_refs[segment_type] = []
                
                cross_refs[segment_type].append({
                    'start': segment.start,
                    'end': segment.end,
                    'data': segment.to_export_dict()
                })
        
        if cross_refs:
            event_data['cross_references'] = cross_refs
    
    def _extract_speaker_summary(self, results: AnalysisResults) -> Dict[str, Any]:
        """Extract speaker summary for API."""
        speaker_segments = results.get_segments_by_type(SpeakerSegment)
        
        if not speaker_segments:
            return {}
        
        # Group by speaker
        speakers = {}
        total_speaking_time = 0.0
        
        for segment in speaker_segments:
            if segment.speaker_id not in speakers:
                speakers[segment.speaker_id] = {
                    'speaking_time': 0.0,
                    'segment_count': 0
                }
            
            speakers[segment.speaker_id]['speaking_time'] += segment.duration
            speakers[segment.speaker_id]['segment_count'] += 1
            total_speaking_time += segment.duration
        
        # Calculate percentages
        for speaker_data in speakers.values():
            speaker_data['speaking_percentage'] = (
                speaker_data['speaking_time'] / results.media_info.duration * 100
                if results.media_info.duration > 0 else 0.0
            )
        
        return {
            'total_speakers': len(speakers),
            'speakers': speakers,
            'total_speaking_time': total_speaking_time
        }
    
    def _extract_emotion_summary(self, results: AnalysisResults) -> Dict[str, Any]:
        """Extract emotion summary for API."""
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        
        if not emotion_segments:
            return {}
        
        # Count emotions
        emotion_counts = {}
        total_confidence = 0.0
        
        for segment in emotion_segments:
            emotion = segment.predicted_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += segment.confidence
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_counts,
            'total_segments': len(emotion_segments),
            'average_confidence': total_confidence / len(emotion_segments) if emotion_segments else 0.0
        }
    
    def _extract_transcription_summary(self, results: AnalysisResults) -> Dict[str, Any]:
        """Extract transcription summary for API."""
        transcription_segments = results.get_segments_by_type(TranscriptionSegment)
        
        if not transcription_segments:
            return {}
        
        total_words = sum(seg.word_count or 0 for seg in transcription_segments)
        languages = set(seg.language for seg in transcription_segments if seg.language)
        
        return {
            'total_segments': len(transcription_segments),
            'total_words': total_words,
            'average_words_per_segment': (
                total_words / len(transcription_segments) 
                if transcription_segments else 0
            ),
            'languages': list(languages)
        }
    
    def _extract_acoustic_summary(self, results: AnalysisResults) -> Dict[str, Any]:
        """Extract acoustic summary for API."""
        acoustic_segments = results.get_segments_by_type(AcousticSegment)
        
        if not acoustic_segments:
            return {}
        
        return {
            'total_segments': len(acoustic_segments),
            'has_pitch_data': any(seg.pitch_mean is not None for seg in acoustic_segments),
            'has_energy_data': any(seg.rms_energy is not None for seg in acoustic_segments),
            'feature_count': len(acoustic_segments[0].features) if acoustic_segments else 0
        }
    
    def _build_emotion_timeline(self, results: AnalysisResults) -> List[Dict[str, Any]]:
        """Build emotion timeline for visualization."""
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        
        return [
            {
                'timestamp': segment.start,
                'emotion': segment.predicted_emotion,
                'confidence': segment.confidence,
                'duration': segment.duration
            }
            for segment in sorted(emotion_segments, key=lambda s: s.start)
        ]
    
    def _build_speaker_distribution(self, statistics: AnalysisStatistics) -> Dict[str, float]:
        """Build speaker distribution for visualization."""
        return statistics.calculate_speaking_percentages()
    
    def _build_acoustic_visualization(self, results: AnalysisResults) -> Dict[str, List[float]]:
        """Build acoustic features for visualization."""
        acoustic_segments = results.get_segments_by_type(AcousticSegment)
        
        if not acoustic_segments:
            return {}
        
        features = {
            'timestamps': [],
            'rms_energy': [],
            'spectral_centroid': [],
            'pitch_mean': []
        }
        
        for segment in sorted(acoustic_segments, key=lambda s: s.start):
            features['timestamps'].append(segment.start)
            features['rms_energy'].append(segment.rms_energy or 0.0)
            features['spectral_centroid'].append(segment.spectral_centroid or 0.0)
            features['pitch_mean'].append(segment.pitch_mean or 0.0)
        
        return features
    
    def _build_word_cloud_data(self, results: AnalysisResults) -> List[Dict[str, Any]]:
        """Build word cloud data from transcriptions."""
        transcription_segments = results.get_segments_by_type(TranscriptionSegment)
        
        if not transcription_segments:
            return []
        
        # Simple word frequency counting
        word_counts = {}
        for segment in transcription_segments:
            if segment.text:
                words = segment.text.lower().split()
                for word in words:
                    # Basic word cleaning
                    word = ''.join(c for c in word if c.isalnum())
                    if len(word) > 2:  # Filter short words
                        word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {'text': word, 'count': count}
            for word, count in sorted_words[:50]  # Top 50 words
        ]
    
    def _build_speaking_time_chart(self, statistics: AnalysisStatistics) -> Dict[str, Any]:
        """Build speaking time chart data."""
        speaking_percentages = statistics.calculate_speaking_percentages()
        
        return {
            'type': 'pie',
            'data': [
                {'speaker': speaker, 'percentage': percentage}
                for speaker, percentage in speaking_percentages.items()
            ]
        }
    
    def _build_emotion_flow_chart(self, results: AnalysisResults) -> Dict[str, Any]:
        """Build emotion flow chart data."""
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        
        if not emotion_segments:
            return {'type': 'line', 'data': []}
        
        # Sample emotions at regular intervals
        duration = results.media_info.duration
        sample_interval = max(1.0, duration / 100)  # 100 samples max
        
        samples = []
        current_time = 0.0
        
        while current_time < duration:
            # Find emotion at current time
            current_emotion = 'neutral'
            for segment in emotion_segments:
                if segment.start <= current_time <= segment.end:
                    current_emotion = segment.predicted_emotion
                    break
            
            samples.append({
                'time': current_time,
                'emotion': current_emotion
            })
            
            current_time += sample_interval
        
        return {
            'type': 'line',
            'data': samples
        }
    
    def _build_acoustic_trends_chart(self, results: AnalysisResults) -> Dict[str, Any]:
        """Build acoustic trends chart data."""
        acoustic_segments = results.get_segments_by_type(AcousticSegment)
        
        if not acoustic_segments:
            return {'type': 'line', 'data': []}
        
        return {
            'type': 'multi_line',
            'data': [
                {
                    'time': segment.start,
                    'pitch': segment.pitch_mean or 0.0,
                    'energy': segment.rms_energy or 0.0,
                    'spectral_centroid': segment.spectral_centroid or 0.0
                }
                for segment in sorted(acoustic_segments, key=lambda s: s.start)
            ]
        }
    
    def _build_subtitle_styles(self, results: AnalysisResults) -> Dict[str, Dict[str, str]]:
        """Build subtitle styles based on emotions and speakers."""
        emotion_segments = results.get_segments_by_type(EmotionSegment)
        
        # Default styles
        styles = {
            'default': {
                'font_family': 'Arial',
                'font_size': '16',
                'color': '#FFFFFF',
                'background_color': '#000000'
            }
        }
        
        # Emotion-based styles
        emotion_colors = {
            'happy': '#FFD700',      # Gold
            'sad': '#4169E1',        # Royal Blue
            'angry': '#FF4500',      # Orange Red
            'fear': '#9370DB',       # Medium Purple
            'surprise': '#FF69B4',   # Hot Pink
            'disgust': '#32CD32',    # Lime Green
            'neutral': '#FFFFFF'     # White
        }
        
        unique_emotions = set(seg.predicted_emotion for seg in emotion_segments)
        for emotion in unique_emotions:
            if emotion in emotion_colors:
                styles[f'emotion_{emotion}'] = {
                    **styles['default'],
                    'color': emotion_colors[emotion]
                }
        
        return styles
    
    def _find_corresponding_emotion(self, 
                                  transcription_seg: TranscriptionSegment,
                                  emotion_segments: List[EmotionSegment]) -> Optional[EmotionSegment]:
        """Find emotion segment that corresponds to transcription segment."""
        for emotion_seg in emotion_segments:
            # Check for overlap
            if (emotion_seg.start <= transcription_seg.end and
                emotion_seg.end >= transcription_seg.start):
                return emotion_seg
        return None