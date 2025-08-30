"""
Unified Output Processing System

This module handles all output generation, formatting, and export functionality.
Combines result merging, format management, and subtitle generation into a cohesive system.

Key Components:
- OutputFormatManager: Handles multiple output formats
- ResultMerger: Merges analysis results from different components
- SubtitleExporter: Generates subtitle files (ASS, VTT, SRT)
- Enhanced formatting with timeline and metadata support
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import re

from config.settings import Config
from .models import (
    AnalysisResults, ProcessingMetadata, MediaInfo, AnalysisTimeline, TimelineEvent,
    SpeakerSegment, EmotionSegment, TranscriptionSegment, AcousticSegment
)


logger = logging.getLogger(__name__)


# ================================
# OUTPUT FORMAT DEFINITIONS
# ================================

class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    ASS = "ass"
    VTT = "vtt"
    SRT = "srt"
    BACKEND_API = "backend_api"
    FRONTEND_JSON = "frontend_json"


# ================================
# RESULT MERGER
# ================================

class ResultMerger:
    """Merges analysis results from different pipeline components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResultMerger")
    
    def merge_results(
        self,
        context,  # PipelineContext
        processing_metadata: ProcessingMetadata,
        use_enhanced_format: bool = False
    ) -> Dict[str, Any]:
        """Merge all analysis results into unified format."""
        
        # Create analysis results container
        results = AnalysisResults()
        
        # Extract results from context
        if context.speaker_results:
            results.speaker_segments = context.speaker_results.get('segments', [])
        
        if context.emotion_results:
            results.emotion_segments = context.emotion_results
        
        if context.transcription_results:
            results.transcription_segments = context.transcription_results
        
        if context.acoustic_results:
            results.acoustic_segments = context.acoustic_results
        
        # Set metadata
        if context.audio_info:
            results.audio_duration = context.audio_info.get('duration', 0.0)
            results.sample_rate = context.audio_info.get('sample_rate', 16000)
        
        results.processing_time = sum(context.step_durations.values())
        
        if use_enhanced_format:
            return self._create_enhanced_output(results, processing_metadata, context)
        else:
            return self._create_standard_output(results, processing_metadata)
    
    def _create_standard_output(
        self, 
        results: AnalysisResults, 
        metadata: ProcessingMetadata
    ) -> Dict[str, Any]:
        """Create standard JSON output format."""
        
        return {
            'metadata': metadata.to_dict(),
            'analysis': results.to_dict(),
            'summary': results.get_analysis_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _create_enhanced_output(
        self,
        results: AnalysisResults,
        metadata: ProcessingMetadata,
        context
    ) -> Dict[str, Any]:
        """Create enhanced output with timeline and advanced features."""
        
        # Create timeline
        timeline = self._build_timeline(results)
        
        # Generate insights
        insights = self._generate_insights(results)
        
        # Create visualization data
        viz_data = self._create_visualization_data(results)
        
        enhanced_output = {
            'format_version': '2.0',
            'processing_info': {
                'pipeline_version': 'enhanced',
                'timestamp': datetime.now().isoformat(),
                'processing_time': results.processing_time,
                'success_rate': metadata.get_success_rate(),
                'features_enabled': metadata.enabled_features
            },
            'media_info': self._extract_media_info(context),
            'analysis_results': results.to_dict(),
            'timeline': timeline.to_dict(),
            'insights': insights,
            'visualization': viz_data,
            'quality_metrics': self._calculate_quality_metrics(results),
            'export_formats': {
                'available': [fmt.value for fmt in OutputFormat],
                'recommended': self._recommend_formats(results)
            }
        }
        
        return enhanced_output
    
    def _build_timeline(self, results: AnalysisResults) -> AnalysisTimeline:
        """Build unified timeline from all analysis results."""
        timeline = AnalysisTimeline(total_duration=results.audio_duration)
        
        # Add speaker events
        for segment in results.speaker_segments:
            event = TimelineEvent(
                start_time=segment.start,
                end_time=segment.end,
                event_type='speaker',
                data={
                    'speaker_id': segment.speaker_id,
                    'confidence': segment.confidence
                },
                confidence=segment.confidence
            )
            timeline.add_event(event)
        
        # Add emotion events
        for segment in results.emotion_segments:
            event = TimelineEvent(
                start_time=segment.start,
                end_time=segment.end,
                event_type='emotion',
                data={
                    'emotion': segment.predicted_emotion,
                    'scores': segment.emotion_scores
                },
                confidence=segment.confidence
            )
            timeline.add_event(event)
        
        # Add transcription events
        for segment in results.transcription_segments:
            event = TimelineEvent(
                start_time=segment.start,
                end_time=segment.end,
                event_type='transcription',
                data={
                    'text': segment.text,
                    'language': segment.language,
                    'word_count': segment.get_word_count()
                },
                confidence=segment.confidence
            )
            timeline.add_event(event)
        
        return timeline
    
    def _generate_insights(self, results: AnalysisResults) -> Dict[str, Any]:
        """Generate analytical insights from results."""
        insights = {}
        
        # Speaker insights
        if results.speaker_segments:
            speaker_stats = self._analyze_speaker_patterns(results.speaker_segments)
            insights['speakers'] = speaker_stats
        
        # Emotion insights
        if results.emotion_segments:
            emotion_stats = self._analyze_emotion_patterns(results.emotion_segments)
            insights['emotions'] = emotion_stats
        
        # Speech insights
        if results.transcription_segments:
            speech_stats = self._analyze_speech_patterns(results.transcription_segments)
            insights['speech'] = speech_stats
        
        return insights
    
    def _analyze_speaker_patterns(self, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """Analyze speaker patterns and generate insights."""
        speaker_stats = {}
        total_duration = sum(s.duration for s in segments)
        
        # Group by speaker
        by_speaker = {}
        for segment in segments:
            if segment.speaker_id not in by_speaker:
                by_speaker[segment.speaker_id] = []
            by_speaker[segment.speaker_id].append(segment)
        
        # Calculate statistics per speaker
        for speaker_id, speaker_segments in by_speaker.items():
            duration = sum(s.duration for s in speaker_segments)
            speaker_stats[speaker_id] = {
                'total_duration': duration,
                'speaking_percentage': (duration / total_duration * 100) if total_duration > 0 else 0,
                'segment_count': len(speaker_segments),
                'average_segment_length': duration / len(speaker_segments) if speaker_segments else 0
            }
        
        return {
            'total_speakers': len(by_speaker),
            'speaker_statistics': speaker_stats,
            'dominant_speaker': max(speaker_stats.keys(), 
                                  key=lambda x: speaker_stats[x]['speaking_percentage']) if speaker_stats else None
        }
    
    def _analyze_emotion_patterns(self, segments: List[EmotionSegment]) -> Dict[str, Any]:
        """Analyze emotion patterns and generate insights."""
        emotion_counts = {}
        total_duration = sum(s.duration for s in segments)
        
        # Count emotions and durations
        for segment in segments:
            emotion = segment.predicted_emotion
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {'count': 0, 'duration': 0.0}
            emotion_counts[emotion]['count'] += 1
            emotion_counts[emotion]['duration'] += segment.duration
        
        # Calculate percentages
        emotion_stats = {}
        for emotion, stats in emotion_counts.items():
            emotion_stats[emotion] = {
                'count': stats['count'],
                'total_duration': stats['duration'],
                'percentage': (stats['duration'] / total_duration * 100) if total_duration > 0 else 0,
                'average_duration': stats['duration'] / stats['count'] if stats['count'] > 0 else 0
            }
        
        return {
            'emotion_distribution': emotion_stats,
            'dominant_emotion': max(emotion_stats.keys(),
                                  key=lambda x: emotion_stats[x]['percentage']) if emotion_stats else None,
            'emotion_variety': len(emotion_stats)
        }
    
    def _analyze_speech_patterns(self, segments: List[TranscriptionSegment]) -> Dict[str, Any]:
        """Analyze speech patterns from transcriptions."""
        total_words = sum(s.get_word_count() for s in segments)
        total_duration = sum(s.duration for s in segments)
        
        # Language analysis
        languages = {}
        for segment in segments:
            if segment.language:
                lang = segment.language
                if lang not in languages:
                    languages[lang] = {'count': 0, 'duration': 0.0, 'words': 0}
                languages[lang]['count'] += 1
                languages[lang]['duration'] += segment.duration
                languages[lang]['words'] += segment.get_word_count()
        
        return {
            'total_words': total_words,
            'speaking_rate': (total_words / total_duration) if total_duration > 0 else 0,  # words per second
            'languages_detected': languages,
            'primary_language': max(languages.keys(),
                                  key=lambda x: languages[x]['duration']) if languages else None
        }
    
    def _create_visualization_data(self, results: AnalysisResults) -> Dict[str, Any]:
        """Create data structures for visualization."""
        return {
            'timeline_data': self._create_timeline_chart_data(results),
            'speaker_chart': self._create_speaker_chart_data(results),
            'emotion_chart': self._create_emotion_chart_data(results),
            'waveform_annotations': self._create_waveform_annotations(results)
        }
    
    def _create_timeline_chart_data(self, results: AnalysisResults) -> List[Dict[str, Any]]:
        """Create timeline chart data."""
        timeline_data = []
        
        # Add speaker data
        for segment in results.speaker_segments:
            timeline_data.append({
                'start': segment.start,
                'end': segment.end,
                'type': 'speaker',
                'label': segment.speaker_id,
                'color': self._get_speaker_color(segment.speaker_id)
            })
        
        # Add emotion data
        for segment in results.emotion_segments:
            timeline_data.append({
                'start': segment.start,
                'end': segment.end,
                'type': 'emotion',
                'label': segment.predicted_emotion,
                'color': self._get_emotion_color(segment.predicted_emotion),
                'confidence': segment.confidence
            })
        
        return sorted(timeline_data, key=lambda x: x['start'])
    
    def _get_speaker_color(self, speaker_id: str) -> str:
        """Get consistent color for speaker."""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        speaker_num = int(re.search(r'\d+', speaker_id).group()) if re.search(r'\d+', speaker_id) else 0
        return colors[speaker_num % len(colors)]
    
    def _get_emotion_color(self, emotion: str) -> str:
        """Get color for emotion."""
        emotion_colors = {
            'happy': '#FFD93D',
            'sad': '#6BCF7F',
            'angry': '#FF6B6B',
            'neutral': '#95A5A6',
            'fear': '#9B59B6',
            'surprise': '#F39C12',
            'disgust': '#E74C3C'
        }
        return emotion_colors.get(emotion, '#BDC3C7')
    
    def _calculate_quality_metrics(self, results: AnalysisResults) -> Dict[str, Any]:
        """Calculate quality metrics for analysis results."""
        metrics = {}
        
        # Overall confidence scores
        if results.emotion_segments:
            emotion_confidence = sum(s.confidence for s in results.emotion_segments) / len(results.emotion_segments)
            metrics['emotion_confidence'] = emotion_confidence
        
        if results.transcription_segments:
            transcription_confidence = sum(s.confidence for s in results.transcription_segments) / len(results.transcription_segments)
            metrics['transcription_confidence'] = transcription_confidence
        
        # Coverage metrics
        total_duration = results.audio_duration
        if total_duration > 0:
            speaker_coverage = sum(s.duration for s in results.speaker_segments) / total_duration
            metrics['speaker_coverage'] = speaker_coverage
            
            if results.transcription_segments:
                transcription_coverage = sum(s.duration for s in results.transcription_segments) / total_duration
                metrics['transcription_coverage'] = transcription_coverage
        
        return metrics
    
    def _extract_media_info(self, context) -> Dict[str, Any]:
        """Extract media information from context."""
        media_info = {
            'source': context.input_source,
            'timestamp': context.timestamp
        }
        
        if context.audio_info:
            media_info['audio'] = context.audio_info
        
        if context.video_metadata:
            media_info['video'] = context.video_metadata
        
        return media_info
    
    def _recommend_formats(self, results: AnalysisResults) -> List[str]:
        """Recommend output formats based on analysis results."""
        recommended = ['json']  # Always recommend JSON
        
        # Recommend subtitles if transcription available
        if results.transcription_segments:
            recommended.extend(['vtt', 'srt'])
            
            # Recommend ASS if emotion data available for styling
            if results.emotion_segments:
                recommended.append('ass')
        
        # Recommend API formats for integration
        recommended.extend(['backend_api', 'frontend_json'])
        
        return recommended


# ================================
# OUTPUT FORMAT MANAGER
# ================================

class OutputFormatManager:
    """Manages multiple output format generation."""
    
    def __init__(self, config: Config, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"{__name__}.OutputFormatManager")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_results(
        self,
        results: Dict[str, Any],
        base_filename: str,
        formats: List[str]
    ) -> Dict[str, Path]:
        """Export results in multiple formats."""
        exported_files = {}
        
        for format_name in formats:
            try:
                output_format = OutputFormat(format_name)
                file_path = self._export_format(results, base_filename, output_format)
                if file_path:
                    exported_files[format_name] = file_path
                    self.logger.info(f"Exported {format_name}: {file_path}")
            except ValueError:
                self.logger.warning(f"Unknown output format: {format_name}")
            except Exception as e:
                self.logger.error(f"Failed to export {format_name}: {e}")
        
        return exported_files
    
    def _export_format(
        self,
        results: Dict[str, Any],
        base_filename: str,
        output_format: OutputFormat
    ) -> Optional[Path]:
        """Export results in specific format."""
        
        if output_format == OutputFormat.JSON:
            return self._export_json(results, f"{base_filename}.json")
        
        elif output_format == OutputFormat.BACKEND_API:
            api_results = self._create_backend_api_format(results)
            return self._export_json(api_results, f"{base_filename}_api.json")
        
        elif output_format == OutputFormat.FRONTEND_JSON:
            frontend_results = self._create_frontend_format(results)
            return self._export_json(frontend_results, f"{base_filename}_frontend.json")
        
        elif output_format in [OutputFormat.ASS, OutputFormat.VTT, OutputFormat.SRT]:
            return self._export_subtitle(results, base_filename, output_format)
        
        return None
    
    def _export_json(self, data: Dict[str, Any], filename: str) -> Path:
        """Export JSON format."""
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return file_path
    
    def _create_backend_api_format(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create backend API compatible format."""
        # Extract key information for API consumption
        api_format = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'summary': results.get('summary', {}),
            'processing_time': results.get('processing_info', {}).get('processing_time', 0),
            'analysis': {
                'speakers': self._extract_speaker_summary(results),
                'emotions': self._extract_emotion_summary(results),
                'transcription': self._extract_transcription_summary(results)
            },
            'metadata': {
                'format': 'backend_api_v1',
                'source': results.get('media_info', {}).get('source', ''),
                'duration': results.get('analysis_results', {}).get('metadata', {}).get('audio_duration', 0)
            }
        }
        
        return api_format
    
    def _create_frontend_format(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create frontend-optimized format with visualization data."""
        frontend_format = {
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'media_info': results.get('media_info', {}),
            'timeline': results.get('timeline', {}),
            'visualization': results.get('visualization', {}),
            'insights': results.get('insights', {}),
            'quality_metrics': results.get('quality_metrics', {}),
            'ui_config': {
                'default_view': 'timeline',
                'available_views': ['timeline', 'speakers', 'emotions', 'transcription'],
                'chart_types': ['timeline', 'speaker_distribution', 'emotion_flow']
            }
        }
        
        return frontend_format
    
    def _export_subtitle(
        self,
        results: Dict[str, Any],
        base_filename: str,
        subtitle_format: OutputFormat
    ) -> Optional[Path]:
        """Export subtitle files."""
        # Extract transcription segments
        transcription_data = results.get('analysis_results', {}).get('transcription_segments', [])
        if not transcription_data:
            self.logger.warning("No transcription data available for subtitle export")
            return None
        
        exporter = SubtitleExporter(self.config)
        
        if subtitle_format == OutputFormat.ASS:
            return exporter.export_ass(transcription_data, self.output_dir / f"{base_filename}.ass")
        elif subtitle_format == OutputFormat.VTT:
            return exporter.export_vtt(transcription_data, self.output_dir / f"{base_filename}.vtt")
        elif subtitle_format == OutputFormat.SRT:
            return exporter.export_srt(transcription_data, self.output_dir / f"{base_filename}.srt")
        
        return None
    
    def _extract_speaker_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract speaker summary for API."""
        insights = results.get('insights', {})
        speaker_data = insights.get('speakers', {})
        
        return {
            'count': speaker_data.get('total_speakers', 0),
            'dominant': speaker_data.get('dominant_speaker'),
            'distribution': speaker_data.get('speaker_statistics', {})
        }
    
    def _extract_emotion_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotion summary for API."""
        insights = results.get('insights', {})
        emotion_data = insights.get('emotions', {})
        
        return {
            'dominant': emotion_data.get('dominant_emotion'),
            'variety': emotion_data.get('emotion_variety', 0),
            'distribution': emotion_data.get('emotion_distribution', {})
        }
    
    def _extract_transcription_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract transcription summary for API."""
        insights = results.get('insights', {})
        speech_data = insights.get('speech', {})
        
        return {
            'total_words': speech_data.get('total_words', 0),
            'speaking_rate': speech_data.get('speaking_rate', 0),
            'primary_language': speech_data.get('primary_language'),
            'languages': list(speech_data.get('languages_detected', {}).keys())
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return [fmt.value for fmt in OutputFormat]


# ================================
# SUBTITLE EXPORTER
# ================================

class SubtitleExporter:
    """Handles export of subtitle files in various formats."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SubtitleExporter")
    
    def export_ass(self, transcription_segments: List[Dict], output_path: Path) -> Path:
        """Export ASS subtitle format with emotion-based styling."""
        
        # ASS file header
        ass_content = [
            "[Script Info]",
            "Title: Audio Analysis Subtitles",
            f"ScriptType: v4.00+",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
            "Style: Happy,Arial,16,&H00ffff,&H00ffff,&H0,&H0,1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
            "Style: Sad,Arial,16,&Hff9999,&Hff9999,&H0,&H0,0,1,0,0,100,100,0,0,1,2,0,2,10,10,10,1",
            "Style: Angry,Arial,16,&H4444ff,&H4444ff,&H0,&H0,1,0,0,0,110,100,0,0,1,2,0,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        # Convert segments to ASS format
        for i, segment in enumerate(transcription_segments):
            start_time = self._format_ass_time(segment.get('start', 0))
            end_time = self._format_ass_time(segment.get('end', 0))
            text = segment.get('text', '').replace('\n', '\\N')
            
            # Try to get emotion for styling
            emotion = segment.get('emotion', 'neutral')
            style = self._get_ass_style_for_emotion(emotion)
            
            ass_line = f"Dialogue: 0,{start_time},{end_time},{style},,0,0,0,,{text}"
            ass_content.append(ass_line)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ass_content))
        
        return output_path
    
    def export_vtt(self, transcription_segments: List[Dict], output_path: Path) -> Path:
        """Export WebVTT subtitle format."""
        
        vtt_content = ["WEBVTT", ""]
        
        for i, segment in enumerate(transcription_segments, 1):
            start_time = self._format_vtt_time(segment.get('start', 0))
            end_time = self._format_vtt_time(segment.get('end', 0))
            text = segment.get('text', '')
            
            vtt_content.extend([
                f"{i}",
                f"{start_time} --> {end_time}",
                text,
                ""
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_content))
        
        return output_path
    
    def export_srt(self, transcription_segments: List[Dict], output_path: Path) -> Path:
        """Export SRT subtitle format."""
        
        srt_content = []
        
        for i, segment in enumerate(transcription_segments, 1):
            start_time = self._format_srt_time(segment.get('start', 0))
            end_time = self._format_srt_time(segment.get('end', 0))
            text = segment.get('text', '')
            
            srt_content.extend([
                str(i),
                f"{start_time} --> {end_time}",
                text,
                ""
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        return output_path
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _get_ass_style_for_emotion(self, emotion: str) -> str:
        """Get ASS style name for emotion."""
        emotion_styles = {
            'happy': 'Happy',
            'sad': 'Sad',
            'angry': 'Angry',
            'neutral': 'Default'
        }
        return emotion_styles.get(emotion.lower(), 'Default')