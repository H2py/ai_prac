"""
Format-specific serialization logic separated from data transformation.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
import logging

from .formats import ExportData, OutputFormat

logger = logging.getLogger(__name__)


class FormatSerializer(ABC):
    """Abstract base class for format-specific serializers."""
    
    def __init__(self, output_format: OutputFormat):
        """Initialize serializer.
        
        Args:
            output_format: The output format this serializer handles
        """
        self.output_format = output_format
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def serialize(self, data: ExportData) -> str:
        """Serialize export data to string format.
        
        Args:
            data: Export data to serialize
            
        Returns:
            Serialized string
        """
        pass
    
    def get_file_extension(self) -> str:
        """Get file extension for this format."""
        return self.output_format.value
    
    def get_filename(self, base_filename: str) -> str:
        """Get full filename with extension.
        
        Args:
            base_filename: Base filename without extension
            
        Returns:
            Full filename with extension
        """
        extension = self.get_file_extension()
        if base_filename.endswith(f".{extension}"):
            return base_filename
        return f"{base_filename}.{extension}"


class JSONSerializer(FormatSerializer):
    """JSON format serializer with comprehensive data structure."""
    
    def __init__(self):
        super().__init__(OutputFormat.JSON)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to enhanced JSON format."""
        json_data = {
            "media": {
                **data.results.media_info.to_dict(),
                "timestamp": data.results.metadata.timestamp,
                "standards": {
                    "speech_api": "W3C Web Speech API 1.0",
                    "timed_text": "SMPTE ST 2052-1:2013", 
                    "phonetic": "IPA (International Phonetic Alphabet)",
                    "emotion": "PAD (Pleasure-Arousal-Dominance) model"
                }
            },
            "processing": data.results.metadata.to_dict(),
            "analysis": {
                "segments": [seg.to_export_dict() for seg in data.results.segments],
                "timeline": [event.to_dict() for event in data.timeline],
                "statistics": data.statistics.to_dict() if data.statistics else {}
            },
            "export": {
                "timestamp": data.export_timestamp,
                "base_filename": data.base_filename,
                "format": self.output_format.value
            }
        }
        
        return json.dumps(json_data, indent=2, ensure_ascii=False, default=str)


class ASSSerializer(FormatSerializer):
    """ASS (Advanced Substation Alpha) subtitle serializer with emotion styling."""
    
    def __init__(self):
        super().__init__(OutputFormat.ASS)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to ASS format with emotion-based styling."""
        if not data.subtitle_data or not data.subtitle_data.get('segments'):
            self.logger.warning("No subtitle segments available for ASS export")
            return self._create_empty_ass()
        
        lines = []
        
        # ASS header
        lines.extend(self._create_ass_header(data))
        
        # Styles section
        lines.extend(self._create_ass_styles(data.subtitle_data.get('styles', {})))
        
        # Events section
        lines.extend(self._create_ass_events(data.subtitle_data['segments']))
        
        return '\n'.join(lines)
    
    def _create_empty_ass(self) -> str:
        """Create empty ASS file."""
        return """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,16,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def _create_ass_header(self, data: ExportData) -> List[str]:
        """Create ASS header section."""
        duration = data.results.media_info.duration
        return [
            "[Script Info]",
            f"Title: Audio Analysis - {data.base_filename}",
            f"ScriptType: v4.00+",
            f"Collisions: Normal",
            f"PlayDepth: 0",
            f"Timer: 100.0000",
            f"Video Aspect Ratio: 0",
            f"Video Zoom: 6",
            f"Video Position: 0",
            f"Last Style Storage: Default",
            f"Audio URI: {data.results.media_info.file_path or ''}",
            f"Video File: {data.results.media_info.file_path or ''}",
            f"Video AR Mode: 4",
            f"Video AR Value: 1.333333",
            f"Scroll Position: 0",
            f"Active Line: 0",
            f"Video Zoom Percent: 1.000000",
            f"",
            f"[V4+ Styles]",
            f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ]
    
    def _create_ass_styles(self, styles: Dict[str, Dict[str, str]]) -> List[str]:
        """Create ASS styles section."""
        style_lines = []
        
        # Default style
        default_style = styles.get('default', {
            'font_family': 'Arial',
            'font_size': '16',
            'color': '#FFFFFF',
            'background_color': '#000000'
        })
        
        style_lines.append(
            f"Style: Default,{default_style['font_family']},{default_style['font_size']},"
            f"&H{self._hex_to_ass_color(default_style['color'])},&H000000FF,&H00000000,"
            f"&H{self._hex_to_ass_color(default_style['background_color'], alpha='80')},0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1"
        )
        
        # Emotion-based styles
        for style_name, style_def in styles.items():
            if style_name != 'default' and style_name.startswith('emotion_'):
                emotion = style_name.replace('emotion_', '')
                style_lines.append(
                    f"Style: {emotion.title()},{style_def.get('font_family', 'Arial')},{style_def.get('font_size', '16')},"
                    f"&H{self._hex_to_ass_color(style_def.get('color', '#FFFFFF'))},&H000000FF,&H00000000,"
                    f"&H{self._hex_to_ass_color(style_def.get('background_color', '#000000'), alpha='80')},0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1"
                )
        
        style_lines.append("")
        style_lines.append("[Events]")
        style_lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
        
        return style_lines
    
    def _create_ass_events(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Create ASS events section."""
        event_lines = []
        
        for i, segment in enumerate(segments):
            start_time = self._format_ass_time(segment['start'])
            end_time = self._format_ass_time(segment['end'])
            text = segment['text'].replace('\n', '\\N')
            speaker = segment.get('speaker', '')
            emotion = segment.get('emotion', 'default')
            
            # Determine style based on emotion
            style = emotion.title() if emotion != 'default' else 'Default'
            
            # Format speaker name
            if speaker and speaker != 'unknown':
                formatted_text = f"{{{speaker}}}{text}"
            else:
                formatted_text = text
            
            # Add emotion indicators if available
            if emotion and emotion != 'neutral':
                formatted_text = f"[{emotion.upper()}] {formatted_text}"
            
            event_line = f"Dialogue: 0,{start_time},{end_time},{style},,0,0,0,,{formatted_text}"
            event_lines.append(event_line)
        
        return event_lines
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    
    def _hex_to_ass_color(self, hex_color: str, alpha: str = '00') -> str:
        """Convert hex color to ASS color format."""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        # ASS uses BGR format
        if len(hex_color) == 6:
            r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
            return f"{alpha}{b.upper()}{g.upper()}{r.upper()}"
        
        return f"{alpha}FFFFFF"  # Default to white


class VTTSerializer(FormatSerializer):
    """WebVTT subtitle serializer with speaker tags."""
    
    def __init__(self):
        super().__init__(OutputFormat.VTT)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to WebVTT format."""
        if not data.subtitle_data or not data.subtitle_data.get('segments'):
            return "WEBVTT\n\nNOTE No transcription data available"
        
        lines = ["WEBVTT", ""]
        
        # Add metadata
        metadata = data.subtitle_data.get('metadata', {})
        lines.append(f"NOTE Created: {data.export_timestamp}")
        lines.append(f"NOTE Duration: {data.results.media_info.duration:.2f}s")
        lines.append(f"NOTE Language: {data.results.media_info.language or 'auto'}")
        if metadata.get('speakers'):
            lines.append(f"NOTE Speakers: {', '.join(metadata['speakers'])}")
        lines.append("")
        
        # Add segments
        for i, segment in enumerate(data.subtitle_data['segments']):
            start_time = self._format_vtt_time(segment['start'])
            end_time = self._format_vtt_time(segment['end'])
            text = segment['text'].strip()
            speaker = segment.get('speaker', '')
            
            lines.append(f"{i + 1}")
            lines.append(f"{start_time} --> {end_time}")
            
            if speaker and speaker != 'unknown':
                lines.append(f"<v {speaker}>{text}")
            else:
                lines.append(text)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for WebVTT format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class SRTSerializer(FormatSerializer):
    """SRT subtitle serializer."""
    
    def __init__(self):
        super().__init__(OutputFormat.SRT)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to SRT format."""
        if not data.subtitle_data or not data.subtitle_data.get('segments'):
            return "1\n00:00:00,000 --> 00:00:03,000\nNo transcription data available\n"
        
        lines = []
        
        for i, segment in enumerate(data.subtitle_data['segments']):
            start_time = self._format_srt_time(segment['start'])
            end_time = self._format_srt_time(segment['end'])
            text = segment['text'].strip()
            speaker = segment.get('speaker', '')
            
            lines.append(str(i + 1))
            lines.append(f"{start_time} --> {end_time}")
            
            if speaker and speaker != 'unknown':
                lines.append(f"[{speaker}] {text}")
            else:
                lines.append(text)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


class BackendAPISerializer(FormatSerializer):
    """Backend API serializer for server integration."""
    
    def __init__(self):
        super().__init__(OutputFormat.BACKEND_API)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to backend API format."""
        if not data.backend_summary:
            return json.dumps({"status": "error", "message": "No backend summary available"})
        
        return json.dumps(data.backend_summary, indent=2, ensure_ascii=False, default=str)
    
    def get_filename(self, base_filename: str) -> str:
        """Get filename for backend API format."""
        return f"{base_filename}_api.json"


class FrontendJSONSerializer(FormatSerializer):
    """Frontend JSON serializer for rich visualization."""
    
    def __init__(self):
        super().__init__(OutputFormat.FRONTEND_JSON)
    
    def serialize(self, data: ExportData) -> str:
        """Serialize to frontend JSON format."""
        if not data.frontend_visualization:
            return json.dumps({"error": "No frontend visualization data available"})
        
        return json.dumps(data.frontend_visualization, indent=2, ensure_ascii=False, default=str)
    
    def get_filename(self, base_filename: str) -> str:
        """Get filename for frontend JSON format."""
        return f"{base_filename}_frontend.json"


# Factory for creating serializers
def create_serializer(output_format: OutputFormat) -> FormatSerializer:
    """Create serializer for given output format.
    
    Args:
        output_format: Output format to create serializer for
        
    Returns:
        Appropriate serializer instance
        
    Raises:
        ValueError: If format is not supported
    """
    serializers = {
        OutputFormat.JSON: JSONSerializer,
        OutputFormat.ASS: ASSSerializer,
        OutputFormat.VTT: VTTSerializer,
        OutputFormat.SRT: SRTSerializer,
        OutputFormat.BACKEND_API: BackendAPISerializer,
        OutputFormat.FRONTEND_JSON: FrontendJSONSerializer
    }
    
    if output_format not in serializers:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    return serializers[output_format]()