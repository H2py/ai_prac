"""
ASS (Advanced SubStation Alpha) subtitle exporter module.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ASSExporter:
    """Export analysis results to ASS subtitle format."""
    
    # Emotion to color mapping (BGR format for ASS)
    EMOTION_COLORS = {
        'happy': '&H00FFFF00',     # Yellow
        'sad': '&H00FF0000',       # Blue
        'angry': '&H000000FF',     # Red
        'fear': '&H00FF00FF',      # Magenta
        'disgust': '&H0000FF00',   # Green
        'surprise': '&H00FFAA00',  # Orange
        'neutral': '&H00FFFFFF'    # White
    }
    
    # Speaker style colors
    SPEAKER_COLORS = {
        'speaker_1': '&H00FFFFFF',  # White
        'speaker_2': '&H00FFFF00',  # Yellow
        'speaker_3': '&H0000FFFF',  # Cyan
        'speaker_4': '&H00FF00FF',  # Magenta
        'speaker_5': '&H0000FF00',  # Green
    }
    
    def __init__(self):
        """Initialize ASS exporter."""
        self.styles = {}
        self.events = []
        
    def export_to_ass(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        video_width: int = 1920,
        video_height: int = 1080,
        font_name: str = "Arial",
        font_size: int = 48,
        use_emotions: bool = True,
        use_speaker_colors: bool = True
    ) -> Path:
        """Export results to ASS subtitle file.
        
        Args:
            results: Analysis results with timeline
            output_path: Output file path
            video_width: Video width for positioning
            video_height: Video height for positioning
            font_name: Font to use
            font_size: Base font size
            use_emotions: Apply emotion-based styling
            use_speaker_colors: Apply speaker-based colors
            
        Returns:
            Path to saved ASS file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Exporting to ASS format: {output_path}")
            
            # Generate ASS content
            ass_content = self._generate_ass(
                results,
                video_width=video_width,
                video_height=video_height,
                font_name=font_name,
                font_size=font_size,
                use_emotions=use_emotions,
                use_speaker_colors=use_speaker_colors
            )
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write(ass_content)
            
            logger.info(f"ASS subtitle exported successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export ASS: {e}")
            raise
    
    def _generate_ass(
        self,
        results: Dict[str, Any],
        video_width: int,
        video_height: int,
        font_name: str,
        font_size: int,
        use_emotions: bool,
        use_speaker_colors: bool
    ) -> str:
        """Generate ASS subtitle content.
        
        Args:
            results: Analysis results
            video_width: Video width
            video_height: Video height
            font_name: Font name
            font_size: Font size
            use_emotions: Use emotion styling
            use_speaker_colors: Use speaker colors
            
        Returns:
            ASS file content as string
        """
        lines = []
        
        # Script Info section
        lines.extend([
            '[Script Info]',
            'Title: Audio Analysis Subtitles',
            'ScriptType: v4.00+',
            f'PlayResX: {video_width}',
            f'PlayResY: {video_height}',
            'YCbCr Matrix: TV.601',
            'WrapStyle: 0',
            ''
        ])
        
        # Styles section
        lines.append('[V4+ Styles]')
        lines.append('Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding')
        
        # Create styles for each speaker and emotion
        styles = self._create_styles(
            results,
            font_name,
            font_size,
            use_emotions,
            use_speaker_colors
        )
        
        for style in styles:
            lines.append(style)
        
        lines.append('')
        
        # Events section
        lines.append('[Events]')
        lines.append('Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text')
        
        # Create dialogue events
        events = self._create_events(
            results,
            use_emotions,
            use_speaker_colors
        )
        
        for event in events:
            lines.append(event)
        
        return '\n'.join(lines)
    
    def _create_styles(
        self,
        results: Dict[str, Any],
        font_name: str,
        font_size: int,
        use_emotions: bool,
        use_speaker_colors: bool
    ) -> List[str]:
        """Create ASS styles.
        
        Args:
            results: Analysis results
            font_name: Font name
            font_size: Font size
            use_emotions: Use emotion styling
            use_speaker_colors: Use speaker colors
            
        Returns:
            List of style definition strings
        """
        styles = []
        
        # Default style
        default_style = f'Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1'
        styles.append(default_style)
        
        # Get unique speakers
        speakers = set()
        if 'speakers' in results:
            speakers.update(results['speakers'].keys())
        
        # Create style for each speaker
        for i, speaker in enumerate(speakers):
            color = self.SPEAKER_COLORS.get(speaker, '&H00FFFFFF')
            if not use_speaker_colors:
                color = '&H00FFFFFF'
            
            style = f'Style: {speaker},{font_name},{font_size},{color},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1'
            styles.append(style)
        
        # Create styles for emotions if needed
        if use_emotions:
            for emotion, color in self.EMOTION_COLORS.items():
                # Slightly larger font for strong emotions
                emotion_size = font_size
                if emotion in ['angry', 'surprise']:
                    emotion_size = int(font_size * 1.1)
                
                style = f'Style: {emotion},{font_name},{emotion_size},{color},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1'
                styles.append(style)
        
        return styles
    
    def _create_events(
        self,
        results: Dict[str, Any],
        use_emotions: bool,
        use_speaker_colors: bool
    ) -> List[str]:
        """Create ASS dialogue events.
        
        Args:
            results: Analysis results
            use_emotions: Use emotion styling
            use_speaker_colors: Use speaker colors
            
        Returns:
            List of dialogue event strings
        """
        events = []
        
        # Get timeline
        timeline = results.get('timeline', [])
        
        # Process timeline entries
        for entry in timeline:
            # Skip non-text entries
            if 'text' not in entry and entry.get('type') != 'transcription':
                continue
            
            start_time = self._format_time(entry.get('start', 0))
            end_time = self._format_time(entry.get('end', 0))
            
            # Get text
            text = entry.get('text', '')
            if not text:
                # Generate placeholder text if no transcription
                speaker = entry.get('speaker', 'speaker_1')
                emotion = entry.get('emotion', 'neutral')
                text = f"[{speaker}: {emotion}]"
            
            # Clean text
            text = self._clean_text(text)
            
            # Determine style
            style = 'Default'
            if use_emotions and 'emotion' in entry:
                style = entry['emotion']
            elif use_speaker_colors and 'speaker' in entry:
                style = entry['speaker']
            
            # Get speaker name
            speaker_name = entry.get('speaker', '')
            
            # Create dialogue event
            event = f'Dialogue: 0,{start_time},{end_time},{style},{speaker_name},0,0,0,,{text}'
            events.append(event)
        
        # If no text entries, create from speaker segments
        if not events and 'speakers' in results:
            for speaker_id, speaker_data in results['speakers'].items():
                for segment in speaker_data.get('segments', []):
                    start_time = self._format_time(segment.get('start', 0))
                    end_time = self._format_time(segment.get('end', 0))
                    
                    # Generate placeholder
                    text = f"[{speaker_id} 발화]"
                    style = speaker_id if use_speaker_colors else 'Default'
                    
                    event = f'Dialogue: 0,{start_time},{end_time},{style},{speaker_id},0,0,0,,{text}'
                    events.append(event)
        
        return events
    
    def _format_time(self, seconds: float) -> str:
        """Format time in ASS format (h:mm:ss.cc).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f'{hours}:{minutes:02d}:{secs:05.2f}'
    
    def _clean_text(self, text: str) -> str:
        """Clean text for ASS format.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        """
        # Remove or escape special characters
        text = text.replace('\\', '\\\\')
        text = text.replace('{', '\\{')
        text = text.replace('}', '\\}')
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit line length (optional - adds line breaks)
        max_length = 80
        if len(text) > max_length:
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_length and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            text = '\\N'.join(lines)  # ASS line break
        
        return text
    
    def create_karaoke_effect(
        self,
        text: str,
        duration: float,
        style: str = "Default"
    ) -> str:
        """Create karaoke effect for text.
        
        Args:
            text: Text to apply effect to
            duration: Duration of the text
            style: Style name
            
        Returns:
            Text with karaoke effect tags
        """
        words = text.split()
        if not words:
            return text
        
        word_duration = duration / len(words) * 100  # Convert to centiseconds
        
        karaoke_text = ""
        for word in words:
            karaoke_text += f"{{\\k{int(word_duration)}}}{word} "
        
        return karaoke_text.strip()
    
    def add_positioning(
        self,
        text: str,
        x: int,
        y: int,
        alignment: int = 2
    ) -> str:
        """Add positioning to text.
        
        Args:
            text: Text to position
            x: X coordinate
            y: Y coordinate
            alignment: Alignment (1-9 numpad style)
            
        Returns:
            Text with position tags
        """
        return f"{{\\pos({x},{y})\\an{alignment}}}{text}"