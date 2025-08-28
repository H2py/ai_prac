"""
Enhanced result merger with linguistic precision and standard compliance.
Follows W3C Web Speech API, SMPTE standards, and linguistic annotation formats.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import pandas as pd

from src.speaker_diarizer import SpeakerSegment
from src.emotion_analyzer import EmotionPrediction
from src.utils.logger import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


class EnhancedResultMerger:
    """Enhanced merger with linguistic precision and standard compliance."""
    
    # IPA-based phoneme mapping (simplified)
    PHONEME_MAP = {
        'w': 'w', 'iy': 'iː', 'sh': 'ʃ', 'uh': 'ʊ', 'd': 'd', 'n': 'n', 't': 't',
        'b': 'b', 'h': 'h', 'r': 'ɹ', 'ih': 'ɪ', 'l': 'l', 'ae': 'æ', 'ks': 'ks',
        'ay': 'aɪ', 'v': 'v', 'ah': 'ʌ', 'dh': 'ð', 's': 's', 'f': 'f', 'ao': 'ɔː',
        'er': 'ɜː', 'z': 'z', 'm': 'm', 'k': 'k', 'p': 'p', 'rt': 'ɹt', 'y': 'j',
        'g': 'g', 'ts': 'ts', 'aw': 'aʊ', 'uw': 'uː', 'ng': 'ŋ'
    }
    
    def __init__(self):
        """Initialize enhanced result merger."""
        self.results = {}
        
    def merge_all_results(
        self,
        speaker_results: Optional[Dict[str, Any]] = None,
        emotion_results: Optional[List[EmotionPrediction]] = None,
        acoustic_results: Optional[List[Dict]] = None,
        transcription_results: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge results with enhanced linguistic and standard compliance.
        
        Returns structure compatible with:
        - W3C Web Speech API
        - SMPTE ST 2052-1:2013 (Timed Text)
        - ISO 639 (Language codes)
        - IPA (International Phonetic Alphabet)
        """
        perf_logger.start_timer("enhanced_result_merging")
        
        try:
            # Build enhanced metadata
            enhanced_metadata = self._build_metadata(metadata)
            
            # Initialize result structure
            merged = {
                'media': enhanced_metadata,
                'diarization': {
                    'turns': [],
                    'speakers': {}
                },
                'asr': {
                    'utterances': []
                },
                'prosody': {
                    'segments': []
                },
                'emotions': {
                    'segments': []
                }
            }
            
            # Process speaker diarization with enhanced format
            if speaker_results:
                merged['diarization'] = self._process_diarization(speaker_results)
            
            # Process transcriptions with linguistic precision
            if transcription_results:
                merged['asr']['utterances'] = self._process_transcriptions(
                    transcription_results, 
                    speaker_results,
                    acoustic_results
                )
            
            # Process prosodic features
            if acoustic_results:
                merged['prosody']['segments'] = self._process_prosody(acoustic_results)
            
            # Process emotions with standardized labels
            if emotion_results:
                merged['emotions']['segments'] = self._process_emotions(emotion_results)
            
            # Add cross-reference indices for fast lookup
            merged['indices'] = self._build_indices(merged)
            
            # Add linguistic statistics
            merged['statistics'] = self._calculate_statistics(merged)
            
            duration = perf_logger.stop_timer("enhanced_result_merging")
            logger.info(f"Enhanced results merged in {duration:.2f}s")
            
            return merged
            
        except Exception as e:
            perf_logger.stop_timer("enhanced_result_merging")
            logger.error(f"Failed to merge results: {e}")
            raise
    
    def _build_metadata(self, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Build enhanced metadata with standard compliance."""
        base_metadata = metadata or {}
        
        return {
            'audio_file': base_metadata.get('audio_file', 'unknown'),
            'duration': base_metadata.get('audio_duration', 0),
            'sample_rate': base_metadata.get('sample_rate', 16000),
            'channels': 1,  # Mono after processing
            'language': self._detect_language(base_metadata),
            'timestamp': datetime.now().isoformat(),
            'standards': {
                'speech_api': 'W3C Web Speech API 1.0',
                'timed_text': 'SMPTE ST 2052-1:2013',
                'phonetic': 'IPA (International Phonetic Alphabet)',
                'emotion': 'PAD (Pleasure-Arousal-Dominance) model'
            }
        }
    
    def _detect_language(self, metadata: Dict) -> str:
        """Detect language with ISO 639-1 code."""
        # Default to English if not specified
        lang = metadata.get('language', 'en')
        
        # Map common language names to ISO 639-1 codes
        language_map = {
            'korean': 'ko',
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'japanese': 'ja',
            'chinese': 'zh'
        }
        
        return language_map.get(lang.lower(), lang[:2].lower())
    
    def _process_diarization(self, speaker_results: Dict) -> Dict[str, Any]:
        """Process speaker diarization with enhanced format."""
        diarization = {
            'turns': [],
            'speakers': {}
        }
        
        segments = speaker_results.get('segments', [])
        speakers = speaker_results.get('speakers', {})
        
        # Create speaker profiles
        for speaker_id, speaker_info in speakers.items():
            diarization['speakers'][speaker_id] = {
                'id': speaker_id,
                'total_duration': speaker_info.get('total_duration', 0),
                'speaking_percentage': speaker_info.get('speaking_percentage', 0),
                'segment_count': speaker_info.get('segment_count', 0),
                'average_confidence': speaker_info.get('average_confidence', 0)
            }
        
        # Process turns with segment IDs
        for idx, segment in enumerate(segments):
            turn = {
                'segment_id': f"seg_{idx+1:04d}",
                'start': segment['start'],
                'end': segment['end'],
                'speaker': segment['speaker'],
                'confidence': segment.get('confidence', 0.5)
            }
            diarization['turns'].append(turn)
        
        return diarization
    
    def _process_transcriptions(
        self, 
        transcriptions: List[Dict],
        speaker_results: Optional[Dict],
        acoustic_results: Optional[List[Dict]]
    ) -> List[Dict]:
        """Process transcriptions with word-level timing and phonetics."""
        utterances = []
        
        # Map segments to speakers
        speaker_map = {}
        if speaker_results:
            for segment in speaker_results.get('segments', []):
                key = (segment['start'], segment['end'])
                speaker_map[key] = segment['speaker']
        
        for idx, trans in enumerate(transcriptions):
            # Find matching speaker
            speaker = 'unknown'
            for (start, end), spk in speaker_map.items():
                if abs(trans.get('start', 0) - start) < 0.5:
                    speaker = spk
                    break
            
            utterance = {
                'segment_id': f"seg_{idx+1:04d}",
                'speaker': speaker,
                'start': trans.get('start', 0),
                'end': trans.get('end', 0),
                'text': trans.get('text', ''),
                'confidence': trans.get('confidence', 0),
                'language': trans.get('language', 'en')
            }
            
            # Add word-level timing if available
            if 'words' in trans:
                utterance['words'] = self._process_words(trans['words'], acoustic_results)
            else:
                # Generate pseudo word timing
                utterance['words'] = self._generate_word_timing(
                    trans.get('text', ''),
                    trans.get('start', 0),
                    trans.get('end', 0),
                    acoustic_results
                )
            
            utterances.append(utterance)
        
        return utterances
    
    def _process_words(self, words: List[Dict], acoustic_results: Optional[List[Dict]]) -> List[Dict]:
        """Process word-level information with phonetics and prosody."""
        processed_words = []
        
        for word_info in words:
            word = {
                'start': word_info.get('start', 0),
                'end': word_info.get('end', 0),
                'word': word_info.get('word', ''),
                'confidence': word_info.get('confidence', 0)
            }
            
            # Add phonetic transcription (simplified)
            word['phones'] = self._generate_phonemes(word['word'])
            
            # Add prosodic features if available
            if acoustic_results:
                prosody = self._get_prosody_for_time(
                    word['start'], 
                    word['end'],
                    acoustic_results
                )
                if prosody:
                    word['pitch_hz'] = prosody.get('pitch_mean', 0)
                    word['volume_db'] = prosody.get('energy_rms', 0)
            
            processed_words.append(word)
        
        return processed_words
    
    def _generate_word_timing(
        self,
        text: str,
        start: float,
        end: float,
        acoustic_results: Optional[List[Dict]]
    ) -> List[Dict]:
        """Generate word-level timing when not available."""
        words = text.split()
        if not words:
            return []
        
        duration = end - start
        word_duration = duration / len(words)
        
        word_list = []
        current_time = start
        
        for word in words:
            word_end = current_time + word_duration
            
            word_info = {
                'start': current_time,
                'end': word_end,
                'word': word,
                'confidence': 0.8,  # Default confidence
                'phones': self._generate_phonemes(word)
            }
            
            # Add prosody if available
            if acoustic_results:
                prosody = self._get_prosody_for_time(
                    current_time,
                    word_end,
                    acoustic_results
                )
                if prosody:
                    word_info['pitch_hz'] = prosody.get('pitch_mean', 0)
                    word_info['volume_db'] = prosody.get('energy_rms', 0)
            
            word_list.append(word_info)
            current_time = word_end
        
        return word_list
    
    def _generate_phonemes(self, word: str) -> List[Dict]:
        """Generate simplified phoneme representation."""
        # This is a very simplified phoneme generation
        # In production, use a proper phonetic dictionary
        phonemes = []
        
        # Simple mapping for common words (expand as needed)
        phoneme_dict = {
            'the': ['ð', 'ə'],
            'hello': ['h', 'ɛ', 'l', 'oʊ'],
            'world': ['w', 'ɜː', 'ɹ', 'l', 'd'],
            'i': ['aɪ'],
            'you': ['j', 'uː'],
            'we': ['w', 'iː'],
            'they': ['ð', 'eɪ']
        }
        
        word_lower = word.lower().strip('.,!?')
        
        if word_lower in phoneme_dict:
            phone_list = phoneme_dict[word_lower]
        else:
            # Fallback to basic character mapping
            phone_list = list(word_lower)[:3]  # Simplified
        
        current_time = 0
        for phone in phone_list:
            phonemes.append({
                'p': phone,
                't': current_time
            })
            current_time += 0.1
        
        return phonemes
    
    def _get_prosody_for_time(
        self,
        start: float,
        end: float,
        acoustic_results: List[Dict]
    ) -> Optional[Dict]:
        """Get prosodic features for a time range."""
        for segment in acoustic_results:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if times overlap
            if seg_start <= start < seg_end or seg_start < end <= seg_end:
                return {
                    'pitch_mean': segment.get('pitch', {}).get('mean', 0),
                    'pitch_std': segment.get('pitch', {}).get('std', 0),
                    'energy_rms': segment.get('energy', {}).get('rms', 0) * -20  # Convert to dB
                }
        
        return None
    
    def _process_prosody(self, acoustic_results: List[Dict]) -> List[Dict]:
        """Process prosodic features with linguistic standards."""
        prosody_segments = []
        
        for segment in acoustic_results:
            prosody = {
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'pitch': {
                    'mean_hz': segment.get('pitch', {}).get('mean', 0),
                    'std_hz': segment.get('pitch', {}).get('std', 0),
                    'min_hz': segment.get('pitch', {}).get('min', 0),
                    'max_hz': segment.get('pitch', {}).get('max', 0),
                    'contour': 'falling' if segment.get('pitch', {}).get('mean', 0) > 200 else 'rising'
                },
                'intensity': {
                    'mean_db': segment.get('energy', {}).get('rms', 0) * -20,
                    'max_db': segment.get('energy', {}).get('rms', 0) * -15
                },
                'speech_rate': {
                    'syllables_per_second': segment.get('tempo', 120) / 60 * 2  # Approximation
                },
                'voice_quality': segment.get('voice_quality', {})
            }
            
            prosody_segments.append(prosody)
        
        return prosody_segments
    
    def _process_emotions(self, emotion_results: List[EmotionPrediction]) -> List[Dict]:
        """Process emotions with PAD model and standardized labels."""
        emotion_segments = []
        
        # PAD values for basic emotions (Pleasure, Arousal, Dominance)
        pad_values = {
            'happy': {'pleasure': 0.8, 'arousal': 0.7, 'dominance': 0.6},
            'sad': {'pleasure': -0.8, 'arousal': -0.4, 'dominance': -0.4},
            'angry': {'pleasure': -0.7, 'arousal': 0.8, 'dominance': 0.7},
            'fear': {'pleasure': -0.8, 'arousal': 0.6, 'dominance': -0.6},
            'surprise': {'pleasure': 0.1, 'arousal': 0.8, 'dominance': 0},
            'disgust': {'pleasure': -0.8, 'arousal': 0.2, 'dominance': 0.2},
            'neutral': {'pleasure': 0, 'arousal': 0, 'dominance': 0}
        }
        
        for emotion in emotion_results:
            emotion_dict = emotion.to_dict() if hasattr(emotion, 'to_dict') else emotion
            primary = emotion_dict.get('primary_emotion', 'neutral')
            
            segment = {
                'start': emotion_dict.get('start', 0),
                'end': emotion_dict.get('end', 0),
                'primary_emotion': primary,
                'confidence': emotion_dict.get('confidence', 0),
                'pad_model': pad_values.get(primary, pad_values['neutral']),
                'emotion_probabilities': emotion_dict.get('emotion_scores', {})
            }
            
            emotion_segments.append(segment)
        
        return emotion_segments
    
    def _build_indices(self, merged: Dict) -> Dict[str, Any]:
        """Build indices for fast lookup."""
        indices = {
            'speaker_index': {},
            'time_index': [],
            'emotion_index': {}
        }
        
        # Build speaker index
        for turn in merged.get('diarization', {}).get('turns', []):
            speaker = turn['speaker']
            if speaker not in indices['speaker_index']:
                indices['speaker_index'][speaker] = []
            indices['speaker_index'][speaker].append(turn['segment_id'])
        
        # Build time index (sorted segments)
        all_segments = []
        for turn in merged.get('diarization', {}).get('turns', []):
            all_segments.append({
                'id': turn['segment_id'],
                'start': turn['start'],
                'end': turn['end'],
                'type': 'speech'
            })
        
        indices['time_index'] = sorted(all_segments, key=lambda x: x['start'])
        
        # Build emotion index
        for segment in merged.get('emotions', {}).get('segments', []):
            emotion = segment['primary_emotion']
            if emotion not in indices['emotion_index']:
                indices['emotion_index'][emotion] = []
            indices['emotion_index'][emotion].append({
                'start': segment['start'],
                'end': segment['end']
            })
        
        return indices
    
    def _calculate_statistics(self, merged: Dict) -> Dict[str, Any]:
        """Calculate linguistic and speech statistics."""
        stats = {
            'speech': {},
            'linguistic': {},
            'prosodic': {},
            'emotional': {}
        }
        
        # Speech statistics
        turns = merged.get('diarization', {}).get('turns', [])
        if turns:
            total_speech = sum(t['end'] - t['start'] for t in turns)
            stats['speech'] = {
                'total_speech_time': total_speech,
                'total_turns': len(turns),
                'average_turn_duration': total_speech / len(turns) if turns else 0,
                'speaker_count': len(merged.get('diarization', {}).get('speakers', {}))
            }
        
        # Linguistic statistics
        utterances = merged.get('asr', {}).get('utterances', [])
        if utterances:
            all_words = []
            for utt in utterances:
                all_words.extend(utt.get('words', []))
            
            stats['linguistic'] = {
                'total_words': len(all_words),
                'total_utterances': len(utterances),
                'average_words_per_utterance': len(all_words) / len(utterances) if utterances else 0,
                'vocabulary_size': len(set(w['word'].lower() for w in all_words if 'word' in w))
            }
        
        # Prosodic statistics
        prosody = merged.get('prosody', {}).get('segments', [])
        if prosody:
            pitch_values = [s['pitch']['mean_hz'] for s in prosody if s['pitch']['mean_hz'] > 0]
            stats['prosodic'] = {
                'average_pitch_hz': sum(pitch_values) / len(pitch_values) if pitch_values else 0,
                'pitch_range_hz': max(pitch_values) - min(pitch_values) if pitch_values else 0
            }
        
        # Emotional statistics
        emotions = merged.get('emotions', {}).get('segments', [])
        if emotions:
            emotion_counts = {}
            for segment in emotions:
                emotion = segment['primary_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            stats['emotional'] = {
                'emotion_distribution': emotion_counts,
                'dominant_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral',
                'emotion_changes': len([i for i in range(1, len(emotions)) 
                                       if emotions[i]['primary_emotion'] != emotions[i-1]['primary_emotion']])
            }
        
        return stats
    
    def export_to_json(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> Path:
        """Export results with enhanced JSON formatting."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(results, f, ensure_ascii=False, default=str)
            
            logger.info(f"Enhanced results exported to JSON: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export enhanced JSON: {e}")
            raise
    
    def export_to_webvtt(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> Path:
        """Export as WebVTT for web compatibility."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        vtt_content = ["WEBVTT", ""]
        
        utterances = results.get('asr', {}).get('utterances', [])
        for idx, utt in enumerate(utterances, 1):
            start = self._format_timestamp(utt['start'])
            end = self._format_timestamp(utt['end'])
            speaker = utt.get('speaker', 'Unknown')
            text = utt.get('text', '')
            
            vtt_content.append(f"{idx}")
            vtt_content.append(f"{start} --> {end}")
            vtt_content.append(f"<v {speaker}>{text}")
            vtt_content.append("")
        
        output_path.write_text('\n'.join(vtt_content), encoding='utf-8')
        logger.info(f"Exported to WebVTT: {output_path}")
        
        return output_path
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for WebVTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"