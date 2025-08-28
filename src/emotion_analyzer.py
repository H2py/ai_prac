"""
Emotion analysis module for detecting emotions in speech segments.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict

try:
    from transformers import (
        AutoModelForAudioClassification,
        AutoFeatureExtractor,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForAudioClassification = None
    AutoFeatureExtractor = None
    pipeline = None

from src.utils.audio_utils import load_audio, split_audio_chunks
from src.utils.logger import PerformanceLogger, ProgressLogger, log_exception
from config.settings import ModelConfig


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger(logger)


# Emotion labels mapping
EMOTION_LABELS = {
    'neutral': 'neutral',
    'calm': 'neutral',
    'happy': 'happy',
    'happiness': 'happy',
    'joy': 'happy',
    'sad': 'sad',
    'sadness': 'sad',
    'angry': 'angry',
    'anger': 'angry',
    'fear': 'fear',
    'fearful': 'fear',
    'disgust': 'disgust',
    'surprised': 'surprise',
    'surprise': 'surprise',
    'excitement': 'happy',
    'frustration': 'angry',
    'boredom': 'neutral'
}

# Emotion colors for visualization
EMOTION_COLORS = {
    'neutral': '#808080',
    'happy': '#FFD700',
    'sad': '#4169E1',
    'angry': '#DC143C',
    'fear': '#8B008B',
    'disgust': '#228B22',
    'surprise': '#FF69B4'
}


@dataclass
class EmotionPrediction:
    """Represents an emotion prediction for a segment."""
    
    start: float
    end: float
    primary_emotion: str
    confidence: float
    all_scores: Dict[str, float]
    speaker: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'primary_emotion': self.primary_emotion,
            'confidence': self.confidence,
            'all_scores': self.all_scores,
            'speaker': self.speaker
        }


class EmotionAnalyzer:
    """Emotion recognition using transformer models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize emotion analyzer.
        
        Args:
            config: Model configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed. "
                "Please install with: pip install transformers"
            )
        
        self.config = config or ModelConfig()
        self.device = self._setup_device()
        self.model = None
        self.feature_extractor = None
        self.emotion_pipeline = None
        self._initialized = False
        
        logger.info(f"EmotionAnalyzer initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device.
        
        Returns:
            torch.device instance
        """
        if self.config.device:
            return torch.device(self.config.device)
        
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Set memory fraction
            if self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.gpu_memory_fraction
                )
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for processing")
        
        return device
    
    def initialize(self) -> None:
        """Initialize the emotion recognition model."""
        if self._initialized:
            return
        
        perf_logger.start_timer("model_initialization")
        
        try:
            model_name = self.config.emotion_model
            logger.info(f"Loading emotion recognition model: {model_name}")
            
            # Try to load the model
            try:
                # Load model and feature extractor
                self.model = AutoModelForAudioClassification.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir
                )
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=self.config.cache_dir
                )
                
                # Move model to device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully: {model_name}")
                
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
                logger.info("Trying alternative model...")
                
                # Try alternative model
                alternative_model = "superb/hubert-base-superb-er"
                try:
                    self.emotion_pipeline = pipeline(
                        "audio-classification",
                        model=alternative_model,
                        device=0 if self.device.type == "cuda" else -1
                    )
                    logger.info(f"Loaded alternative model: {alternative_model}")
                except Exception:
                    logger.warning("Could not load emotion model, using mock predictions")
                    self._setup_mock_model()
            
            self._initialized = True
            duration = perf_logger.stop_timer("model_initialization")
            logger.info(f"Emotion model initialized in {duration:.2f}s")
            
        except Exception as e:
            perf_logger.stop_timer("model_initialization")
            log_exception(logger, e, "Failed to initialize emotion model")
            raise
    
    def _setup_mock_model(self) -> None:
        """Setup mock model for testing when real model unavailable."""
        logger.warning("Using mock emotion model for demonstration")
        self.model = None
        self.feature_extractor = None
        self.emotion_pipeline = None
    
    def predict_emotion(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000
    ) -> EmotionPrediction:
        """Predict emotion for a single audio chunk.
        
        Args:
            audio_chunk: Audio data array
            sample_rate: Sample rate of audio
            
        Returns:
            Emotion prediction
        """
        if not self._initialized:
            self.initialize()
        
        try:
            if self.emotion_pipeline:
                # Use pipeline
                result = self.emotion_pipeline(audio_chunk)
                scores = self._process_pipeline_output(result)
            elif self.model and self.feature_extractor:
                # Use model directly
                scores = self._predict_with_model(audio_chunk, sample_rate)
            else:
                # Use mock prediction
                scores = self._mock_prediction(audio_chunk)
            
            # Get primary emotion
            primary_emotion = max(scores, key=scores.get)
            confidence = scores[primary_emotion]
            
            return EmotionPrediction(
                start=0.0,
                end=len(audio_chunk) / sample_rate,
                primary_emotion=primary_emotion,
                confidence=confidence,
                all_scores=scores
            )
            
        except Exception as e:
            logger.warning(f"Emotion prediction failed: {e}")
            # Return neutral emotion on failure
            return EmotionPrediction(
                start=0.0,
                end=len(audio_chunk) / sample_rate,
                primary_emotion='neutral',
                confidence=0.5,
                all_scores={'neutral': 0.5}
            )
    
    def _predict_with_model(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Predict emotion using loaded model.
        
        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            
        Returns:
            Emotion scores
        """
        # Process audio
        inputs = self.feature_extractor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        # Convert to scores
        probs = probs.squeeze().cpu().numpy()
        labels = self.model.config.id2label
        
        scores = {}
        for idx, prob in enumerate(probs):
            if idx in labels:
                label = labels[idx].lower()
                # Map to standard emotion
                emotion = EMOTION_LABELS.get(label, label)
                if emotion not in scores:
                    scores[emotion] = 0.0
                scores[emotion] = max(scores[emotion], float(prob))
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _process_pipeline_output(self, result: List[Dict]) -> Dict[str, float]:
        """Process output from transformers pipeline.
        
        Args:
            result: Pipeline output
            
        Returns:
            Emotion scores
        """
        scores = defaultdict(float)
        
        for item in result:
            label = item['label'].lower()
            score = item['score']
            
            # Map to standard emotion
            emotion = EMOTION_LABELS.get(label, label)
            scores[emotion] = max(scores[emotion], score)
        
        # Ensure all basic emotions are present
        basic_emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        for emotion in basic_emotions:
            if emotion not in scores:
                scores[emotion] = 0.0
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return dict(scores)
    
    def _mock_prediction(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """Generate mock emotion prediction for testing.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            Mock emotion scores
        """
        # Simple energy-based mock prediction
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Generate pseudo-random but deterministic scores
        np.random.seed(int(energy * 10000) % 100)
        
        emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        raw_scores = np.random.dirichlet(np.ones(len(emotions)))
        
        # Bias towards neutral for low energy
        if energy < 0.01:
            raw_scores[0] += 0.5
        
        # Normalize
        scores = dict(zip(emotions, raw_scores / raw_scores.sum()))
        
        return scores
    
    def analyze_segments(
        self,
        audio_path: Union[str, Path],
        segments: Optional[List[Dict]] = None,
        chunk_duration: float = 3.0,
        **kwargs
    ) -> List[EmotionPrediction]:
        """Analyze emotions for audio segments.
        
        Args:
            audio_path: Path to audio file
            segments: Optional speaker segments to analyze
            chunk_duration: Duration for emotion analysis chunks
            **kwargs: Additional parameters
            
        Returns:
            List of emotion predictions
        """
        if not self._initialized:
            self.initialize()
        
        perf_logger.start_timer("emotion_analysis")
        audio_path = Path(audio_path)
        
        try:
            logger.info(f"Analyzing emotions in: {audio_path}")
            
            # Load audio
            audio_data, sample_rate = load_audio(audio_path, sample_rate=16000)
            
            predictions = []
            
            if segments:
                # Analyze provided segments
                progress = ProgressLogger(logger, total=len(segments))
                
                for segment in segments:
                    # Extract segment audio
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        # Predict emotion
                        pred = self.predict_emotion(segment_audio, sample_rate)
                        
                        # Update with segment info
                        pred.start = segment['start']
                        pred.end = segment['end']
                        pred.speaker = segment.get('speaker')
                        
                        predictions.append(pred)
                    
                    progress.update(message=f"Processed segment {len(predictions)}")
                
                progress.complete()
                
            else:
                # Analyze entire audio in chunks
                chunks = split_audio_chunks(
                    audio_data,
                    sample_rate,
                    chunk_duration=chunk_duration,
                    overlap=0.5
                )
                
                progress = ProgressLogger(logger, total=len(chunks))
                
                for chunk, (start_time, end_time) in chunks:
                    if len(chunk) > 0:
                        # Predict emotion
                        pred = self.predict_emotion(chunk, sample_rate)
                        
                        # Update times
                        pred.start = start_time
                        pred.end = end_time
                        
                        predictions.append(pred)
                    
                    progress.update(message=f"Processed chunk {len(predictions)}")
                
                progress.complete()
            
            # Merge similar adjacent predictions
            predictions = self._merge_predictions(predictions)
            
            duration = perf_logger.stop_timer("emotion_analysis")
            logger.info(
                f"Emotion analysis completed: {len(predictions)} predictions "
                f"in {duration:.2f}s"
            )
            
            return predictions
            
        except Exception as e:
            perf_logger.stop_timer("emotion_analysis")
            log_exception(logger, e, "Emotion analysis failed")
            return []
    
    def _merge_predictions(
        self,
        predictions: List[EmotionPrediction],
        threshold: float = 0.5
    ) -> List[EmotionPrediction]:
        """Merge adjacent predictions with same emotion.
        
        Args:
            predictions: List of predictions
            threshold: Time threshold for merging
            
        Returns:
            Merged predictions
        """
        if not predictions:
            return predictions
        
        merged = []
        current = predictions[0]
        
        for pred in predictions[1:]:
            # Check if same emotion and speaker, and close in time
            same_emotion = pred.primary_emotion == current.primary_emotion
            same_speaker = pred.speaker == current.speaker
            close_time = pred.start - current.end < threshold
            
            if same_emotion and same_speaker and close_time:
                # Merge predictions
                all_scores = {}
                for emotion in set(current.all_scores.keys()) | set(pred.all_scores.keys()):
                    score1 = current.all_scores.get(emotion, 0) * current.duration
                    score2 = pred.all_scores.get(emotion, 0) * pred.duration
                    total_duration = current.duration + pred.duration
                    all_scores[emotion] = (score1 + score2) / total_duration
                
                current = EmotionPrediction(
                    start=current.start,
                    end=pred.end,
                    primary_emotion=current.primary_emotion,
                    confidence=(current.confidence * current.duration + 
                               pred.confidence * pred.duration) / 
                              (current.duration + pred.duration),
                    all_scores=all_scores,
                    speaker=current.speaker
                )
            else:
                merged.append(current)
                current = pred
        
        merged.append(current)
        
        return merged
    
    def batch_process(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000
    ) -> List[EmotionPrediction]:
        """Process multiple audio chunks in batch.
        
        Args:
            audio_chunks: List of audio arrays
            sample_rate: Sample rate
            
        Returns:
            List of emotion predictions
        """
        if not self._initialized:
            self.initialize()
        
        predictions = []
        
        # Process in batches
        batch_size = self.config.emotion_batch_size
        
        for i in range(0, len(audio_chunks), batch_size):
            batch = audio_chunks[i:i + batch_size]
            
            for chunk in batch:
                pred = self.predict_emotion(chunk, sample_rate)
                predictions.append(pred)
        
        return predictions
    
    def get_emotion_statistics(
        self,
        predictions: List[EmotionPrediction]
    ) -> Dict[str, Any]:
        """Calculate emotion statistics from predictions.
        
        Args:
            predictions: List of emotion predictions
            
        Returns:
            Dictionary with emotion statistics
        """
        if not predictions:
            return {
                'dominant_emotion': 'neutral',
                'emotion_distribution': {},
                'average_confidence': 0.0,
                'emotion_changes': 0,
                'total_duration': 0.0
            }
        
        # Calculate total duration
        total_duration = sum(pred.duration for pred in predictions)
        
        # Calculate emotion distribution
        emotion_durations = defaultdict(float)
        confidence_sum = 0.0
        
        for pred in predictions:
            emotion_durations[pred.primary_emotion] += pred.duration
            confidence_sum += pred.confidence * pred.duration
        
        # Normalize to percentages
        emotion_distribution = {
            emotion: (duration / total_duration * 100) if total_duration > 0 else 0
            for emotion, duration in emotion_durations.items()
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotion_durations, key=emotion_durations.get)
        
        # Count emotion changes
        emotion_changes = sum(
            1 for i in range(1, len(predictions))
            if predictions[i].primary_emotion != predictions[i-1].primary_emotion
        )
        
        # Calculate average confidence
        avg_confidence = confidence_sum / total_duration if total_duration > 0 else 0
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_distribution,
            'average_confidence': avg_confidence,
            'emotion_changes': emotion_changes,
            'total_duration': total_duration,
            'emotions_per_minute': (emotion_changes / total_duration * 60) if total_duration > 0 else 0
        }
    
    def visualize_emotions(
        self,
        predictions: List[EmotionPrediction],
        audio_duration: float,
        output_path: Optional[Path] = None
    ) -> None:
        """Create visualization of emotion timeline.
        
        Args:
            predictions: Emotion predictions
            audio_duration: Total audio duration
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            
            # Timeline view
            emotions = list(set(pred.primary_emotion for pred in predictions))
            emotion_to_y = {emotion: i for i, emotion in enumerate(emotions)}
            
            for pred in predictions:
                color = EMOTION_COLORS.get(pred.primary_emotion, '#808080')
                rect = patches.Rectangle(
                    (pred.start, emotion_to_y[pred.primary_emotion]),
                    pred.duration,
                    0.8,
                    linewidth=1,
                    edgecolor='black',
                    facecolor=color,
                    alpha=pred.confidence
                )
                ax1.add_patch(rect)
            
            ax1.set_xlim(0, audio_duration)
            ax1.set_ylim(-0.5, len(emotions) - 0.5)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Emotion', fontsize=12)
            ax1.set_yticks(range(len(emotions)))
            ax1.set_yticklabels(emotions)
            ax1.set_title('Emotion Timeline', fontsize=14, fontweight='bold')
            ax1.grid(True, axis='x', alpha=0.3)
            
            # Confidence over time
            times = [(pred.start + pred.end) / 2 for pred in predictions]
            confidences = [pred.confidence for pred in predictions]
            emotions_colors = [EMOTION_COLORS.get(pred.primary_emotion, '#808080') 
                              for pred in predictions]
            
            ax2.scatter(times, confidences, c=emotions_colors, alpha=0.6, s=50)
            ax2.plot(times, confidences, 'k-', alpha=0.3)
            ax2.set_xlim(0, audio_duration)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Confidence', fontsize=12)
            ax2.set_title('Emotion Confidence Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                logger.info(f"Emotion visualization saved to: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            log_exception(logger, e, "Failed to create visualization")