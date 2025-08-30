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
from src.utils.memory_manager import memory_manager
from src.utils.lazy_loader import LazyModelLoader, model_registry, require_model
from src.utils.cleanup_manager import cleanup_manager, register_for_cleanup
from config.settings import ModelConfig
from src.models.segments import EmotionSegment
from src.models.base import BaseSegment


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




class EmotionAnalyzer:
    """Emotion recognition using transformer models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize emotion analyzer with lazy loading.
        
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
        
        # Set up lazy loaders for models
        self._model_loader = LazyModelLoader(
            loader_func=self._load_emotion_model,
            model_name="emotion_model"
        )
        
        self._feature_extractor_loader = LazyModelLoader(
            loader_func=self._load_feature_extractor,
            model_name="emotion_feature_extractor"
        )
        
        self._pipeline_loader = LazyModelLoader(
            loader_func=self._load_emotion_pipeline,
            model_name="emotion_pipeline"
        )
        
        # Register with global registry for memory management
        model_registry.register("emotion_model", self._load_emotion_model)
        model_registry.register("emotion_feature_extractor", self._load_feature_extractor)
        model_registry.register("emotion_pipeline", self._load_emotion_pipeline)
        
        # Register for cleanup tracking
        register_for_cleanup(self, self.cleanup)
        
        logger.info(f"EmotionAnalyzer initialized with lazy loading (device: {self.device})")
    
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
    
    def _load_emotion_model(self):
        """Load the emotion recognition model.
        
        Returns:
            Loaded emotion model
        """
        perf_logger.start_timer("emotion_model_loading")
        
        try:
            model_name = self.config.emotion_model
            logger.info(f"Loading emotion recognition model: {model_name}")
            
            model = AutoModelForAudioClassification.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Move model to device
            model = model.to(self.device)
            model.eval()
            
            duration = perf_logger.stop_timer("emotion_model_loading")
            logger.info(f"Emotion model loaded in {duration:.2f}s")
            
            return model
            
        except Exception as e:
            perf_logger.stop_timer("emotion_model_loading")
            logger.error(f"Failed to load emotion model: {e}")
            return None
    
    def _load_feature_extractor(self):
        """Load the feature extractor.
        
        Returns:
            Loaded feature extractor
        """
        try:
            model_name = self.config.emotion_model
            logger.info(f"Loading feature extractor: {model_name}")
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            
            logger.info("Feature extractor loaded successfully")
            return feature_extractor
            
        except Exception as e:
            logger.error(f"Failed to load feature extractor: {e}")
            return None
    
    def _load_emotion_pipeline(self):
        """Load alternative emotion pipeline.
        
        Returns:
            Loaded emotion pipeline or None
        """
        try:
            alternative_model = "superb/hubert-base-superb-er"
            logger.info(f"Loading alternative emotion pipeline: {alternative_model}")
            
            emotion_pipeline = pipeline(
                "audio-classification",
                model=alternative_model,
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("Alternative emotion pipeline loaded successfully")
            return emotion_pipeline
            
        except Exception as e:
            logger.error(f"Failed to load emotion pipeline: {e}")
            return None
    
    @property
    def model(self):
        """Lazily loaded emotion model."""
        return self._model_loader.load()
    
    @property  
    def feature_extractor(self):
        """Lazily loaded feature extractor."""
        return self._feature_extractor_loader.load()
    
    @property
    def emotion_pipeline(self):
        """Lazily loaded emotion pipeline."""
        return self._pipeline_loader.load()
    
    def unload_models(self) -> None:
        """Unload all models to free memory."""
        logger.info("Unloading emotion analysis models")
        self._model_loader.unload()
        self._feature_extractor_loader.unload()
        self._pipeline_loader.unload()
        
        # Clear from global registry
        model_registry.unload("emotion_model")
        model_registry.unload("emotion_feature_extractor")
        model_registry.unload("emotion_pipeline")
    
    def cleanup(self) -> None:
        """Comprehensive cleanup of all resources."""
        logger.info("Performing comprehensive cleanup of EmotionAnalyzer")
        
        # Unload models
        self.unload_models()
        
        # Clear memory pools
        memory_manager.clear_memory()
        
        # Check for memory pressure and trigger cleanup if needed
        if cleanup_manager.check_memory_pressure():
            cleanup_manager.periodic_cleanup()
        
        logger.info("EmotionAnalyzer cleanup completed")
    
    def _setup_mock_model(self) -> None:
        """Setup mock model for testing when real model unavailable."""
        logger.warning("Using mock emotion model for demonstration")
        return None
    
    def _calculate_optimal_batch_size(self, audio_duration: float) -> int:
        """Calculate optimal batch size based on available memory and audio duration.
        
        Args:
            audio_duration: Total audio duration in seconds
            
        Returns:
            Optimal batch size for processing
        """
        # Estimate memory per audio second (rough approximation)
        # 16kHz * 4 bytes * processing overhead (5x) = ~320KB per second
        memory_per_second_bytes = 16000 * 4 * 5
        
        # Calculate batch size based on available memory
        batch_size = memory_manager.get_optimal_batch_size(
            sample_size=memory_per_second_bytes,
            max_memory_mb=None,  # Use available system memory
            safety_factor=0.6    # Conservative for emotion processing
        )
        
        # Adjust based on audio duration (longer audio = smaller batches)
        if audio_duration > 300:  # > 5 minutes
            batch_size = max(1, batch_size // 4)
        elif audio_duration > 60:  # > 1 minute
            batch_size = max(1, batch_size // 2)
        
        # Reasonable limits
        batch_size = max(1, min(batch_size, 32))
        
        logger.debug(f"Calculated optimal batch size: {batch_size} for {audio_duration:.1f}s audio")
        return batch_size
    
    def _process_audio_chunks_batched(self, 
                                    audio_chunks: List[Tuple[np.ndarray, Tuple[float, float]]],
                                    sample_rate: int = 16000) -> List[EmotionSegment]:
        """Process audio chunks in optimized batches.
        
        Args:
            audio_chunks: List of (audio_data, (start_time, end_time)) tuples
            sample_rate: Audio sample rate
            
        Returns:
            List of emotion predictions
        """
        if not audio_chunks:
            return []
        
        # Calculate optimal batch size
        total_duration = sum((end - start) for _, (start, end) in audio_chunks)
        batch_size = self._calculate_optimal_batch_size(total_duration)
        
        predictions = []
        
        # Log memory usage before processing
        memory_manager.log_memory_usage("before emotion batch processing")
        
        # Process in batches
        for i in range(0, len(audio_chunks), batch_size):
            batch = audio_chunks[i:i + batch_size]
            
            # Process batch
            batch_predictions = []
            for chunk_audio, (start_time, end_time) in batch:
                pred = self.predict_emotion(chunk_audio, sample_rate)
                pred.start = start_time
                pred.end = end_time
                batch_predictions.append(pred)
            
            predictions.extend(batch_predictions)
            
            # Clear memory after each batch
            if i % (batch_size * 2) == 0:  # Every 2 batches
                memory_manager.clear_memory(force_gc=False)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(audio_chunks)-1)//batch_size + 1}")
        
        # Final memory cleanup
        memory_manager.clear_memory(force_gc=True)
        memory_manager.log_memory_usage("after emotion batch processing")
        
        # Check for memory pressure after batch processing
        cleanup_manager.periodic_cleanup()
        
        return predictions
    
    def predict_emotion(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000
    ) -> EmotionSegment:
        """Predict emotion for a single audio chunk with lazy loading and memory optimization.
        
        Args:
            audio_chunk: Audio data array
            sample_rate: Sample rate of audio
            
        Returns:
            Emotion prediction
        """
        # Models are now lazy loaded through property access
        
        # Use memory pool for processing arrays to reduce allocations
        processing_array = None
        normalized_array = None
        
        try:
            # Get processing array from memory pool
            if audio_chunk.size > 0:
                processing_array = memory_manager.get_memory_pool(
                    "emotion_processing", 
                    audio_chunk.shape, 
                    audio_chunk.dtype
                )
                np.copyto(processing_array, audio_chunk)
                
                # Normalize audio if needed (using pooled array)
                if processing_array.max() > 1.0 or processing_array.min() < -1.0:
                    normalized_array = memory_manager.get_memory_pool(
                        "emotion_normalized",
                        audio_chunk.shape,
                        np.float32
                    )
                    np.copyto(normalized_array, processing_array / np.max(np.abs(processing_array)))
                    audio_data = normalized_array
                else:
                    audio_data = processing_array
            else:
                audio_data = audio_chunk
            
            if self.emotion_pipeline:
                # Use pipeline
                result = self.emotion_pipeline(audio_data)
                scores = self._process_pipeline_output(result)
            elif self.model and self.feature_extractor:
                # Use model directly
                scores = self._predict_with_model(audio_data, sample_rate)
            else:
                # Use mock prediction
                scores = self._mock_prediction(audio_data)
            
            # Get primary emotion
            primary_emotion = max(scores, key=scores.get)
            confidence = scores[primary_emotion]
            
            return EmotionSegment(
                start=0.0,
                end=len(audio_chunk) / sample_rate,
                confidence=confidence,
                predicted_emotion=primary_emotion,
                emotion_scores=scores
            )
            
        except Exception as e:
            logger.warning(f"Emotion prediction failed: {e}")
            # Return neutral emotion on failure
            return EmotionSegment(
                start=0.0,
                end=len(audio_chunk) / sample_rate,
                confidence=0.5,
                predicted_emotion='neutral',
                emotion_scores={'neutral': 0.5}
            )
        finally:
            # Return arrays to memory pool for reuse
            if processing_array is not None:
                memory_manager.return_to_pool("emotion_processing", processing_array)
            if normalized_array is not None:
                memory_manager.return_to_pool("emotion_normalized", normalized_array)
    
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
        segments: Optional[List[Union[Dict, BaseSegment]]] = None,
        chunk_duration: float = 3.0,
        **kwargs
    ) -> List[EmotionSegment]:
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
                    # Handle both dict and BaseSegment types
                    if isinstance(segment, BaseSegment):
                        start_time = segment.start
                        end_time = segment.end
                        speaker = getattr(segment, 'speaker', None)
                    else:
                        start_time = segment['start']
                        end_time = segment['end']
                        speaker = segment.get('speaker')
                    
                    # Extract segment audio
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if len(segment_audio) > 0:
                        # Predict emotion
                        pred = self.predict_emotion(segment_audio, sample_rate)
                        
                        # Update with segment info
                        pred.start = start_time
                        pred.end = end_time
                        pred.speaker = speaker
                        
                        predictions.append(pred)
                    
                    progress.update(message=f"Processed segment {len(predictions)}")
                
                progress.complete()
                
            else:
                # Analyze entire audio in chunks with optimized batching
                chunks = split_audio_chunks(
                    audio_data,
                    sample_rate,
                    chunk_duration=chunk_duration,
                    overlap=0.5
                )
                
                logger.info(f"Processing {len(chunks)} audio chunks with optimized batching")
                
                # Use optimized batch processing
                predictions = self._process_audio_chunks_batched(chunks, sample_rate)
            
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
        predictions: List[EmotionSegment],
        threshold: float = 0.5
    ) -> List[EmotionSegment]:
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
            same_emotion = pred.emotion == current.emotion
            same_speaker = getattr(pred, 'speaker', None) == getattr(current, 'speaker', None)
            close_time = pred.start - current.end < threshold
            
            if same_emotion and same_speaker and close_time:
                # Merge predictions
                all_scores = {}
                current_duration = current.end - current.start
                pred_duration = pred.end - pred.start
                
                for emotion in set(current.scores.keys()) | set(pred.scores.keys()):
                    score1 = current.scores.get(emotion, 0) * current_duration
                    score2 = pred.scores.get(emotion, 0) * pred_duration
                    total_duration = current_duration + pred_duration
                    all_scores[emotion] = (score1 + score2) / total_duration
                
                current = EmotionSegment(
                    start=current.start,
                    end=pred.end,
                    confidence=(current.confidence * current_duration + 
                               pred.confidence * pred_duration) / 
                              (current_duration + pred_duration),
                    predicted_emotion=current.predicted_emotion,
                    emotion_scores=all_scores
                )
                # Preserve speaker info
                current.speaker = getattr(current, 'speaker', None)
            else:
                merged.append(current)
                current = pred
        
        merged.append(current)
        
        return merged
    
    def batch_process(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000
    ) -> List[EmotionSegment]:
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
        predictions: List[EmotionSegment]
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
        total_duration = sum(pred.end - pred.start for pred in predictions)
        
        # Calculate emotion distribution
        emotion_durations = defaultdict(float)
        confidence_sum = 0.0
        
        for pred in predictions:
            duration = pred.end - pred.start
            emotion_durations[pred.emotion] += duration
            confidence_sum += pred.confidence * duration
        
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
            if predictions[i].emotion != predictions[i-1].emotion
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
        predictions: List[EmotionSegment],
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
            emotions = list(set(pred.emotion for pred in predictions))
            emotion_to_y = {emotion: i for i, emotion in enumerate(emotions)}
            
            for pred in predictions:
                duration = pred.end - pred.start
                color = EMOTION_COLORS.get(pred.emotion, '#808080')
                rect = patches.Rectangle(
                    (pred.start, emotion_to_y[pred.emotion]),
                    duration,
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
            emotions_colors = [EMOTION_COLORS.get(pred.emotion, '#808080') 
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