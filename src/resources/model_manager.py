"""
Centralized model resource manager with lazy loading and sharing.
"""

import logging
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import threading
import torch

from .cache import LRUCache

logger = logging.getLogger(__name__)


class ModelResourceManager:
    """Lazy loading and sharing of ML models across processors."""
    
    def __init__(self, cache_size: int = 10):
        """Initialize model resource manager.
        
        Args:
            cache_size: Maximum number of models to cache
        """
        self.cache = LRUCache[Any](max_size=cache_size)
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        
        # Model loading functions
        self._model_loaders: Dict[str, Callable] = {}
        
        logger.info(f"ModelResourceManager initialized with cache size: {cache_size}")
    
    def register_loader(self, model_type: str, loader_func: Callable) -> None:
        """Register a model loading function.
        
        Args:
            model_type: Type identifier for the model
            loader_func: Function to load the model
        """
        self._model_loaders[model_type] = loader_func
        logger.debug(f"Registered loader for model type: {model_type}")
    
    def get_speaker_pipeline(self, auth_token: str, model_name: str = "pyannote/speaker-diarization-3.1") -> Any:
        """Get speaker diarization pipeline with caching.
        
        Args:
            auth_token: HuggingFace authentication token
            model_name: Model name to load
            
        Returns:
            Loaded speaker diarization pipeline
        """
        cache_key = f"speaker_pipeline_{model_name}_{hash(auth_token) % 10000}"
        
        model = self.cache.get(cache_key)
        if model is not None:
            logger.debug("Speaker pipeline cache hit")
            return model
        
        # Load model with thread safety
        with self._get_loading_lock(cache_key):
            # Double-check after acquiring lock
            model = self.cache.get(cache_key)
            if model is not None:
                return model
            
            logger.info(f"Loading speaker diarization pipeline: {model_name}")
            
            try:
                from pyannote.audio import Pipeline
                
                pipeline = Pipeline.from_pretrained(
                    model_name,
                    use_auth_token=auth_token
                )
                
                # Move to appropriate device
                device = self._get_optimal_device()
                if pipeline:
                    pipeline.to(device)
                
                self.cache.put(cache_key, pipeline)
                logger.info(f"Speaker pipeline loaded and cached on {device}")
                
                return pipeline
                
            except Exception as e:
                logger.error(f"Failed to load speaker pipeline: {e}")
                raise
    
    def get_whisper_model(self, model_name: str = "base", device: Optional[str] = None) -> Any:
        """Get Whisper model with caching.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to load on (None for auto)
            
        Returns:
            Loaded Whisper model
        """
        if device is None:
            device = self._get_optimal_device()
        
        cache_key = f"whisper_{model_name}_{device}"
        
        model = self.cache.get(cache_key)
        if model is not None:
            logger.debug("Whisper model cache hit")
            return model
        
        with self._get_loading_lock(cache_key):
            # Double-check after acquiring lock
            model = self.cache.get(cache_key)
            if model is not None:
                return model
            
            logger.info(f"Loading Whisper model: {model_name} on {device}")
            
            try:
                import whisper
                
                model = whisper.load_model(model_name, device=device)
                
                self.cache.put(cache_key, model)
                logger.info(f"Whisper model loaded and cached: {model_name}")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def get_vad_model(self, mode: str = "balanced") -> Any:
        """Get Voice Activity Detection model with caching.
        
        Args:
            mode: VAD mode (fast, balanced, accurate)
            
        Returns:
            Loaded VAD model
        """
        cache_key = f"vad_{mode}"
        
        model = self.cache.get(cache_key)
        if model is not None:
            logger.debug("VAD model cache hit")
            return model
        
        with self._get_loading_lock(cache_key):
            # Double-check after acquiring lock
            model = self.cache.get(cache_key)
            if model is not None:
                return model
            
            logger.info(f"Loading VAD model in {mode} mode")
            
            try:
                # Select model based on mode
                if mode == "fast":
                    model_name = "silero_vad"
                elif mode == "balanced":
                    model_name = "silero_vad_v4"
                else:  # accurate
                    model_name = "silero_vad_v5"
                
                # Load model
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model=model_name,
                    force_reload=False,
                    verbose=False
                )
                
                model.eval()
                
                # Move to appropriate device
                device = self._get_optimal_device()
                model = model.to(device)
                
                self.cache.put(cache_key, model)
                logger.info(f"VAD model loaded and cached: {model_name} on {device}")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise
    
    def get_emotion_model(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") -> Any:
        """Get emotion recognition model with caching.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Loaded emotion model
        """
        cache_key = f"emotion_{model_name.replace('/', '_')}"
        
        model = self.cache.get(cache_key)
        if model is not None:
            logger.debug("Emotion model cache hit")
            return model
        
        with self._get_loading_lock(cache_key):
            # Double-check after acquiring lock
            model = self.cache.get(cache_key)
            if model is not None:
                return model
            
            logger.info(f"Loading emotion model: {model_name}")
            
            try:
                from transformers import pipeline
                
                # Load emotion classification pipeline
                emotion_pipeline = pipeline(
                    "audio-classification",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.cache.put(cache_key, emotion_pipeline)
                logger.info(f"Emotion model loaded and cached: {model_name}")
                
                return emotion_pipeline
                
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                raise
    
    def get_model(self, model_type: str, **kwargs) -> Any:
        """Generic model loading with registered loaders.
        
        Args:
            model_type: Type of model to load
            **kwargs: Arguments for model loader
            
        Returns:
            Loaded model
        """
        if model_type not in self._model_loaders:
            raise ValueError(f"No loader registered for model type: {model_type}")
        
        # Generate cache key from type and kwargs
        cache_key = f"{model_type}_{hash(str(sorted(kwargs.items()))) % 10000}"
        
        model = self.cache.get(cache_key)
        if model is not None:
            logger.debug(f"Model cache hit: {model_type}")
            return model
        
        with self._get_loading_lock(cache_key):
            # Double-check after acquiring lock
            model = self.cache.get(cache_key)
            if model is not None:
                return model
            
            logger.info(f"Loading model: {model_type}")
            
            try:
                loader_func = self._model_loaders[model_type]
                model = loader_func(**kwargs)
                
                self.cache.put(cache_key, model)
                logger.info(f"Model loaded and cached: {model_type}")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_type}: {e}")
                raise
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'cached_models': cache_stats['size'],
            'max_cache_size': cache_stats['max_size'],
            'registered_loaders': list(self._model_loaders.keys()),
            'cached_model_keys': cache_stats['keys']
        }
    
    def _get_loading_lock(self, cache_key: str) -> threading.Lock:
        """Get or create loading lock for a cache key."""
        with self._lock:
            if cache_key not in self._loading_locks:
                self._loading_locks[cache_key] = threading.Lock()
            return self._loading_locks[cache_key]
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for model loading."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _cleanup_locks(self) -> None:
        """Clean up unused locks."""
        with self._lock:
            # Remove locks for cache keys that are no longer cached
            cache_stats = self.cache.get_stats()
            cached_keys = set(cache_stats['keys'])
            
            locks_to_remove = []
            for key in self._loading_locks:
                if key not in cached_keys:
                    locks_to_remove.append(key)
            
            for key in locks_to_remove:
                del self._loading_locks[key]
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.clear_cache()
        except Exception:
            pass  # Ignore errors during cleanup