"""
Lazy loading utilities for AI models to optimize memory usage.
"""

import logging
from typing import Any, Callable, Optional, Dict, TypeVar, Generic
from functools import wraps
import threading
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyModelLoader(Generic[T]):
    """Generic lazy loader for AI models with thread safety."""
    
    def __init__(self, loader_func: Callable[[], T], model_name: str = "model"):
        """Initialize lazy model loader.
        
        Args:
            loader_func: Function that loads and returns the model
            model_name: Name of the model for logging
        """
        self._loader_func = loader_func
        self._model_name = model_name
        self._model: Optional[T] = None
        self._lock = threading.Lock()
        self._loading = False
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def load(self) -> T:
        """Load the model if not already loaded.
        
        Returns:
            The loaded model instance
        """
        if self._model is not None:
            return self._model
        
        with self._lock:
            # Double-check locking pattern
            if self._model is not None:
                return self._model
            
            if self._loading:
                # Another thread is currently loading, wait for it
                logger.debug(f"Waiting for {self._model_name} to finish loading...")
                while self._loading and self._model is None:
                    threading.Event().wait(0.1)  # Small sleep to avoid busy waiting
                
                if self._model is not None:
                    return self._model
            
            try:
                self._loading = True
                logger.info(f"Lazy loading {self._model_name}...")
                self._model = self._loader_func()
                logger.info(f"Successfully loaded {self._model_name}")
                return self._model
            
            except Exception as e:
                logger.error(f"Failed to load {self._model_name}: {e}")
                raise
            
            finally:
                self._loading = False
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self._model is not None:
                logger.info(f"Unloading {self._model_name}")
                # Try to explicitly delete model-specific resources
                if hasattr(self._model, 'cpu'):
                    try:
                        self._model.cpu()  # Move to CPU to free GPU memory
                    except:
                        pass
                
                self._model = None
                
                # Force garbage collection for models
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def __call__(self) -> T:
        """Make the loader callable."""
        return self.load()


class LazyModelRegistry:
    """Registry for managing multiple lazy-loaded models."""
    
    def __init__(self):
        self._models: Dict[str, LazyModelLoader] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, loader_func: Callable[[], Any]) -> LazyModelLoader:
        """Register a new lazy model.
        
        Args:
            name: Model name/identifier
            loader_func: Function to load the model
            
        Returns:
            LazyModelLoader instance
        """
        with self._lock:
            if name in self._models:
                logger.warning(f"Model {name} already registered, overwriting")
            
            lazy_loader = LazyModelLoader(loader_func, name)
            self._models[name] = lazy_loader
            logger.debug(f"Registered lazy model: {name}")
            return lazy_loader
    
    def get(self, name: str) -> Optional[LazyModelLoader]:
        """Get a registered lazy model.
        
        Args:
            name: Model name
            
        Returns:
            LazyModelLoader instance or None if not found
        """
        return self._models.get(name)
    
    def load(self, name: str) -> Any:
        """Load a specific model by name.
        
        Args:
            name: Model name
            
        Returns:
            Loaded model instance
            
        Raises:
            KeyError: If model is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model {name} not registered")
        
        return self._models[name].load()
    
    def unload(self, name: str) -> None:
        """Unload a specific model by name.
        
        Args:
            name: Model name
        """
        if name in self._models:
            self._models[name].unload()
    
    def unload_all(self) -> None:
        """Unload all registered models."""
        logger.info("Unloading all lazy models")
        with self._lock:
            for name, loader in self._models.items():
                try:
                    loader.unload()
                except Exception as e:
                    logger.warning(f"Failed to unload {name}: {e}")
    
    def get_memory_status(self) -> Dict[str, bool]:
        """Get loading status of all models.
        
        Returns:
            Dictionary mapping model names to their loaded status
        """
        return {name: loader.is_loaded for name, loader in self._models.items()}
    
    def list_loaded_models(self) -> list[str]:
        """Get list of currently loaded model names.
        
        Returns:
            List of loaded model names
        """
        return [name for name, loader in self._models.items() if loader.is_loaded]


# Global registry instance
model_registry = LazyModelRegistry()


def lazy_property(loader_func: Callable[[], T]) -> property:
    """Decorator to create a lazy-loaded property.
    
    Args:
        loader_func: Function to load the value
        
    Returns:
        Property that loads value on first access
    """
    attr_name = f'_lazy_{loader_func.__name__}'
    
    def getter(self):
        if not hasattr(self, attr_name):
            logger.debug(f"Lazy loading property: {loader_func.__name__}")
            setattr(self, attr_name, loader_func(self))
        return getattr(self, attr_name)
    
    def deleter(self):
        if hasattr(self, attr_name):
            logger.debug(f"Clearing lazy property: {loader_func.__name__}")
            delattr(self, attr_name)
    
    return property(getter, fdel=deleter)


def require_model(model_attr: str):
    """Decorator to ensure a model is loaded before method execution.
    
    Args:
        model_attr: Name of the model attribute (LazyModelLoader)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            model_loader = getattr(self, model_attr)
            if isinstance(model_loader, LazyModelLoader):
                model_loader.load()  # Ensure model is loaded
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class LazyInitializer:
    """Helper class for lazy initialization of expensive resources."""
    
    def __init__(self):
        self._initializers: Dict[str, Callable] = {}
        self._initialized: Dict[str, bool] = {}
        self._resources: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register_initializer(self, name: str, init_func: Callable[[], Any]):
        """Register an initialization function.
        
        Args:
            name: Resource name
            init_func: Function to initialize the resource
        """
        self._initializers[name] = init_func
        self._initialized[name] = False
    
    def get_resource(self, name: str) -> Any:
        """Get a resource, initializing if necessary.
        
        Args:
            name: Resource name
            
        Returns:
            Initialized resource
        """
        if self._initialized.get(name, False):
            return self._resources[name]
        
        with self._lock:
            # Double-check locking
            if self._initialized.get(name, False):
                return self._resources[name]
            
            if name not in self._initializers:
                raise KeyError(f"No initializer registered for {name}")
            
            logger.debug(f"Lazy initializing resource: {name}")
            self._resources[name] = self._initializers[name]()
            self._initialized[name] = True
            
            return self._resources[name]
    
    def clear_resource(self, name: str):
        """Clear a specific resource to free memory.
        
        Args:
            name: Resource name
        """
        with self._lock:
            if name in self._resources:
                del self._resources[name]
            self._initialized[name] = False
    
    def clear_all(self):
        """Clear all resources."""
        with self._lock:
            self._resources.clear()
            self._initialized = {name: False for name in self._initialized}