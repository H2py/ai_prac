"""
Logging utilities for the audio analysis pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import colorama
from colorama import Fore, Style
import psutil
import time


colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Format the message
        message = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return message


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, list] = {}
        
    def start_timer(self, name: str) -> None:
        """Start a timer for a named operation.
        
        Args:
            name: Name of the operation
        """
        self.start_times[name] = time.time()
        self.logger.debug(f"Started timer for: {name}")
        
    def stop_timer(self, name: str) -> float:
        """Stop a timer and log the duration.
        
        Args:
            name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if name not in self.start_times:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        del self.start_times[name]
        
        # Store metric
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        self.logger.info(f"Completed {name} in {duration:.2f} seconds")
        return duration
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent
        
        self.logger.info(
            f"Memory usage: {memory_mb:.1f} MB "
            f"(System: {system_percent:.1f}%)"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        for name, times in self.metrics.items():
            summary[name] = {
                'count': len(times),
                'total': sum(times),
                'average': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        return summary
    
    def log_summary(self) -> None:
        """Log performance summary."""
        summary = self.get_summary()
        if not summary:
            return
        
        self.logger.info("Performance Summary:")
        self.logger.info("-" * 50)
        
        for name, stats in summary.items():
            self.logger.info(
                f"{name}: "
                f"Count={stats['count']}, "
                f"Total={stats['total']:.2f}s, "
                f"Avg={stats['average']:.2f}s, "
                f"Min={stats['min']:.2f}s, "
                f"Max={stats['max']:.2f}s"
            )


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, total: Optional[int] = None):
        """Initialize progress logger.
        
        Args:
            logger: Base logger instance
            total: Total number of items to process
        """
        self.logger = logger
        self.total = total
        self.current = 0
        self.start_time = time.time()
        
    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """Update progress.
        
        Args:
            increment: Number of items completed
            message: Optional progress message
        """
        self.current += increment
        
        if self.total:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
            else:
                eta = 0
            
            progress_msg = (
                f"Progress: {self.current}/{self.total} "
                f"({percentage:.1f}%) - "
                f"Elapsed: {elapsed:.1f}s - "
                f"ETA: {eta:.1f}s"
            )
        else:
            elapsed = time.time() - self.start_time
            progress_msg = f"Progress: {self.current} items - Elapsed: {elapsed:.1f}s"
        
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark operation as complete.
        
        Args:
            message: Optional completion message
        """
        elapsed = time.time() - self.start_time
        complete_msg = f"Completed {self.current} items in {elapsed:.1f}s"
        
        if message:
            complete_msg += f" - {message}"
        
        self.logger.info(complete_msg)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        log_to_console: Whether to log to console
        log_format: Log message format
        date_format: Date format for log messages
        use_colors: Whether to use colored console output
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if use_colors:
            console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[str] = None,
        suppress: bool = False
    ):
        """Initialize log context.
        
        Args:
            logger: Logger to configure
            level: Temporary log level
            suppress: Whether to suppress all logging
        """
        self.logger = logger
        self.original_level = logger.level
        self.original_disabled = logger.disabled
        
        if suppress:
            self.new_level = logging.CRITICAL + 1
            self.disabled = True
        elif level:
            self.new_level = getattr(logging, level.upper())
            self.disabled = False
        else:
            self.new_level = self.original_level
            self.disabled = self.original_disabled
    
    def __enter__(self):
        """Enter context."""
        self.logger.setLevel(self.new_level)
        self.logger.disabled = self.disabled
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.original_level)
        self.logger.disabled = self.original_disabled


def log_exception(logger: logging.Logger, exception: Exception, context: Optional[str] = None) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Optional context information
    """
    error_msg = f"Exception occurred: {type(exception).__name__}: {str(exception)}"
    
    if context:
        error_msg = f"{context} - {error_msg}"
    
    logger.error(error_msg, exc_info=True)


def create_file_logger(
    name: str,
    log_file: Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """Create a rotating file logger.
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create rotating file handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger