"""
Error handling and policy definitions for processing pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorPolicy(Enum):
    """Error handling policies for processing pipeline."""
    FAIL_FAST = "fail_fast"  # Stop on first error
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Continue with recoverable errors
    CONTINUE = "continue"  # Continue with all errors


class ErrorAction(Enum):
    """Actions to take when error occurs."""
    RAISE = "raise"  # Raise the error
    CONTINUE = "continue"  # Log and continue
    RETRY = "retry"  # Attempt retry


@dataclass
class ProcessingError:
    """Structured error information."""
    component: str
    error_type: str
    message: str
    context: str = ""
    recoverable: bool = True
    
    def __str__(self) -> str:
        context_str = f" ({self.context})" if self.context else ""
        return f"[{self.component}] {self.error_type}: {self.message}{context_str}"


@dataclass
class ProcessingWarning:
    """Structured warning information."""
    component: str
    message: str
    context: str = ""
    
    def __str__(self) -> str:
        context_str = f" ({self.context})" if self.context else ""
        return f"[{self.component}] Warning: {self.message}{context_str}"