"""Request correlation utilities for tracing."""

import uuid
import contextvars
from typing import Optional
import logging

# Context variable to store correlation ID across async calls
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)

logger = logging.getLogger(__name__)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.
    
    Args:
        correlation_id: Optional correlation ID, generates new one if None
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def get_or_create_correlation_id() -> str:
    """Get existing correlation ID or create a new one."""
    correlation_id = get_correlation_id()
    if correlation_id is None:
        correlation_id = set_correlation_id()
    return correlation_id


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class CorrelationContext:
    """Context manager for setting correlation ID."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id
        self.token = None
    
    def __enter__(self) -> str:
        if self.correlation_id is None:
            self.correlation_id = generate_correlation_id()
        
        self.token = correlation_id_var.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            correlation_id_var.reset(self.token)