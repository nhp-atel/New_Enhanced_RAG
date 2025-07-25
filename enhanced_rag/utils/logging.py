"""Structured logging setup for the RAG system."""

import logging
import logging.config
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import traceback


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                ]:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    output_file: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("json" or "text")
        output_file: Optional file to write logs to
        context: Optional context to add to all log records
    """
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    # File handler if specified
    if output_file:
        file_handler = logging.FileHandler(output_file)
        handlers.append(file_handler)
    
    # Configure formatters
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Setup handlers
    for handler in handlers:
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(formatter)
        
        # Add context filter if provided
        if context:
            handler.addFilter(ContextFilter(context))
        
        root_logger.addHandler(handler)
    
    # Configure specific loggers
    _configure_library_loggers()
    
    logging.info(f"Logging configured: level={level}, format={format_type}")


def _configure_library_loggers():
    """Configure logging levels for third-party libraries."""
    
    # Reduce noise from common libraries
    library_levels = {
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "faiss": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
    }
    
    for library, level in library_levels.items():
        logging.getLogger(library).setLevel(level)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name
        context: Optional context to add to all log records
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if context:
        # Add context filter to the logger
        logger.addFilter(ContextFilter(context))
    
    return logger


class LoggingContext:
    """Context manager for adding context to logs within a scope."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.filter = None
    
    def __enter__(self):
        self.filter = ContextFilter(self.context)
        self.logger.addFilter(self.filter)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filter:
            self.logger.removeFilter(self.filter)


# Performance logging utilities
class PerformanceLogger:
    """Utility for logging performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_operation_time(
        self, 
        operation: str, 
        duration_ms: float, 
        **extra_fields
    ):
        """Log operation timing."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "performance_metric": True,
                **extra_fields
            }
        )
    
    def log_token_usage(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int,
        **extra_fields
    ):
        """Log token usage for LLM calls."""
        total_tokens = prompt_tokens + completion_tokens
        
        self.logger.info(
            f"Token usage: {total_tokens} total tokens",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "token_usage_metric": True,
                **extra_fields
            }
        )
    
    def log_embedding_batch(
        self, 
        model: str, 
        batch_size: int, 
        duration_ms: float,
        **extra_fields
    ):
        """Log embedding generation metrics."""
        self.logger.info(
            f"Embedding batch processed: {batch_size} texts",
            extra={
                "model": model,
                "batch_size": batch_size,
                "duration_ms": duration_ms,
                "embeddings_per_second": batch_size / (duration_ms / 1000),
                "embedding_metric": True,
                **extra_fields
            }
        )


# Error logging utilities
def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    **context
):
    """Log an error with additional context."""
    logger.error(
        f"Error in {operation}: {str(error)}",
        extra={
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_occurred": True,
            **context
        },
        exc_info=True
    )


# Audit logging
class AuditLogger:
    """Logger for audit events."""
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
    
    def log_document_ingestion(
        self, 
        document_id: str, 
        source: str, 
        chunks_created: int,
        **metadata
    ):
        """Log document ingestion event."""
        self.logger.info(
            f"Document ingested: {document_id}",
            extra={
                "event_type": "document_ingestion",
                "document_id": document_id,
                "source": source,
                "chunks_created": chunks_created,
                "audit_event": True,
                **metadata
            }
        )
    
    def log_query_event(
        self, 
        correlation_id: str, 
        query: str, 
        retrieved_chunks: int,
        processing_time_ms: float,
        **metadata
    ):
        """Log query processing event."""
        self.logger.info(
            f"Query processed: {correlation_id}",
            extra={
                "event_type": "query_processed",
                "correlation_id": correlation_id,
                "query_length": len(query),
                "retrieved_chunks": retrieved_chunks,
                "processing_time_ms": processing_time_ms,
                "audit_event": True,
                **metadata
            }
        )
    
    def log_system_event(
        self, 
        event_type: str, 
        description: str, 
        **metadata
    ):
        """Log system-level events."""
        self.logger.info(
            f"System event: {event_type}",
            extra={
                "event_type": event_type,
                "description": description,
                "audit_event": True,
                "system_event": True,
                **metadata
            }
        )