"""Unit tests for logging utilities."""

import logging
import json
import tempfile
import sys
from unittest.mock import Mock, patch, mock_open
from io import StringIO
import pytest

from enhanced_rag.utils.logging import (
    JSONFormatter,
    ContextFilter,
    setup_logging,
    get_logger,
    LoggingContext,
    PerformanceLogger,
    log_error_with_context,
    AuditLogger,
    _configure_library_loggers
)


class TestJSONFormatter:
    """Test JSON formatter functionality."""
    
    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line"] == 42
        assert "timestamp" in parsed
        assert parsed["timestamp"].endswith("Z")
    
    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        record.funcName = "test_function"
        record.module = "test_module"
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test error"
        assert "traceback" in parsed["exception"]
        assert isinstance(parsed["exception"]["traceback"], list)
    
    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.correlation_id = "test-123"
        record.user_id = "user-456"
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert parsed["correlation_id"] == "test-123"
        assert parsed["user_id"] == "user-456"
    
    def test_json_formatter_exclude_extra(self):
        """Test JSON formatting without extra fields."""
        formatter = JSONFormatter(include_extra=False)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_function"
        record.module = "test_module"
        record.correlation_id = "test-123"
        
        result = formatter.format(record)
        parsed = json.loads(result)
        
        assert "correlation_id" not in parsed
        assert parsed["message"] == "Test message"


class TestContextFilter:
    """Test context filter functionality."""
    
    def test_context_filter_basic(self):
        """Test basic context filtering."""
        context = {"correlation_id": "test-123", "user_id": "user-456"}
        filter_obj = ContextFilter(context)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.correlation_id == "test-123"
        assert record.user_id == "user-456"
    
    def test_context_filter_empty_context(self):
        """Test context filter with empty context."""
        filter_obj = ContextFilter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
    
    def test_context_filter_none_context(self):
        """Test context filter with None context."""
        filter_obj = ContextFilter(None)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_setup_logging_json_format(self):
        """Test setting up logging with JSON format."""
        with patch('sys.stdout', new_callable=StringIO):
            setup_logging(level="INFO", format_type="json")
            
            root_logger = logging.getLogger()
            assert root_logger.level == logging.INFO
            assert len(root_logger.handlers) == 1
            
            handler = root_logger.handlers[0]
            assert isinstance(handler.formatter, JSONFormatter)
    
    def test_setup_logging_text_format(self):
        """Test setting up logging with text format."""
        with patch('sys.stdout', new_callable=StringIO):
            setup_logging(level="DEBUG", format_type="text")
            
            root_logger = logging.getLogger()
            assert root_logger.level == logging.DEBUG
            
            handler = root_logger.handlers[0]
            assert isinstance(handler.formatter, logging.Formatter)
            assert not isinstance(handler.formatter, JSONFormatter)
    
    def test_setup_logging_with_file(self):
        """Test setting up logging with file output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        with patch('sys.stdout', new_callable=StringIO):
            setup_logging(level="INFO", format_type="json", output_file=temp_filename)
            
            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2  # Console + File
            
            # Check that one handler is FileHandler
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert "FileHandler" in handler_types
            assert "StreamHandler" in handler_types
    
    def test_setup_logging_with_context(self):
        """Test setting up logging with context."""
        context = {"service": "rag-system", "version": "1.0"}
        
        with patch('sys.stdout', new_callable=StringIO):
            setup_logging(level="INFO", format_type="json", context=context)
            
            root_logger = logging.getLogger()
            handler = root_logger.handlers[0]
            
            # Check that context filter was added
            filters = handler.filters
            assert len(filters) == 1
            assert isinstance(filters[0], ContextFilter)
    
    @patch('enhanced_rag.utils.logging._configure_library_loggers')
    def test_setup_logging_configures_libraries(self, mock_configure):
        """Test that setup_logging calls library configuration."""
        with patch('sys.stdout', new_callable=StringIO):
            setup_logging()
            
        mock_configure.assert_called_once()


class TestLibraryLoggerConfiguration:
    """Test library logger configuration."""
    
    def test_configure_library_loggers(self):
        """Test that library loggers are configured properly."""
        _configure_library_loggers()
        
        # Check specific library loggers
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("requests").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("faiss").level == logging.WARNING


class TestGetLogger:
    """Test get_logger functionality."""
    
    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test.logger")
        
        assert logger.name == "test.logger"
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger_with_context(self):
        """Test logger creation with context."""
        context = {"service": "test-service"}
        logger = get_logger("test.logger", context=context)
        
        assert logger.name == "test.logger"
        assert len(logger.filters) == 1
        assert isinstance(logger.filters[0], ContextFilter)


class TestLoggingContext:
    """Test LoggingContext context manager."""
    
    def test_logging_context_manager(self):
        """Test LoggingContext as context manager."""
        logger = logging.getLogger("test.logger")
        initial_filter_count = len(logger.filters)
        
        with LoggingContext(logger, correlation_id="test-123"):
            # Inside context, filter should be added
            assert len(logger.filters) == initial_filter_count + 1
            assert isinstance(logger.filters[-1], ContextFilter)
        
        # After context, filter should be removed
        assert len(logger.filters) == initial_filter_count
    
    def test_logging_context_with_exception(self):
        """Test LoggingContext cleanup when exception occurs."""
        logger = logging.getLogger("test.logger")
        initial_filter_count = len(logger.filters)
        
        try:
            with LoggingContext(logger, correlation_id="test-123"):
                assert len(logger.filters) == initial_filter_count + 1
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Filter should still be removed after exception
        assert len(logger.filters) == initial_filter_count


class TestPerformanceLogger:
    """Test PerformanceLogger functionality."""
    
    def setup_method(self):
        """Setup test logger."""
        self.logger = Mock(spec=logging.Logger)
        self.perf_logger = PerformanceLogger(self.logger)
    
    def test_log_operation_time(self):
        """Test logging operation time."""
        self.perf_logger.log_operation_time(
            operation="test_operation",
            duration_ms=1500.5,
            extra_field="extra_value"
        )
        
        self.logger.info.assert_called_once()
        call_args = self.logger.info.call_args
        
        assert "test_operation" in call_args[0][0]
        extra_data = call_args[1]["extra"]
        assert extra_data["operation"] == "test_operation"
        assert extra_data["duration_ms"] == 1500.5
        assert extra_data["performance_metric"] is True
        assert extra_data["extra_field"] == "extra_value"
    
    def test_log_token_usage(self):
        """Test logging token usage."""
        self.perf_logger.log_token_usage(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            request_id="req-123"
        )
        
        self.logger.info.assert_called_once()
        call_args = self.logger.info.call_args
        
        extra_data = call_args[1]["extra"]
        assert extra_data["model"] == "gpt-4"
        assert extra_data["prompt_tokens"] == 100
        assert extra_data["completion_tokens"] == 50
        assert extra_data["total_tokens"] == 150
        assert extra_data["token_usage_metric"] is True
        assert extra_data["request_id"] == "req-123"
    
    def test_log_embedding_batch(self):
        """Test logging embedding batch metrics."""
        self.perf_logger.log_embedding_batch(
            model="text-embedding-ada-002",
            batch_size=50,
            duration_ms=2000.0,
            batch_id="batch-456"
        )
        
        self.logger.info.assert_called_once()
        call_args = self.logger.info.call_args
        
        extra_data = call_args[1]["extra"]
        assert extra_data["model"] == "text-embedding-ada-002"
        assert extra_data["batch_size"] == 50
        assert extra_data["duration_ms"] == 2000.0
        assert extra_data["embeddings_per_second"] == 25.0  # 50 / 2
        assert extra_data["embedding_metric"] is True
        assert extra_data["batch_id"] == "batch-456"


class TestErrorLogging:
    """Test error logging utilities."""
    
    def test_log_error_with_context(self):
        """Test logging error with context."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error message")
        
        log_error_with_context(
            logger=logger,
            error=error,
            operation="test_operation",
            user_id="user-123",
            request_id="req-456"
        )
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        
        assert "test_operation" in call_args[0][0]
        assert "Test error message" in call_args[0][0]
        
        extra_data = call_args[1]["extra"]
        assert extra_data["operation"] == "test_operation"
        assert extra_data["error_type"] == "ValueError"
        assert extra_data["error_message"] == "Test error message"
        assert extra_data["error_occurred"] is True
        assert extra_data["user_id"] == "user-123"
        assert extra_data["request_id"] == "req-456"
        
        assert call_args[1]["exc_info"] is True


class TestAuditLogger:
    """Test AuditLogger functionality."""
    
    def setup_method(self):
        """Setup test audit logger."""
        with patch('logging.getLogger') as mock_get_logger:
            self.mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = self.mock_logger
            self.audit_logger = AuditLogger("test.audit")
    
    def test_audit_logger_init(self):
        """Test AuditLogger initialization."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            audit_logger = AuditLogger("custom.audit")
            mock_get_logger.assert_called_with("custom.audit")
            assert audit_logger.logger == mock_logger
    
    def test_audit_logger_default_name(self):
        """Test AuditLogger with default name."""
        with patch('logging.getLogger') as mock_get_logger:
            AuditLogger()
            mock_get_logger.assert_called_with("audit")
    
    def test_log_document_ingestion(self):
        """Test logging document ingestion event."""
        self.audit_logger.log_document_ingestion(
            document_id="doc-123",
            source="/path/to/document.pdf",
            chunks_created=5,
            file_size=1024,
            content_type="pdf"
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        assert "doc-123" in call_args[0][0]
        
        extra_data = call_args[1]["extra"]
        assert extra_data["event_type"] == "document_ingestion"
        assert extra_data["document_id"] == "doc-123"
        assert extra_data["source"] == "/path/to/document.pdf"
        assert extra_data["chunks_created"] == 5
        assert extra_data["audit_event"] is True
        assert extra_data["file_size"] == 1024
        assert extra_data["content_type"] == "pdf"
    
    def test_log_query_event(self):
        """Test logging query processing event."""
        query_text = "What is machine learning?"
        
        self.audit_logger.log_query_event(
            correlation_id="corr-789",
            query=query_text,
            retrieved_chunks=3,
            processing_time_ms=1250.5,
            user_id="user-123",
            model="gpt-4"
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        assert "corr-789" in call_args[0][0]
        
        extra_data = call_args[1]["extra"]
        assert extra_data["event_type"] == "query_processed"
        assert extra_data["correlation_id"] == "corr-789"
        assert extra_data["query_length"] == len(query_text)
        assert extra_data["retrieved_chunks"] == 3
        assert extra_data["processing_time_ms"] == 1250.5
        assert extra_data["audit_event"] is True
        assert extra_data["user_id"] == "user-123"
        assert extra_data["model"] == "gpt-4"
    
    def test_log_system_event(self):
        """Test logging system events."""
        self.audit_logger.log_system_event(
            event_type="system_startup",
            description="RAG system started successfully",
            version="1.0.0",
            environment="production"
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args
        
        assert "system_startup" in call_args[0][0]
        
        extra_data = call_args[1]["extra"]
        assert extra_data["event_type"] == "system_startup"
        assert extra_data["description"] == "RAG system started successfully"
        assert extra_data["audit_event"] is True
        assert extra_data["system_event"] is True
        assert extra_data["version"] == "1.0.0"
        assert extra_data["environment"] == "production"


class TestLoggingIntegration:
    """Integration tests for logging components."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_complete_logging_setup_and_usage(self):
        """Test complete logging setup and usage."""
        # Setup logging
        output = StringIO()
        with patch('sys.stdout', output):
            setup_logging(
                level="INFO",
                format_type="json",
                context={"service": "rag-test"}
            )
            
            # Get logger and log message
            logger = get_logger("test.integration")
            logger.info("Test integration message", extra={"operation": "test"})
        
        # Parse JSON output
        log_output = output.getvalue().strip()
        if log_output:
            # Skip the setup message and get the test message
            lines = log_output.split('\n')
            test_log_line = lines[-1] if len(lines) > 1 else lines[0]
            
            try:
                parsed = json.loads(test_log_line)
                assert parsed["message"] == "Test integration message"
                assert parsed["logger"] == "test.integration"
                assert parsed["service"] == "rag-test"
                assert parsed["operation"] == "test"
            except json.JSONDecodeError:
                # If not JSON, check it's at least logging
                assert "Test integration message" in test_log_line
    
    def test_performance_and_audit_logging_together(self):
        """Test performance and audit logging working together."""
        output = StringIO()
        with patch('sys.stdout', output):
            setup_logging(level="INFO", format_type="json")
            
            # Setup loggers
            perf_logger = PerformanceLogger(logging.getLogger("performance"))
            audit_logger = AuditLogger("audit")
            
            # Log performance metric
            perf_logger.log_operation_time("test_operation", 500.0)
            
            # Log audit event
            audit_logger.log_system_event("test_event", "Test description")
        
        log_output = output.getvalue()
        assert "test_operation" in log_output
        assert "test_event" in log_output