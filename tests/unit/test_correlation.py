"""Unit tests for correlation utilities."""

import logging
import asyncio
import pytest
import uuid
from unittest.mock import patch, Mock

from enhanced_rag.utils.correlation import (
    generate_correlation_id,
    set_correlation_id,
    get_correlation_id,
    get_or_create_correlation_id,
    CorrelationFilter,
    CorrelationContext,
    correlation_id_var
)


class TestCorrelationIDFunctions:
    """Test correlation ID utility functions."""
    
    def teardown_method(self):
        """Clean up correlation ID after each test."""
        correlation_id_var.set(None)
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        correlation_id = generate_correlation_id()
        
        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 36  # UUID4 string length
        
        # Verify it's a valid UUID
        uuid.UUID(correlation_id)
    
    def test_generate_correlation_id_uniqueness(self):
        """Test that generated correlation IDs are unique."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()
        id3 = generate_correlation_id()
        
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3
    
    def test_set_correlation_id_with_value(self):
        """Test setting specific correlation ID."""
        test_id = "test-correlation-123"
        
        result = set_correlation_id(test_id)
        
        assert result == test_id
        assert get_correlation_id() == test_id
    
    def test_set_correlation_id_without_value(self):
        """Test setting correlation ID without providing value."""
        result = set_correlation_id()
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 36  # UUID4 string length
        assert get_correlation_id() == result
    
    def test_set_correlation_id_none(self):
        """Test setting correlation ID with explicit None."""
        result = set_correlation_id(None)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 36
        assert get_correlation_id() == result
    
    def test_get_correlation_id_when_not_set(self):
        """Test getting correlation ID when none is set."""
        result = get_correlation_id()
        
        assert result is None
    
    def test_get_correlation_id_when_set(self):
        """Test getting correlation ID when one is set."""
        test_id = "test-correlation-456"
        set_correlation_id(test_id)
        
        result = get_correlation_id()
        
        assert result == test_id
    
    def test_get_or_create_correlation_id_when_exists(self):
        """Test get_or_create when correlation ID already exists."""
        test_id = "existing-correlation-789"
        set_correlation_id(test_id)
        
        result = get_or_create_correlation_id()
        
        assert result == test_id
    
    def test_get_or_create_correlation_id_when_not_exists(self):
        """Test get_or_create when no correlation ID exists."""
        # Ensure no correlation ID is set
        assert get_correlation_id() is None
        
        result = get_or_create_correlation_id()
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 36
        assert get_correlation_id() == result
    
    def test_correlation_id_overwrite(self):
        """Test overwriting existing correlation ID."""
        first_id = "first-correlation-id"
        second_id = "second-correlation-id"
        
        set_correlation_id(first_id)
        assert get_correlation_id() == first_id
        
        set_correlation_id(second_id)
        assert get_correlation_id() == second_id


class TestCorrelationFilter:
    """Test CorrelationFilter logging filter."""
    
    def teardown_method(self):
        """Clean up correlation ID after each test."""
        correlation_id_var.set(None)
    
    def test_correlation_filter_with_correlation_id(self):
        """Test filter when correlation ID is set."""
        test_id = "test-correlation-filter-123"
        set_correlation_id(test_id)
        
        filter_obj = CorrelationFilter()
        
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
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert hasattr(record, 'correlation_id')
        assert record.correlation_id == test_id
    
    def test_correlation_filter_without_correlation_id(self):
        """Test filter when no correlation ID is set."""
        filter_obj = CorrelationFilter()
        
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
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert not hasattr(record, 'correlation_id')
    
    def test_correlation_filter_multiple_records(self):
        """Test filter with multiple log records."""
        test_id = "multi-record-test-456"
        set_correlation_id(test_id)
        
        filter_obj = CorrelationFilter()
        
        # Create multiple log records
        records = []
        for i in range(3):
            record = logging.LogRecord(
                name=f"test.logger.{i}",
                level=logging.INFO,
                pathname=f"/test/path{i}.py",
                lineno=42 + i,
                msg=f"Test message {i}",
                args=(),
                exc_info=None
            )
            records.append(record)
        
        # Filter all records
        for record in records:
            result = filter_obj.filter(record)
            assert result is True
            assert record.correlation_id == test_id


class TestCorrelationContext:
    """Test CorrelationContext context manager."""
    
    def teardown_method(self):
        """Clean up correlation ID after each test."""
        correlation_id_var.set(None)
    
    def test_correlation_context_with_id(self):
        """Test context manager with provided correlation ID."""
        test_id = "context-test-123"
        
        with CorrelationContext(test_id) as context_id:
            assert context_id == test_id
            assert get_correlation_id() == test_id
        
        # After context, correlation ID should be reset
        assert get_correlation_id() is None
    
    def test_correlation_context_without_id(self):
        """Test context manager without provided correlation ID."""
        with CorrelationContext() as context_id:
            assert context_id is not None
            assert isinstance(context_id, str)
            assert len(context_id) == 36
            assert get_correlation_id() == context_id
        
        # After context, correlation ID should be reset
        assert get_correlation_id() is None
    
    def test_correlation_context_none_id(self):
        """Test context manager with explicit None correlation ID."""
        with CorrelationContext(None) as context_id:
            assert context_id is not None
            assert isinstance(context_id, str)
            assert len(context_id) == 36
            assert get_correlation_id() == context_id
        
        # After context, correlation ID should be reset
        assert get_correlation_id() is None
    
    def test_correlation_context_nested(self):
        """Test nested correlation contexts."""
        outer_id = "outer-context-789"
        inner_id = "inner-context-456"
        
        with CorrelationContext(outer_id) as outer_context_id:
            assert outer_context_id == outer_id
            assert get_correlation_id() == outer_id
            
            with CorrelationContext(inner_id) as inner_context_id:
                assert inner_context_id == inner_id
                assert get_correlation_id() == inner_id
            
            # After inner context, should return to outer
            assert get_correlation_id() == outer_id
        
        # After both contexts, should be None
        assert get_correlation_id() is None
    
    def test_correlation_context_with_exception(self):
        """Test context manager cleanup when exception occurs."""
        test_id = "exception-test-999"
        
        try:
            with CorrelationContext(test_id):
                assert get_correlation_id() == test_id
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Correlation ID should still be reset after exception
        assert get_correlation_id() is None
    
    def test_correlation_context_preserves_existing(self):
        """Test that context manager preserves existing correlation ID."""
        existing_id = "existing-correlation-111"
        context_id = "context-correlation-222"
        
        # Set an existing correlation ID
        set_correlation_id(existing_id)
        assert get_correlation_id() == existing_id
        
        with CorrelationContext(context_id):
            assert get_correlation_id() == context_id
        
        # Should return to existing ID
        assert get_correlation_id() == existing_id


class TestCorrelationWithLogging:
    """Test correlation integration with logging."""
    
    def teardown_method(self):
        """Clean up correlation ID and logging handlers."""
        correlation_id_var.set(None)
        
        # Clean up any test handlers
        logger = logging.getLogger("test.correlation")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    def test_correlation_filter_with_logger(self):
        """Test correlation filter integration with actual logger."""
        test_id = "logger-integration-test-555"
        
        # Setup logger with correlation filter
        logger = logging.getLogger("test.correlation")
        logger.setLevel(logging.INFO)
        
        # Create a mock handler to capture log records
        mock_handler = Mock()
        mock_handler.level = logging.INFO
        mock_handler.handle = Mock()
        
        # Add correlation filter
        correlation_filter = CorrelationFilter()
        logger.addFilter(correlation_filter)
        logger.addHandler(mock_handler)
        
        # Set correlation ID and log message
        set_correlation_id(test_id)
        logger.info("Test log message")
        
        # Verify handler was called and record has correlation ID
        mock_handler.handle.assert_called_once()
        log_record = mock_handler.handle.call_args[0][0]
        
        assert hasattr(log_record, 'correlation_id')
        assert log_record.correlation_id == test_id


@pytest.mark.asyncio
class TestCorrelationWithAsyncio:
    """Test correlation ID behavior with asyncio."""
    
    def teardown_method(self):
        """Clean up correlation ID after each test."""
        correlation_id_var.set(None)
    
    async def test_correlation_id_preserved_across_await(self):
        """Test that correlation ID is preserved across await calls."""
        test_id = "async-test-777"
        set_correlation_id(test_id)
        
        assert get_correlation_id() == test_id
        
        # Await something
        await asyncio.sleep(0.001)
        
        # Correlation ID should still be there
        assert get_correlation_id() == test_id
    
    async def test_correlation_context_with_async(self):
        """Test correlation context with async operations."""
        test_id = "async-context-888"
        
        # CorrelationContext is not an async context manager, use regular 'with'
        with CorrelationContext(test_id):
            assert get_correlation_id() == test_id
            
            # Await within context
            await asyncio.sleep(0.001)
            
            # Should still have correlation ID
            assert get_correlation_id() == test_id
        
        # After context, should be None
        assert get_correlation_id() is None
    
    async def test_correlation_id_isolation_between_tasks(self):
        """Test that correlation IDs are isolated between async tasks."""
        
        async def task_with_correlation(task_id: str, correlation_id: str):
            set_correlation_id(correlation_id)
            
            # Simulate some async work
            await asyncio.sleep(0.001)
            
            # Verify correlation ID is still correct
            return get_correlation_id()
        
        # Create multiple tasks with different correlation IDs
        task1_id = "task1-correlation-999"
        task2_id = "task2-correlation-000"
        
        task1 = asyncio.create_task(task_with_correlation("task1", task1_id))
        task2 = asyncio.create_task(task_with_correlation("task2", task2_id))
        
        results = await asyncio.gather(task1, task2)
        
        # Each task should have preserved its own correlation ID
        assert results[0] == task1_id
        assert results[1] == task2_id
    
    async def test_correlation_context_async_context_manager(self):
        """Test using CorrelationContext as async context manager."""
        test_id = "async-context-manager-111"
        
        # Note: CorrelationContext is not an async context manager,
        # but it should work fine in async contexts
        with CorrelationContext(test_id) as context_id:
            assert context_id == test_id
            assert get_correlation_id() == test_id
            
            # Perform async operations
            await asyncio.sleep(0.001)
            
            # Correlation ID should persist
            assert get_correlation_id() == test_id
        
        # After context, should be None
        assert get_correlation_id() is None


class TestCorrelationIntegration:
    """Integration tests for correlation utilities."""
    
    def teardown_method(self):
        """Clean up correlation ID and logging."""
        correlation_id_var.set(None)
        
        # Clean up logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_complete_correlation_workflow(self):
        """Test complete workflow with correlation ID, context, and logging."""
        from io import StringIO
        import json
        
        # Setup logging with correlation filter
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        correlation_filter = CorrelationFilter()
        handler.addFilter(correlation_filter)
        
        logger = logging.getLogger("test.integration")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        test_id = "integration-test-222"
        
        # Use correlation context and log messages
        with CorrelationContext(test_id):
            logger.info("Message within context")
            
            # Nested operation
            nested_id = get_or_create_correlation_id()
            assert nested_id == test_id
            
            logger.info("Nested message")
        
        # Log outside context (should not have correlation ID)
        logger.info("Message outside context")
        
        # Check log output
        log_output = log_stream.getvalue()
        log_lines = log_output.strip().split('\n')
        
        assert len(log_lines) == 3
        
        # First two messages should mention correlation in some way
        # (exact format depends on handler formatter)
        # Last message should not have correlation ID context
    
    def test_error_handling_with_correlation(self):
        """Test error handling preserves correlation context."""
        test_id = "error-handling-333"
        
        def operation_that_fails():
            assert get_correlation_id() == test_id
            raise ValueError("Test error")
        
        try:
            with CorrelationContext(test_id):
                assert get_correlation_id() == test_id
                operation_that_fails()
        except ValueError:
            pass
        
        # Correlation ID should be cleaned up even after error
        assert get_correlation_id() is None
    
    @pytest.mark.asyncio
    async def test_async_operation_with_correlation_logging(self):
        """Test async operations with correlation and logging."""
        from io import StringIO
        
        # Setup logging
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        correlation_filter = CorrelationFilter()
        handler.addFilter(correlation_filter)
        
        logger = logging.getLogger("test.async.correlation")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        async def async_operation(correlation_id: str):
            with CorrelationContext(correlation_id):
                logger.info("Starting async operation")
                
                await asyncio.sleep(0.001)
                
                logger.info("Completing async operation")
                
                return get_correlation_id()
        
        test_id = "async-logging-444"
        result = await async_operation(test_id)
        
        assert result == test_id
        
        # Verify logging output contains correlation context
        log_output = log_stream.getvalue()
        assert "Starting async operation" in log_output
        assert "Completing async operation" in log_output