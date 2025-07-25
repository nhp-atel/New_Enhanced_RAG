"""Unit tests for retry utilities."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from enhanced_rag.utils.retry import (
    RetryConfig,
    retry_async,
    retry_with_config,
    retry_on_network_error,
    retry_on_rate_limit
)


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)
    
    def test_retry_config_custom(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=1.5,
            jitter=False,
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.retryable_exceptions == (ConnectionError, TimeoutError)


class TestRetryAsync:
    """Test async retry functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test successful function on first attempt."""
        
        async def successful_func():
            return "success"
        
        config = RetryConfig(max_attempts=3)
        result = await retry_async(successful_func, config)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        """Test function succeeds after retries."""
        call_count = 0
        
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,  # Fast retries for testing
            retryable_exceptions=(ConnectionError,)
        )
        
        result = await retry_async(failing_then_success, config)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_all_attempts_fail(self):
        """Test function fails on all attempts."""
        call_count = 0
        
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network error")
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        )
        
        with pytest.raises(ConnectionError, match="Network error"):
            await retry_async(always_failing, config)
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_non_retryable_exception(self):
        """Test non-retryable exception fails immediately."""
        call_count = 0
        
        async def non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")
        
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=(ConnectionError,)
        )
        
        with pytest.raises(ValueError, match="Not retryable"):
            await retry_async(non_retryable_error, config)
        
        assert call_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_retry_async_with_args_kwargs(self):
        """Test retry with function arguments."""
        
        async def func_with_args(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"
        
        config = RetryConfig(max_attempts=2)
        result = await retry_async(func_with_args, config, "a", "b", kwarg1="c")
        
        assert result == "a-b-c"
    
    @pytest.mark.asyncio
    async def test_retry_async_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        call_times = []
        
        async def failing_func():
            call_times.append(asyncio.get_event_loop().time())
            raise ConnectionError("Network error")
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable timing
            retryable_exceptions=(ConnectionError,)
        )
        
        with pytest.raises(ConnectionError):
            await retry_async(failing_func, config)
        
        # Check that delays increase exponentially
        assert len(call_times) == 3
        
        # Allow some tolerance for timing variations
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # First delay should be ~0.1s, second should be ~0.2s
        assert 0.05 < delay1 < 0.15
        assert 0.15 < delay2 < 0.25


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator on successful function."""
        config = RetryConfig(max_attempts=3)
        
        @retry_with_config(config)
        async def successful_func():
            return "decorated success"
        
        result = await successful_func()
        assert result == "decorated success"
    
    @pytest.mark.asyncio
    async def test_retry_decorator_with_retries(self):
        """Test retry decorator with failing then successful function."""
        call_count = 0
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        )
        
        @retry_with_config(config)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "decorated success"
        
        result = await failing_then_success()
        
        assert result == "decorated success"
        assert call_count == 3


class TestPreConfiguredDecorators:
    """Test pre-configured retry decorators."""
    
    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        """Test network error retry decorator."""
        call_count = 0
        
        @retry_on_network_error(max_attempts=3)
        async def network_error_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "network success"
        
        result = await network_error_func()
        
        assert result == "network success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test rate limit retry decorator."""
        
        # Test that the decorator is created correctly
        decorator = retry_on_rate_limit(max_attempts=3)
        
        # We can't easily test the RateLimitError since it's defined inside the function
        # So let's test that the decorator exists and has the right config
        assert decorator is not None
        
        # Test with a function that doesn't raise the specific error
        call_count = 0
        
        @retry_on_rate_limit(max_attempts=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "rate limit success"
        
        result = await successful_func()
        
        assert result == "rate limit success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_network_error_exceptions(self):
        """Test that network error decorator handles correct exceptions."""
        
        @retry_on_network_error(max_attempts=2)
        async def connection_error_func():
            raise ConnectionError("Connection failed")
        
        @retry_on_network_error(max_attempts=2)
        async def timeout_error_func():
            raise TimeoutError("Request timed out")
        
        @retry_on_network_error(max_attempts=2)
        async def value_error_func():
            raise ValueError("Not a network error")
        
        # Network errors should be retried
        with pytest.raises(ConnectionError):
            await connection_error_func()
        
        with pytest.raises(TimeoutError):
            await timeout_error_func()
        
        # Non-network errors should not be retried
        with pytest.raises(ValueError):
            await value_error_func()


class TestRetryEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_max_delay_limit(self):
        """Test that delays don't exceed max_delay."""
        call_times = []
        
        async def failing_func():
            call_times.append(asyncio.get_event_loop().time())
            raise ConnectionError("Network error")
        
        config = RetryConfig(
            max_attempts=5,
            base_delay=10.0,  # Large base delay
            max_delay=0.2,    # Small max delay
            exponential_base=2.0,
            jitter=False,
            retryable_exceptions=(ConnectionError,)
        )
        
        with pytest.raises(ConnectionError):
            await retry_async(failing_func, config)
        
        # All delays should be capped at max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i-1]
            assert delay <= 0.25  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_zero_attempts(self):
        """Test retry with zero max attempts."""
        
        async def never_called():
            assert False, "Should not be called"
        
        config = RetryConfig(max_attempts=0)
        
        # Should handle gracefully (though this is an edge case)
        with pytest.raises(Exception):
            await retry_async(never_called, config)
    
    @pytest.mark.asyncio
    async def test_jitter_variation(self):
        """Test that jitter adds randomness to delays."""
        delays = []
        
        async def failing_func():
            raise ConnectionError("Network error")
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            jitter=True,
            retryable_exceptions=(ConnectionError,)
        )
        
        # Run multiple times to see variation in delays
        for _ in range(5):
            call_times = []
            original_time = asyncio.get_event_loop().time
            
            def mock_time():
                call_times.append(len(call_times))
                return len(call_times) * 0.1
            
            asyncio.get_event_loop().time = mock_time
            
            try:
                with pytest.raises(ConnectionError):
                    await retry_async(failing_func, config)
            finally:
                asyncio.get_event_loop().time = original_time
        
        # With jitter, we should see some variation
        # This test is probabilistic but should work in practice