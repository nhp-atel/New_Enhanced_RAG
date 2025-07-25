"""Retry utilities with exponential backoff."""

import asyncio
import logging
import random
from typing import TypeVar, Callable, Any, Optional, Type, Union
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)


async def retry_async(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        config: Retry configuration
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts - 1:
                logger.error(
                    f"Function {func.__name__} failed after {config.max_attempts} attempts",
                    extra={"exception": str(e), "attempt": attempt + 1}
                )
                raise e
            
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            if config.jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(
                f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {delay:.2f}s",
                extra={"exception": str(e), "attempt": attempt + 1, "delay": delay}
            )
            
            await asyncio.sleep(delay)
    
    raise last_exception


def retry_with_config(config: RetryConfig):
    """Decorator for async functions with retry logic."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(func, config, *args, **kwargs)
        return wrapper
    
    return decorator


# Pre-configured retry decorators
def retry_on_network_error(max_attempts: int = 3):
    """Retry decorator for network-related errors."""
    import aiohttp
    import httpx
    
    network_exceptions = (
        aiohttp.ClientError,
        httpx.HTTPError,
        ConnectionError,
        TimeoutError,
    )
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=network_exceptions
    )
    
    return retry_with_config(config)


def retry_on_rate_limit(max_attempts: int = 5):
    """Retry decorator for rate limiting errors."""
    
    class RateLimitError(Exception):
        pass
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=120.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(RateLimitError,)
    )
    
    return retry_with_config(config)