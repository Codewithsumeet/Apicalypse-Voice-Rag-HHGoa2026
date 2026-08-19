"""
Retry logic with exponential backoff.

Used by the harness to retry failed API calls (STT, LLM, etc.)
before falling back or returning an error.
"""

import asyncio
import structlog
from functools import wraps
from typing import Callable, Any

logger = structlog.get_logger(__name__)


async def with_retry(
    func: Callable,
    *args,
    max_retries: int = 2,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    **kwargs,
) -> Any:
    """
    Execute an async function with exponential backoff retry.

    Args:
        func: The async function to call.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries (seconds).
        max_delay: Maximum delay between retries (seconds).
        backoff_factor: Multiplier for delay after each retry.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                logger.warning(
                    "retry_attempt",
                    function=func.__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay_s=round(delay, 3),
                    error=str(e),
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    total_attempts=max_retries + 1,
                    final_error=str(e),
                )

    raise last_exception
