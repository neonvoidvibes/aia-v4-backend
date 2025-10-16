"""
Groq API Rate Limiter - Thread-safe rate limiting for concurrent MLP analysis requests.

This module provides:
1. Thread-safe semaphore-based concurrency control
2. Token bucket algorithm for RPM/TPM limits
3. Exponential backoff retry logic with 429 handling
4. Graceful degradation when rate limits are reached
"""

import os
import time
import logging
import threading
from typing import Optional, Callable, Any, Dict
from datetime import datetime, timezone
from collections import deque
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)


class GroqRateLimiter:
    """
    Thread-safe rate limiter for Groq API calls.

    Uses token bucket algorithm to enforce both RPM (requests per minute)
    and TPM (tokens per minute) limits.

    Features:
    - Semaphore-based concurrency control (max N simultaneous requests)
    - Token bucket for RPM limiting
    - Estimated TPM tracking (conservative estimation)
    - Thread-safe operation for concurrent access
    - Exponential backoff on rate limit errors
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        rpm_limit: int = 500,
        tpm_limit: int = 250000,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0
    ):
        """
        Initialize rate limiter.

        Args:
            max_concurrent: Maximum concurrent Groq requests (default: 10 for gpt-oss-120b)
            rpm_limit: Requests per minute limit (default: 500, very high for gpt-oss-120b)
            tpm_limit: Tokens per minute limit (default: 250K for gpt-oss-120b)
            max_retries: Maximum retry attempts on rate limit errors (default: 3)
            initial_retry_delay: Initial retry delay in seconds (doubles each retry)
        """
        self.max_concurrent = max_concurrent
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        # Semaphore for concurrency control
        self._semaphore = threading.Semaphore(max_concurrent)

        # Token buckets for rate limiting (RPM and TPM)
        self._request_timestamps = deque()  # Track request times for RPM
        self._token_usage = deque()  # Track (timestamp, token_count) for TPM

        # Locks for thread-safe bucket operations
        self._request_lock = threading.Lock()
        self._token_lock = threading.Lock()

        logger.info(
            f"GroqRateLimiter initialized: max_concurrent={max_concurrent}, "
            f"rpm_limit={rpm_limit}, tpm_limit={tpm_limit}"
        )

    def _clean_old_entries(self, bucket: deque, max_age_seconds: float = 60.0):
        """Remove entries older than max_age_seconds from bucket."""
        current_time = time.time()
        while bucket and (current_time - bucket[0][0] if isinstance(bucket[0], tuple) else current_time - bucket[0]) > max_age_seconds:
            bucket.popleft()

    def _can_make_request(self) -> tuple[bool, Optional[float]]:
        """
        Check if a request can be made without exceeding rate limits.

        Returns:
            Tuple of (can_proceed, wait_time_seconds)
            - can_proceed: True if request can be made now
            - wait_time_seconds: Recommended wait time if can_proceed is False
        """
        current_time = time.time()

        with self._request_lock:
            # Clean old request timestamps (older than 1 minute)
            self._clean_old_entries(self._request_timestamps, max_age_seconds=60.0)

            # Check RPM limit
            if len(self._request_timestamps) >= self.rpm_limit:
                # Calculate wait time until oldest request expires
                oldest_request = self._request_timestamps[0]
                wait_time = 60.0 - (current_time - oldest_request) + 0.1  # Add small buffer
                logger.debug(f"RPM limit reached ({len(self._request_timestamps)}/{self.rpm_limit}), wait {wait_time:.1f}s")
                return False, max(0, wait_time)

        with self._token_lock:
            # Clean old token usage entries
            self._clean_old_entries(self._token_usage, max_age_seconds=60.0)

            # Check TPM limit (conservative estimate: 4096 tokens per request)
            estimated_tokens_used = sum(count for _, count in self._token_usage)
            estimated_tokens_per_request = 4096  # Conservative estimate for analysis

            if estimated_tokens_used + estimated_tokens_per_request > self.tpm_limit:
                # Calculate wait time until enough tokens available
                if self._token_usage:
                    oldest_token_entry = self._token_usage[0]
                    wait_time = 60.0 - (current_time - oldest_token_entry[0]) + 0.1
                    logger.debug(
                        f"TPM limit approaching ({estimated_tokens_used}/{self.tpm_limit}), "
                        f"wait {wait_time:.1f}s"
                    )
                    return False, max(0, wait_time)

        return True, None

    def _record_request(self, estimated_tokens: int = 4096):
        """
        Record a request in the rate limiter buckets.

        Args:
            estimated_tokens: Estimated token count for this request (default: 4096)
        """
        current_time = time.time()

        with self._request_lock:
            self._request_timestamps.append(current_time)

        with self._token_lock:
            self._token_usage.append((current_time, estimated_tokens))

    def execute_with_rate_limit(
        self,
        groq_call: Callable[[], Any],
        context: str = "groq_call"
    ) -> Optional[Any]:
        """
        Execute a Groq API call with rate limiting and retry logic.

        Args:
            groq_call: Callable that makes the Groq API call
            context: Description of the call for logging (e.g., "mirror_analysis")

        Returns:
            Result from groq_call, or None if all retries failed
        """
        # Acquire semaphore (blocks if max concurrent reached)
        logger.debug(f"[{context}] Acquiring semaphore for concurrent control")
        with self._semaphore:
            # Check rate limits before proceeding
            can_proceed, wait_time = self._can_make_request()
            if not can_proceed and wait_time:
                logger.info(f"[{context}] Rate limit pre-check: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

            # Retry loop with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    # Record this request attempt
                    self._record_request()

                    # Execute the Groq call
                    logger.debug(f"[{context}] Executing Groq call (attempt {attempt + 1}/{self.max_retries})")
                    result = groq_call()

                    logger.debug(f"[{context}] Groq call succeeded")
                    return result

                except RateLimitError as rate_err:
                    error_msg = str(rate_err)
                    logger.warning(
                        f"[{context}] Groq rate limit hit (attempt {attempt + 1}/{self.max_retries}): "
                        f"{error_msg}"
                    )

                    # If this is the last attempt, give up
                    if attempt == self.max_retries - 1:
                        logger.error(
                            f"[{context}] Groq rate limit exceeded after {self.max_retries} attempts. "
                            "Analysis will be skipped."
                        )
                        return None

                    # Exponential backoff: 1s, 2s, 4s, ...
                    wait_time = self.initial_retry_delay * (2 ** attempt)
                    logger.info(f"[{context}] Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue

                except Exception as e:
                    logger.error(f"[{context}] Unexpected error in Groq call: {e}", exc_info=True)
                    return None

        # Should never reach here
        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current usage statistics
        """
        current_time = time.time()

        with self._request_lock:
            self._clean_old_entries(self._request_timestamps)
            current_rpm = len(self._request_timestamps)

        with self._token_lock:
            self._clean_old_entries(self._token_usage)
            current_tpm = sum(count for _, count in self._token_usage)

        return {
            'current_rpm': current_rpm,
            'rpm_limit': self.rpm_limit,
            'rpm_utilization': f"{(current_rpm / self.rpm_limit) * 100:.1f}%",
            'current_tpm': current_tpm,
            'tpm_limit': self.tpm_limit,
            'tpm_utilization': f"{(current_tpm / self.tpm_limit) * 100:.1f}%",
            'max_concurrent': self.max_concurrent
        }


# Global rate limiter instance (singleton pattern)
# Configured based on environment variables for flexibility
_GLOBAL_RATE_LIMITER: Optional[GroqRateLimiter] = None
_RATE_LIMITER_LOCK = threading.Lock()


def get_groq_rate_limiter() -> GroqRateLimiter:
    """
    Get or create the global Groq rate limiter instance.

    Returns:
        Global GroqRateLimiter instance
    """
    global _GLOBAL_RATE_LIMITER

    if _GLOBAL_RATE_LIMITER is None:
        with _RATE_LIMITER_LOCK:
            # Double-check pattern for thread safety
            if _GLOBAL_RATE_LIMITER is None:
                # Read configuration from environment variables
                # Defaults tuned for gpt-oss-120b (250K TPM, very high RPM/RPD)
                max_concurrent = int(os.getenv('GROQ_MAX_CONCURRENT', '10'))
                rpm_limit = int(os.getenv('GROQ_RPM_LIMIT', '500'))
                tpm_limit = int(os.getenv('GROQ_TPM_LIMIT', '250000'))
                max_retries = int(os.getenv('GROQ_MAX_RETRIES', '3'))

                _GLOBAL_RATE_LIMITER = GroqRateLimiter(
                    max_concurrent=max_concurrent,
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                    max_retries=max_retries
                )

                logger.info("Global Groq rate limiter initialized")

    return _GLOBAL_RATE_LIMITER


def reset_groq_rate_limiter():
    """Reset the global rate limiter (useful for testing)."""
    global _GLOBAL_RATE_LIMITER
    with _RATE_LIMITER_LOCK:
        _GLOBAL_RATE_LIMITER = None
        logger.info("Global Groq rate limiter reset")
