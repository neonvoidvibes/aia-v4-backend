import os
import logging
import threading
import time
import random
from typing import Optional, Callable, TypeVar, Iterable

import httpx
import httpcore
from supabase import create_client, Client
from gotrue.errors import AuthRetryableError

T = TypeVar("T")

logger = logging.getLogger(__name__)

supabase_client: Optional[Client] = None
supabase_lock = threading.Lock()

# Subset of network exceptions that indicate the underlying HTTP/2 session was torn down.
_RETRYABLE_ERROR_TYPES = (
    httpx.RequestError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpcore.RemoteProtocolError,
    ConnectionError,
    OSError,
    AuthRetryableError,
)

_RETRYABLE_ERROR_STRINGS = (
    "Server disconnected",
    "Connection closed",
    "Connection reset",
    "Connection aborted",
    "ConnectionTerminated",
    "Connection broken",
)


class SupabaseUnavailableError(RuntimeError):
    """Raised when the Supabase client is unavailable."""


def _iter_exception_chain(exc: Exception) -> Iterable[Exception]:
    """Yield an exception and its causes/contexts to simplify checks."""
    seen: set[int] = set()
    current: Optional[Exception] = exc
    while current and id(current) not in seen:
        yield current
        seen.add(id(current))
        if current.__cause__:
            current = current.__cause__  # Prefer direct cause if present
        elif current.__context__:
            current = current.__context__
        else:
            break


def _should_reset_supabase_client(exc: Exception) -> bool:
    """Return True when an exception indicates a broken Supabase connection."""
    for item in _iter_exception_chain(exc):
        if isinstance(item, _RETRYABLE_ERROR_TYPES):
            return True
        message = str(item) if item else ""
        if message:
            lowered = message.lower()
            if any(marker.lower() in lowered for marker in _RETRYABLE_ERROR_STRINGS):
                return True
    return False


def is_supabase_disconnect_error(exc: Exception) -> bool:
    """Public helper so callers can detect connection resets."""
    return _should_reset_supabase_client(exc)


def mark_supabase_client_stale(reason: str) -> None:
    """
    Mark the cached Supabase client as stale, forcing the next call to re-initialize it.
    This is used when httpx/httpcore tears down the underlying HTTP/2 connection.
    """
    global supabase_client
    with supabase_lock:
        if supabase_client is None:
            return
        logger.warning("Supabase client marked stale (%s); resetting session.", reason)
        try:
            session = getattr(getattr(supabase_client, "auth", None), "session", None)
            if session and hasattr(session, "close") and not getattr(session, "_is_closed", False):
                session.close()
        except Exception as close_exc:
            logger.debug("Error while closing Supabase session: %s", close_exc)
        supabase_client = None


def execute_supabase_operation(
    operation: Callable[[Client], T],
    *,
    context: str,
    max_attempts: int = 3,
    base_delay: float = 0.35,
) -> T:
    """
    Execute a Supabase operation with automatic client reset and retry when we detect
    HTTP/2 disconnects. The callable receives a fresh `Client` instance per attempt.
    """
    attempt = 0
    last_exc: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        client = get_supabase_client()
        if client is None:
            raise SupabaseUnavailableError("Supabase client is not available")
        try:
            return operation(client)
        except Exception as exc:  # noqa: BLE001 - we intentionally inspect and re-raise
            last_exc = exc
            if not _should_reset_supabase_client(exc):
                raise
            mark_supabase_client_stale(f"{context} attempt {attempt} failed: {exc}")
            if attempt >= max_attempts:
                raise
            sleep_for = base_delay * (2 ** (attempt - 1))
            sleep_for += random.uniform(0.0, 0.15)
            logger.warning(
                "Supabase %s retry %s/%s after %s (sleep %.2fs)",
                context,
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    # If we exhausted attempts without returning, raise the last exception to preserve behaviour.
    if last_exc:
        raise last_exc
    raise RuntimeError(f"execute_supabase_operation({context}) failed without raising an exception")

def get_supabase_client() -> Optional[Client]:
    """
    Gets a thread-safe, resilient Supabase client.

    This function ensures the client is initialized and attempts to recover
    from stale connections by checking the health of the underlying session.
    """
    global supabase_client
    with supabase_lock:
        # Health check: If the client exists, check if its session is closed.
        # The `auth.session` is a good proxy for the health of the httpx client pool.
        # If `_is_closed` is True, it means the client can no longer make requests.
        if supabase_client and hasattr(supabase_client.auth, 'session') and supabase_client.auth.session._is_closed:
            logger.warning("Supabase client session is closed. Forcing re-initialization.")
            supabase_client = None # Force re-initialization

        if supabase_client is None:
            logger.info("Supabase client is None or was stale, attempting to initialize.")
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                if not supabase_url or not supabase_key:
                    logger.error("Cannot initialize Supabase client: URL or Key is missing.")
                    return None
                
                # Create a new client instance
                supabase_client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully.")

            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                supabase_client = None # Ensure it's None on failure
                
    return supabase_client
