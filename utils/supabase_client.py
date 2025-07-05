import os
import logging
import threading
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

supabase_client: Optional[Client] = None
supabase_lock = threading.Lock()

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
