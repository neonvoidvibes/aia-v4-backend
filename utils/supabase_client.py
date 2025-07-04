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
    Gets a thread-safe, resilient Supabase client, re-initializing if necessary.
    """
    global supabase_client
    with supabase_lock:
        # Check if the client is None or if the session might be stale/closed
        if supabase_client is None:
            logger.info("Supabase client is None, attempting to initialize.")
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                if not supabase_url or not supabase_key:
                    logger.error("Cannot initialize Supabase client: URL or Key is missing.")
                    return None
                supabase_client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                supabase_client = None # Ensure it's None on failure
    return supabase_client

# Initial connection attempt at startup
get_supabase_client()
