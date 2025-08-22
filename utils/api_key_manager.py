import os
import logging
import time
from typing import Optional, Dict, Any
from supabase import Client
import threading
from utils.supabase_client import get_supabase_client # Import from the new centralized module

logger = logging.getLogger(__name__)

# In-memory cache for API keys
# Structure: { "agent_name:service_name": {"key": "...", "expiry": timestamp} }
API_KEY_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()
CACHE_TTL_SECONDS = 900  # 15 minutes

def get_api_key(agent_name: str, service_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given agent and service.
    1. Checks a local cache.
    2. If not cached, queries the 'agent_api_keys' table in Supabase.
    3. If no custom key is found in the DB, falls back to the environment variable.
    4. Caches the result (DB key or fallback) to speed up subsequent requests.

    Args:
        agent_name (str): The name of the agent.
        service_name (str): The service name ('anthropic', 'openai', or 'google').

    Returns:
        str: The appropriate API key, or None if not found.
    """
    cache_key = f"{agent_name}:{service_name}"
    fallback_env_var = ""
    if service_name == 'anthropic':
        fallback_env_var = 'ANTHROPIC_API_KEY'
    elif service_name == 'openai':
        fallback_env_var = 'OPENAI_API_KEY'
    elif service_name == 'google':
        fallback_env_var = 'GOOGLE_API_KEY'
    elif service_name == 'groq':
        fallback_env_var = 'GROQ_API_KEY'
    else:
        raise ValueError(f"Unsupported service_name: {service_name}")
        
    fallback_key = os.getenv(fallback_env_var)

    # 1. Check cache
    with CACHE_LOCK:
        cached_entry = API_KEY_CACHE.get(cache_key)
        if cached_entry and cached_entry['expiry'] > time.time():
            logger.info(f"CACHE HIT for API key: {cache_key}")
            return cached_entry['key']

    # 2. Cache miss, attempt to query database
    logger.info(f"CACHE MISS for API key: {cache_key}. Querying database.")
    final_key = fallback_key # Default to fallback

    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.error(f"Cannot query for API key {cache_key}: Supabase client is not available. Using fallback.")
    else:
        try:
            # First, get the agent_id from the agent_name
            agent_res = supabase_client.table("agents").select("id").eq("name", agent_name).limit(1).execute()
            
            if agent_res.data:
                agent_id = agent_res.data[0]['id']
                
                # Now, query for the API key using the agent_id
                key_res = supabase_client.table("agent_api_keys") \
                    .select("api_key") \
                    .eq("agent_id", agent_id) \
                    .eq("service_name", service_name) \
                    .limit(1) \
                    .execute()

                if key_res.data:
                    custom_key = key_res.data[0]['api_key']
                    if custom_key:
                        final_key = custom_key
                        logger.info(f"Found custom '{service_name}' key for agent '{agent_name}'.")
                    else:
                        logger.warning(f"Found DB entry for agent '{agent_name}' but key is empty. Using fallback.")
                else:
                    logger.info(f"No custom '{service_name}' key found for agent '{agent_name}'. Using fallback.")
            else:
                logger.warning(f"Agent '{agent_name}' not found in DB. Using fallback key for '{service_name}'.")

        except Exception as e:
            logger.error(f"Error fetching API key for {cache_key} from Supabase: {e}", exc_info=True)
            # On DB error, fall back to the environment variable key
            final_key = fallback_key

    # 3. Update cache
    with CACHE_LOCK:
        API_KEY_CACHE[cache_key] = {
            "key": final_key,
            "expiry": time.time() + CACHE_TTL_SECONDS
        }
        # logger.debug(f"CACHE SET for API key: {cache_key}")

    if not final_key:
        logger.error(f"CRITICAL: API key for service '{service_name}' (agent: {agent_name}) is NOT SET, neither in DB nor in env var '{fallback_env_var}'.")
        return None

    return final_key
