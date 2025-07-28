# utils/s3_utils.py
"""Utilities for interacting with AWS S3."""

import os
import boto3
from botocore.config import Config as BotoConfig # For S3 timeouts
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Callable
import threading # For s3_client_lock
from werkzeug.utils import secure_filename

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global S3 client instance (lazy loaded)
s3_client_instance = None # Renamed to avoid confusion with boto3.client
s3_client_lock = threading.Lock()

# Global Cache for S3 files
S3_FILE_CACHE: Dict[str, Dict[str, Any]] = {}
S3_CACHE_LOCK = threading.Lock()
S3_CACHE_TTL_DEFAULT = timedelta(minutes=120)
S3_CACHE_TTL_NOT_FOUND = timedelta(minutes=1)

def get_s3_client(config: Optional[BotoConfig] = None) -> Optional[boto3.client]:
    """
    Initializes and returns an S3 client, or None on failure.
    Reads env vars inside the function.
    If a BotoConfig is provided, it's used for this specific client instance (not cached globally).
    The global client is initialized without specific config for general use.
    """
    global s3_client_instance
    
    if config: # If a specific config is requested, create a new client with it
        aws_region_cfg = os.getenv('AWS_REGION')
        if not aws_region_cfg:
            logger.error("AWS_REGION env var not found for S3 client with custom config.")
            return None
        try:
            logger.debug(f"Creating new S3 client with custom config for region {aws_region_cfg}")
            return boto3.client('s3', region_name=aws_region_cfg, config=config)
        except Exception as e:
            logger.error(f"Failed to initialize S3 client with custom config: {e}", exc_info=True)
            return None

    # For global instance (no specific config)
    with s3_client_lock:
        if s3_client_instance is None:
            aws_region_global = os.getenv('AWS_REGION')
            aws_s3_bucket_global = os.getenv('AWS_S3_BUCKET')

            if not aws_region_global or not aws_s3_bucket_global:
                logger.error("Global S3 Client: AWS_REGION or AWS_S3_BUCKET env vars not found.")
                return None
            try:
                # Default config for the global client
                default_boto_config = BotoConfig(
                    connect_timeout=15, # Increased connect timeout
                    read_timeout=45,    # Increased read timeout
                    retries={'max_attempts': 3}
                )
                s3_client_instance = boto3.client(
                    's3',
                    region_name=aws_region_global,
                    config=default_boto_config
                )
                logger.info(f"Global S3 client initialized for region {aws_region_global} with default timeouts.")
            except Exception as e:
                logger.error(f"Failed to initialize global S3 client: {e}", exc_info=True)
                s3_client_instance = None
    return s3_client_instance

def get_cached_s3_file(
    cache_key: str,
    fetch_function: Callable[[], Optional[Tuple[str, str]]],
    description: str
    ) -> Optional[str]:
    """
    Generic caching wrapper for S3 file content fetching functions.
    """
    # Check cache first
    with S3_CACHE_LOCK:
        cached_item = S3_FILE_CACHE.get(cache_key)
        if cached_item and cached_item['expiry'] > datetime.now(timezone.utc):
            logger.info(f"CACHE HIT for {description} ('{cache_key}')")
            return cached_item['content']

    # If not in cache or expired, fetch from S3
    logger.info(f"CACHE MISS for {description} ('{cache_key}')")
    result = fetch_function()
    content = result[1] if result else None

    # Update cache
    with S3_CACHE_LOCK:
        if content is not None:
            S3_FILE_CACHE[cache_key] = {
                'content': content,
                'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_DEFAULT
            }
            logger.info(f"CACHE SET for {description} ('{cache_key}').")
        else:
            # Cache "not found" to avoid repeated lookups for a short time
            S3_FILE_CACHE[cache_key] = {
                'content': None,
                'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_NOT_FOUND
            }
            logger.info(f"CACHE SET (None) for {description} ('{cache_key}').")
    
    return content


def read_file_content(file_key: str, description: str) -> Optional[str]:
    """Read content from S3 file, handling potential errors."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket name not available for reading {description}.")
        return None
    try:
        logger.debug(f"Reading {description} from S3: s3://{aws_s3_bucket}/{file_key}")
        response = s3.get_object(Bucket=aws_s3_bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        logger.debug(f"Successfully read {description} ({len(content)} chars)")
        return content
    except s3.exceptions.NoSuchKey:
        logger.warning(f"{description} file not found at S3 key: {file_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading {description} from {file_key}: {e}", exc_info=True)
        return None


def list_s3_objects_metadata(base_key_prefix: str) -> List[Dict[str, Any]]:
    """Lists objects under a given S3 prefix, returning their Key, Size, and LastModified."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not available for listing objects.")
        return []

    objects_metadata = []
    try:
        logger.info(f"S3 LIST [BEGIN]: Bucket='{aws_s3_bucket}', Prefix='{base_key_prefix}'")
        paginator = s3.get_paginator('list_objects_v2')
        raw_object_count = 0
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=base_key_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    raw_object_count += 1
                    object_key = obj['Key']
                    logger.debug(f"S3 LIST [RAW_OBJ]: Key='{object_key}', Size={obj.get('Size', 0)}")

                    if object_key == base_key_prefix:
                        logger.debug(f"S3 LIST [SKIP_FOLDER_OBJ]: Skipped object key same as prefix (folder itself): {object_key}")
                        continue
                    
                    relative_path = object_key[len(base_key_prefix):]
                    
                    if '/' in relative_path:
                        logger.debug(f"S3 LIST [SKIP_SUB_DIR_OBJ]: Skipped object in subdirectory. Key='{object_key}', Relative='{relative_path}', Prefix='{base_key_prefix}'")
                        continue
                    
                    if not relative_path and obj.get('Size', 0) == 0 and object_key.endswith('/'):
                         logger.debug(f"S3 LIST [SKIP_EMPTY_RELATIVE_FOLDER_MARKER]: Skipped empty relative path folder marker: {object_key}")
                         continue
                    
                    objects_metadata.append({
                        'Key': object_key,
                        'Size': obj.get('Size', 0),
                        'LastModified': obj.get('LastModified')
                    })
        logger.info(f"S3 LIST [SUMMARY]: Bucket='{aws_s3_bucket}', Prefix='{base_key_prefix}'. Raw S3 objects found: {raw_object_count}. Filtered objects_metadata count: {len(objects_metadata)}.")
        if raw_object_count > 0 and not objects_metadata:
             logger.warning(f"S3 LIST [WARN]: All {raw_object_count} raw S3 objects were filtered out for prefix '{base_key_prefix}'. Check filtering logic or S3 structure.")
        elif not objects_metadata: 
            logger.warning(f"S3 LIST [WARN]: No objects (raw or filtered) found for bucket '{aws_s3_bucket}' and prefix '{base_key_prefix}'.")

        return objects_metadata
    except Exception as e:
        logger.error(f"S3 LIST [ERROR]: Error listing objects in S3 for prefix '{base_key_prefix}': {e}", exc_info=True)
        return []


def find_file_any_extension(base_pattern: str, description: str) -> Optional[Tuple[str, str]]:
    """Find the most recent file matching base pattern with any extension in S3."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket name not available for finding {description}.")
        return None

    try:
        prefix = ""
        base_name = base_pattern
        if '/' in base_pattern:
             prefix = base_pattern.rsplit('/', 1)[0] + '/'
             base_name = base_pattern.rsplit('/', 1)[1]

        logger.debug(f"Searching for {description} with prefix '{prefix}' and base name '{base_name}' in bucket '{aws_s3_bucket}'")
        paginator = s3.get_paginator('list_objects_v2')
        matching_files = []

        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename_part = key[len(prefix):] 
                    if '/' in filename_part: continue 

                    name_only, ext = os.path.splitext(filename_part)

                    if name_only == base_name:
                        matching_files.append(obj)

        if not matching_files:
             logger.warning(f"No files found matching base pattern '{base_pattern}' for {description}.")
             return None

        logger.debug(f"Found {len(matching_files)} potential files for {description} matching base '{base_name}'.")
        matching_files.sort(key=lambda obj: obj['LastModified'], reverse=True)
        latest_file_key = matching_files[0]['Key']

        logger.debug(f"Latest file found for {description}: {latest_file_key}")
        content = read_file_content(latest_file_key, description)
        if content is not None:
            logger.debug(f"Successfully loaded content for {description}, length: {len(content)}")
            return latest_file_key, content
        else:
            logger.error(f"Failed to read content from {latest_file_key} for {description}.")
            return None

    except Exception as e:
        logger.error(f"Error finding {description} file for pattern '{base_pattern}': {e}", exc_info=True)
        return None

def get_latest_system_prompt(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine system prompts from S3, using a cache."""
    logger.debug(f"Getting system prompt (agent: {agent_name})")
    
    # Use caching for base prompt
    base_prompt_pattern = '_config/systemprompt_base'
    base_prompt = get_cached_s3_file(
        cache_key=base_prompt_pattern,
        description="base system prompt",
        fetch_function=lambda: find_file_any_extension(base_prompt_pattern, "base system prompt")
    )

    if not base_prompt:
         logger.error("Base system prompt '_config/systemprompt_base' not found or failed to load.")
         return "You are a helpful assistant." # Critical fallback

    agent_prompt = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}'
        # Use caching for agent-specific prompt
        agent_prompt = get_cached_s3_file(
            cache_key=agent_pattern,
            description=f"agent system prompt for {agent_name}",
            fetch_function=lambda: find_file_any_extension(agent_pattern, f"agent system prompt for {agent_name}")
        )
        if agent_prompt: logger.info(f"Loaded agent-specific system prompt for '{agent_name}'.")
        else: logger.warning(f"No agent-specific system prompt found using pattern '{agent_pattern}'.")

    system_prompt = base_prompt
    if agent_prompt: system_prompt += "\n\n" + agent_prompt
    logger.info(f"Final system prompt length: {len(system_prompt)}")
    return system_prompt

def get_latest_frameworks(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine frameworks from S3, using a cache."""
    logger.debug(f"Getting frameworks (agent: {agent_name})")

    base_framework_pattern = '_config/frameworks_base'
    base_frameworks = get_cached_s3_file(
        cache_key=base_framework_pattern,
        description="base frameworks",
        fetch_function=lambda: find_file_any_extension(base_framework_pattern, "base frameworks")
    ) or ""

    agent_frameworks = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}'
        agent_frameworks = get_cached_s3_file(
            cache_key=agent_pattern,
            description=f"agent frameworks for {agent_name}",
            fetch_function=lambda: find_file_any_extension(agent_pattern, f"agent frameworks for {agent_name}")
        ) or ""
        if agent_frameworks: logger.info(f"Loaded agent-specific frameworks for '{agent_name}'.")
        else: logger.warning(f"No agent-specific frameworks found using pattern '{agent_pattern}'.")

    frameworks = base_frameworks
    if agent_frameworks: frameworks += ("\n\n" + agent_frameworks) if frameworks else agent_frameworks

    if frameworks: logger.info(f"Loaded frameworks, total length: {len(frameworks)}")
    else: logger.warning("No base or agent-specific frameworks found.")
    return frameworks if frameworks else None


def get_latest_context(agent_name: str, event_id: Optional[str] = None) -> Optional[str]:
    """Get and combine agent-specific and event-specific contexts from S3, with caching."""
    logger.debug(f"Getting context (agent: {agent_name}, event: {event_id})")
    
    agent_context_pattern = f'organizations/river/agents/{agent_name}/_config/context_aID-{agent_name}'
    agent_primary_context = get_cached_s3_file(
        cache_key=agent_context_pattern,
        description=f"agent primary context for {agent_name}",
        fetch_function=lambda: find_file_any_extension(agent_context_pattern, "agent primary context")
    ) or ""
    if agent_primary_context: logger.info(f"Loaded agent-specific primary context for agent '{agent_name}'.")
    else: logger.warning(f"No agent-specific primary context found for agent '{agent_name}'.")

    event_context = ""
    if event_id and event_id != '0000':
        event_pattern = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}'
        event_context = get_cached_s3_file(
            cache_key=event_pattern,
            description=f"event context for {agent_name}/{event_id}",
            fetch_function=lambda: find_file_any_extension(event_pattern, "event context")
        ) or ""
        if event_context: logger.info(f"Loaded event-specific context for event '{event_id}'.")
        else: logger.warning(f"No event-specific context found for event '{event_id}'.")

    final_context = agent_primary_context
    if event_context:
        final_context += ("\n\n" + event_context) if final_context else event_context
            
    if final_context: logger.info(f"Final combined context loaded. Total length: {len(final_context)}")
    else: logger.warning(f"No agent primary or event-specific context found for agent '{agent_name}', event '{event_id}'.")
        
    return final_context if final_context else None

def get_agent_docs(agent_name: str) -> Optional[str]:
    """Get documentation files for the specified agent, with caching."""
    
    docs_cache_key = f"agent_docs_{agent_name}"

    def fetch_docs():
        s3 = get_s3_client()
        aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
        if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for getting agent docs."); return None
        try:
            prefix = f'organizations/river/agents/{agent_name}/docs/'
            logger.debug(f"Searching for agent docs in S3 prefix '{prefix}'")
            paginator = s3.get_paginator('list_objects_v2')
            docs = []
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                 if 'Contents' in page:
                      for obj in page['Contents']:
                          key = obj['Key']
                          if key == prefix or key.endswith('/'): continue
                          relative_path = key[len(prefix):]
                          if '/' in relative_path: continue
                          filename = os.path.basename(key)
                          content = read_file_content(key, f'agent doc ({filename})')
                          if content: docs.append(f"--- START Doc: {filename} ---\n{content}\n--- END Doc: {filename} ---")
            if not docs: logger.warning(f"No documentation files found directly in '{prefix}'"); return None
            logger.info(f"Loaded {len(docs)} documentation files for agent '{agent_name}'.")
            # For caching, we need to return a tuple, but this function only returns the content string
            # The get_cached_s3_file wrapper is designed for find_file_any_extension which returns (key, content)
            # We'll return a dummy key here to satisfy the wrapper's expectation.
            return docs_cache_key, "\n\n".join(docs)
        except Exception as e: logger.error(f"Error getting agent docs for '{agent_name}': {e}", exc_info=True); return None

    # We need to adapt this for get_cached_s3_file. Let's make it simpler.
    with S3_CACHE_LOCK:
        cached_item = S3_FILE_CACHE.get(docs_cache_key)
        if cached_item and cached_item['expiry'] > datetime.now(timezone.utc):
            logger.info(f"CACHE HIT for agent docs ('{docs_cache_key}')")
            return cached_item['content']

    logger.info(f"CACHE MISS for agent docs ('{docs_cache_key}')")
    fetch_result = fetch_docs()
    content = fetch_result[1] if fetch_result else None
    
    with S3_CACHE_LOCK:
        if content is not None:
            S3_FILE_CACHE[docs_cache_key] = {
                'content': content,
                'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_DEFAULT
            }
            logger.info(f"CACHE SET for agent docs ('{docs_cache_key}').")
        else:
            S3_FILE_CACHE[docs_cache_key] = {
                'content': None,
                'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_NOT_FOUND
            }
            logger.info(f"CACHE SET (None) for agent docs ('{docs_cache_key}').")
    
    return content


def get_objective_function(agent_name: Optional[str] = None) -> Optional[str]:
    """
    Get the objective function from S3, which is a stable, core directive.
    It checks for an agent-specific override first, then falls back to a base file.
    Uses caching.
    """
    logger.debug(f"Getting objective function (agent: {agent_name})")
    
    # 1. Look for agent-specific objective function (optional override)
    agent_objective_function = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/objective_function_aID-{agent_name}'
        agent_objective_function = get_cached_s3_file(
            cache_key=agent_pattern,
            description=f"agent objective function for {agent_name}",
            fetch_function=lambda: find_file_any_extension(agent_pattern, f"agent objective function for {agent_name}")
        ) or ""
        if agent_objective_function:
            logger.info(f"Loaded agent-specific objective function for '{agent_name}'.")
        else:
            logger.debug(f"No agent-specific objective function found for '{agent_name}'.")

    # 2. Look for base objective function
    base_pattern = '_config/objective_function'
    base_objective_function = get_cached_s3_file(
        cache_key=base_pattern,
        description="base objective function",
        fetch_function=lambda: find_file_any_extension(base_pattern, "base objective function")
    ) or ""
    if base_objective_function: logger.info("Loaded base objective function.")
    
    # Prioritize agent-specific, then base.
    final_objective_function = agent_objective_function or base_objective_function
    if final_objective_function: logger.info(f"Final objective function loaded, length: {len(final_objective_function)}")
    else: logger.warning("No objective function file found (neither agent-specific nor base).")
    return final_objective_function if final_objective_function else None


def write_agent_doc(agent_name: str, doc_name: str, content: str) -> bool:
    """
    Creates or updates a specific documentation file for an agent and invalidates the cache.
    The doc_name should be a simple filename, e.g., 'project_overview.md'.
    """
    logger.info(f"Attempting to write agent doc '{doc_name}' for agent '{agent_name}'.")
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket not available for writing agent doc. S3 Client: {'Exists' if s3 else 'None'}, Bucket: {aws_s3_bucket}")
        return False

    # Enforce the directory structure for security.
    # Sanitize doc_name to prevent path traversal attacks (e.g., '..').
    safe_doc_name = secure_filename(doc_name)
    if not safe_doc_name:
        logger.error(f"Invalid doc_name provided: '{doc_name}'")
        return False

    file_key = f"organizations/river/agents/{agent_name}/docs/{safe_doc_name}"
    
    try:
        logger.info(f"Executing S3 PutObject. Bucket: '{aws_s3_bucket}', Key: '{file_key}'")
        s3.put_object(Bucket=aws_s3_bucket, Key=file_key, Body=content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
        logger.info(f"Successfully wrote agent doc to S3 key: {file_key}")

        # CRITICAL: Invalidate the cache for the entire agent docs collection.
        # The get_agent_docs function caches all docs under a single key.
        docs_cache_key = f"agent_docs_{agent_name}"
        with S3_CACHE_LOCK:
            if docs_cache_key in S3_FILE_CACHE:
                del S3_FILE_CACHE[docs_cache_key]
                logger.info(f"CACHE INVALIDATED for agent docs: '{docs_cache_key}' after write.")

        return True
    except Exception as e:
        logger.error(f"Error writing agent doc to {file_key}: {e}", exc_info=True)
        return False


def save_chat_to_s3(agent_name: str, chat_content: str, event_id: Optional[str], is_saved: bool = False, filename: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Save chat content to S3 bucket (archive or saved folder)."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for saving chat."); return False, None
    event_id = event_id or '0000'

    try:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d-T%H%M%S')
            filename = f"chat_D{timestamp}_aID-{agent_name}_eID-{event_id}.txt"
            logger.debug(f"Generated chat filename: {filename}")

        base_path = f"organizations/river/agents/{agent_name}/events/{event_id}/chats"
        archive_key = f"{base_path}/archive/{filename}"
        saved_key = f"{base_path}/saved/{filename}"

        if is_saved:
            logger.info(f"Saving chat: Copying from archive '{archive_key}' to saved '{saved_key}'")
            try:
                 s3.head_object(Bucket=aws_s3_bucket, Key=archive_key) 
                 copy_source = {'Bucket': aws_s3_bucket, 'Key': archive_key}
                 s3.copy_object(CopySource=copy_source, Bucket=aws_s3_bucket, Key=saved_key)
                 logger.info(f"Successfully copied chat to saved: {saved_key}")
                 return True, filename
            except s3.exceptions.ClientError as e:
                 if e.response['Error']['Code'] == '404': logger.error(f"Cannot save: Archive {archive_key} missing.")
                 else: logger.error(f"S3 ClientError checking/copying {archive_key}: {e}", exc_info=True)
                 return False, None
            except Exception as e: logger.error(f"Error copying chat {archive_key}: {e}", exc_info=True); return False, None
        else:
            logger.info(f"Saving chat to archive: {archive_key}")
            full_content = chat_content.strip(); existing_content = ""
            try:
                existing_obj = s3.get_object(Bucket=aws_s3_bucket, Key=archive_key)
                existing_content = existing_obj['Body'].read().decode('utf-8')
                full_content = existing_content.strip() + '\n\n' + chat_content.strip()
                logger.debug(f"Appending {len(chat_content)} chars to archive.")
            except s3.exceptions.NoSuchKey: logger.debug(f"Archive {archive_key} not found. Creating new.")
            except Exception as get_e: logger.error(f"Error reading archive {archive_key}: {get_e}", exc_info=True); return False, None

            try:
                s3.put_object(Bucket=aws_s3_bucket, Key=archive_key, Body=full_content.encode('utf-8'), ContentType='text/plain; charset=utf-8')
                logger.info(f"Successfully saved to archive: {archive_key}")
                return True, filename
            except Exception as put_e: logger.error(f"Error writing to archive {archive_key}: {put_e}", exc_info=True); return False, None

    except Exception as e: logger.error(f"Error in save_chat_to_s3 ({filename}): {e}", exc_info=True); return False, None

def parse_text_chat(chat_content_str: str) -> List[Dict[str, str]]:
    """Parses chat content from text format (**User:**/**Agent:**)."""
    messages = []; current_role = None; current_content = []
    for line in chat_content_str.splitlines():
        line_strip = line.strip(); role_found = None; content_start = 0
        if line_strip.startswith('**User:**'): role_found = 'user'; content_start = len('**User:**')
        elif line_strip.startswith('**Agent:**') or line_strip.startswith('**Assistant:**'): 
            role_found = 'assistant'; content_start = len('**Agent:**') if line_strip.startswith('**Agent:**') else len('**Assistant:**')
        if role_found:
            if current_role: messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
            current_role = role_found; current_content = [line_strip[content_start:].strip()]
        elif current_role: current_content.append(line) 
    if current_role: messages.append({'role': current_role, 'content': '\n'.join(current_content).strip()})
    return [msg for msg in messages if msg.get('content')]

def load_existing_chats_from_s3(agent_name: str, memory_agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load chat history from S3 'saved' directory."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: logger.error("S3 unavailable for loading chats."); return []
    chat_histories = []; agents_to_load = memory_agents or [agent_name]
    logger.info(f"Loading saved chat history for agents: {agents_to_load}")
    for agent in agents_to_load:
        prefix = f'organizations/river/agents/{agent}/events/0000/chats/saved/' 
        logger.debug(f"Checking for saved chats in: {prefix}")
        try:
            paginator = s3.get_paginator('list_objects_v2'); chat_files = []
            for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix):
                if 'Contents' in page: chat_files.extend(obj for obj in page['Contents'] if not obj['Key'].endswith('/'))
            if not chat_files: logger.debug(f"No saved chats for {agent}."); continue
            chat_files.sort(key=lambda obj: obj['LastModified'], reverse=True)
            logger.info(f"Found {len(chat_files)} saved chats for {agent}.")
            for chat_obj in chat_files:
                file_key = chat_obj['Key']; logger.debug(f"Reading saved chat: {file_key}")
                try:
                    content_str = read_file_content(file_key, f"saved chat {file_key}")
                    if not content_str: logger.warning(f"Empty chat file: {file_key}"); continue
                    messages = parse_text_chat(content_str) 
                    if messages: chat_histories.append({'agent': agent, 'file': file_key, 'messages': messages})
                    else: logger.warning(f"No messages extracted from {file_key}")
                except Exception as read_err: logger.error(f"Error reading/parsing chat {file_key}: {read_err}", exc_info=True)
        except Exception as list_err: logger.error(f"Error listing chats for {agent}: {list_err}", exc_info=True)
    logger.info(f"Loaded {len(chat_histories)} chat history files total.")
    return chat_histories

def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Formats messages list into a string for saving."""
    chat_content = ""
    for msg in messages:
        role = msg.get("role", "unknown").capitalize(); content = msg.get("content", "")
        if content: chat_content += f"**{role}:**\n{content}\n\n"
    return chat_content.strip()

def list_agent_names_from_s3() -> Optional[List[str]]:
    """Lists potential agent names by looking at directories under organizations/river/agents/."""
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not available for listing agents.")
        return None

    base_prefix = 'organizations/river/agents/'
    agent_names = set()

    try:
        paginator = s3.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=aws_s3_bucket, Prefix=base_prefix, Delimiter='/')

        for page in result:
            if 'CommonPrefixes' in page:
                for prefix_data in page['CommonPrefixes']:
                    full_prefix = prefix_data.get('Prefix', '')
                    if full_prefix.startswith(base_prefix) and full_prefix.endswith('/'):
                        agent_name = full_prefix[len(base_prefix):].strip('/')
                        if agent_name: 
                            agent_names.add(agent_name)

        found_agents = list(agent_names)
        logger.info(f"Found {len(found_agents)} potential agent directories in S3: {found_agents}")
        return found_agents

    except Exception as e:
        logger.error(f"Error listing agent directories in S3 prefix '{base_prefix}': {e}", exc_info=True)
        return None

def get_transcript_summaries(agent_name: str, event_id: str) -> List[Dict[str, Any]]:
    """
    Fetches and parses all JSON transcript summaries for a given agent and event.
    Uses an in-memory cache to avoid repeated S3 calls.
    Returns a list of dictionaries, where each dictionary is a parsed JSON summary.
    Includes filename in the returned metadata for each summary.
    """
    cache_key = f"transcript_summaries_{agent_name}_{event_id}"
    description = f"transcript summaries for {agent_name}/{event_id}"

    # 1. Check cache first
    with S3_CACHE_LOCK:
        cached_item = S3_FILE_CACHE.get(cache_key)
        if cached_item and cached_item['expiry'] > datetime.now(timezone.utc):
            logger.info(f"CACHE HIT for {description} ('{cache_key}')")
            if cached_item['content'] is None:
                return [] # Cached "not found"
            try:
                # Content is a JSON string, deserialize it
                return json.loads(cached_item['content'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"CACHE CORRUPT for {description}: Could not parse cached JSON. Refetching. Error: {e}")
                # Fall through to re-fetch if cache is corrupt

    # 2. Cache miss, fetch from S3
    logger.info(f"CACHE MISS for {description} ('{cache_key}')")
    
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not available for fetching transcript summaries.")
        return []

    summaries_prefix = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/summarized/"
    logger.info(f"Fetching transcript summaries from S3 prefix: {summaries_prefix}")
    
    parsed_summaries = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=summaries_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    if not s3_key.endswith('.json'):
                        continue # Skip non-JSON files

                    filename = os.path.basename(s3_key)
                    logger.info(f"Processing summary candidate file: {s3_key}") # Changed from debug to info
                    try:
                        summary_content_str = read_file_content(s3_key, f"summary file {filename}")
                        if summary_content_str:
                            logger.debug(f"Successfully read content for {s3_key}, length {len(summary_content_str)}")
                            summary_data = json.loads(summary_content_str)
                            # Add filename to the summary data itself for easier reference later
                            if 'metadata' in summary_data and isinstance(summary_data['metadata'], dict):
                                summary_data['metadata']['summary_filename'] = filename
                            else:
                                summary_data['metadata'] = {'summary_filename': filename}
                            parsed_summaries.append(summary_data)
                            logger.info(f"Successfully parsed and added summary from {filename} to list. Current total parsed: {len(parsed_summaries)}")
                        else:
                            logger.warning(f"Empty content for summary file: {s3_key}")
                    except json.JSONDecodeError as e_json:
                        logger.error(f"Failed to parse JSON for summary file {s3_key}: {e_json}")
                    except Exception as e_read:
                        logger.error(f"Failed to read or process summary file {s3_key}: {e_read}")
        
        logger.info(f"Fetched and parsed {len(parsed_summaries)} transcript summaries for {agent_name}/{event_id}.")
        # Optional: Sort summaries, e.g., by a timestamp within their metadata if available
        # For now, returning in S3 list order.

        # 3. Update cache
        with S3_CACHE_LOCK:
            if parsed_summaries:
                # Serialize list of dicts to a JSON string for caching
                content_to_cache = json.dumps(parsed_summaries)
                S3_FILE_CACHE[cache_key] = {
                    'content': content_to_cache,
                    'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_DEFAULT
                }
                logger.info(f"CACHE SET for {description} ('{cache_key}').")
            else:
                # Cache "not found"
                S3_FILE_CACHE[cache_key] = {
                    'content': None,
                    'expiry': datetime.now(timezone.utc) + S3_CACHE_TTL_NOT_FOUND
                }
                logger.info(f"CACHE SET (None) for {description} ('{cache_key}').")
        
        return parsed_summaries

    except Exception as e:
        logger.error(f"Error listing or fetching transcript summaries from '{summaries_prefix}': {e}", exc_info=True)
        return []
