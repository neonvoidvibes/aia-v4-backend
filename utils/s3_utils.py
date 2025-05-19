# utils/s3_utils.py
"""Utilities for interacting with AWS S3."""

import os
import boto3
from botocore.config import Config as BotoConfig # For S3 timeouts
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import threading # For s3_client_lock

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global S3 client instance (lazy loaded)
s3_client_instance = None # Renamed to avoid confusion with boto3.client
s3_client_lock = threading.Lock()


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

def read_file_content(file_key: str, description: str) -> Optional[str]:
    """Read content from S3 file, handling potential errors."""
    s3 = get_s3_client()
    # Read bucket name here as well, in case it changes or needs checking per call
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error(f"S3 client or bucket name not available for reading {description}.")
        return None
    try:
        logger.debug(f"Reading {description} from S3: s3://{aws_s3_bucket}/{file_key}")
        response = s3.get_object(Bucket=aws_s3_bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        # No need to log full content, length is sufficient for debug
        logger.debug(f"Successfully read {description} ({len(content)} chars)")
        return content
    except s3.exceptions.NoSuchKey:
        logger.warning(f"{description} file not found at S3 key: {file_key}")
        return None
    except Exception as e: # Generic catch after specific S3 exceptions
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

# --- Functions moved from magic_chat.py ---

def get_latest_system_prompt(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine system prompts from S3"""
    logger.debug(f"Getting system prompt (agent: {agent_name})")
    base_result = find_file_any_extension('_config/systemprompt_base', "base system prompt")
    base_prompt = base_result[1] if base_result else None

    if not base_prompt:
         logger.error("Base system prompt '_config/systemprompt_base' not found or failed to load.")
         return "You are a helpful assistant." # Critical fallback

    agent_prompt = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/systemprompt_aID-{agent_name}'
        agent_result = find_file_any_extension(agent_pattern, "agent system prompt")
        if agent_result:
            agent_prompt = agent_result[1]
            logger.info(f"Loaded agent-specific system prompt for '{agent_name}'.")
        else:
             logger.warning(f"No agent-specific system prompt found using pattern '{agent_pattern}'.")

    system_prompt = base_prompt
    if agent_prompt: system_prompt += "\n\n" + agent_prompt
    logger.info(f"Final system prompt length: {len(system_prompt)}")
    return system_prompt

def get_latest_frameworks(agent_name: Optional[str] = None) -> Optional[str]:
    """Get and combine frameworks from S3"""
    logger.debug(f"Getting frameworks (agent: {agent_name})")
    base_result = find_file_any_extension('_config/frameworks_base', "base frameworks")
    base_frameworks = base_result[1] if base_result else ""

    agent_frameworks = ""
    if agent_name:
        agent_pattern = f'organizations/river/agents/{agent_name}/_config/frameworks_aID-{agent_name}'
        agent_result = find_file_any_extension(agent_pattern, "agent frameworks")
        if agent_result:
            agent_frameworks = agent_result[1]
            logger.info(f"Loaded agent-specific frameworks for '{agent_name}'.")
        else:
             logger.warning(f"No agent-specific frameworks found using pattern '{agent_pattern}'.")

    frameworks = base_frameworks
    if agent_frameworks: frameworks += ("\n\n" + agent_frameworks) if frameworks else agent_frameworks

    if frameworks: logger.info(f"Loaded frameworks, total length: {len(frameworks)}")
    else: logger.warning("No base or agent-specific frameworks found.")
    return frameworks if frameworks else None

def get_latest_context(agent_name: str, event_id: Optional[str] = None) -> Optional[str]:
    """Get and combine agent-specific and event-specific contexts from S3."""
    logger.debug(f"Getting context (agent: {agent_name}, event: {event_id})")
    
    agent_primary_context = ""
    # Load agent-specific context from the agent's _config folder
    # Pattern: organizations/river/agents/{agent_name}/_config/context_aID-{agent_name}.[txt|json|md]
    agent_context_pattern = f'organizations/river/agents/{agent_name}/_config/context_aID-{agent_name}'
    logger.info(f"Attempting agent-specific context load using pattern: '{agent_context_pattern}'.")
    agent_context_result = find_file_any_extension(agent_context_pattern, "agent primary context")
    if agent_context_result:
        agent_primary_context = agent_context_result[1]
        logger.info(f"Loaded agent-specific primary context for agent '{agent_name}'. Length: {len(agent_primary_context)}")
    else:
        logger.warning(f"No agent-specific primary context found for agent '{agent_name}' using pattern '{agent_context_pattern}'.")

    event_context = ""
    if event_id and event_id != '0000':
        # Event-specific context path already includes agent_name and event_id.
        # This pattern remains consistent with the original code for event context.
        event_pattern = f'organizations/river/agents/{agent_name}/events/{event_id}/_config/context_aID-{agent_name}_eID-{event_id}'
        logger.info(f"Attempting event-specific context load using pattern: '{event_pattern}'.")
        event_result = find_file_any_extension(event_pattern, "event context")
        if event_result:
            event_context = event_result[1]
            logger.info(f"Loaded event-specific context for event '{event_id}'. Length: {len(event_context)}")
        else:
            logger.warning(f"No event-specific context found for event '{event_id}' using pattern '{event_pattern}'.")

    # Combine agent primary context and event context
    final_context = agent_primary_context
    if event_context:
        if final_context: # If agent_primary_context was found, append event_context
            final_context += "\n\n" + event_context
        else: # If no agent_primary_context, event_context becomes the final_context
            final_context = event_context
            
    if final_context:
        logger.info(f"Final combined context loaded. Total length: {len(final_context)}")
    else:
        logger.warning(f"No agent primary or event-specific context found for agent '{agent_name}', event '{event_id}'.")
        
    return final_context if final_context else None

def get_agent_docs(agent_name: str) -> Optional[str]:
    """Get documentation files for the specified agent."""
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
        return "\n\n".join(docs)
    except Exception as e: logger.error(f"Error getting agent docs for '{agent_name}': {e}", exc_info=True); return None

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