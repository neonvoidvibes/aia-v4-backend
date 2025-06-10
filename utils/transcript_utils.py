# utils/transcript_utils.py
import logging
import boto3
import os
from datetime import datetime, timezone # Ensure timezone is imported
from typing import Optional, Dict, Any, List, Tuple

from .s3_utils import get_s3_client

logger = logging.getLogger(__name__)

TRANSCRIPT_MODE = os.getenv('TRANSCRIPT_MODE', 'regular') # Read from env, default to 'regular'
logger.info(f"Transcript mode set to: {TRANSCRIPT_MODE}")


def get_transcript_mode() -> str:
    """Get the current transcript mode."""
    return TRANSCRIPT_MODE

# The TranscriptState class is no longer needed and has been removed.

def get_latest_transcript_file(agent_name: Optional[str] = None, event_id: Optional[str] = None, s3_client_provided: Optional[boto3.client] = None) -> Optional[str]:
    """Get the latest transcript file key based on TRANSCRIPT_MODE setting and priority."""
    s3 = s3_client_provided if s3_client_provided else get_s3_client()
    if not s3: 
        logger.error("get_latest_transcript_file: S3 client unavailable.")
        return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: 
        logger.error("get_latest_transcript_file: AWS_S3_BUCKET not set.")
        return None

    candidate_files = []
    # Define base path for transcripts for the given agent and event
    base_transcript_path = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        logger.debug(f"Listing S3 objects with prefix: {base_transcript_path} for latest transcript file.")
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=base_transcript_path):
             if 'Contents' in page:
                 for obj in page['Contents']:
                     key = obj['Key']
                     # Ensure it's a file directly in the folder (not a sub-folder's content)
                     if key.startswith(base_transcript_path) and key != base_transcript_path and key.endswith('.txt'):
                          relative_path = key[len(base_transcript_path):]
                          if '/' not in relative_path: # Check if it's directly in the target folder
                               filename = os.path.basename(key)
                               is_rolling = filename.startswith('rolling-')
                               # Filter based on TRANSCRIPT_MODE
                               if (TRANSCRIPT_MODE == 'rolling' and is_rolling) or \
                                  (TRANSCRIPT_MODE == 'regular' and not is_rolling):
                                    candidate_files.append(obj)
    except Exception as e:
        logger.error(f"Error listing S3 objects under prefix '{base_transcript_path}': {e}", exc_info=True)
        return None # Return None on error

    if not candidate_files:
        logger.warning(f"No transcript files found (Mode: '{TRANSCRIPT_MODE}') in {base_transcript_path}")
        return None
    
    # Sort by LastModified to find the most recent file
    candidate_files.sort(key=lambda x: x['LastModified'], reverse=True)
    latest_file = candidate_files[0]
    
    logger.debug(f"Latest transcript file selected (Mode: '{TRANSCRIPT_MODE}'): {latest_file['Key']} (Mod: {latest_file['LastModified']})")
    return latest_file['Key']


def read_new_transcript_content(agent_name: str, event_id: str) -> Tuple[Optional[str], bool]:
    """
    Read new transcript content.
    This function is now stateless for 'latest' mode. It finds the single latest
    transcript file and reads its entire content on every call.

    Returns a tuple: (content: Optional[str], success: bool).
    - content: The full content of the latest file, or None.
    - success: True if the operation completed (even if no file was found), False on an error.
    """
    s3 = get_s3_client()
    if not s3: 
        logger.error("read_new_transcript_content: S3 client unavailable.")
        return None, False

    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket:
        logger.error("read_new_transcript_content: AWS_S3_BUCKET not set.")
        return None, False

    latest_key_on_s3 = get_latest_transcript_file(agent_name, event_id, s3)

    if not latest_key_on_s3:
        logger.info(f"Listen: Latest - No transcript file found for agent '{agent_name}', event '{event_id}' (Mode: {TRANSCRIPT_MODE}).")
        return None, True  # True: the operation was successful, but no file was found

    try:
        logger.info(f"Listen: Latest - Reading full content of latest file: {latest_key_on_s3}")
        response = s3.get_object(Bucket=aws_s3_bucket, Key=latest_key_on_s3)
        file_content_bytes = response['Body'].read()
        file_content_str = file_content_bytes.decode('utf-8', errors='replace')

        filename = os.path.basename(latest_key_on_s3)
        # Prepend a standard header to the content.
        labeled_content = f"(Source File: {filename})\n{file_content_str}"

        logger.info(f"Listen: Latest - Read {len(labeled_content)} chars from {latest_key_on_s3}.")
        return labeled_content, True # True: success

    except Exception as e:
        logger.error(f"Listen: Latest - Error reading file {latest_key_on_s3}: {e}", exc_info=True)
        return None, False # False: there was an operational error


def read_all_transcripts_in_folder(agent_name: str, event_id: str) -> Optional[str]:
    """Read and combine content of all relevant transcripts in the folder. Respects TRANSCRIPT_MODE."""
    s3 = get_s3_client()
    if not s3: 
        logger.error("read_all_transcripts: S3 client unavailable.")
        return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: 
        logger.error("read_all_transcripts: AWS_S3_BUCKET not set.")
        return None

    base_prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
    logger.info(f"Reading all transcripts from: {base_prefix} (Mode: {TRANSCRIPT_MODE})")
    
    all_content_parts = []
    transcript_files_metadata: List[Dict[str, Any]] = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=aws_s3_bucket, Prefix=base_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.startswith(base_prefix) and key != base_prefix and key.endswith('.txt'):
                         relative_path = key[len(base_prefix):]
                         if '/' not in relative_path: # Files directly in folder
                            filename = os.path.basename(key)
                            is_rolling = filename.startswith('rolling-')
                            # Adhere to TRANSCRIPT_MODE
                            if (TRANSCRIPT_MODE == 'rolling' and is_rolling) or \
                               (TRANSCRIPT_MODE == 'regular' and not is_rolling):
                                 transcript_files_metadata.append(obj)
                                 
        if not transcript_files_metadata:
            logger.warning(f"No transcript files found in {base_prefix} matching mode '{TRANSCRIPT_MODE}'.")
            return None
            
        # Sort by LastModified to process chronologically
        transcript_files_metadata.sort(key=lambda x: x['LastModified'])
        logger.info(f"Found {len(transcript_files_metadata)} transcript files to combine.")

        for s3_obj in transcript_files_metadata:
            key = s3_obj['Key']
            filename = os.path.basename(key)
            try:
                logger.debug(f"Reading content from {key} for combined transcript.")
                response = s3.get_object(Bucket=aws_s3_bucket, Key=key)
                text_content = response['Body'].read().decode('utf-8', errors='replace') # Use replace
                all_content_parts.append(f"--- Transcript Source: {filename} ---\n{text_content}")
            except Exception as read_e:
                logger.error(f"Error reading file {key} for combined transcript: {read_e}")
        
        if all_content_parts:
            logger.info(f"Successfully combined content from {len(all_content_parts)} transcript files.")
            return "\n\n".join(all_content_parts)
        else:
            logger.warning("No content was read from any transcript files during 'read_all'.")
            return None
            
    except Exception as e:
        logger.error(f"Error in read_all_transcripts_in_folder for {base_prefix}: {e}", exc_info=True)
        return None