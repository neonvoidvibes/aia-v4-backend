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

class TranscriptState:
    """Tracks position across multiple transcript files."""
    def __init__(self):
        self.file_positions: Dict[str, int] = {}
        self.last_modified: Dict[str, datetime] = {}
        self.current_latest_key: Optional[str] = None

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


def read_new_transcript_content(state: TranscriptState, agent_name: str, event_id: str) -> Optional[str]:
    """
    Read new transcript content.
    If it's the first call for this state (agent/event where state.file_positions is empty
    and state.current_latest_key is None), it reads all relevant historical transcripts.
    Otherwise, it reads new delta content from the latest transcript file.
    """
    s3 = get_s3_client()
    if not s3: 
        logger.error("read_new_transcript_content: S3 client unavailable.")
        return None
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not aws_s3_bucket: 
        logger.error("read_new_transcript_content: AWS_S3_BUCKET not set.")
        return None

    # --- Initial Full Load Logic ---
    # Trigger initial load if file_positions is empty AND current_latest_key is None (truly fresh state)
    if not state.file_positions and state.current_latest_key is None:
        logger.info(f"Performing initial full transcript load for agent '{agent_name}', event '{event_id}'...")
        
        base_prefix = f'organizations/river/agents/{agent_name}/events/{event_id}/transcripts/'
        
        all_transcript_objects_metadata: List[Dict[str, Any]] = []
        try:
            paginator = s3.get_paginator('list_objects_v2')
            logger.debug(f"Initial load: Listing S3 objects with prefix: {base_prefix}")
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
                                    all_transcript_objects_metadata.append(obj)
        except Exception as e:
            logger.error(f"Initial load: Error listing S3 objects for {base_prefix}: {e}", exc_info=True)
            return None # Abort if S3 listing fails

        if not all_transcript_objects_metadata:
            logger.info(f"Initial load: No transcript files found for agent '{agent_name}', event '{event_id}' (Mode: {TRANSCRIPT_MODE}).")
            # Mark state to indicate initial load attempted, even if no files found, to prevent re-triggering.
            # A non-None value that's not a real key signifies this.
            state.current_latest_key = f"__INITIAL_LOAD_COMPLETE_NO_FILES_FOR_{agent_name}_{event_id}__"
            return None
            
        # Sort by LastModified to process chronologically for initial load
        all_transcript_objects_metadata.sort(key=lambda x: x['LastModified'])
        logger.info(f"Initial load: Found {len(all_transcript_objects_metadata)} transcript files to process.")

        combined_content_parts = []
        actual_latest_file_key_from_all: Optional[str] = None
        actual_latest_mod_time: Optional[datetime] = None # Ensure it's timezone-aware if comparing

        for s3_obj in all_transcript_objects_metadata:
            key = s3_obj['Key']
            s3_size = s3_obj['ContentLength']
            s3_mod_time_utc = s3_obj['LastModified'] # S3 LastModified is already timezone-aware (UTC)

            try:
                logger.debug(f"Initial load: Reading full content of {key} (Size: {s3_size}, Modified: {s3_mod_time_utc})")
                response = s3.get_object(Bucket=aws_s3_bucket, Key=key)
                file_content_bytes = response['Body'].read()
                file_content_str = file_content_bytes.decode('utf-8', errors='replace') # Use replace for robustness
                
                filename = os.path.basename(key)
                labeled_content = f"(Source File: {filename})\n{file_content_str}"
                combined_content_parts.append(labeled_content)

                # Update state for this specific file
                state.file_positions[key] = s3_size
                state.last_modified[key] = s3_mod_time_utc

                # Track the truly latest file among those processed in this initial batch
                if actual_latest_mod_time is None or s3_mod_time_utc > actual_latest_mod_time:
                    actual_latest_mod_time = s3_mod_time_utc
                    actual_latest_file_key_from_all = key
                
            except Exception as e:
                logger.error(f"Initial load: Error reading or processing file {key}: {e}", exc_info=True)
                # Optionally, decide if one failed read should abort the whole initial load.
                # For now, log and continue to give as much context as possible.
                # If critical, could `return None` here.

        state.current_latest_key = actual_latest_file_key_from_all # Set after processing all
        
        if not combined_content_parts:
            logger.info(f"Initial load: No content aggregated from files for agent '{agent_name}', event '{event_id}'.")
            # This path might be hit if all files read were empty or errored out.
            # state.current_latest_key might be None or set from the loop. If set, it's fine.
            # If None, the next delta check will re-evaluate.
            return None
        
        final_content = "\n\n".join(combined_content_parts)
        logger.info(f"Initial load completed. Total content length: {len(final_content)} chars. Latest key set to: {state.current_latest_key}")
        return final_content

    # --- Delta Load Logic ---
    # This part executes if `state.file_positions` was not empty or `state.current_latest_key` was not None (i.e., not a fresh state).
    latest_key_on_s3 = get_latest_transcript_file(agent_name, event_id, s3)
    
    if not latest_key_on_s3:
        logger.debug(f"Delta load: No latest transcript file found on S3 for {agent_name}/{event_id}. Nothing to read.")
        return None

    try:
        s3_metadata = s3.head_object(Bucket=aws_s3_bucket, Key=latest_key_on_s3)
        current_s3_size = s3_metadata['ContentLength']
        current_s3_modified = s3_metadata['LastModified'] # Already timezone-aware

        last_known_pos_for_this_file = state.file_positions.get(latest_key_on_s3, 0)
        last_known_mod_for_this_file = state.last_modified.get(latest_key_on_s3)

        is_different_file_than_tracked_latest = (latest_key_on_s3 != state.current_latest_key)
        
        file_has_new_data = (last_known_mod_for_this_file is None or current_s3_modified > last_known_mod_for_this_file) or \
                             (current_s3_size > last_known_pos_for_this_file)

        logger.debug(f"Delta Check: Key='{os.path.basename(latest_key_on_s3)}', "
                     f"S3Size={current_s3_size}, S3Mod={current_s3_modified}, "
                     f"StateKnownPosForThisFile={last_known_pos_for_this_file}, StateKnownModForThisFile={last_known_mod_for_this_file}, "
                     f"IsDifferentFileThanTrackedLatest={is_different_file_than_tracked_latest}, FileHasNewData={file_has_new_data}, "
                     f"StateCurrentLatestKeyTracked={state.current_latest_key}")
        
        start_read_pos = 0
        read_this_file = False

        if is_different_file_than_tracked_latest:
            logger.info(f"Delta: Latest file on S3 ({latest_key_on_s3}) is different from last tracked ({state.current_latest_key}). Reading new file from start.")
            start_read_pos = 0 
            read_this_file = True
        elif file_has_new_data:
            if current_s3_size > last_known_pos_for_this_file:
                logger.info(f"Delta: File {latest_key_on_s3} has new content. Reading from pos {last_known_pos_for_this_file} (S3 size: {current_s3_size}).")
                start_read_pos = last_known_pos_for_this_file
                read_this_file = True
            elif last_known_mod_for_this_file is None or current_s3_modified > last_known_mod_for_this_file:
                logger.warning(f"Delta: File {latest_key_on_s3} modified (ModTime: {current_s3_modified} > StateMod: {last_known_mod_for_this_file}) "
                               f"but S3 size ({current_s3_size}) not greater than known position ({last_known_pos_for_this_file}). "
                               f"Assuming potential replacement/overwrite, reading from start.")
                start_read_pos = 0
                read_this_file = True
            else: # Should not be reached if file_has_new_data is true based on current logic
                logger.debug(f"Delta: File {latest_key_on_s3} met 'file_has_new_data' but no specific read condition. No read needed.")
                read_this_file = False
        else:
            logger.debug(f"Delta: No changes detected for {latest_key_on_s3}. No read needed.")
            read_this_file = False

        if not read_this_file:
            # Update mod time for the current latest key if it was checked and no read was needed
            if latest_key_on_s3 == state.current_latest_key:
                state.last_modified[latest_key_on_s3] = current_s3_modified
            return None

        # Pre-read check: ensure start_read_pos is not beyond the current file size
        if start_read_pos >= current_s3_size and current_s3_size > 0:
             logger.warning(f"Delta: Calculated start_read_pos ({start_read_pos}) is >= S3 size ({current_s3_size}) for {latest_key_on_s3}. Skipping read.")
             state.file_positions[latest_key_on_s3] = current_s3_size 
             state.last_modified[latest_key_on_s3] = current_s3_modified
             if is_different_file_than_tracked_latest: state.current_latest_key = latest_key_on_s3
             return None
        
        new_content_str = ""; bytes_read = 0
        if current_s3_size > start_read_pos: # Only read if there are potential new bytes
            read_range = f"bytes={start_read_pos}-"
            try:
                logger.debug(f"Delta: S3 GET Key={latest_key_on_s3}, Range={read_range}")
                response = s3.get_object(Bucket=aws_s3_bucket, Key=latest_key_on_s3, Range=read_range)
                new_content_bytes = response['Body'].read()
                bytes_read = len(new_content_bytes)
                if bytes_read > 0:
                    new_content_str = new_content_bytes.decode('utf-8', errors='replace')
                    logger.info(f"Delta: Read {bytes_read} bytes, decoded {len(new_content_str)} chars from {latest_key_on_s3}.")
                else:
                    logger.info(f"Delta: Read 0 new bytes from {latest_key_on_s3} (Range: {read_range}). Current S3 size {current_s3_size}, start_read_pos {start_read_pos}.")
            except s3.exceptions.InvalidRange: # Check specific exception for S3
                logger.warning(f"Delta: S3 InvalidRange for {latest_key_on_s3} at {start_read_pos} (S3 size {current_s3_size}). Resetting pos for this file.")
                state.file_positions[latest_key_on_s3] = 0 
                state.last_modified[latest_key_on_s3] = current_s3_modified
                # current_latest_key should be updated if this was a new file
                if is_different_file_than_tracked_latest: state.current_latest_key = latest_key_on_s3
                return None # No content this time
            except Exception as get_e:
                logger.error(f"Delta: Error reading {latest_key_on_s3} (range {read_range}): {get_e}", exc_info=True)
                return None # Error during read
        elif current_s3_size == 0 and start_read_pos == 0:
             logger.info(f"Delta: File {latest_key_on_s3} is empty. No content to read.")
        else: # start_read_pos >= current_s3_size (and not the specific warning case above)
            logger.debug(f"Delta: No new bytes to read from {latest_key_on_s3} as S3_size ({current_s3_size}) <= start_pos ({start_read_pos}).")

        # Update state for the file that was processed (latest_key_on_s3)
        state.file_positions[latest_key_on_s3] = start_read_pos + bytes_read
        state.last_modified[latest_key_on_s3] = current_s3_modified
        state.current_latest_key = latest_key_on_s3 # Ensure this points to the file we just processed

        logger.info(f"Delta: State for {latest_key_on_s3} updated. NewPos={state.file_positions[latest_key_on_s3]}, Mod={state.last_modified[latest_key_on_s3]}. CurrentLatestKey is now {state.current_latest_key}.")

        if new_content_str:
            filename = os.path.basename(latest_key_on_s3)
            # For deltas, consider if the label is always needed or only if it's a new file.
            # For now, always label for clarity, especially if the latest file changes.
            labeled_content = f"(Source File: {filename})\n{new_content_str}"
            return labeled_content
        else:
            return None # No new content decoded

    except Exception as e: # Catch-all for the delta path
        logger.error(f"Delta: Unhandled error for {latest_key_on_s3 if latest_key_on_s3 else 'N/A'}: {e}", exc_info=True)
        return None


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