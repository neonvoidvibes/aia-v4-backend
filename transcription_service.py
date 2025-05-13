# transcription_service.py
import os
import io
import logging
import json
import re
import time # For sleep in transcribe_audio retry
from datetime import datetime, timezone, timedelta
import boto3
from botocore.config import Config as BotoConfig # For S3 timeouts
import openai
import requests # For OpenAI API call
from typing import Dict, Any, Optional, List
import threading # For s3_lock type hint
import subprocess # For ffprobe fallback

# Configure logging
logger = logging.getLogger(__name__)

# --- Utility Functions (adapted from magic_audio.py) ---

def get_s3_client() -> Optional[boto3.client]:
    """Initializes and returns an S3 client, or None on failure."""
    try:
        aws_region = os.getenv('AWS_REGION')
        if not aws_region:
            logger.error("AWS_REGION environment variable not found.")
            return None
        
        # Credentials should be handled by the AWS SDK's default credential chain
        # (IAM role, environment variables, shared credential file, etc.)
        # Apply default timeouts for the global client
        default_boto_config = BotoConfig(
            connect_timeout=15, # Increased connect timeout
            read_timeout=45,    # Increased read timeout for potentially larger files
            retries={'max_attempts': 3}
        )
        client = boto3.client('s3', region_name=aws_region, config=default_boto_config)
        logger.debug(f"S3 client initialized for region {aws_region} in transcription_service with timeouts.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client in transcription_service: {e}", exc_info=True)
        return None

def _format_time_delta(seconds_delta: float, base_datetime_utc: datetime) -> str:
    """Helper to format elapsed seconds relative to a base UTC datetime into HH:MM:SS.mmm"""
    if seconds_delta < 0: seconds_delta = 0 # Ensure non-negative
    # Create a new datetime object by adding the delta to the base UTC time
    absolute_time_utc = base_datetime_utc + timedelta(seconds=seconds_delta)
    # Format as HH:MM:SS.mmm - always UTC for consistency in transcript backend
    return absolute_time_utc.strftime("%H:%M:%S.%f")[:-3]


def format_timestamp_range(
    start_seconds_delta: float, 
    end_seconds_delta: float, 
    base_datetime_utc: datetime,
    timezone_name: str = "UTC" # Keep internal timestamps UTC for simplicity
    ) -> str:
    """Format a time range based on deltas from a base UTC time."""
    start_time_str = _format_time_delta(start_seconds_delta, base_datetime_utc)
    end_time_str = _format_time_delta(end_seconds_delta, base_datetime_utc)
    return f"[{start_time_str} - {end_time_str} {timezone_name}]"

def filter_hallucinations(text: str) -> str:
    """Remove known hallucination patterns from transcribed text."""
    patterns = [
        r"^\s*Över\.?\s*$",                          
        r"Översättning av.*",                       
        r"www\.sdimedia\.com",                      
        r"^\s*Svensktextning\.nu\s*$",               
        r"^\s*Tack (för|till).*(tittade|hjälpte).*", 
        r"^\s*Trio(,\s*Trio)*\.?\s*$",               
        r"^\s*(Ja|Nej)(,\s*(Ja|Nej))*\.?\s*$",      
        r"^\s*Thank(s?).*watching.*",
        r"^\s*Thank(s?).*listening.*",
        # Add more specific patterns here if needed
    ]
    original_text_repr = repr(text)
    normalized_text = ' '.join(text.lower().split())
    for pattern in patterns:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            logger.debug(f"Filter: Matched pattern '{pattern}' on text {original_text_repr}. Filtering out.")
            return "" 
    return text

def is_valid_transcription(text: str) -> bool:
    if not text: return False
    if re.fullmatch(r'[^\w\s]+', text): return False # Only emojis or non-alphanumerics
    if len(text) < 2: return False # Too short
    return True

def _transcribe_audio_segment_openai(
    audio_file_path: str, 
    openai_api_key: str, 
    language: Optional[str] = None,
    chunk_duration: float = 15.0 # Default, should match segmenting logic
    ) -> Optional[Dict[str, Any]]:
    """
    Transcribes a single audio file segment using OpenAI Whisper API.
    Returns verbose JSON format if successful.
    """
    if not openai_api_key:
        logger.error("OpenAI API key not provided for transcription.")
        return None

    openai.api_key = openai_api_key # Ensure it's set for the openai library if used directly
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            # Using requests directly as per previous MagicAudio structure
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {openai_api_key}"}
            
            data_payload: Dict[str, Any] = {
                'model': 'whisper-1',
                'response_format': 'verbose_json', # Critical for timestamps
                'temperature': 0.0,
            }
            if language: data_payload['language'] = language
            else: data_payload['initial_prompt'] = "Please transcribe the speech accurately." # Generic prompt
            
            data_payload['logprob_threshold'] = -0.7 
            data_payload['compression_ratio_threshold'] = 2.2
            data_payload['no_speech_threshold'] = 0.7 

            files_param = {'file': (os.path.basename(audio_file_path), audio_file, 'audio/wav')} # Assuming WAV
            
            max_retries = 3
            transcription_timeout = 60 # Increased timeout for potentially larger segments
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting transcription for {audio_file_path} (attempt {attempt + 1})...")
                    response = requests.post(url, headers=headers, data=data_payload, files=files_param, timeout=transcription_timeout)
                    response.raise_for_status()
                    transcription = response.json()
                    
                    # Validate transcription structure
                    if 'segments' in transcription and isinstance(transcription['segments'], list):
                        logger.info(f"Successfully transcribed {audio_file_path}.")
                        return transcription
                    else:
                        logger.warning(f"Unexpected transcription format for {audio_file_path}: {transcription}")
                        return None # Or handle differently
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout transcribing {audio_file_path} on attempt {attempt + 1}.")
                    if attempt == max_retries - 1: raise
                except requests.exceptions.RequestException as e:
                    logger.error(f"RequestException transcribing {audio_file_path} on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1: raise
                except Exception as e: # Catch other potential errors like JSONDecodeError early
                    logger.error(f"Generic error transcribing {audio_file_path} on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1: raise
                time.sleep(min(2 ** attempt, 8)) # Exponential backoff
            return None # Should be unreachable if retries fail and raise

    except Exception as e:
        logger.error(f"Transcription error for {audio_file_path}: {e}", exc_info=True)
        return None

# --- Main Service Function ---
def process_audio_segment_and_update_s3(
    temp_segment_wav_path: str, 
    session_data: Dict[str, Any], 
    s3_lock: threading.Lock
    ) -> bool:
    """
    Processes a single audio segment: transcribes it, calculates absolute timestamps,
    and appends the result to the main transcript file on S3.
    """
    s3_transcript_key = session_data.get('s3_transcript_key')
    session_start_time_utc = session_data.get('session_start_time_utc')
    # This is the total duration of audio ALREADY processed and included in S3 transcript
    segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0) 
    language = session_data.get('language') # Assuming language is part of session_data if set

    if not all([temp_segment_wav_path, s3_transcript_key, isinstance(session_start_time_utc, datetime)]):
        logger.error("Missing critical data for processing segment (path, S3 key, or start time).")
        return False
    
    # Duration is now expected to be pre-calculated and stored in session_data
    segment_actual_duration = session_data.get('actual_segment_duration_seconds', 0.0)
    if segment_actual_duration <= 0:
        # This could happen if ffprobe failed in api_server and fallback was also 0.
        # Or if 'actual_segment_duration_seconds' was not set.
        # Attempt to get duration from the WAV file itself as a last resort.
        logger.warning(f"Segment duration from session_data is {segment_actual_duration:.2f}s for {temp_segment_wav_path}. Attempting ffprobe locally.")
        try:
            ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_segment_wav_path]
            duration_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
            segment_actual_duration = float(duration_result.stdout.strip())
            logger.info(f"Local ffprobe successful for {temp_segment_wav_path}: duration {segment_actual_duration:.2f}s")
        except Exception as e_ffprobe:
            logger.error(f"Local ffprobe failed for {temp_segment_wav_path}: {e_ffprobe}. Cannot determine segment duration. Skipping S3 duration update for this segment.")
            # If we can't determine duration, we can still transcribe, but the total processed duration won't be accurate.
            # Setting to a small placeholder or 0 to avoid massive offset errors if transcription is short.
            segment_actual_duration = 0.1 # Small placeholder to show *something* was processed if transcription occurs.

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot transcribe.")
        return False

    s3 = get_s3_client() # This will now use the client with default timeouts
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not configured. Cannot update transcript.")
        return False
    
    logger.info(f"Processing segment {temp_segment_wav_path} for S3 key {s3_transcript_key}. Segment offset: {segment_offset_seconds:.2f}s")

    # 1. Transcribe the audio segment
    transcription_result = _transcribe_audio_segment_openai(temp_segment_wav_path, openai_api_key, language)

    if not transcription_result: # Check if transcription_result itself is None
        logger.warning(f"Transcription call returned None for {temp_segment_wav_path}.")
        session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
        logger.info(f"Updated session {session_data.get('session_id')} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s (transcription call failed)")
        return True
    
    logger.info(f"Transcription result for {temp_segment_wav_path} received. Segments found: {len(transcription_result.get('segments', []))}")

    if not transcription_result.get('segments'): # Check if segments list is empty or missing
        logger.warning(f"Transcription returned no segments for {temp_segment_wav_path}.")
        # Even if transcription fails, we should update the processed duration based on the actual segment length
        # to prevent this segment from being re-processed or causing an offset error for subsequent segments.
        session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
        logger.info(f"Updated session {session_data.get('session_id')} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s (transcription failed/empty)")
        return True # Considered success in terms of processing the audio chunk, even if no text.

    # 2. Acquire lock, download current S3 transcript, append new, upload
    new_transcript_lines = []
    
    logger.debug(f"Preparing to process {len(transcription_result.get('segments', []))} transcribed segments for {temp_segment_wav_path}")
    for segment_idx, segment in enumerate(transcription_result.get('segments', [])):
        logger.debug(f"Processing segment data {segment_idx+1}/{len(transcription_result.get('segments', []))} for {temp_segment_wav_path}")
        raw_text = segment.get('text', '').strip()
        filtered_text = filter_hallucinations(raw_text)

        if is_valid_transcription(filtered_text):
            whisper_start_time = segment.get('start', 0.0)
            whisper_end_time = segment.get('end', 0.0)
            absolute_start_seconds = segment_offset_seconds + whisper_start_time
            absolute_end_seconds = segment_offset_seconds + whisper_end_time
            timestamp_str = format_timestamp_range(absolute_start_seconds, absolute_end_seconds, session_start_time_utc)
            new_transcript_lines.append(f"{timestamp_str} {filtered_text}")
        else:
            logger.debug(f"Segment filtered out: '{raw_text}'")

    if not new_transcript_lines:
        logger.info(f"No valid new transcript lines generated for {temp_segment_wav_path} after filtering.")
        session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
        logger.info(f"Updated session {session_data.get('session_id')} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s (empty transcript segment)")
        return True

    appended_text = "\n".join(new_transcript_lines) + "\n"

    logger.info(f"Finished processing transcribed segments for {temp_segment_wav_path}. Appended text length: {len(appended_text)}. Attempting to acquire S3 lock.")
    with s3_lock:
        session_id_for_log = session_data.get('session_id', 'UNKNOWN_SESSION')
        logger.debug(f"S3_LOCK_ACQUIRED for session {session_id_for_log}, updating {s3_transcript_key}")
        
        try:
            existing_content = ""
            logger.debug(f"S3_GET_ATTEMPT: Bucket='{aws_s3_bucket}', Key='{s3_transcript_key}' for session {session_id_for_log}")
            try:
                obj = s3.get_object(Bucket=aws_s3_bucket, Key=s3_transcript_key) 
                logger.debug(f"S3_GET_SUCCESS: Reading body for session {session_id_for_log}. ContentLength: {obj.get('ContentLength', 'N/A')}")
                body_bytes = obj['Body'].read()
                logger.debug(f"S3_GET_READ_BODY_SUCCESS: Read {len(body_bytes)} bytes for session {session_id_for_log}.")
                existing_content = body_bytes.decode('utf-8')
                logger.info(f"S3_GET_DECODE_SUCCESS: Downloaded existing transcript (length {len(existing_content)}) from {s3_transcript_key}. ETag: {obj.get('ETag')}")
            except s3.exceptions.NoSuchKey:
                logger.info(f"S3_GET_NOSUCHKEY: Transcript {s3_transcript_key} not found for session {session_id_for_log}. Will create new.")
                if not existing_content.startswith("# Transcript"): 
                     header = f"# Transcript - Session {session_id_for_log}\n"
                     header += f"Agent: {session_data.get('agent_name', 'N/A')}, Event: {session_data.get('event_id', 'N/A')}\n"
                     header += f"Session Started (UTC): {session_start_time_utc.isoformat()}\n\n"
                     existing_content = header
            except Exception as e_get:
                logger.error(f"S3_GET_FAIL: Error downloading transcript {s3_transcript_key} for session {session_id_for_log}: {e_get}", exc_info=True)
                return False

            logger.debug(f"S3_PRE_APPEND: Existing length: {len(existing_content)}, Appended text length: {len(appended_text)} for session {session_id_for_log}")
            updated_content = existing_content + appended_text
            
            logger.debug(f"S3_PUT_ATTEMPT: Bucket='{aws_s3_bucket}', Key='{s3_transcript_key}', NewTotalLength={len(updated_content)} for session {session_id_for_log}")
            put_response = s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8')) 
            logger.info(f"S3_PUT_SUCCESS: Appended {len(appended_text)} chars (total {len(updated_content)} chars) to {s3_transcript_key}. ETag: {put_response.get('ETag')}")

            session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
            logger.info(f"Updated session {session_id_for_log} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s using segment duration {segment_actual_duration:.2f}s")

            return True
        except Exception as e_s3_update: 
            logger.error(f"S3_OPERATION_FAIL: Overall error during S3 transcript update for {s3_transcript_key} (session {session_id_for_log}): {e_s3_update}", exc_info=True)
            return False
        finally:
            logger.debug(f"S3_LOCK_RELEASED for session {session_id_for_log}")