# transcription_service.py
import os
import io
import logging
import json
import re
import time # For sleep in transcribe_audio retry
from datetime import datetime, timezone, timedelta
import boto3
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
        client = boto3.client('s3', region_name=aws_region)
        logger.debug(f"S3 client initialized for region {aws_region} in transcription_service.")
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

    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not configured. Cannot update transcript.")
        return False
    
    logger.info(f"Processing segment {temp_segment_wav_path} for S3 key {s3_transcript_key}. Segment offset: {segment_offset_seconds:.2f}s")

    # 1. Transcribe the audio segment
    transcription_result = _transcribe_audio_segment_openai(temp_segment_wav_path, openai_api_key, language)

    if not transcription_result or not transcription_result.get('segments'):
        logger.warning(f"Transcription failed or returned no segments for {temp_segment_wav_path}.")
        return False # Or handle as partial success if some segments transcribed

    # 2. Acquire lock, download current S3 transcript, append new, upload
    new_transcript_lines = []
    processed_segment_duration = 0.0

    for segment in transcription_result.get('segments', []):
        raw_text = segment.get('text', '').strip()
        filtered_text = filter_hallucinations(raw_text)

        if is_valid_transcription(filtered_text):
            # Timestamps from Whisper are relative to the start of the current segment
            whisper_start_time = segment.get('start', 0.0)
            whisper_end_time = segment.get('end', 0.0)

            # Calculate absolute timestamps relative to the session start
            absolute_start_seconds = segment_offset_seconds + whisper_start_time
            absolute_end_seconds = segment_offset_seconds + whisper_end_time
            
            timestamp_str = format_timestamp_range(absolute_start_seconds, absolute_end_seconds, session_start_time_utc)
            new_transcript_lines.append(f"{timestamp_str} {filtered_text}")
            
            # Track the end of the last valid segment to update processed duration
            # whisper_end_time is relative to the current segment.
            # The actual duration of the *entire segment file* is what matters for the offset update.
            # We now use `segment_actual_duration` calculated by ffprobe in api_server.py.
            # No need to use `processed_segment_duration` based on whisper internal segment times anymore.
            pass # Placeholder, `segment_actual_duration` is used later

        else:
            logger.debug(f"Segment filtered out: '{raw_text}'")


    if not new_transcript_lines:
        logger.info(f"No valid new transcript lines generated for {temp_segment_wav_path} after filtering.")
        # Update the processed duration using the actual segment duration if available
        session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
        logger.info(f"Updated session {session_data.get('session_id')} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s (empty transcript segment)")
        return True

    appended_text = "\n".join(new_transcript_lines) + "\n"

    with s3_lock:
        logger.debug(f"Acquired S3 lock for session {session_data.get('session_id')}, updating {s3_transcript_key}")
        try:
            # Download existing content
            existing_content = ""
            try:
                obj = s3.get_object(Bucket=aws_s3_bucket, Key=s3_transcript_key)
                existing_content = obj['Body'].read().decode('utf-8')
                logger.debug(f"Downloaded existing transcript (length {len(existing_content)}) from {s3_transcript_key}")
            except s3.exceptions.NoSuchKey:
                logger.info(f"S3 transcript {s3_transcript_key} not found. Will create new.")
                # Header should have been created by /start endpoint. If not, add it here or ensure /start does it.
                if not existing_content.startswith("# Transcript"): # Basic check
                     header = f"# Transcript - Session {session_data.get('session_id', 'UNKNOWN_SESSION')}\n"
                     header += f"Agent: {session_data.get('agent_name', 'N/A')}, Event: {session_data.get('event_id', 'N/A')}\n"
                     header += f"Session Started (UTC): {session_start_time_utc.isoformat()}\n\n"
                     existing_content = header
            except Exception as e:
                logger.error(f"Error downloading existing transcript {s3_transcript_key}: {e}", exc_info=True)
                return False # Fail if we can't download

            updated_content = existing_content + appended_text
            
            # Upload updated content
            s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8'))
            logger.info(f"Successfully appended {len(appended_text)} chars to S3 transcript {s3_transcript_key}")

            # Update the cumulative processed duration for the session in session_data
            # Use the actual_segment_duration obtained from ffprobe (or its fallback)
            session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
            logger.info(f"Updated session {session_data.get('session_id')} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s using segment duration {segment_actual_duration:.2f}s")

            return True
        except Exception as e:
            logger.error(f"Error during S3 transcript update for {s3_transcript_key}: {e}", exc_info=True)
            return False
        finally:
            logger.debug(f"Released S3 lock for session {session_data.get('session_id')}")