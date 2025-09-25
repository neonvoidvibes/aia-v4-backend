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
import mimetypes # Import the mimetypes library
import math # For chunking calculations
import concurrent.futures # For parallel processing

from utils.hallucination_detector import DetectorState, Word, maybe_trim_repetition

# Configure logging
logger = logging.getLogger(__name__)

# Import session state model
from models.session_state import SessionState

def is_transient_provider_error(e) -> bool:
    """Classify errors as transient (retryable) or permanent."""
    import socket
    from requests.exceptions import ConnectionError, Timeout, HTTPError

    # Network and DNS errors
    if isinstance(e, (socket.gaierror, ConnectionError, Timeout,
                      ConnectionResetError, ConnectionAbortedError)):
        return True

    # HTTP status codes
    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
        status_code = e.response.status_code
        return status_code >= 500 or status_code == 429

    if hasattr(e, 'status_code'):
        return e.status_code >= 500 or e.status_code == 429

    # OpenAI specific errors
    if 'openai' in str(type(e)).lower():
        error_msg = str(e).lower()
        if any(term in error_msg for term in ['timeout', 'connection', 'network', 'rate limit']):
            return True

    return False

def save_for_retry(wav_path: str, session_data: Dict[str, Any], segment_offset_seconds: float, seq: int) -> bool:
    """Save failed segment to disk queue for retry."""
    try:
        session_id = session_data.get("session_id")
        if not session_id:
            return False

        # Create retry directory
        retry_dir = f"tmp/retry_segments/{session_id}"
        os.makedirs(retry_dir, exist_ok=True)

        # Generate unique filename with sequence number
        retry_base = f"{seq:06d}_{int(time.time())}"
        retry_json_path = os.path.join(retry_dir, f"{retry_base}.json")
        retry_wav_path = os.path.join(retry_dir, f"{retry_base}.wav")

        # Copy WAV file to retry location
        import shutil
        shutil.copy2(wav_path, retry_wav_path)

        # Create retry metadata
        retry_metadata = {
            "session_id": session_id,
            "seq": seq,
            "segment_offset_seconds": segment_offset_seconds,
            "wav_path": retry_wav_path,
            "original_wav_path": wav_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "last_error": None,
            "session_data_snapshot": {
                "language_setting_from_client": session_data.get("language_setting_from_client"),
                "vad_aggressiveness_from_client": session_data.get("vad_aggressiveness_from_client"),
                "last_successful_transcript": session_data.get("last_successful_transcript", ""),
                "s3_transcript_key": session_data.get("s3_transcript_key"),
                "session_start_time_utc": session_data.get("session_start_time_utc").isoformat() if session_data.get("session_start_time_utc") else None,
                "agent_name": session_data.get("agent_name"),
                "event_id": session_data.get("event_id"),
                "user_id": session_data.get("user_id"),
            }
        }

        # Save metadata
        with open(retry_json_path, 'w') as f:
            json.dump(retry_metadata, f, indent=2)

        logger.info(f"Saved segment seq={seq} for retry in session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to save segment for retry: {e}")
        return False

def load_retry_segment(retry_json_path: str) -> Optional[Dict[str, Any]]:
    """Load retry segment metadata from disk."""
    try:
        if not os.path.exists(retry_json_path):
            return None

        with open(retry_json_path, 'r') as f:
            retry_data = json.load(f)

        # Validate required fields
        required_fields = ["session_id", "seq", "wav_path", "created_at"]
        if not all(field in retry_data for field in required_fields):
            logger.warning(f"Invalid retry metadata in {retry_json_path}")
            return None

        # Check if WAV file exists
        if not os.path.exists(retry_data["wav_path"]):
            logger.warning(f"Missing WAV file for retry: {retry_data['wav_path']}")
            return None

        return retry_data

    except Exception as e:
        logger.error(f"Failed to load retry segment {retry_json_path}: {e}")
        return None

def cleanup_retry_files(retry_json_path: str) -> None:
    """Clean up retry files after successful processing or max attempts."""
    try:
        retry_data = load_retry_segment(retry_json_path)
        if retry_data:
            wav_path = retry_data.get("wav_path")
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        if os.path.exists(retry_json_path):
            os.remove(retry_json_path)

        logger.debug(f"Cleaned up retry files for {retry_json_path}")

    except Exception as e:
        logger.error(f"Failed to cleanup retry files {retry_json_path}: {e}")

# Provider router with fallback support
from transcription_providers.base import TranscriptionProvider
_PRIMARY_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "deepgram").strip().lower()
_FALLBACK_PROVIDER = os.getenv("TRANSCRIBE_FALLBACK", "whisper").strip().lower()
_PROVIDER_NAME = _PRIMARY_PROVIDER  # For backward compatibility

logger.info(f"Initializing transcription providers: primary={_PRIMARY_PROVIDER}, fallback={_FALLBACK_PROVIDER}")

# Initialize primary provider
if _PRIMARY_PROVIDER == "deepgram":
    from transcription_providers.deepgram_provider import DeepgramProvider as _PrimaryProvider
    _primary_provider: TranscriptionProvider = _PrimaryProvider(os.getenv("DEEPGRAM_API_KEY"))
    logger.info("Deepgram primary provider initialized successfully")
else:
    from transcription_providers.whisper_provider import WhisperProvider as _PrimaryProvider
    _primary_provider: TranscriptionProvider = _PrimaryProvider(os.getenv("OPENAI_API_KEY"))
    logger.info("Whisper primary provider initialized successfully")

# Initialize fallback provider
_fallback_provider: Optional[TranscriptionProvider] = None
if _FALLBACK_PROVIDER and _FALLBACK_PROVIDER != _PRIMARY_PROVIDER:
    if _FALLBACK_PROVIDER == "whisper":
        from transcription_providers.whisper_provider import WhisperProvider as _FallbackProvider
        _fallback_provider = _FallbackProvider(os.getenv("OPENAI_API_KEY"))
        logger.info("Whisper fallback provider initialized successfully")
    elif _FALLBACK_PROVIDER == "deepgram":
        from transcription_providers.deepgram_provider import DeepgramProvider as _FallbackProvider
        _fallback_provider = _FallbackProvider(os.getenv("DEEPGRAM_API_KEY"))
        logger.info("Deepgram fallback provider initialized successfully")

# Backward compatibility
_provider = _primary_provider

# Sticky fallback with timed Deepgram probe functions
def _mark_deepgram_failure(state: SessionState):
    """Mark Deepgram as failed and switch to Whisper fallback."""
    now = time.time()
    # switch to whisper and set retry time if not already in fallback
    if state.provider_current != "whisper":
        state.provider_current = "whisper"
        state.provider_fallback_at = now
        state.provider_retry_after = now + state.provider_cooldown_sec
        logger.info(f"Session {state.id}: Switched to Whisper fallback, retry after {state.provider_retry_after}")

def _maybe_probe_deepgram(state: SessionState) -> bool:
    """Check if we should probe Deepgram for availability."""
    # only after cooldown, only one probe at a time
    if state.provider_current != "whisper":
        return False
    if state.provider_retry_after is None or time.time() < state.provider_retry_after:
        return False
    if state.provider_probe_inflight:
        return False
    state.provider_probe_inflight = True
    logger.info(f"Session {state.id}: Starting Deepgram probe")
    return True

def _complete_probe(state: SessionState, success: bool):
    """Complete a Deepgram probe attempt."""
    state.provider_probe_inflight = False
    if success:
        state.provider_current = "deepgram"
        state.provider_fallback_at = None
        state.provider_retry_after = None
        logger.info(f"Session {state.id}: Deepgram probe successful, switched back to Deepgram")
    else:
        # push retry window forward (exponential-ish backoff)
        state.provider_retry_after = time.time() + min(state.provider_cooldown_sec, 1800)
        logger.info(f"Session {state.id}: Deepgram probe failed, retry after {state.provider_retry_after}")

def _get_session_websocket(session_id: str):
    """Get WebSocket connection for a session."""
    try:
        # Import here to avoid circular imports
        from api_server import active_sessions
        if session_id in active_sessions:
            return active_sessions[session_id].get("websocket_connection")
    except Exception:
        pass
    return None

def _send_ws_status(state: SessionState, paused: bool, reason: str):
    """Send WebSocket status update (PAUSED/RESUMED)."""
    if not bool(os.getenv("TRANSCRIBE_WS_STATUS_EVENTS", "true").lower() in ("1", "true", "yes")):
        return

    now = time.time()
    debounce_ms = int(os.getenv("TRANSCRIBE_WS_DEBOUNCE_MS", "1500"))

    if paused:
        if state.net_paused:
            return
        # debounce PAUSED if sent very recently
        if state.last_pause_sent_at and (now - state.last_pause_sent_at) * 1000 < debounce_ms:
            return

        payload = {"type": "transcription_status", "state": "PAUSED", "reason": reason}
        state.net_paused = True
        state.last_pause_sent_at = now
    else:
        if not state.net_paused:
            return

        payload = {"type": "transcription_status", "state": "RESUMED"}
        state.net_paused = False

    # Send WebSocket message
    ws = _get_session_websocket(state.id)
    if ws:
        try:
            ws.send(json.dumps(payload))
            logger.info(f"Session {state.id}: Sent transcription status {payload['state']} ({reason})")
        except Exception as e:
            logger.warning(f"Session {state.id}: Failed to send WebSocket status: {e}")
    else:
        logger.debug(f"Session {state.id}: No WebSocket connection for status {payload['state']}")

def _on_provider_success(state: SessionState):
    """Handle provider success - clear PAUSED state."""
    _send_ws_status(state, paused=False, reason="")

def _on_provider_failure(state: SessionState, e: Exception):
    """Handle provider failure - notify UI once."""
    logger.error(f"Session {state.id}: Transcription error: {e}")
    # Notify UI once; reason is generic "network"
    _send_ws_status(state, paused=True, reason="network")

def _get_deepgram_params(language: Optional[str], vad_aggressiveness: Optional[int]) -> Dict[str, Any]:
    """Get Deepgram parameters with language detection logic."""
    base = {
        'model': 'nova-3',
        'smart_format': 'true',
        'punctuate': 'true',
        'words': 'true',
        'timestamps': 'true'
    }

    # Add VAD mode if specified
    if vad_aggressiveness is not None:
        vad_mode = {1: 1, 2: 2, 3: 3}.get(vad_aggressiveness, 2)
        base['vad_mode'] = str(vad_mode)

    # Handle language parameter
    if language and language != "any":
        base['language'] = language
        base['detect_language'] = 'false'
    else:
        base['detect_language'] = 'true'

    return base

def get_current_transcription_provider() -> str:
    """Get the name of the currently active transcription provider."""
    return _PROVIDER_NAME.capitalize()

def get_default_vad_aggressiveness() -> int:
    """Get the default VAD aggressiveness level for the current transcription provider."""
    if _PROVIDER_NAME == "deepgram":
        return 1  # Quiet environment - Deepgram has good built-in noise handling
    else:  # whisper
        return 2  # Mid environment - Whisper needs more VAD filtering

def format_language_for_header(language_code: str) -> str:
    """Format language code for display in transcript headers."""
    language_map = {
        "any": "Auto-detect",
        "sv": "Swedish",
        "en": "English",
        "da": "Danish",
        "no": "Norwegian",
        "fi": "Finnish",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean"
    }
    return language_map.get(language_code, language_code.upper())

# --- Utility Functions (adapted from magic_audio.py) ---

def get_s3_client() -> Optional[boto3.client]:
    """Initializes and returns an S3 client, or None on failure."""
    try:
        aws_region = os.getenv('AWS_REGION')
        if not aws_region:
            logger.error("AWS_REGION environment variable not found.")
            return None
        
        default_boto_config = BotoConfig(
            connect_timeout=15, 
            read_timeout=45,    
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
    if seconds_delta < 0: seconds_delta = 0 
    absolute_time_utc = base_datetime_utc + timedelta(seconds=seconds_delta)
    return absolute_time_utc.strftime("%H:%M:%S.%f")[:-3]


def format_timestamp_range(
    start_seconds_delta: float, 
    end_seconds_delta: float, 
    base_datetime_utc: datetime,
    timezone_name: str = "UTC" 
    ) -> str:
    """Format a time range based on deltas from a base UTC time."""
    start_time_str = _format_time_delta(start_seconds_delta, base_datetime_utc)
    end_time_str = _format_time_delta(end_seconds_delta, base_datetime_utc)
    return f"[{start_time_str} - {end_time_str} {timezone_name}]"

def detect_repetition_hallucination(text: str, threshold: float = 0.7) -> bool:
    """Detect if text is likely a hallucination based on repetition patterns"""
    words = text.split()
    if len(words) < 3:
        return False
    
    # Check for phrase repetition within the text
    word_count = len(words)
    for phrase_len in range(2, min(word_count // 2 + 1, 6)):  # Check 2-5 word phrases
        phrases = [' '.join(words[i:i+phrase_len]) for i in range(word_count - phrase_len + 1)]
        unique_phrases = set(phrases)
        repetition_ratio = 1 - (len(unique_phrases) / len(phrases))
        if repetition_ratio > threshold:
            logger.debug(f"Detected repetition hallucination in text: '{text[:100]}...' (ratio: {repetition_ratio:.2f})")
            return True
    
    return False

def detect_cross_segment_repetition(segments: List[Dict], window_size: int = 10) -> List[Dict]:
    """Filter out segments that repeat recent content"""
    filtered_segments = []
    recent_texts = []
    
    for segment in segments:
        text = segment.get('text', '').strip()
        
        # Check if this text appeared recently
        if text in recent_texts[-window_size:]:
            logger.debug(f"Filtering repetitive segment: '{text}'")
            continue
            
        filtered_segments.append(segment)
        recent_texts.append(text)
        
        # Keep only recent history
        if len(recent_texts) > window_size:
            recent_texts.pop(0)
    
    return filtered_segments

def detect_single_word_loops(segments: List[Dict], session_data: Dict[str, Any], segment_offset_seconds: float) -> List[Dict]:
    """Filter out single words that appear too frequently in loops (like 'Okej' repeating)"""
    filtered_segments = []
    
    # Get persistent word frequency tracker from session data
    if 'word_frequency_tracker' not in session_data:
        session_data['word_frequency_tracker'] = {}
    word_frequency_tracker = session_data['word_frequency_tracker']
    
    max_single_word_frequency = 3  # Allow 3 occurrences, filter 4th+
    time_window_minutes = 3.0  # 3-minute sliding window
    
    for segment in segments:
        text = segment.get('text', '').strip()
        # Calculate absolute timestamp relative to session start
        absolute_timestamp = segment_offset_seconds + segment.get('start', 0.0)
        
        # Normalize text for comparison (remove punctuation, convert to lowercase)
        normalized_text = re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        # Only apply this filter to single words (no spaces = single word)
        if normalized_text and ' ' not in normalized_text:
            # Clean up old entries outside the time window
            cutoff_time = absolute_timestamp - (time_window_minutes * 60)
            if normalized_text in word_frequency_tracker:
                word_frequency_tracker[normalized_text] = [
                    timestamp for timestamp in word_frequency_tracker[normalized_text] 
                    if timestamp > cutoff_time
                ]
            
            # Count current occurrences within the time window
            current_count = len(word_frequency_tracker.get(normalized_text, []))
            
            # Filter if we already have max occurrences
            if current_count >= max_single_word_frequency:
                logger.debug(f"Filtering looping single word: '{text}' (would be occurrence #{current_count + 1} in last {time_window_minutes} minutes)")
                continue
            
            # Track this occurrence
            if normalized_text not in word_frequency_tracker:
                word_frequency_tracker[normalized_text] = []
            word_frequency_tracker[normalized_text].append(absolute_timestamp)
        
        filtered_segments.append(segment)
    
    return filtered_segments

def transcribe_chunk_wrapper(args):
    """Wrapper function for parallel chunk transcription."""
    chunk_path, chunk_index, total_chunks, openai_api_key, language_setting, chunk_context, progress_callback = args
    
    # Add staggered delay to reduce concurrent load on OpenAI API
    import time
    import random
    stagger_delay = chunk_index * 0.3 + random.uniform(0, 0.2)  # 0.3s per chunk + jitter
    if stagger_delay > 0:
        logger.debug(f"Staggering chunk {chunk_index + 1} by {stagger_delay:.1f}s to reduce API load")
        time.sleep(stagger_delay)
    
    logger.info(f"Starting parallel transcription of chunk {chunk_index + 1}/{total_chunks}: {os.path.basename(chunk_path)}")
    
    # Update progress if callback provided - report start of processing with percentage only
    if progress_callback:
        try:
            progress_percent = (chunk_index / total_chunks) * 0.8 + 0.1  # 10-90% range
            progress_callback(chunk_index, total_chunks, f"Processing... {int(progress_percent * 100)}%")
        except Exception as e:
            # If progress callback raises exception (e.g., cancellation), stop processing
            logger.info(f"Chunk {chunk_index + 1} processing stopped: {e}")
            return None
    
    # Smart chunk retry with improved error handling  
    chunk_max_retries = 3  # Reduced for parallel processing
    chunk_result = None
    
    for chunk_attempt in range(chunk_max_retries):
        try:
            chunk_result = _transcribe_audio_segment_openai(
                chunk_path, openai_api_key, language_setting, chunk_context
            )
            if chunk_result and chunk_result.get('segments'):
                logger.info(f"Chunk {chunk_index + 1}/{total_chunks} transcribed successfully on attempt {chunk_attempt + 1}")
                return {
                    'chunk_index': chunk_index,
                    'chunk_path': chunk_path,
                    'result': chunk_result,
                    'success': True
                }
            else:
                raise Exception("No segments returned from transcription")
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Chunk {chunk_index + 1} attempt {chunk_attempt + 1} failed: {error_msg}")
            
            # Don't retry on certain errors
            if any(term in error_msg.lower() for term in ['client error', 'file too large', 'api key', 'unauthorized', 'cancelled']):
                logger.error(f"Non-recoverable error for chunk {chunk_index + 1}: {error_msg}")
                return {
                    'chunk_index': chunk_index,
                    'chunk_path': chunk_path,
                    'result': None,
                    'success': False,
                    'error': error_msg
                }
            
            if chunk_attempt == chunk_max_retries - 1:
                logger.error(f"Failed to transcribe chunk {chunk_index + 1} after {chunk_max_retries} attempts: {error_msg}")
                return {
                    'chunk_index': chunk_index,
                    'chunk_path': chunk_path,
                    'result': None,
                    'success': False,
                    'error': error_msg
                }
            
            # Smart backoff for chunk retries - shorter delays for parallel processing
            if 'rate limit' in error_msg.lower():
                chunk_delay = min(15, 3 * (2 ** chunk_attempt))
            elif 'server error' in error_msg.lower() or '5' in error_msg[:3]:
                chunk_delay = min(10, 2 * (2 ** chunk_attempt))
            else:
                chunk_delay = min(5, 2 ** chunk_attempt)
            
            logger.info(f"Retrying chunk {chunk_index + 1} in {chunk_delay} seconds...")
            time.sleep(chunk_delay)
    
    return {
        'chunk_index': chunk_index,
        'chunk_path': chunk_path,
        'result': None,
        'success': False,
        'error': 'Max retries exceeded'
    }

def _transcribe_chunks_parallel(chunk_paths, openai_api_key, language_setting, rolling_context_prompt, chunk_duration, progress_callback):
    """Transcribe audio chunks in parallel for better performance."""
    all_segments = []
    combined_text_parts = []
    current_time_offset = 0.0
    
    # Prepare arguments for parallel processing
    chunk_args = []
    for i, chunk_path in enumerate(chunk_paths):
        # Use minimal context to avoid hallucinations in parallel processing
        chunk_context = rolling_context_prompt if i == 0 else ""
        chunk_args.append((
            chunk_path, i, len(chunk_paths), openai_api_key, 
            language_setting, chunk_context, progress_callback
        ))
    
    # Process chunks in parallel with controlled concurrency - reduced for better stability
    max_workers = min(2, len(chunk_paths))  # Limit to 2 concurrent to reduce OpenAI API errors
    completed_chunks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk jobs
        future_to_chunk = {
            executor.submit(transcribe_chunk_wrapper, args): args[1] 
            for args in chunk_args
        }
        
        # Collect results as they complete and report progress
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                if result is None:  # Cancelled
                    logger.info("Parallel transcription cancelled")
                    executor.shutdown(wait=False)
                    return [], []
                    
                completed_chunks.append(result)
                completed_count += 1
                
                # Report completion progress with percentage only
                if progress_callback:
                    try:
                        progress_percent = (completed_count / len(chunk_paths)) * 0.8 + 0.1  # 10-90% range for transcription
                        progress_callback(completed_count, len(chunk_paths), f"Processing... {int(progress_percent * 100)}%")
                    except:
                        pass  # Don't fail on callback errors
                
                if result['success']:
                    logger.info(f"Chunk {chunk_index + 1} completed successfully ({completed_count}/{len(chunk_paths)})")
                else:
                    logger.warning(f"Chunk {chunk_index + 1} failed: {result.get('error', 'Unknown error')} ({completed_count}/{len(chunk_paths)})")
                    
            except Exception as e:
                logger.error(f"Exception in parallel chunk processing: {e}")
                completed_count += 1
                completed_chunks.append({
                    'chunk_index': chunk_index,
                    'chunk_path': chunk_paths[chunk_index],
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
                
                # Report completion progress even for failures
                if progress_callback:
                    try:
                        progress_percent = (completed_count / len(chunk_paths)) * 0.8 + 0.1
                        progress_callback(completed_count, len(chunk_paths), f"Processing... {int(progress_percent * 100)}%")
                    except:
                        pass
    
    # Sort results by chunk index to maintain order
    completed_chunks.sort(key=lambda x: x['chunk_index'])
    
    # Combine results and adjust timestamps
    for chunk_result in completed_chunks:
        if chunk_result['success'] and chunk_result['result']:
            chunk_data = chunk_result['result']
            chunk_segments = chunk_data.get('segments', [])
            
            # Adjust timestamps to account for chunk offset
            for segment in chunk_segments:
                segment['start'] += current_time_offset
                segment['end'] += current_time_offset
            
            all_segments.extend(chunk_segments)
            
            # Add text from this chunk
            chunk_text = chunk_data.get('text', '').strip()
            if chunk_text:
                combined_text_parts.append(chunk_text)
            
            # Update time offset for next chunk
            if chunk_segments:
                last_segment_end = max(seg.get('end', 0) for seg in chunk_segments)
                current_time_offset = last_segment_end
            else:
                current_time_offset += chunk_duration
        else:
            # Failed chunk - still advance time offset
            current_time_offset += chunk_duration
    
    return all_segments, combined_text_parts

def _transcribe_chunks_sequential(chunk_paths, openai_api_key, language_setting, rolling_context_prompt, chunk_duration, progress_callback):
    """Transcribe audio chunks sequentially (fallback for single chunks or when parallel fails)."""
    all_segments = []
    combined_text_parts = []
    current_time_offset = 0.0
    
    try:
        for i, chunk_path in enumerate(chunk_paths):
            if progress_callback:
                progress_callback(i, len(chunk_paths), f"Processing chunk {i+1}/{len(chunk_paths)}")
            
            logger.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
            
            # Use context from previous chunk for continuity
            chunk_context = rolling_context_prompt if i == 0 else " ".join(combined_text_parts[-1:])
            
            # Use the parallel wrapper for consistency
            chunk_args = (chunk_path, i, len(chunk_paths), openai_api_key, language_setting, chunk_context, progress_callback)
            chunk_result = transcribe_chunk_wrapper(chunk_args)
            
            if chunk_result is None:  # Cancelled
                logger.info("Sequential transcription cancelled")
                return [], []
            
            if chunk_result['success'] and chunk_result['result']:
                chunk_data = chunk_result['result']
                chunk_segments = chunk_data.get('segments', [])
                
                # Adjust timestamps to account for chunk offset
                for segment in chunk_segments:
                    segment['start'] += current_time_offset
                    segment['end'] += current_time_offset
                
                all_segments.extend(chunk_segments)
                
                # Add text from this chunk
                chunk_text = chunk_data.get('text', '').strip()
                if chunk_text:
                    combined_text_parts.append(chunk_text)
                
                # Update time offset for next chunk
                if chunk_segments:
                    last_segment_end = max(seg.get('end', 0) for seg in chunk_segments)
                    current_time_offset = last_segment_end
                else:
                    current_time_offset += chunk_duration
            else:
                logger.warning(f"Failed to transcribe chunk {i+1}: {chunk_result.get('error', 'Unknown error')}")
                current_time_offset += chunk_duration
                
    except Exception as e:
        logger.error(f"Exception in sequential chunk processing: {e}")
        
    return all_segments, combined_text_parts

def _generate_failure_advice(chunk_paths, processing_time):
    """Generate intelligent advice for transcription failures."""
    file_count = len(chunk_paths) if chunk_paths else 0
    
    if processing_time < 30:
        return "API connection issue. Try again in a few minutes."
    elif processing_time > 300:  # 5+ minutes
        return "Request timeout. Try splitting into smaller files or check your internet connection."
    elif file_count > 5:
        return "Large file processing failed. Consider using smaller audio files (under 50MB)."
    else:
        return "OpenAI Whisper service may be experiencing issues. Please try again later."

def _should_use_fallback_strategy(consecutive_failures, file_size_mb, processing_time):
    """Determine if we should use fallback strategy based on failure patterns."""
    # Use fallback if:
    # 1. Multiple consecutive failures
    # 2. Large file with repeated timeouts  
    # 3. Extended processing time with poor results
    
    if consecutive_failures >= 2:
        return True, "Multiple API failures detected"
    
    if file_size_mb > 100 and processing_time > 600:  # 10+ minutes for large files
        return True, "Large file processing taking too long"
        
    if processing_time > 300 and consecutive_failures > 0:  # 5+ minutes with failures
        return True, "Extended processing time with failures"
    
    return False, None

def filter_by_duration_and_confidence(segment: Dict) -> bool:
    """Filter segments that are too short or have low confidence"""
    duration = segment.get('end', 0) - segment.get('start', 0)
    avg_logprob = segment.get('avg_logprob', 0)
    no_speech_prob = segment.get('no_speech_prob', 0)
    
    # Filter very short segments with low confidence
    if duration < 1.0 and (avg_logprob < -0.5 or no_speech_prob > 0.3):
        logger.debug(f"Filtering short low-confidence segment: duration={duration:.2f}s, logprob={avg_logprob:.2f}, no_speech={no_speech_prob:.2f}")
        return False
    
    # Filter segments that are likely silent
    if no_speech_prob > 0.8:
        logger.debug(f"Filtering likely silent segment: no_speech_prob={no_speech_prob:.2f}")
        return False
        
    return True

def analyze_silence_gaps(segments: List[Dict]) -> List[Dict]:
    """Filter segments that appear after long silence gaps (likely hallucinations)"""
    filtered = []
    
    for i, segment in enumerate(segments):
        if i == 0:
            filtered.append(segment)
            continue
            
        prev_end = segments[i-1].get('end', 0)
        curr_start = segment.get('start', 0)
        gap = curr_start - prev_end
        
        # If there's a large gap (>5 seconds), be more suspicious
        if gap > 5.0:
            no_speech_prob = segment.get('no_speech_prob', 0)
            if no_speech_prob > 0.5:  # Lower threshold after silence
                logger.debug(f"Filtering post-silence segment: '{segment.get('text', '')}' (gap={gap:.2f}s, no_speech={no_speech_prob:.2f})")
                continue
                
        filtered.append(segment)
    
    return filtered

def filter_hallucinations(text: str) -> str:
    """Remove known hallucination patterns from transcribed text."""
    # First check for repetition-based hallucinations
    if detect_repetition_hallucination(text):
        logger.debug(f"Filtering repetition-based hallucination: '{text[:100]}...'")
        return ""
    
    patterns = [
        # Existing patterns
        r"^\s*Över\.?\s*$",                          
        r"Översättning av.*",                       
        r"www\.sdimedia\.com",                      
        r"^\s*Svensktextning\.nu\s*$",               
        r"^\s*Tack (för|till).*(tittade|hjälpte).*", 
        r"^\s*Trio(,\s*Trio)*\.?\s*$",               
        r"^\s*(Ja|Nej)(,\s*(Ja|Nej))*\.?\s*$",      
        r"^\s*Thank(s?).*watching.*", # Made more general below
        r"^\s*Thank(s?).*listening.*", # Made more general below

        # New patterns from examples
        r"(?i)^\s*Share this video with your friends.*",
        r"(?i)^\s*Якщо вам подобається їжа.*", # Ukrainian
        r"(?i)^\s*今日の映像はここまでです!.*", # Japanese
        r"(?i)^\s*最後まで視聴してくださって 本当にありがとうございます。.*", # Japanese
        r"(?i)^\s*次の映像でまた会いましょう!.*", # Japanese
        r"(?i)^\s*Дякуємо за перегляд і до зустрічі у наступному відео!.*", # Ukrainian
        r"^\s*[ಠ]+$", # Symbol repetitions
        r"^\s*[୧]+$",
        r"^\s*[ស្ក]+$", # Khmer-like symbols
        r"^\s*([0-9]\.){5,}[0-9]?\s*$", # For "1.1.1.1..."
        r"(?i)^\s*Thank you for joining us.*",
        r"(?i)^\s*I'll see you next time.*",
        r"(?i)^\s*「パパ!」.*", # Japanese "Papa!"
        r"(?i)^\s*「ネズミがいない!」.*", # Japanese "Nezumi ga inai!"
        r"(?i)^\s*If you enjoyed this video, please subscribe, like, and leave a comment.*",
        r"(?i)^\s*1\.5cm x 1\.5cm\s*$",
        r"(?i)^\s*チャンネル登録をお願いいたします.*", # Japanese "Channel registration please"
        r"(?i)^\s*If you enjoyed the video, please subscribe, like, and set an alarm.*",
        r"(?i)^\s*If you like(d)? the video, don't forget to like it and subscribe to the channel\s*\.?$", # Covers multiple variations
        r"(?i)^\s*If you like this video, don't forget to like it and subscribe to my channel\s*\.?$",
        r"(?i)^\s*1\.5 tbsp of lemon juice\s*$",
        r"(?i)^\s*Thank you very much\s*\.?$",
        r"(?i)^\s*If you like this video, don't forget to give it a thumbs up and subscribe to my channel\s*\.?$",
        r"(?i)^\s*Om du gillar den här videon, gör gärna en tumme upp.*", # Swedish like/comment
        r"(?i)^\s*プレイヤーを選択すると.*", # Japanese player select
        r"(?i)^\s*3Dプロジェクターを作成するプロジェクトを作成するプロジェクト.*", # Japanese 3D projector
        r"(?i)^\s*ご視聴いただきありがとうございます.*", # Japanese "Thank you for watching"
        r"(?i)^\s*Thank you\s*\.?$",
        r"(?i)^\s*I hope you have a great day, and I'll see you in the next video\s*\.?$",
        r"(?i)^\s*Bye!\s*$",
        r"(?i)^\s*I hope you have a wonderful day, and I will see you in the next video\s*\.?$",
        r"(?i)^\s*Якщо вам сподобалося це відео, не забудьте дати менI лайк.*", # Ukrainian like/subscribe
        r"(?i)^\s*Дякую за перегляд!\s*$", # Ukrainian "Thanks for watching"
        r"(?i)^\s*하지만 이것은 가장 어려운 과정입니다.*", # Korean
        r"(?i)^\s*이 과정에 대해 더 자세히 알아보시기 바랍니다.*", # Korean
        r"(?i)^\s*감사합니다\s*\.?$", # Korean "Thank you"
        r"(?i)^\s*Я амірую!\s*$", # Cyrillic phrase
        
        # Enhanced language-specific patterns
        r"(?i)^\s*Tack för att du har tittat.*", # Swedish thanks variations
        r"(?i)^\s*(Ja|Nej),?\s*(men samtidigt).*", # Swedish filler phrases  
        r"(?i)^\s*이 영상은 유료광고.*", # Korean ad disclaimers
        r"(?i)^\s*\d+\.?\d*\s*cm\s*x\s*\d+\.?\d*\s*cm\s*$", # Measurement patterns
        r"(?i)^\s*\d+\.\d+kg\s+.*썰어주세요.*", # Korean cooking instructions
        
        # Common standalone filler words that are often hallucinated
        r"(?i)^\s*okay\.?\s*$", # Standalone "okay"
        r"(?i)^\s*okej\.?\s*$", # Standalone Swedish "okej"
        
        # Short, potentially out-of-context phrases - to be used cautiously, rely on thresholds first
        # r"^\s*What\s*\?\s*$",
        # r"^\s*Oh\s*\.?\s*$",
        # r"^\s*Sorry\s*\.?\s*$",
        # r"^\s*Have a nice day!\s*$"
    ]
    original_text_repr = repr(text)
    # For case-insensitive matching on the whole text for some patterns
    text_lower_for_some_checks = text.lower()
    
    # For most patterns, we compile them with re.IGNORECASE and check directly.
    # No need to manually lower `text` for those.
    # Let's refine how normalized_text is used.
    # Keep original text for direct pattern application if pattern handles case.

    for pattern_str in patterns:
        try:
            # Most patterns now include (?i) for case-insensitivity directly
            if re.search(pattern_str, text):
                logger.debug(f"Filter: Matched pattern '{pattern_str}' on text {original_text_repr}. Filtering out.")
                return ""
        except re.error as e:
            logger.error(f"Regex error with pattern '{pattern_str}': {e}")
            continue # Skip faulty regex
            
    return text

def is_prompt_echo(new_text: str, prompt_text: str, threshold: float = 0.7) -> bool:
    """
    Detect if new text is too similar to the prompt, indicating a feedback loop.
    
    Args:
        new_text: The newly transcribed text
        prompt_text: The prompt that was sent to Whisper
        threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        True if the text appears to be echoing the prompt
    """
    if not prompt_text or not new_text:
        return False
    
    new_text_lower = new_text.lower().strip()
    prompt_text_lower = prompt_text.lower().strip()
    
    # Check if prompt appears at start of new text (exact substring match)
    if new_text_lower.startswith(prompt_text_lower):
        logger.debug(f"Prompt echo detected: New text starts with prompt. New: '{new_text[:100]}...', Prompt: '{prompt_text}'")
        return True
    
    # Check if new text is contained in prompt (reverse echo)
    if prompt_text_lower.startswith(new_text_lower) and len(new_text_lower) > 5:
        logger.debug(f"Reverse prompt echo detected: Prompt starts with new text. New: '{new_text}', Prompt: '{prompt_text}'")
        return True
    
    # Check overall similarity using sequence matcher
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, new_text_lower, prompt_text_lower).ratio()
    if similarity > threshold:
        logger.debug(f"High similarity prompt echo detected: similarity={similarity:.3f}, threshold={threshold}. New: '{new_text}', Prompt: '{prompt_text}'")
        return True
    
    # Check if all words from prompt appear in new text (word-level echo)
    prompt_words = set(prompt_text_lower.split())
    new_words = set(new_text_lower.split())
    if len(prompt_words) >= 2 and prompt_words.issubset(new_words):
        logger.debug(f"Word-level prompt echo detected: All prompt words found in new text. New: '{new_text}', Prompt: '{prompt_text}'")
        return True
    
    return False

def is_valid_transcription(text: str) -> bool:
    if not text: return False
    if re.fullmatch(r'[^\w\s]+', text): return False 
    if len(text) < 2: return False 
    return True

def get_audio_duration(file_path: str) -> float:
    """Get audio file duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0

def split_audio_file(input_path: str, chunk_duration: float = 600) -> List[str]:
    """
    Split audio file into smaller chunks for transcription.
    
    Args:
        input_path: Path to the audio file to split
        chunk_duration: Duration of each chunk in seconds (default 10 minutes)
    
    Returns:
        List of paths to the chunk files
    """
    total_duration = get_audio_duration(input_path)
    if total_duration <= 0:
        logger.error(f"Could not determine duration for {input_path}, cannot split")
        return []
    
    num_chunks = math.ceil(total_duration / chunk_duration)
    logger.info(f"Splitting {input_path} ({total_duration:.2f}s) into {num_chunks} chunks of {chunk_duration}s each")
    
    chunk_paths = []
    base_name = os.path.splitext(input_path)[0]
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = f"{base_name}_chunk_{i+1:03d}.m4a"
        
        # Use ffmpeg to extract the chunk
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output files
            chunk_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            chunk_paths.append(chunk_path)
            logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create chunk {i+1}: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            # Continue with other chunks even if one fails
    
    return chunk_paths

def check_file_size_for_openai(file_path: str, max_size_mb: int = 24) -> bool:
    """
    Check if file size is within OpenAI's limits.
    Using 24MB as limit instead of 25MB to provide buffer for API overhead.
    """
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"File {os.path.basename(file_path)} size: {file_size_mb:.2f} MB")
        return file_size_mb <= max_size_mb
    except OSError as e:
        logger.error(f"Could not check file size for {file_path}: {e}")
        return False

def transcribe_large_audio_file_with_progress(
    audio_file_path: str,
    openai_api_key: str,
    language_setting_from_client: Optional[str] = "any",
    progress_callback=None,
    chunk_size_mb: int = 20,
    rolling_context_prompt: Optional[str] = None,
    chunk_duration: float = 600  # Smart chunking will optimize this based on file size
) -> Optional[Dict[str, Any]]:
    """
    Transcribe large audio files by splitting them into smaller chunks with progress tracking.
    
    Args:
        audio_file_path: Path to the audio file
        openai_api_key: OpenAI API key for Whisper
        language_setting_from_client: Language setting ('any' or ISO language code)
        progress_callback: Function to call with progress updates (completed_chunks, total_chunks, current_step)
        chunk_size_mb: Maximum chunk size in MB (default 20MB for OpenAI Whisper safety)
        rolling_context_prompt: Context for transcription
        chunk_duration: Duration of each chunk in seconds (default 5 minutes for better progress)
    
    Returns:
        Dictionary with 'text' and 'segments' keys, or None on failure
    """
    start_time = time.time()
    
    # Check if file is within OpenAI limits
    if check_file_size_for_openai(audio_file_path):
        logger.info(f"File {os.path.basename(audio_file_path)} is within size limits, using direct transcription")
        if progress_callback:
            progress_callback(0, 1, "Transcribing single file...")
        result = _transcribe_audio_segment_openai(
            audio_file_path, openai_api_key, language_setting_from_client, rolling_context_prompt
        )
        if progress_callback and result:
            progress_callback(1, 1, "Transcription completed")
        return result
    
    logger.info(f"File {os.path.basename(audio_file_path)} exceeds size limits, splitting into chunks")
    
    # Smart chunking: Use file size to determine optimal chunk duration
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    
    if file_size_mb <= 50:   # Small files: use larger chunks for efficiency
        adjusted_chunk_duration = 600   # 10 minutes per chunk
    elif file_size_mb <= 200: # Medium files: balance efficiency and progress
        adjusted_chunk_duration = 450   # 7.5 minutes per chunk  
    else:  # Large files: smaller chunks for better progress tracking and memory management
        adjusted_chunk_duration = 300   # 5 minutes per chunk
    
    logger.info(f"File size: {file_size_mb:.1f}MB, using {adjusted_chunk_duration}s chunks")
    
    # Split the file into chunks
    chunk_paths = split_audio_file(audio_file_path, adjusted_chunk_duration)
    if not chunk_paths:
        logger.error("Failed to create audio chunks")
        return None
    
    if progress_callback:
        progress_callback(0, len(chunk_paths), f"Starting transcription of {len(chunk_paths)} chunks...")
    
    # Performance optimization: Use parallel processing for multiple chunks  
    try:
        if len(chunk_paths) > 1:
            logger.info(f"Using parallel processing for {len(chunk_paths)} chunks (max 2 concurrent)")
            all_segments, combined_text_parts = _transcribe_chunks_parallel(
                chunk_paths, openai_api_key, language_setting_from_client, 
                rolling_context_prompt, adjusted_chunk_duration, progress_callback
            )
        else:
            # Single chunk - use sequential processing
            all_segments, combined_text_parts = _transcribe_chunks_sequential(
                chunk_paths, openai_api_key, language_setting_from_client, 
                rolling_context_prompt, adjusted_chunk_duration, progress_callback
            )
    
    finally:
        # Clean up chunk files
        for chunk_path in chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    logger.debug(f"Cleaned up chunk file: {chunk_path}")
            except OSError as e:
                logger.error(f"Error cleaning up chunk file {chunk_path}: {e}")
    
    # Combine results
    if all_segments:
        combined_result = {
            'text': ' '.join(combined_text_parts),
            'segments': all_segments
        }
        processing_time = time.time() - start_time
        total_duration = get_audio_duration(audio_file_path)
        
        # Enhanced success metrics - fix calculation
        successful_chunks = sum(1 for result in combined_text_parts if result.strip())  # Count non-empty results
        success_rate = successful_chunks / len(chunk_paths) if chunk_paths else 0
        logger.info(f"Successfully transcribed large file: {len(all_segments)} segments from {len(chunk_paths)} chunks ({successful_chunks}/{len(chunk_paths)} successful chunks, {success_rate:.1%} success rate). Processing time: {processing_time:.1f}s for {total_duration:.1f}s of audio")
        
        if progress_callback:
            progress_callback(len(chunk_paths), len(chunk_paths), "Transcription completed successfully!")
        
        return combined_result
    elif len(all_segments) > 0:
        # INTELLIGENT FALLBACK: Partial success - return what we have with warning  
        processing_time = time.time() - start_time
        successful_chunks = sum(1 for result in combined_text_parts if result.strip())
        partial_success_rate = successful_chunks / len(chunk_paths) if chunk_paths else 0
        
        logger.warning(f"Partial transcription success: {len(all_segments)} segments from {len(chunk_paths)} chunks ({partial_success_rate:.1%} success rate)")
        
        # Create partial result with metadata about missing chunks
        combined_result = {
            'text': ' '.join(combined_text_parts),
            'segments': all_segments,
            'partial': True,
            'success_rate': partial_success_rate,
            'total_chunks': len(chunk_paths),
            'successful_chunks': len([s for s in all_segments]),
            'warning': f'Partial transcription: {len(chunk_paths) - len(all_segments)} chunks failed due to API issues'
        }
        
        if progress_callback:
            progress_callback(len(chunk_paths), len(chunk_paths), f"Partial transcription completed ({partial_success_rate:.0%} success)")
        
        return combined_result
    else:
        # INTELLIGENT FALLBACK: Complete failure - provide actionable guidance
        processing_time = time.time() - start_time
        logger.error(f"Complete transcription failure. Processing time: {processing_time:.1f}s")
        
        # Analyze failure patterns for better user guidance
        failure_advice = _generate_failure_advice(chunk_paths, processing_time)
        
        if progress_callback:
            try:
                progress_callback(0, len(chunk_paths), f"Transcription failed: {failure_advice}")
            except:
                pass  # Don't fail on callback errors
        
        return None

def transcribe_large_audio_file(
    audio_file_path: str,
    openai_api_key: str,
    language_setting_from_client: Optional[str] = "any",
    rolling_context_prompt: Optional[str] = None,
    chunk_duration: float = 600
) -> Optional[Dict[str, Any]]:
    """
    Transcribe large audio files by splitting them into chunks if necessary.
    
    Args:
        audio_file_path: Path to the audio file
        openai_api_key: OpenAI API key
        language_setting_from_client: Language setting
        rolling_context_prompt: Context for transcription
        chunk_duration: Duration of each chunk in seconds (default 10 minutes)
    
    Returns:
        Combined transcription result or None if failed
    """
    start_time = time.time()
    
    # Check if file is within OpenAI limits
    if check_file_size_for_openai(audio_file_path):
        logger.info(f"File {os.path.basename(audio_file_path)} is within size limits, using direct transcription")
        return _transcribe_audio_segment_openai(
            audio_file_path, openai_api_key, language_setting_from_client, rolling_context_prompt
        )
    
    logger.info(f"File {os.path.basename(audio_file_path)} exceeds size limits, splitting into chunks")
    
    # Split the file into chunks
    chunk_paths = split_audio_file(audio_file_path, chunk_duration)
    if not chunk_paths:
        logger.error("Failed to create audio chunks")
        return None
    
    # Transcribe each chunk
    all_segments = []
    combined_text_parts = []
    current_time_offset = 0.0
    
    try:
        for i, chunk_path in enumerate(chunk_paths):
            progress_pct = ((i + 1) / len(chunk_paths)) * 100
            logger.info(f"Transcribing chunk {i+1}/{len(chunk_paths)} ({progress_pct:.1f}%): {os.path.basename(chunk_path)}")
            
            # Use context from previous chunk for continuity
            chunk_context = rolling_context_prompt if i == 0 else " ".join(combined_text_parts[-1:])
            
            # Smart chunk retry with improved error handling
            chunk_max_retries = 4  # Increased for better reliability
            chunk_result = None
            
            for chunk_attempt in range(chunk_max_retries):
                try:
                    chunk_result = _transcribe_audio_segment_openai(
                        chunk_path, openai_api_key, language_setting_from_client, chunk_context
                    )
                    if chunk_result and chunk_result.get('segments'):
                        logger.info(f"Chunk {i+1}/{len(chunk_paths)} transcribed successfully on attempt {chunk_attempt + 1}")
                        break  # Success, exit retry loop
                    else:
                        raise Exception("No segments returned from transcription")
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Chunk {i+1} attempt {chunk_attempt + 1} failed: {error_msg}")
                    
                    # Don't retry on certain errors
                    if any(term in error_msg.lower() for term in ['client error', 'file too large', 'api key', 'unauthorized']):
                        logger.error(f"Non-recoverable error for chunk {i+1}: {error_msg}")
                        chunk_result = None
                        break
                    
                    if chunk_attempt == chunk_max_retries - 1:
                        logger.error(f"Failed to transcribe chunk {i+1} after {chunk_max_retries} attempts: {error_msg}")
                        chunk_result = None
                        break
                    
                    # Smart backoff for chunk retries
                    if 'rate limit' in error_msg.lower():
                        chunk_delay = min(30, 5 * (2 ** chunk_attempt))  # Longer delay for rate limits
                    elif 'server error' in error_msg.lower() or '5' in error_msg[:3]:  # Server errors
                        chunk_delay = min(20, 3 * (2 ** chunk_attempt))
                    else:
                        chunk_delay = min(10, 2 ** chunk_attempt)  # Standard exponential backoff
                    
                    logger.info(f"Retrying chunk {i+1} in {chunk_delay} seconds...")
                    time.sleep(chunk_delay)
            
            if chunk_result and chunk_result.get('segments'):
                # Adjust timestamps to account for chunk offset
                chunk_segments = chunk_result['segments']
                for segment in chunk_segments:
                    segment['start'] += current_time_offset
                    segment['end'] += current_time_offset
                
                all_segments.extend(chunk_segments)
                
                # Add text from this chunk
                chunk_text = chunk_result.get('text', '').strip()
                if chunk_text:
                    combined_text_parts.append(chunk_text)
                
                # Update time offset for next chunk
                if chunk_segments:
                    last_segment_end = max(seg.get('end', 0) for seg in chunk_segments)
                    current_time_offset = last_segment_end
                else:
                    current_time_offset += chunk_duration
            else:
                logger.warning(f"Failed to transcribe chunk {i+1} after all retries, continuing with others")
                current_time_offset += chunk_duration
    
    finally:
        # Clean up chunk files
        for chunk_path in chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    logger.debug(f"Cleaned up chunk file: {chunk_path}")
            except OSError as e:
                logger.error(f"Error cleaning up chunk file {chunk_path}: {e}")
    
    # Combine results
    if all_segments:
        combined_result = {
            'text': ' '.join(combined_text_parts),
            'segments': all_segments
        }
        processing_time = time.time() - start_time
        total_duration = get_audio_duration(audio_file_path)
        logger.info(f"Successfully transcribed large file in {len(chunk_paths)} chunks. Total segments: {len(all_segments)}. Processing time: {processing_time:.1f}s for {total_duration:.1f}s of audio")
        return combined_result
    else:
        processing_time = time.time() - start_time
        logger.error(f"No successful transcriptions from any chunks. Processing time: {processing_time:.1f}s")
        return None

def _transcribe_audio_segment_openai(
    audio_file_path: str,
    openai_api_key: str,
    language_setting_from_client: Optional[str] = "any",
    rolling_context_prompt: Optional[str] = None,
    vad_aggressiveness: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
    # Provider with fallback support
    try:
        result = _primary_provider.transcribe_file(audio_file_path, language=language_setting_from_client, prompt=rolling_context_prompt, vad_aggressiveness=vad_aggressiveness)
        if result:
            return result
    except Exception as e:
        logger.error(f"Primary provider ({_PRIMARY_PROVIDER}) failed for {audio_file_path}: {e}. Falling back to {_FALLBACK_PROVIDER}.")

    # Try fallback provider if available
    if _fallback_provider:
        try:
            result = _fallback_provider.transcribe_file(audio_file_path, language="auto" if language_setting_from_client == "any" else language_setting_from_client)
            if result:
                logger.info(f"Fallback provider ({_FALLBACK_PROVIDER}) succeeded for {audio_file_path}")
                return result
        except Exception as ee:
            logger.error(f"Fallback provider ({_FALLBACK_PROVIDER}) failed for {audio_file_path}: {ee}")

    return None

def transcribe_chunk_with_sticky_fallback(state: SessionState, wav_path: str, language: Optional[str] = None, vad_aggressiveness: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Transcribe a chunk with sticky provider fallback logic.
    """
    provider = state.provider_current
    # If we are in Whisper fallback and cooldown elapsed, try a single Deepgram probe on this chunk
    probing = False
    if provider == "whisper" and _maybe_probe_deepgram(state):
        provider = "deepgram"
        probing = True

    try:
        if provider == "deepgram":
            # Use Deepgram with proper parameters
            result = _primary_provider.transcribe_file(wav_path, language=language, vad_aggressiveness=vad_aggressiveness)
        else:
            # Use Whisper
            result = _fallback_provider.transcribe_file(wav_path, language=language or "auto")

        if result:
            if probing:
                _complete_probe(state, success=True)
            _on_provider_success(state)
            return result
        else:
            # No result but no exception - treat as failure
            raise RuntimeError(f"{provider} returned no result")

    except Exception as e:
        if provider == "deepgram":
            _mark_deepgram_failure(state)
            if probing:
                _complete_probe(state, success=False)

        _on_provider_failure(state, e)

        # sticky fallback: if we are not already in Whisper, do the switch now for this same chunk
        if provider == "deepgram":
            try:
                result = _fallback_provider.transcribe_file(wav_path, language=language or "auto")
                if result:
                    _on_provider_success(state)
                    return result
                else:
                    raise RuntimeError("Whisper fallback returned no result")
            except Exception as ee:
                _on_provider_failure(state, ee)
                return None

        # if Whisper failed, just bubble up (no further fallback)
        return None

def process_audio_segment_and_update_s3(
    temp_segment_wav_path: str,
    session_data: Dict[str, Any],
    session_lock: threading.Lock,
    openai_api_key: Optional[str],
    anthropic_api_key: Optional[str]
) -> bool:
    from session_cleanup import session_id_from_path, release_transcription_ref
    import time

    # Guard: if the producer handed us a path, it must exist.
    session_id = session_data.get("session_id")
    if not os.path.exists(temp_segment_wav_path):
        # Give a very brief grace period in case of FS latency
        for _ in range(3):
            time.sleep(0.05)
            if os.path.exists(temp_segment_wav_path):
                break
        else:
            logger.warning(f"Transcription skipped; file missing: {temp_segment_wav_path}")
            if session_id:
                release_transcription_ref(session_id)  # decrement refcount
            return False

    # --- Part 1a: Calculate actual duration before anything else ---
    # This is now the authoritative source of duration for the segment.
    actual_duration_from_ffprobe = 0.0
    try:
        ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_segment_wav_path]
        duration_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
        actual_duration_from_ffprobe = float(duration_result.stdout.strip())
        logger.info(f"ffprobe successful for {os.path.basename(temp_segment_wav_path)}: duration {actual_duration_from_ffprobe:.2f}s")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as ffprobe_err:
        logger.error(f"ffprobe failed for {temp_segment_wav_path}: {ffprobe_err}. Estimating duration from file size.")
        try:
            file_size_bytes = os.path.getsize(temp_segment_wav_path)
            # For 16-bit mono 16kHz WAV
            bytes_per_second = 16000 * 2
            if bytes_per_second > 0:
                actual_duration_from_ffprobe = file_size_bytes / bytes_per_second
                logger.warning(f"Using estimated duration from file size: {actual_duration_from_ffprobe:.2f}s")
        except Exception as size_err:
             logger.error(f"Could not get file size for fallback duration estimation: {size_err}")
             actual_duration_from_ffprobe = 1.5 # Final fallback


    # --- Part 1: Setup and Slow Operations (outside lock) ---
    session_id = session_data.get("session_id")
    s3_transcript_key = session_data.get('s3_transcript_key')
    session_start_time_utc = session_data.get('session_start_time_utc')
    language_setting_from_client = session_data.get('language_setting_from_client', 'any')
    language_hint_fallback = 'en' # Used if language is 'any'

    if not all([session_id, temp_segment_wav_path, s3_transcript_key, isinstance(session_start_time_utc, datetime)]):
        logger.error("Missing critical data for processing segment (session_id, path, S3 key, or start time).")
        # Clean up the temp file even on failure to prevent disk fill
        if temp_segment_wav_path and os.path.exists(temp_segment_wav_path):
             try: os.remove(temp_segment_wav_path)
             except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path} after pre-check fail: {e_del}")
        if session_id:
            release_transcription_ref(session_id)
        return False

    if not openai_api_key:
        logger.error("OpenAI API key was not provided to process_audio_segment_and_update_s3. Cannot transcribe.")
        return False
        
    # Assign sequence number for this segment (before transcription)
    with session_lock:
        current_seq = session_data.get("next_segment_seq", 1)
        session_data["next_segment_seq"] = current_seq + 1
        segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0)

    # Transcribe audio (slow network call), using sticky provider fallback
    # Note: last_transcript is accessed here, before the lock, which is fine.
    last_transcript = session_data.get("last_successful_transcript", "")
    vad_aggressiveness_from_client = session_data.get("vad_aggressiveness_from_client")

    # Get session state for sticky provider fallback
    session_state = session_data.get("session_state")
    transcription_result = None

    try:
        if session_state:
            # Use new sticky fallback logic
            transcription_result = transcribe_chunk_with_sticky_fallback(
                session_state,
                temp_segment_wav_path,
                language=language_setting_from_client,
                vad_aggressiveness=vad_aggressiveness_from_client
            )
        else:
            # Fallback to legacy logic if no session state
            transcription_result = _transcribe_audio_segment_openai(
                temp_segment_wav_path,
                openai_api_key,
                language_setting_from_client,
                rolling_context_prompt=last_transcript,
                vad_aggressiveness=vad_aggressiveness_from_client
            )
    except Exception as e:
        # Check if this is a transient error that should be retried
        if is_transient_provider_error(e):
            logger.warning(f"Session {session_id}: Transient transcription error for seq={current_seq}: {e}")

            # Save segment for retry
            if save_for_retry(temp_segment_wav_path, session_data, segment_offset_seconds, current_seq):
                # Return success - segment will be processed by retry worker
                if session_id:
                    release_transcription_ref(session_id)
                return True
            else:
                logger.error(f"Session {session_id}: Failed to save segment seq={current_seq} for retry")

        # Re-raise for permanent errors or retry save failures
        raise

    # If transcription fails, exit early.
    if not transcription_result:
        logger.info(f"Transcription skipped (missing or failed): {temp_segment_wav_path}")
        if os.path.exists(temp_segment_wav_path):
            try:
                os.remove(temp_segment_wav_path)
            except OSError as e:
                logger.error(f"Error removing failed segment file {temp_segment_wav_path}: {e}")
        if session_id:
            release_transcription_ref(session_id)
        return False

    # Pre-process segments that don't depend on the atomic offset
    filtered_segments_pre_lock = []
    if transcription_result and transcription_result.get('segments'):
        raw_segments = transcription_result['segments']
        logger.debug(f"Raw transcription returned {len(raw_segments)} segments for {temp_segment_wav_path}")
        
        # These filters don't depend on session state/offset.
        filtered_segments_pre_lock = analyze_silence_gaps(raw_segments)
        filtered_segments_pre_lock = detect_cross_segment_repetition(filtered_segments_pre_lock)
        filtered_segments_pre_lock = [s for s in filtered_segments_pre_lock if filter_by_duration_and_confidence(s)]

    # Instantiate a transient PII client if PII filtering is enabled
    pii_llm_client_for_service = None
    if os.getenv('ENABLE_TRANSCRIPT_PII_FILTERING', 'false').lower() == 'true':
        if anthropic_api_key:
            from anthropic import Anthropic
            try:
                pii_llm_client_for_service = Anthropic(api_key=anthropic_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize transient Anthropic client for PII filtering: {e}")
        else:
            logger.warning("PII filtering is enabled, but no Anthropic API key was provided.")

    from utils.pii_filter import anonymize_transcript_chunk
    
    # --- Part 2: Critical Section for State and S3 Update (inside lock) ---
    with session_lock:
        session_id_for_log = session_data.get("session_id", "FALLBACK_UNKNOWN_SESSION")
        logger.debug(f"SESSION_LOCK_ACQUIRED for session {session_id_for_log}")

        # CRITICAL FIX: Update duration BEFORE timestamp calculation to fix frozen timestamps
        # Use the duration calculated at the start of this function. This is the key fix.
        actual_segment_duration = actual_duration_from_ffprobe
        
        # ATOMIC READ of current offset for this segment's timestamps
        segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0)
        
        # Update total duration immediately after reading the offset for timestamps
        if actual_segment_duration > 0:
            session_data['current_total_audio_duration_processed_seconds'] += actual_segment_duration
            # Logged at a higher level of INFO because this is a critical value for correct timestamps.
            logger.info(f"Updated session {session_id_for_log} total processed duration: +{actual_segment_duration:.2f}s = {session_data['current_total_audio_duration_processed_seconds']:.2f}s total")
        else:
            # This case should now be rare due to ffprobe fallback logic, but it's safe to keep.
            logger.warning(f"Session {session_id_for_log}: actual_segment_duration was 0. Timestamps might not advance.")
        
        # Run filters that depend on session state/offset
        final_filtered_segments = detect_single_word_loops(filtered_segments_pre_lock, session_data, segment_offset_seconds)
        logger.debug(f"After all filtering, {len(final_filtered_segments)} segments remain for processing.")

        # --- NEW: Combine segments into a single block before writing ---
        lines_to_append_to_s3 = []
        if final_filtered_segments:
            # Combine text from all valid segments
            combined_text = " ".join([s.get('text', '').strip() for s in final_filtered_segments])
            
            # Get the full text from the original, unsegmented transcription result
            # This is a better source for the hallucination detector as it has more context
            full_unsegmented_text = transcription_result.get("text", "").strip()

            # Convert provider words to Word[]
            curr_words = []
            if transcription_result.get('words'):
                curr_words = [
                    Word(w.get('text') or w.get('word', ''), w['start'], w['end'], w.get('confidence', 1.0))
                    for w in transcription_result['words']
                ]

            curr_text_raw = full_unsegmented_text

            # Initialize detector state if not exists
            if 'hallu_state' not in session_data:
                from config import HALLU
                session_data['hallu_state'] = DetectorState(
                    prev_tail=[],
                    cooldown_until_ts=0.0,
                    tail_window_s=HALLU.TAIL_WINDOW_S,
                    head_window_s=HALLU.HEAD_WINDOW_S,
                    min_tokens=HALLU.MIN_TOKENS,
                    jaccard_thresh=HALLU.JACCARD,
                    prefix_char_sim_thresh=HALLU.PREFIX,
                    min_conf=HALLU.MIN_CONF,
                    cooldown_s=HALLU.COOLDOWN_S,
                )
                session_data['prev_delivered_words'] = []
                session_data['stream_time_s'] = 0.0

            # Guard: only consider trimming if prev tail ended within MAX_GAP_S of current start
            within_gap = False
            if session_data['prev_delivered_words'] and curr_words:
                from config import HALLU
                prev_end = session_data['prev_delivered_words'][-1].end
                curr_start = curr_words[0].start
                within_gap = (curr_start - prev_end) <= HALLU.MAX_GAP_S

            trimmed_text, reason, cut_s = (curr_text_raw, "", -1.0)
            if within_gap:
                trimmed_text, reason, cut_s = maybe_trim_repetition(
                    state=session_data['hallu_state'],
                    stream_time_s=session_data['stream_time_s'],
                    prev_delivered_words=session_data['prev_delivered_words'],
                    curr_words=curr_words,
                    curr_text_raw=curr_text_raw,
                )
            final_text_for_s3 = trimmed_text
            if reason:
                logger.warning(f"hallucination_trim reason={reason} orig={curr_text_raw[:120]!r} kept={final_text_for_s3[:120]!r}")
                # Track metrics
                try:
                    from metrics import HALLU_TRIMS
                    provider = session_data.get("session_state", {}).get("provider_current", "unknown")
                    HALLU_TRIMS.labels(reason=reason, provider=provider).inc()
                except Exception as e:
                    logger.debug(f"Failed to record hallucination trim metric: {e}")

            # Update rolling tail using delivered words (not raw)
            # small epsilon avoids jitter re-emission on boundary
            delivered_words = curr_words if cut_s < 0 else [w for w in curr_words if w.end > cut_s + 1e-3]
            session_data['prev_delivered_words'] = (session_data['prev_delivered_words'] + delivered_words)[-200:]
            session_data['stream_time_s'] = delivered_words[-1].end if delivered_words else session_data['stream_time_s']

            # Apply PII filtering to the final, corrected, combined text
            if os.getenv('ENABLE_TRANSCRIPT_PII_FILTERING', 'false').lower() == 'true':
                lang_hint = language_setting_from_client if language_setting_from_client != 'any' else language_hint_fallback
                model_name_pii = os.getenv("PII_REDACTION_MODEL_NAME", "claude-3-haiku-20240307")
                final_text_for_s3 = anonymize_transcript_chunk(final_text_for_s3, pii_llm_client_for_service, model_name_pii, language_hint=lang_hint)

            if is_valid_transcription(final_text_for_s3):
                # Create a single timestamp from the first segment's start to the last segment's end
                first_segment = final_filtered_segments[0]
                last_segment = final_filtered_segments[-1]

                abs_start = segment_offset_seconds + first_segment.get('start', 0.0)
                abs_end = segment_offset_seconds + last_segment.get('end', 0.0)

                timestamp_str = format_timestamp_range(abs_start, abs_end, session_start_time_utc)

                # Update context with this new valid line
                session_data["last_successful_transcript"] = final_text_for_s3

                # Try ordered delivery first, fallback to direct S3 append
                delivered_via_ordering = False
                logger.info(f"Session {session_id_for_log}: Processing transcript for seq={current_seq}, text='{final_text_for_s3[:50]}...'")
                try:
                    # Import the ordering function from api_server
                    from api_server import try_deliver_ordered_results, SEGMENT_RETRY_ENABLED

                    # Only use ordering system if gap recovery is enabled and session exists
                    if SEGMENT_RETRY_ENABLED:
                        try_deliver_ordered_results(
                            session_id=session_id_for_log,
                            seq=current_seq,
                            transcript_text=final_text_for_s3,
                            timestamp_str=timestamp_str,
                            session_lock=session_lock
                        )
                        delivered_via_ordering = True
                        logger.info(f"Session {session_id_for_log}: Delivered seq={current_seq} via ordering system")

                except Exception as delivery_e:
                    logger.warning(f"Session {session_id_for_log}: Failed to deliver seq={current_seq} via ordering system: {delivery_e}")
                    delivered_via_ordering = False

                # Fallback to direct S3 append if ordering failed or is disabled
                if not delivered_via_ordering:
                    lines_to_append_to_s3.append(f"{timestamp_str} {final_text_for_s3}")
                    logger.info(f"Session {session_id_for_log}: Using fallback S3 append for seq={current_seq}")
                else:
                    logger.info(f"Session {session_id_for_log}: Successfully delivered seq={current_seq} via ordering system (no S3 buffer needed)")
            else:
                logger.warning(f"Session {session_id_for_log}: Combined text was invalid after PII filtering. Final text: '{final_text_for_s3}'")
        else:
            logger.info(f"Session {session_id_for_log}: No valid segments remained after filtering, nothing to append.")
        # --- END: New segment combination logic ---

        # Process markers (this logic remains the same)
        marker_to_write = session_data.get("pause_marker_to_write")
        if marker_to_write:
            offset = session_data.get("pause_event_timestamp_offset", segment_offset_seconds)
            timestamp = _format_time_delta(offset, session_start_time_utc)
            lines_to_append_to_s3.insert(0, f"[{timestamp} UTC] {marker_to_write}") # Insert at the beginning
            session_data["pause_marker_to_write"] = None # Clear after processing


        # Add verification log before persistence
        joined_lines = '\n'.join(lines_to_append_to_s3)
        logger.info(f"Session {session_id_for_log}: Persisting transcript: chars={len(joined_lines)}, lines={len(lines_to_append_to_s3)}")

        # Perform S3 append if there's new content
        if lines_to_append_to_s3:
            s3 = get_s3_client()
            aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
            if not s3 or not aws_s3_bucket:
                logger.error("S3 client/bucket unavailable inside lock. Cannot write.")
                # Don't update duration if write fails
            else:
                try:
                    obj = s3.get_object(Bucket=aws_s3_bucket, Key=s3_transcript_key)
                    existing_content = obj['Body'].read().decode('utf-8')
                except s3.exceptions.NoSuchKey:
                    header = f"# Transcript - Session {session_id_for_log}\nAgent: {session_data.get('agent_name', 'N/A')}, Event: {session_data.get('event_id', 'N/A')}\nLanguage: {format_language_for_header(language_setting_from_client)}\nProvider: {get_current_transcription_provider()}\nSession Started (UTC): {session_start_time_utc.isoformat()}\n\n"
                    existing_content = header
                
                appended_text = "\n".join(lines_to_append_to_s3) + "\n"
                updated_content = existing_content + appended_text
                
                s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8'))
                logger.info(f"Appended {len(lines_to_append_to_s3)} lines to S3 transcript {s3_transcript_key}.")
        
        logger.debug(f"SESSION_LOCK_RELEASED for session {session_id_for_log}")

    # --- Part 3: Cleanup ---
    if os.path.exists(temp_segment_wav_path):
        try: os.remove(temp_segment_wav_path); logger.debug(f"Cleaned up temp WAV: {temp_segment_wav_path}")
        except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path}: {e_del}")

    # Release the transcription reference count and try cleanup
    if session_id:
        from session_cleanup import release_transcription_ref, maybe_cleanup_session
        release_transcription_ref(session_id)
        maybe_cleanup_session(session_id)

    return True
