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

from utils.hallucination_detector import get_hallucination_manager # For hallucination detection

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

def is_valid_transcription(text: str) -> bool:
    if not text: return False
    if re.fullmatch(r'[^\w\s]+', text): return False 
    if len(text) < 2: return False 
    return True

def _transcribe_audio_segment_openai(
    audio_file_path: str, 
    openai_api_key: str, 
    language_setting_from_client: Optional[str] = "any",
    rolling_context_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
    if not openai_api_key:
        logger.error("OpenAI API key not provided for transcription.")
        return None

    openai.api_key = openai_api_key 

    try:
        with open(audio_file_path, "rb") as audio_file_obj: # Renamed to audio_file_obj
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {openai_api_key}"}

            data_payload: Dict[str, Any] = {
                'model': 'whisper-1',
                'response_format': 'verbose_json', 
                'temperature': 0.0,
                'no_speech_threshold': 0.9,        # Increased from 0.85 to 0.9 (stricter)
                'logprob_threshold': -0.5,         # Increased from -0.7 to -0.5 (stricter)
                'compression_ratio_threshold': 1.8, # Decreased from 2.0 to 1.8 (stricter)
            }

            # Handle language setting from client
            if language_setting_from_client == "en" or language_setting_from_client == "sv":
                data_payload['language'] = language_setting_from_client
                logger.info(f"Whisper: Language explicitly set to '{language_setting_from_client}'.")
            elif language_setting_from_client == "any" or language_setting_from_client is None: # Explicitly handle None as 'any'
                logger.info(f"Whisper: Language set to '{language_setting_from_client if language_setting_from_client else 'any (default)'}' (auto-detect by omitting language param).")
                # No 'language' key is added to data_payload, Whisper will auto-detect
            else: # Unrecognized, treat as auto-detect
                logger.info(f"Whisper: Language setting '{language_setting_from_client}' unrecognized, defaulting to auto-detect.")
                # No 'language' key added for auto-detect

            # Use the rolling context as the primary prompt to prevent repetition.
            # Fallback to a generic prompt only if no context is available.
            if rolling_context_prompt:
                initial_prompt_text = rolling_context_prompt
                logger.info(f"Whisper: Using rolling context as initial_prompt: '{initial_prompt_text[-150:]}'")
            else:
                initial_prompt_text_base = (
                    "This is the first segment of a professional business meeting. Please focus on transcribing accurately. "
                    "The primary languages are English and Swedish."
                )
                initial_prompt_text = initial_prompt_text_base
                logger.info("Whisper: No rolling context available, using generic initial prompt.")

            data_payload['prompt'] = initial_prompt_text # Use 'prompt' which is the recommended parameter


            # === DYNAMIC MIME TYPE DETECTION ===
            file_name_for_api = os.path.basename(audio_file_path)
            mime_type, _ = mimetypes.guess_type(audio_file_path)

            # Fallback for common audio types if mimetypes fails or is too generic
            # OpenAI Whisper supports: mp3, mp4, mpeg, mpga, m4a, wav, webm
            if not mime_type or mime_type == 'application/octet-stream':
                ext = os.path.splitext(file_name_for_api)[1].lower()
                if ext == '.mp3':
                    mime_type = 'audio/mpeg'
                elif ext == '.mp4':
                    mime_type = 'audio/mp4'
                elif ext == '.m4a':
                    mime_type = 'audio/mp4' # M4A files are MP4 containers
                elif ext == '.wav':
                    mime_type = 'audio/wav'
                elif ext == '.webm':
                    mime_type = 'audio/webm'
                elif ext == '.mpeg':
                    mime_type = 'audio/mpeg'
                elif ext == '.mpga':
                    mime_type = 'audio/mpeg'
                else:
                    logger.warning(f"Could not determine specific audio MIME type for {file_name_for_api} (ext: {ext}). Using 'application/octet-stream'. OpenAI might infer.")
                    mime_type = 'application/octet-stream'
            
            logger.info(f"Preparing to send {file_name_for_api} to OpenAI with detected MIME type: {mime_type}")
            files_param = {'file': (file_name_for_api, audio_file_obj, mime_type)}
            # === END DYNAMIC MIME TYPE DETECTION ===

            max_retries = 3
            transcription_timeout = 900 # Increased from 60 to 900 (15 minutes)
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting transcription for {audio_file_path} (attempt {attempt + 1})...")
                    # Reset file pointer for retries if the file object is being reused
                    audio_file_obj.seek(0)
                    response = requests.post(url, headers=headers, data=data_payload, files=files_param, timeout=transcription_timeout)
                    response.raise_for_status()
                    transcription = response.json()

                    if 'segments' in transcription and isinstance(transcription['segments'], list):
                        logger.info(f"Successfully transcribed {audio_file_path}.")
                        return transcription
                    else:
                        logger.warning(f"Unexpected transcription format for {audio_file_path}: {transcription}")
                        return None 
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout transcribing {audio_file_path} on attempt {attempt + 1}.")
                    if attempt == max_retries - 1: raise
                except requests.exceptions.RequestException as e:
                    logger.error(f"RequestException transcribing {audio_file_path} on attempt {attempt + 1}: {e}")
                    # Check if 'response' exists before trying to access it
                    if 'response' in locals() and response and response.content:
                        try:
                            error_detail = response.json()
                            logger.error(f"OpenAI API error detail: {error_detail}")
                        except json.JSONDecodeError:
                            logger.error(f"OpenAI API error response (non-JSON): {response.text[:500]}")
                    if attempt == max_retries - 1: raise
                except Exception as e: 
                    logger.error(f"Generic error transcribing {audio_file_path} on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1: raise
                time.sleep(min(2 ** attempt, 8)) 
            return None 

    except Exception as e:
        logger.error(f"Transcription error for {audio_file_path}: {e}", exc_info=True)
        return None

def process_audio_segment_and_update_s3(
    temp_segment_wav_path: str, 
    session_data: Dict[str, Any], 
    session_lock: threading.Lock # Renamed from s3_lock for clarity
    ) -> bool:
    
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
        return False
    
    # Determine segment duration (can be slow if ffprobe is needed)
    segment_actual_duration = session_data.get('actual_segment_duration_seconds', 0.0)
    if segment_actual_duration <= 0:
        logger.warning(f"Segment duration from session_data is {segment_actual_duration:.2f}s for {temp_segment_wav_path}. Attempting ffprobe locally.")
        try:
            ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_segment_wav_path]
            duration_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
            segment_actual_duration = float(duration_result.stdout.strip())
            logger.info(f"Local ffprobe successful for {temp_segment_wav_path}: duration {segment_actual_duration:.2f}s")
        except Exception as e_ffprobe:
            logger.error(f"Local ffprobe failed for {temp_segment_wav_path}: {e_ffprobe}. Using fallback duration.")
            segment_actual_duration = 15.0 # Fallback to a reasonable default if ffprobe fails

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot transcribe.")
        return False
        
    # Transcribe audio (slow network call), passing in the rolling context
    # Note: last_transcript is accessed here, before the lock, which is fine.
    last_transcript = session_data.get("last_successful_transcript", "")
    transcription_result = _transcribe_audio_segment_openai(
        temp_segment_wav_path,
        openai_api_key,
        language_setting_from_client,
        rolling_context_prompt=last_transcript
    )

    # === Hallucination Detection & Context Update
    # Check for hallucinations before further processing
    if transcription_result and transcription_result.get("text"):
        hallucination_manager = get_hallucination_manager(session_id)
        is_valid, reason = hallucination_manager.process_transcript(transcription_result.get("text", "").strip())
        if not is_valid:
            logger.warning(f"Session {session_id}: Hallucination detected for transcription from {temp_segment_wav_path}. Reason: {reason}. Skipping S3 update.")
            stats = hallucination_manager.get_statistics()
            logger.info(f"Session {session_id} hallucination stats - Total: {stats['total_transcripts']}, Valid: {stats['valid_transcripts']}, Hallucinations: {stats['hallucinations_detected']}, Rate: {stats['hallucination_rate']:.1%}")
            
            # CRITICAL FIX: Clear the last successful transcript to prevent a feedback loop.
            with session_lock:
                session_data["last_successful_transcript"] = ""
            logger.warning(f"Session {session_id}: Cleared rolling context due to hallucination.")

            # Clean up the WAV file and exit early
            if os.path.exists(temp_segment_wav_path):
                try: os.remove(temp_segment_wav_path)
                except OSError as e_del: logger.error(f"Error deleting hallucinated temp WAV {temp_segment_wav_path}: {e_del}")
            return True # Return True as the "error" was handled (by not processing a hallucination)
    # === End Hallucination Detection & Context Update

    # Pre-process segments that don't depend on the atomic offset
    filtered_segments_pre_lock = []
    if transcription_result and transcription_result.get('segments'):
        raw_segments = transcription_result['segments']
        logger.debug(f"Raw transcription returned {len(raw_segments)} segments for {temp_segment_wav_path}")
        
        # These filters don't depend on session state/offset.
        filtered_segments_pre_lock = analyze_silence_gaps(raw_segments)
        filtered_segments_pre_lock = detect_cross_segment_repetition(filtered_segments_pre_lock)
        filtered_segments_pre_lock = [s for s in filtered_segments_pre_lock if filter_by_duration_and_confidence(s)]

    # Get PII client ready inside the function to avoid circular import at module level
    pii_llm_client_for_service = None
    try:
        from api_server import anthropic_client as global_anthropic_client
        pii_llm_client_for_service = global_anthropic_client
    except (ImportError, Exception) as e:
        logger.warning(f"Could not get global Anthropic client for PII filter: {e}")

    from utils.pii_filter import anonymize_transcript_chunk
    
    # --- Part 2: Critical Section for State and S3 Update (inside lock) ---
    with session_lock:
        session_id_for_log = session_data.get("session_id", "FALLBACK_UNKNOWN_SESSION")
        logger.debug(f"SESSION_LOCK_ACQUIRED for session {session_id_for_log}")

        # Correctly update the shared session_data object with the duration for this segment.
        session_data['actual_segment_duration_seconds'] = segment_actual_duration
        
        # ATOMIC READ of current offset
        segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0)
        
        # Run filters that depend on session state/offset
        segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0)
        
        # Run filters that depend on session state/offset
        final_filtered_segments = detect_single_word_loops(filtered_segments_pre_lock, session_data, segment_offset_seconds)
        logger.debug(f"After all filtering, {len(final_filtered_segments)} segments remain for processing.")

        processed_transcript_lines = []
        for segment in final_filtered_segments:
            raw_text = segment.get('text', '').strip()
            filtered_text = filter_hallucinations(raw_text)
            
            if is_valid_transcription(filtered_text):
                final_text_for_s3 = filtered_text
                if os.getenv('ENABLE_TRANSCRIPT_PII_FILTERING', 'false').lower() == 'true':
                    lang_hint = language_setting_from_client if language_setting_from_client != 'any' else language_hint_fallback
                    model_name_pii = os.getenv("PII_REDACTION_MODEL_NAME", "claude-3-haiku-20240307")
                    final_text_for_s3 = anonymize_transcript_chunk(filtered_text, pii_llm_client_for_service, model_name_pii, language_hint=lang_hint)
                
                if is_valid_transcription(final_text_for_s3):
                    # Generate timestamp using the ATOMICALLY read offset
                    abs_start = segment_offset_seconds + segment.get('start', 0.0)
                    abs_end = segment_offset_seconds + segment.get('end', 0.0)
                    timestamp_str = format_timestamp_range(abs_start, abs_end, session_start_time_utc)
                    processed_transcript_lines.append(f"{timestamp_str} {final_text_for_s3}")

        # Process markers
        lines_to_append_to_s3 = []
        marker_to_write = session_data.get("pause_marker_to_write")
        if marker_to_write:
            offset = session_data.get("pause_event_timestamp_offset", segment_offset_seconds)
            timestamp = _format_time_delta(offset, session_start_time_utc)
            lines_to_append_to_s3.append(f"[{timestamp} UTC] {marker_to_write}")
            session_data["pause_marker_to_write"] = None # Clear after processing

        lines_to_append_to_s3.extend(processed_transcript_lines)
        
        # --- Update rolling context for next API call ---
        if processed_transcript_lines:
            # Get the text from the last valid segment to use as context
            last_line_text = processed_transcript_lines[-1].split("] ", 1)[-1]
            session_data["last_successful_transcript"] = last_line_text
            logger.debug(f"Updated rolling context for session {session_id_for_log}: '{last_line_text}'")
        # --- End context update ---

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
                    header = f"# Transcript - Session {session_id_for_log}\nAgent: {session_data.get('agent_name', 'N/A')}, Event: {session_data.get('event_id', 'N/A')}\nSession Started (UTC): {session_start_time_utc.isoformat()}\n\n"
                    existing_content = header
                
                appended_text = "\n".join(lines_to_append_to_s3) + "\n"
                updated_content = existing_content + appended_text
                
                s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8'))
                logger.info(f"Appended {len(lines_to_append_to_s3)} lines to S3 transcript {s3_transcript_key}.")
        
        # ATOMIC WRITE of new total duration, happens regardless of whether content was written
        # as long as the segment was processed.
        session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
        logger.info(f"Updated session {session_id_for_log} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s")
        
        logger.debug(f"SESSION_LOCK_RELEASED for session {session_id_for_log}")

    # --- Part 3: Cleanup ---
    if os.path.exists(temp_segment_wav_path):
        try: os.remove(temp_segment_wav_path); logger.debug(f"Cleaned up temp WAV: {temp_segment_wav_path}")
        except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path}: {e_del}")
        
    return True