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
    language_setting_from_client: Optional[str] = None # Changed from 'language'
    # chunk_duration: float = 15.0 # chunk_duration is not used by Whisper file API
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
            elif language_setting_from_client == "any":
                logger.info("Whisper: Language set to 'any' (auto-detect by omitting language param).")
                # No 'language' key is added to data_payload, Whisper will auto-detect
            else: # Default or unrecognized, treat as auto-detect or a sensible default like 'en'
                logger.info(f"Whisper: Language setting '{language_setting_from_client}' unrecognized or not provided, defaulting to auto-detect.")
                # No 'language' key added for auto-detect

            # Construct initial_prompt
            initial_prompt_text_base = (
                "Please focus on transcribing professional business discourse accurately. "
                "Avoid common conversational filler phrases if they obscure meaning. "
                "Do not transcribe generic social media phrases, common video outros like 'subscribe to my channel' "
                "or 'thanks for watching', or phrases in languages clearly unrelated to a business meeting context "
                "unless they are direct quotes or proper nouns. If unsure, prioritize clear speech over interpreting noise."
            )
            
            initial_prompt_text = ""
            if language_setting_from_client == "en":
                initial_prompt_text = "The primary language is English. " + initial_prompt_text_base
            elif language_setting_from_client == "sv":
                initial_prompt_text = "The primary language is Swedish. " + initial_prompt_text_base
            else: # "any" or default/unrecognized
                initial_prompt_text = "The primary languages expected are English and Swedish. " + initial_prompt_text_base
            
            data_payload['initial_prompt'] = initial_prompt_text
            logger.info(f"Whisper: Using initial_prompt: '{initial_prompt_text[:100]}...'")


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
                    if response and response.content: # Check if response object exists
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
    s3_lock: threading.Lock
    ) -> bool:
    s3_transcript_key = session_data.get('s3_transcript_key')
    session_start_time_utc = session_data.get('session_start_time_utc')
    segment_offset_seconds = session_data.get('current_total_audio_duration_processed_seconds', 0.0) 
    language_setting_from_client = session_data.get('language_setting_from_client', 'en') # Use new key, default 'en'
    
    # Define language_hint_fallback (now primarily for PII filter if language setting is 'any')
    language_hint_fallback = 'en' # Default language hint if not in session_data or if 'any'

    if not all([temp_segment_wav_path, s3_transcript_key, isinstance(session_start_time_utc, datetime)]):
        logger.error("Missing critical data for processing segment (path, S3 key, or start time).")
        return False
    
    segment_actual_duration = session_data.get('actual_segment_duration_seconds', 0.0)
    if segment_actual_duration <= 0:
        logger.warning(f"Segment duration from session_data is {segment_actual_duration:.2f}s for {temp_segment_wav_path}. Attempting ffprobe locally.")
        try:
            ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_segment_wav_path]
            duration_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
            segment_actual_duration = float(duration_result.stdout.strip())
            logger.info(f"Local ffprobe successful for {temp_segment_wav_path}: duration {segment_actual_duration:.2f}s")
        except Exception as e_ffprobe:
            logger.error(f"Local ffprobe failed for {temp_segment_wav_path}: {e_ffprobe}. Cannot determine segment duration. Skipping S3 duration update for this segment.")
            segment_actual_duration = 0.1 

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot transcribe.")
        return False

    s3 = get_s3_client() 
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("S3 client or bucket not configured. Cannot update transcript.")
        if os.path.exists(temp_segment_wav_path):
            try: os.remove(temp_segment_wav_path); logger.debug(f"Cleaned up temp WAV: {temp_segment_wav_path}")
            except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path}: {e_del}")
        return False

    # Attempt to get the Anthropic client for PII filtering
    # This assumes anthropic_client is importable from api_server or initialized/passed here.
    pii_llm_client_for_service = None
    try:
        from api_server import anthropic_client as global_anthropic_client # Try importing from api_server
        if global_anthropic_client:
            pii_llm_client_for_service = global_anthropic_client
            logger.debug("PII Filter: Using global Anthropic client from api_server.")
        else:
            logger.warning("PII Filter: Global Anthropic client from api_server is None. LLM PII redaction might be skipped.")
    except ImportError:
        logger.warning("PII Filter: Could not import anthropic_client from api_server. LLM PII redaction likely skipped unless initialized locally.")
    except Exception as e:
        logger.error(f"PII Filter: Error accessing Anthropic client: {e}. LLM PII redaction may be skipped.")

    # Import PII filter utility
    from utils.pii_filter import anonymize_transcript_chunk
    
    # Transcribe audio first (can be outside the S3 lock)
    transcription_result = _transcribe_audio_segment_openai(temp_segment_wav_path, openai_api_key, language_setting_from_client)
    
    processed_transcript_lines = [] # Stores lines from audio transcription
    if transcription_result and transcription_result.get('segments'):
        raw_segments = transcription_result['segments']
        logger.debug(f"Raw transcription returned {len(raw_segments)} segments for {temp_segment_wav_path}")
        
        # Apply new hallucination mitigation filters
        filtered_segments = analyze_silence_gaps(raw_segments)
        logger.debug(f"After silence gap analysis: {len(filtered_segments)} segments remain")
        
        filtered_segments = detect_cross_segment_repetition(filtered_segments)
        logger.debug(f"After cross-segment repetition filter: {len(filtered_segments)} segments remain")
        
        filtered_segments = detect_single_word_loops(filtered_segments, session_data, segment_offset_seconds)
        logger.debug(f"After single word loop filter: {len(filtered_segments)} segments remain")
        
        filtered_segments = [s for s in filtered_segments if filter_by_duration_and_confidence(s)]
        logger.debug(f"After duration/confidence filter: {len(filtered_segments)} segments remain")
        
        logger.debug(f"Preparing to process {len(filtered_segments)} filtered segments for {temp_segment_wav_path}")
        for segment_idx, segment in enumerate(filtered_segments):
            raw_text = segment.get('text', '').strip()
            filtered_text_stage1 = filter_hallucinations(raw_text) # Basic hallucination filter
            is_valid_stage1 = is_valid_transcription(filtered_text_stage1)

            final_text_for_s3 = filtered_text_stage1

            if is_valid_stage1 and os.getenv('ENABLE_TRANSCRIPT_PII_FILTERING', 'false').lower() == 'true':
                logger.debug(f"Attempting PII filtering for chunk: '{filtered_text_stage1[:100]}...'")
                # Use the specific language setting if 'en' or 'sv', otherwise fallback for 'any'
                if language_setting_from_client == "en" or language_setting_from_client == "sv":
                    language_hint_for_pii = language_setting_from_client
                else: # 'any' or other unexpected value
                    language_hint_for_pii = language_hint_fallback # PII filter needs a concrete hint
                
                pii_model_name_config = os.getenv("PII_REDACTION_MODEL_NAME", "claude-3-haiku-20240307")
                
                if pii_llm_client_for_service:
                    try:
                        anonymized_chunk = anonymize_transcript_chunk(
                            filtered_text_stage1,
                            pii_llm_client_for_service,
                            pii_model_name_config,
                            language_hint=language_hint_for_pii
                        )
                        if anonymized_chunk != filtered_text_stage1:
                             logger.info(f"PII filter applied changes. Original len: {len(filtered_text_stage1)}, New len: {len(anonymized_chunk)}")
                        else:
                             logger.debug("PII filter made no LLM changes to the chunk.")
                        final_text_for_s3 = anonymized_chunk
                    except Exception as pii_ex:
                        logger.error(f"Error during PII anonymization call: {pii_ex}. Using regex-filtered/original text.", exc_info=True)
                        # Fallback: final_text_for_s3 remains filtered_text_stage1 (which might have had regex applied by anonymize_transcript_chunk)
                else:
                    logger.warning("PII LLM client instance not available in transcription_service. Attempting regex-only PII filtering.")
                    # Call anonymize_transcript_chunk with None client, it will do regex only
                    final_text_for_s3 = anonymize_transcript_chunk(
                        filtered_text_stage1, 
                        None, # Pass None for client
                        pii_model_name_config, # Model name not used if client is None
                        language_hint=language_hint_for_pii
                    )


            # Re-check validity if PII filter could alter it fundamentally (e.g., make it too short)
            # For now, assume PII filter doesn't invalidate a previously valid chunk.
            is_valid_final = is_valid_transcription(final_text_for_s3)

            if is_valid_final:
                whisper_start_time = segment.get('start', 0.0)
                whisper_end_time = segment.get('end', 0.0)
                absolute_start_seconds = segment_offset_seconds + whisper_start_time
                absolute_end_seconds = segment_offset_seconds + whisper_end_time
                timestamp_str = format_timestamp_range(absolute_start_seconds, absolute_end_seconds, session_start_time_utc)
                processed_transcript_lines.append(f"{timestamp_str} {final_text_for_s3}")
            elif is_valid_stage1 and not is_valid_final: # Was valid, PII filter made it invalid
                logger.warning(f"Chunk became invalid after PII filtering. Original: '{raw_text}', Post-PII: '{final_text_for_s3}'")
            else: # Was not valid initially or still not valid
                 logger.debug(f"Audio segment content not valid or filtered out completely. Original: '{raw_text}', Final for S3 attempt: '{final_text_for_s3}'")

    elif not transcription_result:
        logger.warning(f"Transcription call returned None for {temp_segment_wav_path}.")
    else: 
        logger.warning(f"Transcription returned no segments in .get('segments') for {temp_segment_wav_path}.")

    # Critical section: Update S3 with markers and/or transcribed text
    # This entire block needs to be atomic for a given session's transcript file.
    with s3_lock:
        session_id_for_log = session_data.get("session_id", "FALLBACK_UNKNOWN_SESSION")
        logger.debug(f"S3_LOCK_ACQUIRED for session {session_id_for_log} in process_audio_segment_and_update_s3")

        lines_to_append_to_s3 = []
        marker_processed_and_cleared = False

        # Check for and prepare pause/resume marker
        # This must happen before appending audio transcript lines to maintain chronological order if possible
        # The marker's timestamp (event_offset_for_marker) is relative to the start of the session,
        # just like the audio segment's offset (segment_offset_seconds).
        
        marker_to_write = session_data.get("pause_marker_to_write")
        event_offset_for_marker = session_data.get("pause_event_timestamp_offset")

        if marker_to_write is not None and event_offset_for_marker is not None:
            # Check if this marker should be written *before* the current audio segment's transcription
            # This simple check assumes markers are processed close to their occurrence.
            # A more robust solution might involve a sorted queue of events (markers and transcriptions).
            if event_offset_for_marker <= segment_offset_seconds:
                logger.info(f"Session {session_id_for_log}: Processing marker '{marker_to_write}' for offset {event_offset_for_marker:.2f}s (before current audio segment).")
                formatted_timestamp_for_marker = _format_time_delta(event_offset_for_marker, session_start_time_utc)
                marker_line = f"[{formatted_timestamp_for_marker} UTC] {marker_to_write}"
                lines_to_append_to_s3.append(marker_line)
                
                # Clear the marker from session_data *only if it's being added now*
                session_data["pause_marker_to_write"] = None
                session_data["pause_event_timestamp_offset"] = None
                marker_processed_and_cleared = True
            else:
                logger.info(f"Session {session_id_for_log}: Marker '{marker_to_write}' for offset {event_offset_for_marker:.2f}s is *after* current audio segment offset {segment_offset_seconds:.2f}s. Will process later.")
        
        # Add processed audio transcript lines
        if processed_transcript_lines:
            lines_to_append_to_s3.extend(processed_transcript_lines)

        # If a marker was not processed because its offset was too high,
        # it remains in session_data for the next call to process_audio_segment_and_update_s3.

        if not lines_to_append_to_s3:
            logger.info(f"No new content (marker or transcript) to append for session {session_id_for_log}.")
            # Still update duration as audio segment was processed (even if transcription was empty)
            session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
            logger.info(f"Updated session {session_id_for_log} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s (no new lines to S3)")
        else:
            # If there are lines to append, perform S3 operations
            try:
                existing_content = ""
                try:
                    obj = s3.get_object(Bucket=aws_s3_bucket, Key=s3_transcript_key)
                    existing_content = obj['Body'].read().decode('utf-8')
                except s3.exceptions.NoSuchKey:
                    logger.info(f"Transcript {s3_transcript_key} not found. Creating with header.")
                    if not existing_content.startswith("# Transcript"):
                         header = f"# Transcript - Session {session_id_for_log}\n"
                         header += f"Agent: {session_data.get('agent_name', 'N/A')}, Event: {session_data.get('event_id', 'N/A')}\n"
                         header += f"Session Started (UTC): {session_start_time_utc.isoformat()}\n\n"
                         existing_content = header
                
                appended_text = "\n".join(lines_to_append_to_s3) + "\n"
                updated_content = existing_content + appended_text
                
                s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8'))
                logger.info(f"Appended {len(lines_to_append_to_s3)} lines to S3 transcript {s3_transcript_key}.")

                # Update duration after successful S3 write
                session_data['current_total_audio_duration_processed_seconds'] = segment_offset_seconds + segment_actual_duration
                logger.info(f"Updated session {session_id_for_log} processed duration to {session_data['current_total_audio_duration_processed_seconds']:.2f}s")
                
            except Exception as e_s3_update:
                logger.error(f"Error during S3 transcript update for {s3_transcript_key}: {e_s3_update}", exc_info=True)
                # If S3 write failed, and we had prepared a marker, we should NOT clear it from session_data
                # so it can be retried. The `marker_processed_and_cleared` flag helps here.
                if marker_processed_and_cleared: # If we thought we processed it
                    logger.warning(f"S3 write failed for session {session_id_for_log}, but marker was cleared. This marker might be lost if not re-queued.")
                    # Potentially re-queue the marker if S3 fails - complex, for now, log it.
                
                # Cleanup temp WAV file and release lock, then return False
                if os.path.exists(temp_segment_wav_path):
                    try: os.remove(temp_segment_wav_path); logger.debug(f"Cleaned up temp WAV after S3 error: {temp_segment_wav_path}")
                    except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path} after S3 error: {e_del}")
                logger.debug(f"S3_LOCK_RELEASED for session {session_id_for_log} (S3 write error)")
                return False # Indicate S3 update failure

        # Cleanup temp WAV file
        if os.path.exists(temp_segment_wav_path):
            try: os.remove(temp_segment_wav_path); logger.debug(f"Cleaned up temp WAV: {temp_segment_wav_path}")
            except OSError as e_del: logger.error(f"Error deleting temp WAV {temp_segment_wav_path}: {e_del}")
        
        logger.debug(f"S3_LOCK_RELEASED for session {session_id_for_log}")
        return True
