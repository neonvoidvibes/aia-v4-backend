import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables at the very top

# Ensure module is accessible as "api_server" even when executed as __main__
import sys as _sys
_sys.modules.setdefault("api_server", _sys.modules[__name__])

# Quiet gRPC/absl noise unless explicitly overridden
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GRPC_LOG_SEVERITY_LEVEL', 'ERROR')

import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context, g
from uuid import uuid4
import threading 
from concurrent.futures import ThreadPoolExecutor
import time
import json
from datetime import datetime, timezone, timedelta # Ensure datetime, timezone, timedelta are imported
from zoneinfo import ZoneInfo
import urllib.parse 
from functools import wraps 
from typing import Optional, List, Dict, Any, Tuple, Union 
import uuid 
from collections import defaultdict, deque
import subprocess 
import re
import math # For chunking calculations
from werkzeug.utils import secure_filename # Added for file uploads

from utils.api_key_manager import get_api_key # Import the new key manager
from event_bus import set_emitter
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from openai import OpenAI, APIError as OpenAI_APIError, APIStatusError as OpenAI_APIStatusError, APIConnectionError as OpenAI_APIConnectionError
from flask_sock import Sock 
from simple_websocket.errors import ConnectionClosed

from supabase import create_client, Client
from gotrue.errors import AuthApiError, AuthRetryableError
from gotrue.types import User as SupabaseUser
from utils.supabase_client import get_supabase_client

from langchain_core.documents import Document
from utils.retrieval_handler import RetrievalHandler
from utils.prompt_builder import prompt_builder
from utils.transcript_utils import read_new_transcript_content, read_all_transcripts_in_folder, list_saved_transcripts
from utils.transcript_format import format_transcript_line
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, get_event_docs, parse_event_doc_key, save_chat_to_s3, format_chat_history, get_s3_client,
    list_agent_names_from_s3, list_s3_objects_metadata, get_transcript_summaries, get_objective_function,
    write_agent_doc, write_event_doc, S3_CACHE_LOCK, S3_FILE_CACHE, create_agent_structure
)
from utils.transcript_summarizer import generate_transcript_summary # Added import
from utils.multi_agent_summarizer.pipeline import summarize_transcript as ma_summarize_transcript
from utils.pinecone_utils import init_pinecone, create_namespace
from utils.embedding_handler import EmbeddingHandler
from utils.webm_header import extract_webm_header
from pinecone.exceptions import NotFoundException
from anthropic import Anthropic, APIStatusError, AnthropicError, APIConnectionError
from groq import APIError as GroqAPIError
import anthropic # Need the module itself for type hints
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
from flask_cors import CORS
import boto3

from app.session_adapter import SessionAdapter
from services.deepgram_sdk import DeepgramSDK
from services.openai_whisper_sdk import OpenAIWhisper

import tempfile

# Provider client factories used by the new append-only pipeline
def make_deepgram_client():
    return DeepgramSDK(api_key=os.environ.get('DEEPGRAM_API_KEY'))


def make_whisper_client():
    return OpenAIWhisper(api_key=os.environ.get('OPENAI_API_KEY'))


def make_s3_client():
    return boto3.client('s3')

# Append-only transcript adapter wiring
DG_CLIENT = make_deepgram_client()
WHISPER_CLIENT = make_whisper_client()
S3_CLIENT = make_s3_client()
TRANSCRIPT_BUCKET = os.environ.get('TRANSCRIPT_BUCKET', 'aiademomagicaudio')
TRANSCRIPT_PREFIX = os.environ.get('TRANSCRIPT_PREFIX', 'organizations/river/agents')
SESSION_ADAPTER = SessionAdapter(dg_client=DG_CLIENT, whisper_client=WHISPER_CLIENT,
                                 s3_client=S3_CLIENT, bucket=TRANSCRIPT_BUCKET,
                                 base_prefix=TRANSCRIPT_PREFIX)

_TRANSCRIPTION_PROVIDER_NAME = os.environ.get('TRANSCRIPTION_PROVIDER', 'deepgram').strip().lower()
STRICT_WEBM_HEADER_ENABLED = os.environ.get('WEBM_STRICT_HEADER', '0').lower() in ('1', 'true', 'yes')


def _resolve_webm_header(session_id: str, session_data: dict[str, Any]) -> tuple[bytes, str]:
    """Return the header bytes and effective mode for the session."""

    header_mode = session_data.get("webm_header_mode", "legacy")
    if header_mode == "pure":
        header_bytes = session_data.get("webm_pure_header_bytes")
        if header_bytes:
            return header_bytes, "pure"
        if not session_data.get("webm_header_warned_missing"):
            logger.warning(
                f"Session {session_id}: Pure header requested but unavailable; falling back to legacy header bytes."
            )
            session_data["webm_header_warned_missing"] = True

    header_bytes = session_data.get("webm_global_header_bytes", b"")
    return header_bytes, "legacy"


def _store_first_webm_blob(session_id: str, session_data: dict[str, Any], blob: bytes) -> None:
    """Persist header bytes for later segments, respecting strict mode."""

    session_data["webm_global_header_bytes"] = blob
    session_data["webm_header_mode"] = "legacy"

    if STRICT_WEBM_HEADER_ENABLED:
        pure_header = extract_webm_header(blob)
        if pure_header:
            session_data["webm_pure_header_bytes"] = pure_header
            session_data["webm_header_mode"] = "pure"
            session_data["webm_header_stats"] = {
                "mode": "pure",
                "header_len": len(pure_header),
                "fallback": False
            }
            logger.info(
                f"Session {session_id}: Stored pure WebM header ({len(pure_header)} bytes) for subsequent segments."
            )
            return

        session_data["webm_header_stats"] = {
            "mode": "legacy",
            "header_len": len(blob),
            "fallback": True
        }
        logger.warning(
            f"Session {session_id}: Strict header extraction failed; reverting to legacy header handling."
        )
    else:
        session_data["webm_header_stats"] = {
            "mode": "legacy",
            "header_len": len(blob),
            "fallback": False
        }


def get_current_transcription_provider() -> str:
    return _TRANSCRIPTION_PROVIDER_NAME.capitalize()


def get_default_vad_aggressiveness() -> int:
    return 1 if _TRANSCRIPTION_PROVIDER_NAME == 'deepgram' else 2


def transcribe_large_audio_file_with_progress(
    audio_file_path: str,
    openai_api_key: str,
    language_setting_from_client: str,
    progress_callback,
    chunk_size_mb: int = 20
):
    whisper = OpenAIWhisper(api_key=openai_api_key)
    progress_callback(0, 1, 'Transcribing audio...')
    result = whisper.transcribe_file(wav_path=audio_file_path, language=language_setting_from_client)
    progress_callback(1, 1, 'Transcription complete')
    if not result:
        return None
    return {
        'text': result.get('text', ''),
        'segments': result.get('segments') or [],
        'partial': False
    }


def transcribe_large_audio_file(audio_file_path: str, openai_api_key: str, language_setting_from_client: str):
    whisper = OpenAIWhisper(api_key=openai_api_key)
    result = whisper.transcribe_file(wav_path=audio_file_path, language=language_setting_from_client)
    if not result:
        return None
    return {
        'text': result.get('text', ''),
        'segments': result.get('segments') or []
    }

def format_language_for_header(language_code: str) -> str:
    language_map = {
        'any': 'Auto-detect',
        'sv': 'Swedish',
        'en': 'English',
        'da': 'Danish',
        'no': 'Norwegian',
        'fi': 'Finnish',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    return language_map.get(language_code, language_code.upper())


def _get_default_timezone_name() -> str:
    env_tz = os.getenv("DEFAULT_TRANSCRIPT_TIMEZONE")
    if env_tz:
        try:
            ZoneInfo(env_tz)
            return env_tz
        except Exception:
            logger.warning(f"Environment DEFAULT_TRANSCRIPT_TIMEZONE='{env_tz}' is invalid; ignoring")

    try:
        local_tz = datetime.now().astimezone().tzinfo
        if isinstance(local_tz, ZoneInfo):
            return getattr(local_tz, "key", "UTC") or "UTC"
    except Exception:
        pass

    return "UTC"


def _resolve_client_timezone(tz_name: Optional[str], reference_utc: datetime) -> Tuple[str, str, datetime]:
    """Resolve a client-supplied timezone into canonical name, abbreviation, and local start time."""
    fallback_zone_name = _get_default_timezone_name()
    zone_name = fallback_zone_name

    try:
        zone = ZoneInfo(fallback_zone_name)
    except Exception:
        logger.warning(f"Failed to initialize fallback timezone '{fallback_zone_name}', defaulting to UTC")
        zone = ZoneInfo("UTC")
        zone_name = "UTC"

    if tz_name:
        try:
            zone = ZoneInfo(tz_name)
            zone_name = tz_name
        except Exception:
            logger.warning(f"Received invalid timezone '{tz_name}', falling back to '{zone_name}'")

    local_dt = reference_utc.astimezone(zone)
    tz_abbr = local_dt.tzname() or zone_name or "UTC"
    return zone_name, tz_abbr, local_dt

# Mobile recording uses custom magic number detection (no external dependencies needed)

# ---------------------------
# New: Multi-Agent Transcript Save Endpoint helpers/imports
# ---------------------------
from utils.multi_agent_summarizer.pipeline import summarize_transcript as ma_summarize_transcript  # re-import for clarity
import boto3 as _boto3

def _read_transcript_text_for_ma(s3_key: str) -> str:
    if s3_key.startswith("s3://"):
        _, _, bucket_and_key = s3_key.partition("s3://")
        bucket, _, key = bucket_and_key.partition("/")
        s3c = _boto3.client("s3")
        obj = s3c.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    with open(s3_key, "r", encoding="utf-8") as f:
        return f.read()

# Mobile Recording Enhancement Functions
def _detect_audio_codec(audio_bytes: bytes) -> str:
    """Detect audio codec from binary data using file magic numbers."""
    try:
        # Check for common audio format signatures
        if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            # WebM/EBML signature
            return "audio/webm"
        elif audio_bytes[4:8] == b'ftyp':
            # MP4 signature
            if b'M4A ' in audio_bytes[:32] or b'mp41' in audio_bytes[:32]:
                return "audio/mp4"
            return "audio/mp4"
        elif audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:12]:
            # WAV signature
            return "audio/wav"
        elif audio_bytes.startswith(b'\xff\xfb') or audio_bytes.startswith(b'\xff\xf3') or audio_bytes.startswith(b'\xff\xf2'):
            # MP3 signature
            return "audio/mpeg"
        elif len(audio_bytes) >= 2 and all(abs(int.from_bytes(audio_bytes[i:i+2], 'little', signed=True)) < 32767 for i in range(0, min(100, len(audio_bytes)-1), 2)):
            # Heuristic for PCM: check if values are in valid 16-bit range
            return "audio/pcm"
        else:
            # Default fallback
            return "audio/webm"
    except Exception as e:
        logger.error(f"Codec detection failed: {e}")
        return "audio/webm"

def _normalize_audio_to_wav(audio_bytes: bytes, source_format: str, target_rate: int = 16000, target_channels: int = 1) -> bytes:
    """
    Normalize audio to 16kHz mono WAV format using ffmpeg.

    Args:
        audio_bytes: Raw audio data
        source_format: Detected format (audio/webm, audio/mp4, etc.)
        target_rate: Target sample rate (default 16kHz for STT)
        target_channels: Target channel count (default mono)

    Returns:
        Normalized WAV audio bytes
    """
    try:
        # Map content types to ffmpeg input formats
        format_map = {
            "audio/webm": "webm",
            "audio/mp4": "mp4",
            "audio/wav": "wav",
            "audio/mpeg": "mp3",
            "audio/pcm": "s16le"  # 16-bit signed little-endian PCM
        }

        input_format = format_map.get(source_format, "webm")

        with tempfile.NamedTemporaryFile(suffix=f".{input_format}") as input_file, \
             tempfile.NamedTemporaryFile(suffix=".wav") as output_file:

            # Write input audio
            input_file.write(audio_bytes)
            input_file.flush()

            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite output
                "-i", input_file.name,
                "-ar", str(target_rate),  # Sample rate
                "-ac", str(target_channels),  # Channel count
                "-c:a", "pcm_s16le",  # 16-bit PCM codec
                "-f", "wav",  # WAV format
                output_file.name
            ]

            # Special handling for raw PCM input
            if source_format == "audio/pcm":
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "s16le",  # Input format
                    "-ar", "16000",  # Assume 16kHz input for PCM
                    "-ac", "1",     # Assume mono for PCM
                    "-i", input_file.name,
                    "-ar", str(target_rate),
                    "-ac", str(target_channels),
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    output_file.name
                ]

            # Execute ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"FFmpeg normalization failed: {result.stderr}")
                raise Exception(f"FFmpeg error: {result.stderr}")

            # Read normalized audio
            output_file.seek(0)
            return output_file.read()

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg normalization timed out")
        raise Exception("Audio normalization timeout")
    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        raise

def _handle_mobile_audio_processing(session_id: str, audio_bytes: bytes, session_data: dict):
    """
    Handle mobile audio processing with format normalization.

    This function processes audio from mobile devices, normalizes it to a standard format,
    and sends it for transcription.
    """
    try:
        content_type = session_data.get("content_type", "audio/webm")
        mobile_telemetry = session_data.get("mobile_telemetry", {})

        # Update telemetry
        mobile_telemetry["transcode_attempts"] = mobile_telemetry.get("transcode_attempts", 0) + 1

        # Normalize audio if needed
        if content_type in ["audio/mp4", "audio/pcm"] or content_type.startswith("audio/webm") and "opus" in content_type:
            logger.debug(f"Session {session_id}: Normalizing {content_type} to WAV")
            normalized_audio = _normalize_audio_to_wav(audio_bytes, content_type)
            mobile_telemetry["transcode_successes"] = mobile_telemetry.get("transcode_successes", 0) + 1
        else:
            # Use original audio (WebM is supported by OpenAI)
            normalized_audio = audio_bytes

        # Process with existing transcription service
        session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(normalized_audio)

        # Feed normalized audio into ring buffer for reconnection
        try:
            recent_audio = session_data.get("recent_audio")
            if recent_audio is not None:
                recent_audio.append(normalized_audio)
        except Exception as e:
            logger.debug(f"Session {session_id}: Failed to add audio to ring buffer: {e}")

        # Estimate duration (rough approximation based on content type)
        duration_estimate = _estimate_audio_duration(normalized_audio, content_type)
        session_data["accumulated_audio_duration_for_current_segment_seconds"] += duration_estimate

        logger.debug(f"Session {session_id}: Added {len(normalized_audio)} normalized bytes, estimated duration: {duration_estimate:.2f}s")

    except Exception as e:
        logger.error(f"Mobile audio processing failed for session {session_id}: {e}")
        mobile_telemetry["error_count"] = mobile_telemetry.get("error_count", 0) + 1
        # Fall back to treating as WebM
        session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(audio_bytes)
        session_data["accumulated_audio_duration_for_current_segment_seconds"] += 1.0  # Conservative estimate

def _estimate_audio_duration(audio_bytes: bytes, content_type: str) -> float:
    """Estimate audio duration in seconds based on format and size."""
    try:
        if content_type == "audio/wav":
            # WAV has header with duration info
            if len(audio_bytes) >= 44:
                # Basic WAV duration calculation: data_size / (sample_rate * channels * bytes_per_sample)
                # For 16kHz mono 16-bit: ~32KB per second
                return max(0.1, (len(audio_bytes) - 44) / 32000)
        elif content_type == "audio/pcm":
            # PCM: 16kHz mono 16-bit = 32KB per second
            return max(0.1, len(audio_bytes) / 32000)
        elif "mp4" in content_type:
            # AAC compression ~8:1, estimate based on size
            return max(0.1, len(audio_bytes) / 4000)  # ~4KB per second for AAC
        else:
            # WebM/Opus compression ~10:1, estimate based on size
            return max(0.1, len(audio_bytes) / 3000)  # ~3KB per second for Opus
    except:
        return 1.0  # Conservative fallback

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
        }
        # Add request_id if it's in the context
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        return json.dumps(log_record)

class RedactFilter(logging.Filter):
    SENSITIVE_KEYS = ("authorization", "apikey", "api-key", "x-api-key", "cookie", "set-cookie")
    # Match "Header: value" or "header=value" and redact the value up to a delimiter
    _pattern = re.compile("(" + "|".join(SENSITIVE_KEYS) + r")(\s*[:=]\s*)([^,;\s'\"]+)", re.IGNORECASE)

    def _redact(self, text: str) -> str:
        try:
            return self._pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", text)
        except Exception:
            return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self._redact(str(record.msg))
            if record.args:
                record.args = tuple(self._redact(str(a)) for a in record.args)
        except Exception:
            pass
        return True

def setup_logging(debug=False):
    log_filename = 'api_server.log'; root_logger = logging.getLogger(); log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8'); fh.setLevel(log_level)
        json_formatter = JsonFormatter()
        fh.setFormatter(json_formatter); root_logger.addHandler(fh)
    except Exception as e: 
        print(f"Error setting up file logger: {e}", file=sys.stderr)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level)
    cf = logging.Formatter('[%(levelname)-8s] %(name)s: %(message)s'); ch.setFormatter(cf); root_logger.addHandler(ch)
    # Add redaction filter globally (only once)
    if not any(isinstance(f, RedactFilter) for f in root_logger.filters):
        root_logger.addFilter(RedactFilter())

    for lib in ['anthropic', 'httpx', 'httpcore', 'hpack', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone', 'werkzeug', 'flask_sock', 'google.generativeai', 'google.api_core']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('utils').setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger

s3_bucket_on_startup = os.getenv('AWS_S3_BUCKET')
startup_logger = logging.getLogger(__name__ + ".startup") 
if s3_bucket_on_startup:
    startup_logger.info(f"AWS_S3_BUCKET on startup: '{s3_bucket_on_startup}'")
else:
    startup_logger.error("AWS_S3_BUCKET environment variable is NOT SET at startup!")

logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

# Import VAD integration components after logger is set up
try:
    from vad_integration_bridge import (
        initialize_vad_bridge, get_vad_bridge, cleanup_vad_bridge, 
        is_vad_enabled, log_vad_configuration
    )
    VAD_IMPORT_SUCCESS = True
    logger.info("VAD integration components imported successfully")
except ImportError as e:
    VAD_IMPORT_SUCCESS = False
    logger.warning(f"VAD integration not available: {e}")

app = Flask(__name__)
CORS(app) 
sock = Sock(app) 
# Create a global thread pool for handling slow transcription tasks.
# The number of workers can be tuned based on server resources.
app.executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 2)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))


def push_session_event(session_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Record a session-scoped event and fan it out to connected clients."""
    payload = payload or {}
    timestamp = datetime.now(timezone.utc).isoformat()
    event_record = {"type": event_type, "payload": payload, "ts": timestamp}

    ws_to_notify: Optional[Any] = None

    lock = session_locks[session_id]
    with lock:
        sess = active_sessions.get(session_id)
        if not sess:
            return

        events_buf = sess.setdefault("recent_events", deque(maxlen=200))
        events_buf.append(event_record)

        if event_type == "provider_fallback":
            sess["current_provider"] = payload.get("to") or payload.get("provider") or sess.get("current_provider")
            sess["fallback_active"] = True
        elif event_type in {"provider_primary_resumed", "provider_probe_recovered"}:
            sess["current_provider"] = payload.get("provider") or sess.get("current_provider")
            sess["fallback_active"] = False

        sess["last_event_ts"] = timestamp
        ws_to_notify = sess.get("websocket_connection")

    if ws_to_notify:
        message = {"type": "event", "event": event_type, "ts": timestamp}
        if payload:
            message.update(payload)
        try:
            ws_to_notify.send(json.dumps(message))
        except Exception as e:
            logger.debug(f"Session {session_id}: Failed to send event {event_type}: {e}")


app.config["SESSION_EVENT_EMITTER"] = push_session_event
set_emitter(push_session_event)

# Request ID middleware
@app.before_request
def add_request_id():
    # Check if request ID is provided by frontend, otherwise generate one
    request_id = request.headers.get('X-Request-ID', str(uuid4()))
    g.request_id = request_id
    
    # Add request ID to all log records for this request
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            record.request_id = request_id
            return True
    
    # Add the filter to the root logger for this request
    root_logger = logging.getLogger()
    request_filter = RequestIdFilter()
    root_logger.addFilter(request_filter)
    
    # Store the filter so we can remove it after the request
    g.request_filter = request_filter

@app.after_request
def remove_request_id_filter(response):
    # Remove the request ID filter after the request
    if hasattr(g, 'request_filter'):
        root_logger = logging.getLogger()
        root_logger.removeFilter(g.request_filter)
    
    # Add request ID to response headers for debugging
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    return response

# Clear any stale sessions at startup, which can happen in some deployment environments
active_sessions: Dict[str, Dict[str, Any]] = {}
session_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
logger.info("Initialized and cleared active_sessions and session_locks at startup.")

# Reattach grace window for unexpected disconnects (seconds)
REATTACH_GRACE_SECONDS = int(os.getenv("REATTACH_GRACE_SECONDS", "90"))
RINGBUF_SECONDS = int(os.getenv("RINGBUF_SECONDS", "45"))
RINGBUF_MAX_RESULTS = int(os.getenv("RINGBUF_MAX_RESULTS", "200"))

# Gap recovery configuration
SEGMENT_RETRY_ENABLED = os.getenv("SEGMENT_RETRY_ENABLED", "false").lower() == "true"
SEGMENT_RETRY_MAX_ATTEMPTS = int(os.getenv("SEGMENT_RETRY_MAX_ATTEMPTS", "3"))
MAX_PENDING_SEGMENTS = int(os.getenv("MAX_PENDING_SEGMENTS", "20"))
SEGMENT_GAP_TIMEOUT_SEC = int(os.getenv("SEGMENT_GAP_TIMEOUT_SEC", "120"))

def _init_session_reconnect_state(session_id: str) -> Dict[str, Any]:
    """Initialize session state for reconnection support with ring buffers."""
    return {
        "reconnect_token": uuid4().hex,
        "last_disconnect_ts": None,
        "grace_deadline": None,
        "recent_audio": deque(maxlen=RINGBUF_SECONDS * 3),  # rough chunks @ ~3 items/s
        "recent_results": deque(maxlen=RINGBUF_MAX_RESULTS),
        "next_seq": 1,
        "last_client_ack": 0,
        # Gap recovery sequence tracking
        "next_segment_seq": 1,       # Next sequence number for transcription segments
        "expected_seq": 1,           # Next sequence we expect to deliver
        "pending_seqs": {},          # {seq: transcript_data} waiting for gaps to fill
        "max_delivered_seq": 0,      # Highest sequence sent to client
        "last_audio_ts": None,
        "silence_seconds": 0,
        "recent_events": deque(maxlen=200),
    }

# Job tracking system for async transcription
transcription_jobs: Dict[str, Dict[str, Any]] = {}
job_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
logger.info("Initialized transcription job tracking system.")


@app.get("/session/<session_id>/status")
def get_session_status(session_id: str):
    """Return authoritative session state for the chat view."""
    sess = active_sessions.get(session_id)
    if not sess:
        return jsonify({"exists": False}), 404

    lock = session_locks[session_id]
    with lock:
        seq_state = {
            "next": sess.get("next_seq", 1),
            "expected": sess.get("expected_seq", 1),
            "max_delivered": sess.get("max_delivered_seq", 0),
            "pending": len(sess.get("pending_seqs") or {}),
        }

        provider_state = {
            "name": sess.get("current_provider", "primary"),
            "fallback_active": bool(sess.get("fallback_active", False)),
        }

        pending_segments = int(sess.get("pending_segments", 0))
        max_pending = int(sess.get("max_pending_segments", MAX_PENDING_SEGMENTS))

        grace_deadline = sess.get("grace_deadline")
        if isinstance(grace_deadline, (int, float)):
            grace_iso = datetime.fromtimestamp(grace_deadline, tz=timezone.utc).isoformat()
        else:
            grace_iso = grace_deadline

        status_payload = {
            "exists": True,
            "connected": bool(sess.get("websocket_connection")) and not bool(sess.get("ws_disconnected")),
            "grace_deadline": grace_iso,
            "last_audio_ts": sess.get("last_audio_ts"),
            "silence_seconds": int(sess.get("silence_seconds", 0)),
            "seq": seq_state,
            "backlog": {
                "pending_segments": pending_segments,
                "max_pending": max_pending,
            },
            "provider": provider_state,
            "reconnect_token": sess.get("reconnect_token"),
        }

    return jsonify(status_payload), 200

def update_job_progress(job_id: str, **kwargs):
    """Update job progress in a thread-safe manner."""
    with job_locks[job_id]:
        if job_id in transcription_jobs:
            transcription_jobs[job_id].update(kwargs)
            transcription_jobs[job_id]['updated_at'] = datetime.now(timezone.utc).isoformat()

def create_transcription_job(agent_name: str, s3_key: str, original_filename: str, user_id: str) -> str:
    """Create a new transcription job and return job ID."""
    job_id = str(uuid4())
    job_data = {
        'status': 'queued',
        'progress': 0.0,
        'current_step': 'Preparing transcription...',
        'total_chunks': 0,
        'completed_chunks': 0,
        'result': None,
        'error': None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'agent_name': agent_name,
        's3_key': s3_key,
        'original_filename': original_filename,
        'user_id': user_id
    }
    
    with job_locks[job_id]:
        transcription_jobs[job_id] = job_data
    
    logger.info(f"Created transcription job {job_id} for {original_filename}")
    return job_id

def on_shutdown():
    """Ensure the executor is shut down gracefully when the app exits."""
    logger.info("Shutting down ThreadPoolExecutor...")
    app.executor.shutdown(wait=True)
    logger.info("Executor shut down successfully.")

import atexit
atexit.register(on_shutdown)

def has_user_agent_access(user_id: str, agent_name: str) -> bool:
    """
    Check if a user has access to a specific agent.
    Returns True if the user has access, False otherwise.
    """
    client = get_supabase_client()
    if not client:
        logger.error("Cannot check agent access: Supabase client not available.")
        return False
    
    try:
        # Get agent ID from name
        agent_res = client.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            logger.error(f"Database error finding agent_id for name '{agent_name}': {agent_res.error}")
            return False
        if not agent_res.data:
            logger.debug(f"Agent with name '{agent_name}' not found in DB.")
            return False
        
        agent_id = agent_res.data[0]['id']
        
        # Check user access to agent
        access_res = client.table("user_agent_access") \
            .select("agent_id") \
            .eq("user_id", user_id) \
            .eq("agent_id", agent_id) \
            .limit(1) \
            .execute()
        
        if hasattr(access_res, 'error') and access_res.error:
            logger.error(f"Database error checking access for user {user_id} / agent {agent_name}: {access_res.error}")
            return False
        
        has_access = bool(access_res.data)
        logger.debug(f"User {user_id} {'has' if has_access else 'does not have'} access to agent {agent_name}")
        return has_access
        
    except Exception as e:
        logger.error(f"Exception checking user agent access: {e}")
        return False

def verify_s3_key_ownership(s3_key: str, user: SupabaseUser) -> bool:
    """
    Verifies S3 object access. For user-specific files, it checks for user ID or agent permissions.
    For shared files (like agent docs and transcripts), it checks for valid path and agent access.
    This is a security measure to prevent Insecure Direct Object Reference (IDOR).
    """
    if not s3_key or not user:
        return False
    
    # Allow access to agent documentation, as it's not user-specific data.
    # The path is organizations/river/agents/{agent_name}/docs/{filename}
    if '/agents/' in s3_key and '/docs/' in s3_key:
        logger.debug(f"S3 Ownership check PASSED for user {user.id} on agent doc key {s3_key}")
        return True

    # Allow access to agent event docs if user has access to the agent and the path matches the flat docs prefix.
    event_doc_match = parse_event_doc_key(s3_key)
    if event_doc_match:
        agent_name, event_id, filename = event_doc_match
        if has_user_agent_access(user.id, agent_name):
            logger.debug(
                f"S3 Ownership check PASSED for user {user.id} on agent {agent_name} event doc key {s3_key}"
            )
            return True
        logger.warning(
            f"SECURITY: User {user.id} tried to access {agent_name} event docs without agent access. Key: {s3_key}"
        )
        return False

    # Allow access to agent transcripts/recordings if user has access to the agent
    # Path patterns: organizations/river/agents/{agent_name}/events/{event_id}/transcripts/...
    #                organizations/river/agents/{agent_name}/events/{event_id}/recordings/...
    agent_transcript_match = re.search(r'/agents/([^/]+)/events/[^/]+/(transcripts|recordings)/', s3_key)
    if agent_transcript_match:
        agent_name = agent_transcript_match.group(1)
        # Check if user has access to this agent
        if has_user_agent_access(user.id, agent_name):
            logger.debug(f"S3 Ownership check PASSED for user {user.id} on agent {agent_name} transcript/recording key {s3_key}")
            return True
        else:
            logger.warning(f"SECURITY: User {user.id} tried to access {agent_name} transcript/recording without agent access. Key: {s3_key}")
            return False

    # For other user-specific files, check the uID pattern
    match = re.search(r'_uID-([a-f0-9\-]+)', s3_key)
    if match:
        owner_id = match.group(1)
        if owner_id == user.id:
            logger.debug(f"S3 Ownership check PASSED for user {user.id} on key {s3_key}")
            return True
        else:
            logger.warning(f"SECURITY: IDOR attempt - User {user.id} tried to access S3 key owned by {owner_id}. Key: {s3_key}")
            logger.info(f"Current user ID: {user.id}, File owner ID: {owner_id}")
            return False
            
    # Deny by default if the key does not conform to any of the allowed patterns.
    logger.warning(f"SECURITY: Access denied for user {user.id} on key {s3_key} because it lacks an ownership identifier (uID) and is not a recognized shared resource path.")
    return False

UPLOAD_FOLDER = 'tmp/uploaded_transcriptions/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CircuitBreaker:
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int, recovery_timeout: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        logger.info(f"Circuit Breaker initialized: Threshold={failure_threshold}, Timeout={recovery_timeout}s")

    def is_open(self) -> bool:
        with self._lock:
            if self.state == self.STATE_OPEN:
                if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    logger.warning("Circuit Breaker: State changed to HALF_OPEN. Allowing a test request.")
                    return False
                return True
            return False

    def record_failure(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.last_failure_time = time.time()
                logger.error("Circuit Breaker: Failure in HALF_OPEN state. Tripping back to OPEN.")
            else: # CLOSED
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = self.STATE_OPEN
                    self.last_failure_time = time.time()
                    logger.error(f"Circuit Breaker: Failure threshold ({self.failure_threshold}) reached. Tripping to OPEN state for {self.recovery_timeout}s.")
    
    def record_success(self):
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                logger.info("Circuit Breaker: Success in HALF_OPEN state. Resetting to CLOSED.")
            elif self.state == self.STATE_CLOSED and self.failure_count > 0:
                 logger.info("Circuit Breaker: Success recorded, resetting failure count.")
            self.failure_count = 0
            self.state = self.STATE_CLOSED

# Note: Circuit breaker instances and CircuitBreakerOpen are provided by utils.llm_api_utils

SIMPLE_QUERIES_TO_BYPASS_RAG = frozenset([
    "hello", "hello :)", "hi", "hi :)", "hey", "hey :)", "yo", "greetings", "good morning", "good afternoon",
    "good evening", "morning", "afternoon", "evening", "bye", "goodbye", "see you", "see ya", "take care", "farewell", "bye bye",
    "later", "cheers", "good night", "thanks", "thanks :)", "thank you", "thank you :)", "ty", "thx", "appreciate it", "thanks so much",
    "thank you very much", "much appreciated", "thanks a lot", "ok", "ok :)", "okay", "okay :)", "k", "kk", "got it", "understood", "sounds good", "perfect",
    "great", "cool", "alright", "roger", "fine", "sure", "yes", "yes :)", "yes please", "yes please :)",
    "yep", "yeah", "yeah :)", "indeed", "affirmative", "certainly", "absolutely", "definitely",
    "exactly", "right", "correct", "i see", "makes sense", "fair enough", "will do",
    "you bet", "of course", "agreed", "true", "no", "nope", "nah", "negative", "not really", "i disagree", "disagree",
    "false", "incorrect", "please", "pardon", "excuse me", "you're welcome", "yw", "no problem", "np",
    "my pleasure", "don't worry", "it's ok", "it's okay", "no worries", "wow", "oops", "nice", "awesome", "excellent", "fantastic", "amazing",
    "brilliant", "sweet", "sorry", "my apologies", "apologies", "my bad", "maybe", "perhaps", "i don't know", "idk", "not sure", "hard to say",
    "really", "are you sure", "is that right", "correct",
    "hej", "hejsan", "god morgon", "god kväll", "hej då", "vi ses", "ha det bra",
    "tack", "tack så mycket", "okej", "perfekt", "toppen", "ja", "nej", "jepp",
    "inga problem", "kanske", "jag vet inte", "är du säker", "precis",
    "hej :)", "hejsan :)", "god morgon :)", "god kväll :)", "hej då :)", "vi ses :)",
    "tack :)", "tack så mycket :)", "okej :)", "perfekt :)", "toppen :)", "ja :)", "ja tack :)",
    "jepp :)", "inga problem :)", "precis :)"
])

anthropic_client: Optional[Anthropic] = None
gemini_client: Optional[genai.GenerativeModel] = None # Placeholder, will be configured
try: 
    init_pinecone()
    logger.info("Pinecone initialized (or skipped).")
except Exception as e: 
    logger.warning(f"Pinecone initialization failed: {e}")

# The get_supabase_client logic is now in utils/supabase_client.py
# We just need to call it to ensure it's initialized at startup.
supabase = get_supabase_client()

try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not found")
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized.")
except Exception as e: 
    logger.critical(f"Failed Anthropic client init: {e}", exc_info=True)
    anthropic_client = None # Keep it None on failure

try:
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key: raise ValueError("GOOGLE_API_KEY not found")
    genai.configure(api_key=google_api_key)
    logger.info("Google Generative AI client configured.")
except Exception as e:
    logger.critical(f"Failed Google GenAI client init: {e}", exc_info=True)

# Initialize VAD integration if enabled and available
vad_bridge = None
if VAD_IMPORT_SUCCESS and os.getenv('ENABLE_VAD_TRANSCRIPTION', 'true').lower() == 'true':
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            vad_bridge = initialize_vad_bridge(openai_api_key)
            logger.info("VAD transcription bridge initialized successfully")
        else:
            logger.error("VAD transcription enabled but OPENAI_API_KEY not found")
    except Exception as e:
        logger.error(f"Failed to initialize VAD transcription bridge: {e}", exc_info=True)
        vad_bridge = None

# Log VAD configuration for debugging
if VAD_IMPORT_SUCCESS:
    log_vad_configuration()
else:
    logger.info("VAD transcription not available (import failed)")

def log_retry_error(retry_state): 
    logger.warning(f"Retrying API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")
retry_strategy_anthropic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((anthropic.APIStatusError, anthropic.APIConnectionError)))
)

# A separate strategy for Gemini, as it has different retry-able exceptions
retry_strategy_gemini = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    # Gemini can throw ResourceExhausted or other service unavailable errors
    retry=(retry_if_exception_type((
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError
    )))
)

retry_strategy_openai = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type((OpenAI_APIStatusError, OpenAI_APIConnectionError)))
)

# Specific retry for Supabase auth transient errors
retry_strategy_auth = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=retry_if_exception_type(AuthRetryableError)
)

@retry_strategy_auth
def verify_user(token: Optional[str]) -> Optional[SupabaseUser]:
    client = get_supabase_client()
    if not client:
        logger.error("Auth check failed: Supabase client not available.")
        return None

    if not token:
        logger.warning("Auth check failed: No token provided.")
        return None

    # Bound the auth call; retry once on timeout
    def _get_user_with_timeout(jwt: str, timeout_s: float = 2.5):
        import httpx, time
        t0 = time.time()
        try:
            return client.auth.get_user(jwt)  # Note: supabase client may not support timeout kw directly
        except Exception as e:
            # Retry once if this smells like a timeout
            if "timeout" in str(e).lower() or isinstance(e, (httpx.ReadTimeout, TimeoutError)):
                logger.info(f"Auth timeout, retrying after {time.time() - t0:.2f}s")
                return client.auth.get_user(jwt)
            raise

    try:
        user_resp = _get_user_with_timeout(token)
        if user_resp and user_resp.user:
            logger.debug(f"Token verified for user ID: {user_resp.user.id}")
            return user_resp.user
        else:
            logger.warning("Auth check failed: Invalid token or user not found in response.")
            return None
    except AuthRetryableError as e:
        logger.warning(f"Auth connection error, will retry: {e}")
        raise # Re-raise to be handled by the tenacity decorator
    except AuthApiError as e:
        logger.warning(f"Auth API Error during token verification (non-retryable): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
        return None

def verify_user_agent_access(token: Optional[str], agent_name: Optional[str]) -> Tuple[Optional[SupabaseUser], Optional[Response]]:
    client = get_supabase_client()
    if not client:
        logger.error("Auth check failed: Supabase client not available.")
        return None, jsonify({"error": "Auth service unavailable"}, 503)

    user = verify_user(token)
    if not user: 
        return None, jsonify({"error": "Unauthorized: Invalid or missing token"}, 401)
    
    if not agent_name:
        logger.debug(f"Authorization check: User {user.id} authenticated. No specific agent access check required for this route.")
        return user, None

    try:
        agent_res = client.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            logger.error(f"Database error finding agent_id for name '{agent_name}': {agent_res.error}")
            return None, jsonify({"error": "Database error checking agent"}), 500
        if not agent_res.data:
            logger.warning(f"Authorization check failed: Agent with name '{agent_name}' not found in DB.")
            return None, jsonify({"error": "Forbidden: Agent not found"}), 403
        
        agent_id = agent_res.data[0]['id']
        logger.debug(f"Found agent_id '{agent_id}' for name '{agent_name}'.")

        access_res = client.table("user_agent_access") \
            .select("agent_id") \
            .eq("user_id", user.id) \
            .eq("agent_id", agent_id) \
            .limit(1) \
            .execute()

        logger.debug(f"DB Check for user {user.id} accessing agent {agent_name} (ID: {agent_id}): {access_res.data}")

        if hasattr(access_res, 'error') and access_res.error:
            logger.error(f"Database error checking access for user {user.id} / agent {agent_name}: {access_res.error}")
            return None, jsonify({"error": "Database error checking permissions"}), 500

        if not access_res.data:
            logger.warning(f"Access Denied: User {user.id} does not have access to agent {agent_name} (ID: {agent_id}).")
            return None, jsonify({"error": "Forbidden: Access denied to this agent"}), 403

        logger.info(f"Access Granted: User {user.id} authorized for agent {agent_name} (ID: {agent_id}).")
        return user, None 

    except (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        # Re-raise exceptions that the retry decorator should handle
        logger.warning(f"Retryable network error during agent access check: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during agent access check for user {user.id} / agent {agent_name}: {e}", exc_info=True)
        return None, jsonify({"error": "Internal server error during authorization"}, 500)

import httpx

def get_user_role(user_id: str) -> Optional[str]:
    """Retrieves a user's role from the database."""
    client = get_supabase_client()
    if not client:
        logger.error(f"Cannot get role for user {user_id}: Supabase client not available.")
        return None
    try:
        res = client.table("user_roles").select("role").eq("user_id", user_id).single().execute()
        if hasattr(res, 'error') and res.error:
            # It's normal for a user to not have a role, so we log this at a debug level.
            logger.debug(f"Could not retrieve role for user {user_id}: {res.error}")
            return None
        return res.data.get("role") if res.data else None
    except Exception as e:
        logger.error(f"Unexpected error fetching role for user {user.id}: {e}", exc_info=True)
        return None

def admin_or_super_user_required(f):
    """Decorator to protect routes, allowing access only to 'admin' or 'super user' roles."""
    @wraps(f)
    def decorated_function(user: SupabaseUser, *args, **kwargs):
        role = get_user_role(user.id)
        if role not in ['admin', 'super user']:
            logger.warning(f"Forbidden: User {user.id} with role '{role}' tried to access an admin route.")
            return jsonify({"error": "Forbidden: Administrator or super user access required."}), 403
        
        logger.info(f"Access granted for admin route to user {user.id} with role '{role}'.")
        # The original function expects 'user' as a keyword argument from the supabase_auth_required decorator.
        return f(user=user, *args, **kwargs)
    return decorated_function

# Generic retry strategy for Supabase calls that might face transient network issues
retry_strategy_supabase = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry_error_callback=log_retry_error,
    retry=retry_if_exception_type((
        httpx.RequestError, httpx.TimeoutException, httpx.ConnectError, 
        httpx.RemoteProtocolError, ConnectionError, OSError
    ))
)

@retry_strategy_supabase
def verify_user_agent_access_with_retry(token: Optional[str], agent_name: Optional[str]) -> Tuple[Optional[SupabaseUser], Optional[Response]]:
    return verify_user_agent_access(token, agent_name)

def supabase_auth_required(agent_required: bool = True):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # For POST/PUT requests, parse the JSON body once and store it in Flask's request-bound global 'g'.
            # This prevents consuming the request stream multiple times.
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    g.json_data = request.get_json(silent=True) or {}
                except Exception as e:
                    logger.warning(f"Could not parse request body as JSON: {e}")
                    g.json_data = {}
            else:
                g.json_data = {}

            auth_header = request.headers.get("Authorization")
            token = None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]

            agent_name_from_payload = None
            if agent_required:
                # Use the already-parsed data from g
                if 'agent' in g.json_data:
                    agent_name_from_payload = g.json_data.get('agent')
                elif 'agentName' in g.json_data: # For manage-file route
                    agent_name_from_payload = g.json_data.get('agentName')
                elif 'agent' in request.args: # Fallback for GET requests
                    agent_name_from_payload = request.args.get('agent')

            # Special case for the immutable agent creator: user must be an admin/super user.
            if agent_name_from_payload == '_aicreator':
                user = verify_user(token)
                if not user:
                    return jsonify({"error": "Unauthorized: Invalid or missing token for _aicreator"}), 401
                
                role = get_user_role(user.id)
                if role not in ['admin', 'super user']:
                    logger.warning(f"Forbidden: User {user.id} with role '{role}' tried to access special agent '_aicreator'.")
                    return jsonify({"error": "Forbidden: Access to this agent requires administrative privileges."}), 403

                logger.info(f"Access granted for special agent '_aicreator' to user {user.id} with role '{role}'.")
                return f(user=user, *args, **kwargs)
            
            try:
                user, error_response = verify_user_agent_access_with_retry(token, agent_name_from_payload if agent_required else None)
                if error_response:
                    return error_response
                if not user: # Should be caught by error_response, but as a safeguard
                    return jsonify({"error": "Unauthorized"}), 401
                return f(user=user, *args, **kwargs)
            except RetryError as e:
                logger.error(f"Authorization failed after multiple retries: {e}", exc_info=True)
                return jsonify({"error": "Authorization service is temporarily unavailable. Please try again."}), 503
            except Exception as e:
                logger.error(f"Unhandled exception during authorization: {e}", exc_info=True)
                return jsonify({"error": "Internal server error during authorization."}), 500
        return decorated_function
    return decorator

@app.route('/api/users/list', methods=['GET'])
@supabase_auth_required(agent_required=False)
@admin_or_super_user_required
def list_users(user: SupabaseUser):
    """Lists all users for admin dashboard."""
    client = get_supabase_client()
    if not client:
        return jsonify({"error": "Database service unavailable"}), 503
    try:
        users_response = client.auth.admin.list_users()
        # The response is an iterator, so we consume it into a list
        users = [{'id': u.id, 'email': u.email} for u in users_response]
        return jsonify(users), 200
    except Exception as e:
        logger.error(f"Error listing users: {e}", exc_info=True)
        return jsonify({"error": "Failed to list users"}), 500

@app.route('/api/users/create', methods=['POST'])
@supabase_auth_required(agent_required=False)
@admin_or_super_user_required
def create_user_admin(user: SupabaseUser):
    """Creates a new user with auto-confirmed email."""
    data = g.get('json_data', {})
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    client = get_supabase_client()
    if not client:
        return jsonify({"error": "Database service unavailable"}), 503
    
    try:
        new_user_res = client.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True, # Auto-confirm user as per spec
        })
        
        new_user = new_user_res.user
        if not new_user:
             logger.error(f"Admin user creation did not return a user object for email {email}.")
             return jsonify({"error": "Failed to create user, no user object returned."}), 500

        logger.info(f"Admin {user.id} created new user {new_user.id} ({email}).")
        
        return jsonify({
            "id": new_user.id,
            "email": new_user.email,
            "created_at": new_user.created_at,
        }), 201
    except Exception as e:
        logger.error(f"Error creating user {email}: {e}", exc_info=True)
        if "User already exists" in str(e):
            return jsonify({"error": "A user with this email already exists."}), 409
        return jsonify({"error": f"Failed to create user: {str(e)}"}), 500

from utils.llm_api_utils import (
    _call_anthropic_stream_with_retry, _call_gemini_stream_with_retry,
    _call_openai_stream_with_retry, _call_gemini_non_stream_with_retry,
    _call_groq_stream_with_retry, _call_groq_non_stream_with_retry,
    anthropic_circuit_breaker, gemini_circuit_breaker, openai_circuit_breaker,
    groq_circuit_breaker,
    CircuitBreakerOpen
)


@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    # Check if the GOOGLE_API_KEY is set, as a proxy for client health
    google_client_ok = os.getenv('GOOGLE_API_KEY') is not None
    return jsonify({
        "status": "ok",
        "message": "Backend is running",
        "anthropic_client_initialized": anthropic_client is not None,
        "gemini_client_initialized": google_client_ok,
        "s3_client_initialized": get_s3_client() is not None
    }), 200

@app.route('/api/config/defaults', methods=['GET'])
def get_config_defaults():
    """Get default configuration values for the UI based on current provider settings."""
    return jsonify({
        "transcriptionProvider": get_current_transcription_provider(),
        "defaultVadAggressiveness": get_default_vad_aggressiveness(),
        "defaultTranscriptionLanguage": "any"
    }), 200

@app.route('/healthz', methods=['GET'])
def healthz():
    # Constant-time, no external calls
    resp = jsonify({"status": "ok"})
    resp.headers['Cache-Control'] = 'no-store'
    return resp, 200

def _get_current_recording_status_snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    if session_id and session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "session_id": session_id,
            "is_recording": session.get('is_active', False),
            "is_backend_processing_paused": session.get('is_backend_processing_paused', False),
            "session_start_time_utc": session.get('session_start_time_utc').isoformat() if session.get('session_start_time_utc') else None,
            "agent": session.get('agent_name'),
            "event": session.get('event_id'),
            "current_total_audio_duration_processed_seconds": session.get('current_total_audio_duration_processed_seconds', 0)
        }
    if not active_sessions: return {"is_recording": False, "is_backend_processing_paused": False, "message": "No active sessions"}
    
    first_active_session_id = next((s_id for s_id, s_data in active_sessions.items() if s_data.get('is_active')), None)
    if first_active_session_id:
        return _get_current_recording_status_snapshot(first_active_session_id) 
    
    return {"is_recording": False, "is_backend_processing_paused": False, "message": "No currently active recording sessions"}

@app.route('/api/recording/start', methods=['POST'])
@supabase_auth_required(agent_required=True) 
def start_recording_route(user: SupabaseUser): 
    # Mutual Exclusivity Check
    for session in active_sessions.values():
        if session.get('session_type') == 'recording' and session.get('is_active'):
            logger.warning("Attempted to start a transcript session while a recording session is active.")
            return jsonify({"status": "error", "message": "A recording session is already in progress. Cannot start a transcript session."}), 409

    data = g.get('json_data', {})
    agent_name = data.get('agent')
    event_id = data.get('event')
    # 'language' is the old key, 'transcriptionLanguage' is the new one. Prioritize new one.
    language_setting = data.get('transcriptionLanguage', data.get('language', 'any')) # Default to 'any'
    vad_aggressiveness = data.get('vadAggressiveness') or get_default_vad_aggressiveness()  # VAD aggressiveness level (1, 2, 3)

    logger.info(f"Start recording: agent='{agent_name}', event='{event_id}', language_setting='{language_setting}', vad_aggressiveness='{vad_aggressiveness}'")

    if not event_id: 
        return jsonify({"status": "error", "message": "Missing event ID"}), 400

    session_id = uuid.uuid4().hex
    session_start_time_utc = datetime.now(timezone.utc)
    client_timezone_raw = data.get('clientTimezone')
    client_timezone_name, client_timezone_abbr, session_start_time_local = _resolve_client_timezone(
        client_timezone_raw, session_start_time_utc
    )
    timezone_source = "client" if client_timezone_raw else "default"
    logger.info(
        f"Transcript session {session_id}: timezone set to {client_timezone_name} ({client_timezone_abbr}), source={timezone_source}"
    )
    # Fetch agent-specific API keys or fall back to globals
    agent_openai_key = get_api_key(agent_name, 'openai')
    agent_anthropic_key = get_api_key(agent_name, 'anthropic')
    
    s3_transcript_base_filename = f"transcript_D{session_start_time_utc.strftime('%Y%m%d')}-T{session_start_time_utc.strftime('%H%M%S')}_uID-{user.id}_oID-river_aID-{agent_name}_eID-{event_id}_sID-{session_id}.txt"
    s3_transcript_key = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/{s3_transcript_base_filename}"
    
    temp_audio_base_dir = os.path.join('tmp', 'audio_sessions', session_id)
    os.makedirs(temp_audio_base_dir, exist_ok=True)
    
    # Create session state for sticky provider fallback
    from models.session_state import SessionState
    session_state = SessionState(
        id=session_id,
        user_id=user.id,
        language_hint=language_setting,
        provider_cooldown_sec=int(os.getenv("TRANSCRIBE_PROVIDER_COOLDOWN_SEC", "900"))
    )

    # Initialize reconnection state
    reconnect_state = _init_session_reconnect_state(session_id)

    active_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user.id,
        "agent_name": agent_name,
        "event_id": event_id,
        "session_type": "transcript", # Differentiate session type
        "language_setting_from_client": language_setting, # Store the new setting
        "vad_aggressiveness_from_client": vad_aggressiveness, # Store VAD aggressiveness level
        "session_start_time_utc": session_start_time_utc,
        "session_start_time_local": session_start_time_local,
        "s3_transcript_key": s3_transcript_key,
        "temp_audio_session_dir": temp_audio_base_dir,
        "openai_api_key": agent_openai_key,       # Store the potentially agent-specific key
        "anthropic_api_key": agent_anthropic_key, # Store the potentially agent-specific key
        "is_backend_processing_paused": False,
        "current_total_audio_duration_processed_seconds": 0.0,
        "websocket_connection": None,
        "last_activity_timestamp": time.time(),
        "is_active": True,
        "ws_disconnected": False,
        "session_state": session_state,  # Add session state for sticky provider fallback
        "is_finalizing": False,
        "current_segment_raw_bytes": bytearray(),
        "accumulated_audio_duration_for_current_segment_seconds": 0.0,
        "actual_segment_duration_seconds": 0.0,
        "webm_global_header_bytes": None,
        "is_first_blob_received": False,
        "vad_enabled": False,  # Will be set to True if VAD session is created successfully
        "last_successful_transcript": "", # For providing rolling context to Whisper
        "actual_segment_duration_seconds": 0.0, # To track duration of processed segments
        "pending_segments": 0,
        "max_pending_segments": MAX_PENDING_SEGMENTS,
        "current_provider": get_current_transcription_provider(),
        "fallback_active": False,
        "client_timezone_name": client_timezone_name,
        "client_timezone_abbr": client_timezone_abbr,
        **reconnect_state  # Add reconnection fields
    }
    SESSION_ADAPTER.register_session(session_id, s3_transcript_key, client_timezone_name)
    logger.info(f"Transcript session {session_id} started for agent {agent_name}, event {event_id} by user {user.id}.")
    logger.info(f"Session temp audio dir: {temp_audio_base_dir}, S3 transcript key: {s3_transcript_key}")
    if SESSION_ADAPTER.silence_gate_enabled():
        gate_cfg = SESSION_ADAPTER.gate_config_for(vad_aggressiveness)
        margin_frames = SESSION_ADAPTER.trim_margin_frames()
        logger.info(
            f"Session {session_id}: silence gate armed (vad_aggressiveness={vad_aggressiveness}, "
            f"min_ratio={gate_cfg.min_speech_ratio:.3f}, rms_floor={gate_cfg.rms_floor:.1f}, "
            f"confirm={gate_cfg.confirm_silence_windows}, trim_margin_frames={margin_frames})"
        )
    else:
        logger.info(
            f"Session {session_id}: silence gate disabled (vad_aggressiveness={vad_aggressiveness})"
        )
    
    # Initialize VAD session if enabled
    if VAD_IMPORT_SUCCESS and vad_bridge: # Force VAD activation if available to fix hallucinations
        try:
            vad_success = vad_bridge.create_vad_session(
                session_id=session_id,
                existing_session_data=active_sessions[session_id],
                main_session_lock=session_locks[session_id]
            )
            if vad_success:
                active_sessions[session_id]["vad_enabled"] = True
                logger.info(f"VAD session {session_id} created successfully")
            else:
                logger.warning(f"Failed to create VAD session {session_id}, falling back to original transcription")
        except Exception as e:
            logger.error(f"Error creating VAD session {session_id}: {e}", exc_info=True)
            logger.warning(f"VAD session creation failed for {session_id}, falling back to original transcription")
    
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if s3 and aws_s3_bucket:
        timezone_line = client_timezone_name if client_timezone_name == client_timezone_abbr else f"{client_timezone_name} ({client_timezone_abbr})"
        header = (
            f"# Transcript - Session {session_id}\n"
            f"Agent: {agent_name}, Event: {event_id}\n"
            f"User: {user.id}\n"
            f"Language: {format_language_for_header(language_setting)}\n"
            f"Provider: {get_current_transcription_provider()}\n"
            f"Session Started (UTC): {session_start_time_utc.isoformat()}\n"
            f"Session Timezone: {timezone_line}\n"
            f"Session Started (Local): {session_start_time_local.isoformat()}\n\n"
        )
        try:
            s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=header.encode('utf-8'))
            logger.info(f"Initialized S3 transcript file: {s3_transcript_key}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 transcript file {s3_transcript_key}: {e}")

    return jsonify({
        "status": "success", 
        "message": "Recording session initiated", 
        "session_id": session_id,
        "session_start_time_utc": session_start_time_utc.isoformat(),
        "recording_status": _get_current_recording_status_snapshot(session_id)
    })

@app.route('/api/audio-recording/start', methods=['POST'])
@supabase_auth_required(agent_required=True)
def start_audio_recording(user: SupabaseUser):
    # Mutual Exclusivity Check
    for session in active_sessions.values():
        if session.get('session_type') == 'transcript' and session.get('is_active'):
            logger.warning("Attempted to start a recording session while a transcript session is active.")
            return jsonify({"status": "error", "message": "A transcript session is already in progress. Cannot start a recording session."}), 409

    data = g.get('json_data', {})
    agent_name = data.get('agent')
    language_setting = data.get('transcriptionLanguage', 'any')
    vad_aggressiveness = data.get('vadAggressiveness') or get_default_vad_aggressiveness()  # VAD aggressiveness level (1, 2, 3)

    logger.info(f"Start audio recording: agent='{agent_name}', language='{language_setting}', vad_aggressiveness='{vad_aggressiveness}'")

    session_id = uuid.uuid4().hex
    session_start_time_utc = datetime.now(timezone.utc)
    client_timezone_raw = data.get('clientTimezone')
    client_timezone_name, client_timezone_abbr, session_start_time_local = _resolve_client_timezone(
        client_timezone_raw, session_start_time_utc
    )
    timezone_source = "client" if client_timezone_raw else "default"
    logger.info(
        f"Recording session {session_id}: timezone set to {client_timezone_name} ({client_timezone_abbr}), source={timezone_source}"
    )
    
    agent_openai_key = get_api_key(agent_name, 'openai')
    agent_anthropic_key = get_api_key(agent_name, 'anthropic')
    
    s3_recording_filename = f"recording_D{session_start_time_utc.strftime('%Y%m%d')}-T{session_start_time_utc.strftime('%H%M%S')}_uID-{user.id}_oID-river_aID-{agent_name}_sID-{session_id}.txt"
    s3_recording_key = f"organizations/river/agents/{agent_name}/recordings/{s3_recording_filename}"
    
    temp_audio_base_dir = os.path.join('tmp', 'audio_sessions', session_id)
    os.makedirs(temp_audio_base_dir, exist_ok=True)
    
    active_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user.id,
        "agent_name": agent_name,
        "session_type": "recording", # Differentiate session type
        "language_setting_from_client": language_setting,
        "vad_aggressiveness_from_client": vad_aggressiveness, # Store VAD aggressiveness level
        "session_start_time_utc": session_start_time_utc,
        "session_start_time_local": session_start_time_local,
        "s3_transcript_key": s3_recording_key, # Re-use the same key name for compatibility
        "temp_audio_session_dir": temp_audio_base_dir,
        "openai_api_key": agent_openai_key,
        "anthropic_api_key": agent_anthropic_key,
        "is_backend_processing_paused": False,
        "current_total_audio_duration_processed_seconds": 0.0,
        "websocket_connection": None,
        "last_activity_timestamp": time.time(),
        "is_active": True,
        "ws_disconnected": False,
        "is_finalizing": False,
        "current_segment_raw_bytes": bytearray(),
        "accumulated_audio_duration_for_current_segment_seconds": 0.0,
        "actual_segment_duration_seconds": 0.0,
        "webm_global_header_bytes": None,
        "is_first_blob_received": False,
        "vad_enabled": False,
        "last_successful_transcript": "",
        "pending_segments": 0,
        "max_pending_segments": MAX_PENDING_SEGMENTS,
        "current_provider": get_current_transcription_provider(),
        "fallback_active": False,
        "silence_seconds": 0,
        "last_audio_ts": None,
        "recent_events": deque(maxlen=200),
        "client_timezone_name": client_timezone_name,
        "client_timezone_abbr": client_timezone_abbr,
    }
    SESSION_ADAPTER.register_session(session_id, s3_recording_key, client_timezone_name)
    logger.info(f"Audio recording session {session_id} started for agent {agent_name} by user {user.id}.")
    
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if s3 and aws_s3_bucket:
        timezone_line = client_timezone_name if client_timezone_name == client_timezone_abbr else f"{client_timezone_name} ({client_timezone_abbr})"
        header = (
            f"# Recording - Session {session_id}\n"
            f"Agent: {agent_name}\n"
            f"User: {user.id}\n"
            f"Session Started (UTC): {session_start_time_utc.isoformat()}\n"
            f"Session Timezone: {timezone_line}\n"
            f"Session Started (Local): {session_start_time_local.isoformat()}\n\n"
        )
        try:
            s3.put_object(Bucket=aws_s3_bucket, Key=s3_recording_key, Body=header.encode('utf-8'))
            logger.info(f"Initialized S3 recording file: {s3_recording_key}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 recording file {s3_recording_key}: {e}")

    return jsonify({
        "status": "success",
        "message": "Audio recording session initiated",
        "session_id": session_id,
        "session_start_time_utc": session_start_time_utc.isoformat(),
        "recording_status": _get_current_recording_status_snapshot(session_id)
    })

def _finalize_session(session_id: str):
    logger.info(f"Attempting to finalize session {session_id}...")
    with session_locks[session_id]: 
        if session_id not in active_sessions:
            logger.warning(f"Finalize: Session {session_id} not found or already cleaned up (checked after lock). Aborting.")
            return

        session_data = active_sessions[session_id]
        
        if session_data.get("is_finalizing", False):
            logger.warning(f"Finalize: Session {session_id} is already being finalized. Aborting redundant call.")
            return
            
        session_data["is_finalizing"] = True 
        session_data["is_active"] = False 
        session_data["finalization_timestamp"] = time.time() # Mark when finalization started
        logger.info(f"Finalizing session {session_id} (marked is_finalizing=True, is_active=False).")
        
        current_fragment_bytes_final = bytes(session_data.get("current_segment_raw_bytes", bytearray()))
        global_header_bytes_final, finalize_header_mode = _resolve_webm_header(session_id, session_data)

        all_final_segment_bytes = b''
        if global_header_bytes_final and current_fragment_bytes_final:
            if current_fragment_bytes_final.startswith(global_header_bytes_final): 
                all_final_segment_bytes = current_fragment_bytes_final
            else:
                all_final_segment_bytes = global_header_bytes_final + current_fragment_bytes_final
                logger.debug(
                    f"Session {session_id}: Prepended header ({len(global_header_bytes_final)} bytes, mode={finalize_header_mode}) during finalize."
                )
        elif current_fragment_bytes_final: 
             all_final_segment_bytes = current_fragment_bytes_final

        if all_final_segment_bytes:
            logger.info(f"Processing {len(all_final_segment_bytes)} remaining combined audio bytes for session {session_id} during finalization.")
            
            session_data["current_segment_raw_bytes"] = bytearray()
            session_data["actual_segment_duration_seconds"] = 0.0

            temp_processing_dir = os.path.join(session_data['temp_audio_session_dir'], "segments_processing")
            os.makedirs(temp_processing_dir, exist_ok=True)
            
            final_segment_uuid = f"final_{uuid.uuid4().hex}"
            final_output_wav_path = os.path.join(temp_processing_dir, f"final_audio_{final_segment_uuid}.wav")
            
            try:
                ffmpeg_command = [
                    'ffmpeg',
                    '-loglevel', 'warning',
                    '-fflags', '+discardcorrupt',
                    '-err_detect', 'ignore_err',
                    '-y',
                    '-i', 'pipe:0',
                    '-ar', '16000',
                    '-ac', '1',
                    '-acodec', 'pcm_s16le',
                    final_output_wav_path
                ]
                logger.info(f"Session {session_id} Finalize: Executing ffmpeg direct WAV: {' '.join(ffmpeg_command)}")
                process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate(input=all_final_segment_bytes)

                if process.returncode == 0:
                    logger.info(f"Session {session_id} Finalize: Successfully converted final piped stream to WAV: {final_output_wav_path}")
                    
                    ffprobe_command_final = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', final_output_wav_path]
                    try:
                        duration_result_final = subprocess.run(ffprobe_command_final, capture_output=True, text=True, check=True)
                        actual_duration_final = float(duration_result_final.stdout.strip())
                        session_data['actual_segment_duration_seconds'] = actual_duration_final
                        logger.info(f"Session {session_id} Finalize: Actual duration of final WAV segment {final_output_wav_path} is {actual_duration_final:.2f}s")
                    except (subprocess.CalledProcessError, ValueError) as ffprobe_err_final:
                        logger.error(f"Session {session_id} Finalize: ffprobe failed for final WAV {final_output_wav_path}: {ffprobe_err_final}. Estimating duration.")
                        estimated_duration_final = len(all_final_segment_bytes) / (16000 * 2 * 0.2) 
                        session_data['actual_segment_duration_seconds'] = estimated_duration_final
                        logger.warning(f"Using rough estimated duration for final segment: {estimated_duration_final:.2f}s")

                    SESSION_ADAPTER.on_segment(
                        session_id=session_id,
                        raw_path=final_output_wav_path,
                        captured_ts=time.time(),
                        duration_s=session_data.get("actual_segment_duration_seconds", actual_duration_final),
                        language=session_data.get("language_setting_from_client"),
                        vad_aggressiveness=session_data.get("vad_aggressiveness_from_client")
                    )
                else:
                    logger.error(f"Session {session_id} Finalize: ffmpeg direct WAV conversion failed. RC: {process.returncode}, Err: {stderr.decode('utf-8', 'ignore')}")
            except Exception as e:
                logger.error(f"Error processing final audio segment (piped) for session {session_id}: {e}", exc_info=True)
            finally:
                pass
        else: 
            logger.info(f"Session {session_id} Finalize: No remaining audio bytes to process.")
        
        # Clean up VAD session if it exists
        if VAD_IMPORT_SUCCESS and session_data.get("vad_enabled") and vad_bridge:
            try:
                logger.info(f"Session {session_id} Finalize: Destroying VAD session")
                vad_bridge.destroy_vad_session(session_id)
                logger.info(f"Session {session_id} Finalize: VAD session destroyed successfully")
            except Exception as e:
                logger.warning(f"Session {session_id} Finalize: Error destroying VAD session: {e}", exc_info=True)
        
        ws = session_data.get("websocket_connection")
        if ws:
            logger.info(f"Attempting to close WebSocket for session {session_id} during finalization.")
            try:
                ws.close(1000, "Session stopped by server (finalize)")
                logger.info(f"WebSocket for session {session_id} close() called.")
            except ConnectionClosed as e:
                logger.info(
                    f"WebSocket for session {session_id} already closed before finalize close(): {e}"
                )
            except Exception as e: 
                logger.warning(f"Error closing WebSocket for session {session_id} during finalization: {e}", exc_info=True)
            finally:
                session_data["websocket_connection"] = None

        SESSION_ADAPTER.on_finalize(session_id=session_id)

        logger.info(f"Session {session_id} has been marked as finalized. It will be cleaned up by the idle thread later.")

@app.route('/api/s3/summarize-transcript', methods=['POST'])
@supabase_auth_required(agent_required=False) # agentName will be in payload
def summarize_transcript_route(user: SupabaseUser):
    logger.info(f"Received request /api/s3/summarize-transcript from user: {user.id}")
    data = g.get('json_data', {})
    s3_key_original_transcript = data.get('s3Key')
    agent_name = data.get('agentName')
    event_id = data.get('eventId')
    original_filename = data.get('originalFilename') # Expecting this from frontend

    if not all([s3_key_original_transcript, agent_name, event_id, original_filename]):
        missing_params = [
            k for k, v in {
                "s3Key": s3_key_original_transcript, "agentName": agent_name,
                "eventId": event_id, "originalFilename": original_filename
            }.items() if not v
        ]
        logger.error(f"SummarizeTranscript: Missing parameters: {', '.join(missing_params)}")
        return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("SummarizeTranscript: S3 client or bucket not configured.")
        return jsonify({"error": "S3 service not available"}), 503
    
    agent_anthropic_key = get_api_key(agent_name, 'anthropic')
    if not agent_anthropic_key:
        logger.error(f"SummarizeTranscript: Could not retrieve Anthropic API key for agent '{agent_name}'.")
        return jsonify({"error": "LLM service not available (key retrieval failed)"}), 503

    try:
        # Initialize a transient client for this request
        summary_llm_client = Anthropic(api_key=agent_anthropic_key)
    except Exception as e:
        logger.error(f"SummarizeTranscript: Failed to initialize Anthropic client: {e}")
        return jsonify({"error": "LLM service initialization failed"}), 503

    try:
        # 1. Read original transcript content
        from utils.s3_utils import read_file_content as s3_read_content # Local import for clarity
        transcript_content = s3_read_content(s3_key_original_transcript, f"transcript for summarization ({original_filename})")
        if transcript_content is None:
            logger.error(f"SummarizeTranscript: Failed to read original transcript from {s3_key_original_transcript}")
            return jsonify({"error": "Could not read original transcript from S3"}), 404

        # 2. Generate summary
        summary_data = generate_transcript_summary(
            transcript_content=transcript_content,
            original_filename=original_filename,
            agent_name=agent_name,
            event_id=event_id,
            source_s3_key=s3_key_original_transcript, # Pass the original S3 key
            llm_client=summary_llm_client
            # model_name and max_tokens will use defaults in generate_transcript_summary or be overridden by env vars there
        )

        if summary_data is None:
            logger.error(f"SummarizeTranscript: Failed to generate summary for {original_filename}")
            return jsonify({"error": "Failed to generate transcript summary using LLM"}), 500

        # 3. Save summary JSON to S3
        # New filename convention: summary_{original_transcript_main_part}.json
        # Example: if original is "transcript_D20250519-T075616_...sID-xyz.txt"
        # summary will be "summary_D20250519-T075616_...sID-xyz.json"
        
        original_base, _ = os.path.splitext(original_filename) # e.g., "transcript_D2025...sID-xyz"
        if original_base.startswith("transcript_"):
            summary_base_name = original_base[len("transcript_"):] # e.g., "D2025...sID-xyz"
        else:
            summary_base_name = original_base # Fallback if "transcript_" prefix is missing
            
        summary_filename = f"summary_{summary_base_name}.json"
        
        # Corrected path: include /transcripts/ before /summarized/
        summary_s3_key = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/summarized/{summary_filename}"
        
        try:
            summary_json_string = json.dumps(summary_data, indent=2, ensure_ascii=False)
            logger.info(f"SummarizeTranscript: Attempting to save summary. Bucket: '{aws_s3_bucket}', Key: '{summary_s3_key}', ContentType: 'application/json; charset=utf-8', JSON_String_Length: {len(summary_json_string)}")
            put_response = s3.put_object(
                Bucket=aws_s3_bucket,
                Key=summary_s3_key,
                Body=summary_json_string.encode('utf-8'),
                ContentType='application/json; charset=utf-8'
            )
            logger.info(f"SummarizeTranscript: S3 put_object response: {put_response}")
            
            # Check the response metadata for success
            if put_response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
                logger.info(f"SummarizeTranscript: Successfully saved summary to {summary_s3_key} (HTTP 200 OK).")

                # Invalidate the cache for this agent/event's summaries
                summaries_cache_key = f"transcript_summaries_{agent_name}_{event_id}"
                with S3_CACHE_LOCK:
                    if summaries_cache_key in S3_FILE_CACHE:
                        del S3_FILE_CACHE[summaries_cache_key]
                        logger.info(f"CACHE INVALIDATED for transcript summaries: '{summaries_cache_key}'")
                
                # Step 3: Move original transcript to a /saved subfolder
                try:
                    original_transcript_dir = os.path.dirname(s3_key_original_transcript)
                    original_transcript_basename = os.path.basename(s3_key_original_transcript)
                    saved_transcript_key = f"{original_transcript_dir}/saved/{original_transcript_basename}"
                    
                    logger.info(f"SummarizeTranscript: Moving original transcript from '{s3_key_original_transcript}' to '{saved_transcript_key}'.")
                    
                    copy_source = {'Bucket': aws_s3_bucket, 'Key': s3_key_original_transcript}
                    s3.copy_object(CopySource=copy_source, Bucket=aws_s3_bucket, Key=saved_transcript_key)
                    logger.info(f"SummarizeTranscript: Successfully copied original to '{saved_transcript_key}'.")

                    s3.delete_object(Bucket=aws_s3_bucket, Key=s3_key_original_transcript)
                    logger.info(f"SummarizeTranscript: Successfully deleted original transcript at '{s3_key_original_transcript}'.")

                except Exception as e_move:
                    logger.error(f"SummarizeTranscript: Error moving original transcript {s3_key_original_transcript} to saved location: {e_move}", exc_info=True)
                    # If moving fails, the summary is still created. We might want to indicate a partial success
                    # or attempt to clean up the summary if the original can't be moved.
                    # For now, return success for the summary part
                    pass # Continue to return success for the summary part

            else:
                logger.error(f"SummarizeTranscript: S3 put_object for {summary_s3_key} did NOT return HTTP 200. Full response: {put_response}")
                # This case might not be hit if an exception is raised first for non-200, but good for robustness.
                return jsonify({"error": "Failed to save summary to S3 - S3 returned non-200 status."}), 500
                
        except Exception as e_put:
            logger.error(f"SummarizeTranscript: Exception during S3 put_object for {summary_s3_key}: {e_put}", exc_info=True)
            return jsonify({"error": f"Failed to save summary to S3: {str(e_put)}"}), 500

        return jsonify({
            "message": "Transcript summarized and saved successfully. Original transcript has been moved.",
            "original_transcript_s3_key": s3_key_original_transcript, # This key now points to a non-existent object if move was successful
            "moved_to_s3_key": saved_transcript_key if 'saved_transcript_key' in locals() else None,
            "summary_s3_key": summary_s3_key,
            "summary_filename": summary_filename
        }), 200

    except Exception as e:
        logger.error(f"SummarizeTranscript: Unexpected error for {original_filename}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during summarization."}), 500


@app.route('/api/recording/stop', methods=['POST'])
@supabase_auth_required(agent_required=False) 
def stop_recording_route(user: SupabaseUser): 
    data = g.get('json_data', {})
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400

    logger.info(f"Stop recording request for session {session_id} by user {user.id}") 
    
    if session_id not in active_sessions:
        logger.warning(f"Stop request for session {session_id}, but session not found in active_sessions. It might have already been finalized or never started.")
        return jsonify({"status": "success", "message": "Session already stopped or not found", "session_id": session_id}), 200 
    
    session_data = active_sessions.get(session_id)
    if session_data and session_data.get('session_type') != 'transcript':
        logger.warning(f"Stop recording request for session {session_id} which is not a transcript session.")
        # Decide if we should still stop it or return an error. For now, let's stop it.
    
    _finalize_session(session_id)
    
    return jsonify({
        "status": "success", 
        "message": "Recording session stopped", 
        "session_id": session_id,
        "recording_status": {"is_recording": False, "is_backend_processing_paused": False} 
    })

from utils.multi_agent_summarizer.pipeline import run_pipeline_steps as _run_pipeline_steps

# New: Multi-Agent Transcript Save Endpoint
@app.post("/api/transcripts/save")
@supabase_auth_required(agent_required=False)
def save_transcript(user: SupabaseUser):
    data = g.get('json_data', {}) or {}
    agent = data.get("agentName")
    event = data.get("eventId") or "0000"
    s3_key = data.get("s3Key")
    dry = bool(data.get("dryRun"))
    return_steps = bool(data.get("returnSteps"))
    if not agent or not s3_key:
        return jsonify({"error": "agentName and s3Key required"}), 400

    logger.info(f"tx.save.start {{agent:{agent}, event:{event}, s3Key:{s3_key}}}")
    text = _read_transcript_text_for_ma(s3_key)
    steps = _run_pipeline_steps(text)
    full = steps.get("full", {})

    if not dry:
        md = steps.get("full_md") or "# No Summary Generated\n"
        EmbeddingHandler(index_name="river", namespace=f"{agent}").embed_and_upsert(
            content=md,
            metadata={
                "agent_name": agent,
                "event_id": event,
                "transcript": event,
                "source": "transcript_summary",
                "source_type": "transcript",
                "source_identifier": s3_key,
                "file_name": "transcript_summary.md",
                "doc_id": f"{s3_key}:summary",
            },
        )

    payload = {"ok": True, "upserted": (not dry)}
    if return_steps:
        # Include compact intermediates for debugging
        payload["steps"] = {
            "segments": steps.get("segments"),
            "story_md": steps.get("story_md"),
            "mirror_md": steps.get("mirror_md"),
            "lens_md": steps.get("lens_md"),
            "portal_md": steps.get("portal_md"),
            "layer3_md": steps.get("layer3_md"),
            "layer4_md": steps.get("layer4_md"),
            "full_md": steps.get("full_md"),
            "full": full,
        }
    return jsonify(payload), 200

@app.route('/api/audio-recording/stop', methods=['POST'])
@supabase_auth_required(agent_required=False)
def stop_audio_recording_route(user: SupabaseUser):
    data = g.get('json_data', {})
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400

    logger.info(f"Stop audio recording request for session {session_id} by user {user.id}")

    with session_locks[session_id]:
        if session_id not in active_sessions:
            logger.warning(f"Stop request for audio recording session {session_id}, but session not found (already cleaned up).")
            return jsonify({"status": "success", "message": "Session already stopped or not found", "session_id": session_id}), 200

        session_data = active_sessions[session_id]
        
        if session_data.get("is_finalizing"):
            logger.info(f"Stop request for audio recording session {session_id}, which is already finalizing. Acknowledging.")
            return jsonify({
                "status": "success",
                "message": "Audio recording session is already stopping.",
                "session_id": session_id,
                "s3Key": session_data.get('s3_transcript_key'),
            }), 200

        if session_data.get('session_type') != 'recording':
            logger.warning(f"Stop audio recording request for session {session_id} which is not a recording session.")
            # Let's stop it anyway to be safe.
        
        s3_key = session_data.get('s3_transcript_key') # Get the key before finalizing
        
        _finalize_session(session_id)
        
        return jsonify({
            "status": "success",
            "message": "Audio recording session stopped",
            "session_id": session_id,
            "s3Key": s3_key, # Return the S3 key of the finished recording
            "recording_status": {"is_recording": False, "is_backend_processing_paused": False}
        })

@app.route('/api/audio-recording/pause', methods=['POST'])
@supabase_auth_required(agent_required=False)
def pause_audio_recording_route(user: SupabaseUser):
    data = g.get('json_data', {})
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400

    logger.info(f"Pause audio recording request for session {session_id} by user {user.id}")

    with session_locks[session_id]:
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            if session_data.get('session_type') == 'recording':
                session_data['is_backend_processing_paused'] = True
                logger.info(f"Audio recording session {session_id} paused.")
                return jsonify({"status": "success", "message": "Audio recording paused."}), 200
            else:
                return jsonify({"status": "error", "message": "Session is not an audio recording session."}), 400
        else:
            return jsonify({"status": "error", "message": "Session not found."}), 404

@app.route('/api/audio-recording/resume', methods=['POST'])
@supabase_auth_required(agent_required=False)
def resume_audio_recording_route(user: SupabaseUser):
    data = g.get('json_data', {})
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400

    logger.info(f"Resume audio recording request for session {session_id} by user {user.id}")

    with session_locks[session_id]:
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            if session_data.get('session_type') == 'recording':
                session_data['is_backend_processing_paused'] = False
                logger.info(f"Audio recording session {session_id} resumed.")
                return jsonify({"status": "success", "message": "Audio recording resumed."}), 200
            else:
                return jsonify({"status": "error", "message": "Session is not an audio recording session."}), 400
        else:
            return jsonify({"status": "error", "message": "Session not found."}), 404

@sock.route('/api/recording/attach')
def attach_ws(ws):
    """WebSocket attach endpoint with reconnection support."""
    session_id = request.args.get("session_id")
    rt = request.args.get("rt")  # reconnect_token
    client_from_seq = int(request.args.get("client_from_seq") or 0)
    token = request.args.get('token')

    # Authenticate user
    user = verify_user(token)
    if not user:
        logger.warning(f"WebSocket attach failed: Invalid token for session {session_id}. Closing.")
        ws.close(code=1008, reason='Authentication failed')
        return

    if not session_id or session_id not in active_sessions:
        logger.warning(f"WebSocket attach failed: Session {session_id} not found. Closing.")
        ws.close(code=1011, reason='Session not found')
        return

    with session_locks[session_id]:
        sess = active_sessions[session_id]

        # Verify user owns session
        if sess.get("user_id") != user.id:
            logger.warning(f"WebSocket attach failed: Access denied for session {session_id}. Closing.")
            ws.close(code=1008, reason='Access denied')
            return

        # Verify reconnect token
        if sess.get("reconnect_token") != rt:
            logger.warning(f"WebSocket attach failed: Invalid reconnect token for session {session_id}. Closing.")
            ws.close(code=1008, reason='Invalid reconnect token')
            return

        now = time.time()
        grace_deadline = sess.get("grace_deadline")

        # Check if grace period has expired
        if grace_deadline and now > grace_deadline:
            # Grace expired: reset session with fresh token
            logger.info(f"WebSocket attach: Grace period expired for session {session_id}. Resetting session.")
            reconnect_state = _init_session_reconnect_state(session_id)
            sess.update(reconnect_state)

        # Attach WebSocket to session
        sess["websocket_connection"] = ws
        sess["last_disconnect_ts"] = None
        sess["grace_deadline"] = None
        sess["last_activity_timestamp"] = time.time()
        sess["is_active"] = True

        # Update session state for status tracking
        session_state = sess.get("session_state")
        if session_state:
            session_state.mark_ws_reconnected()
            session_state.last_status = {"type": "status", "state": "RESUMED"}

        # Replay missed results
        recent_results = sess.get("recent_results", [])
        missed = [r for r in recent_results if r.get("seq", 0) > client_from_seq]

        logger.info(f"WebSocket attached for session {session_id}, replaying {len(missed)} missed messages")

    # Send missed results to client
    for result in missed:
        try:
            ws.send(json.dumps({
                "type": "transcript",
                "seq": result.get("seq", 0),
                "text": result.get("text", ""),
                "ts": result.get("ts", "")
            }))
        except Exception as e:
            logger.warning(f"Failed to send missed result to session {session_id}: {e}")
            break

    # Send RESUMED status
    try:
        ws.send(json.dumps({"type": "status", "state": "RESUMED"}))
    except Exception as e:
        logger.warning(f"Failed to send RESUMED status to session {session_id}: {e}")

    # Basic message handling loop for pings/pongs
    try:
        while True:
            message = ws.receive(timeout=30)
            if message is None:
                continue

            # Handle control messages
            if isinstance(message, str):
                try:
                    control_msg = json.loads(message)
                    msg_type = control_msg.get("type")
                    if msg_type == "ping":
                        ws.send(json.dumps({"type": "pong"}))
                    elif msg_type == "pong":
                        pass  # Acknowledge pong
                    elif msg_type == "ack":
                        # Client acknowledging receipt of sequence number
                        with session_locks[session_id]:
                            sess = active_sessions.get(session_id)
                            if sess:
                                sess["last_client_ack"] = control_msg.get("seq", 0)
                except json.JSONDecodeError:
                    logger.warning(f"WebSocket attach session {session_id}: Invalid JSON message")
    except Exception as e:
        logger.info(f"WebSocket attach for session {session_id} closed: {e}")
    finally:
        # Mark as disconnected but keep grace period
        _mark_ws_disconnected(session_id)

def _mark_ws_disconnected(session_id: str):
    """Mark WebSocket as disconnected and start grace period."""
    with session_locks[session_id]:
        sess = active_sessions.get(session_id)
        if not sess:
            return

        sess["websocket_connection"] = None
        sess["last_disconnect_ts"] = time.time()
        sess["grace_deadline"] = sess["last_disconnect_ts"] + REATTACH_GRACE_SECONDS
        sess["ws_disconnected"] = True

        # Update session state
        session_state = sess.get("session_state")
        if session_state:
            session_state.mark_ws_disconnected(REATTACH_GRACE_SECONDS)
            session_state.last_status = {"type": "status", "state": "PAUSED", "reason": "network"}

        logger.info(f"WebSocket disconnected for session {session_id}. Grace period until {sess['grace_deadline']}")

def try_deliver_ordered_results(
    session_id: str,
    seq: int,
    transcript_text: str,
    timestamp_str: str,
    session_lock: threading.Lock = None,
    metadata: Optional[dict] = None,
) -> None:
    """Deliver transcript results in perfect order, handling gaps gracefully."""
    if not SEGMENT_RETRY_ENABLED:
        # Fallback to direct delivery if gap recovery disabled
        _deliver_to_client_and_s3(session_id, seq, transcript_text, timestamp_str, metadata)
        return

    lock_to_use = session_lock or session_locks.get(session_id, threading.Lock())

    with lock_to_use:
        sess = active_sessions.get(session_id)
        if not sess:
            logger.warning(f"Session {session_id} not found when trying to deliver seq={seq}")
            raise RuntimeError(f"Session {session_id} not found, triggering fallback delivery")

        expected_seq = sess.get("expected_seq", 1)
        pending_seqs = sess.get("pending_seqs", {})
        max_delivered_seq = sess.get("max_delivered_seq", 0)

        if seq == expected_seq:
            # Perfect order - deliver immediately
            _deliver_to_client_and_s3(session_id, seq, transcript_text, timestamp_str, metadata)
            sess["expected_seq"] = seq + 1
            sess["max_delivered_seq"] = seq

            # Deliver any consecutive pending sequences
            while sess["expected_seq"] in pending_seqs:
                next_seq = sess["expected_seq"]
                pending_data = pending_seqs.pop(next_seq)
                _deliver_to_client_and_s3(session_id, next_seq,
                                        pending_data["text"],
                                        pending_data["timestamp_str"],
                                        pending_data.get("metadata"))
                sess["expected_seq"] = next_seq + 1
                sess["max_delivered_seq"] = next_seq

            logger.debug(f"Session {session_id}: Delivered seq={seq} and {len(pending_seqs)} pending seqs")

        elif seq > expected_seq:
            # Out of order - store for later
            pending_seqs[seq] = {
                "text": transcript_text,
                "timestamp_str": timestamp_str,
                "metadata": metadata or {},
                "received_at": time.time()
            }
            sess["pending_seqs"] = pending_seqs

            # Prevent unbounded pending queue
            if len(pending_seqs) > MAX_PENDING_SEGMENTS:
                # Deliver oldest pending with gap marker
                oldest_seq = min(pending_seqs.keys())
                oldest_data = pending_seqs.pop(oldest_seq)

                # Insert gap marker in S3
                gap_marker = f"[Gap: Missing segments {expected_seq}-{oldest_seq-1}]"
                _deliver_to_client_and_s3(session_id, oldest_seq,
                                        gap_marker + " " + oldest_data["text"],
                                        oldest_data["timestamp_str"],
                                        oldest_data.get("metadata"))

                sess["expected_seq"] = oldest_seq + 1
                sess["max_delivered_seq"] = oldest_seq
                logger.warning(f"Session {session_id}: Forced delivery of seq={oldest_seq} due to queue limit")

            logger.debug(f"Session {session_id}: Queued seq={seq} for later (expecting seq={expected_seq})")

        else:
            # Sequence already delivered or too old
            logger.debug(f"Session {session_id}: Ignoring duplicate/old seq={seq} (max_delivered={max_delivered_seq})")

        # Check for timed-out gaps
        current_time = time.time()
        expired_seqs = []
        for pending_seq, pending_data in pending_seqs.items():
            if current_time - pending_data["received_at"] > SEGMENT_GAP_TIMEOUT_SEC:
                expired_seqs.append(pending_seq)

        # Handle expired sequences
        for expired_seq in expired_seqs:
            expired_data = pending_seqs.pop(expired_seq)
            gap_marker = f"[Delayed segment seq={expired_seq}]"
            _deliver_to_client_and_s3(session_id, expired_seq,
                                    gap_marker + " " + expired_data["text"],
                                    expired_data["timestamp_str"],
                                    expired_data.get("metadata"))
            logger.warning(f"Session {session_id}: Delivered expired seq={expired_seq} after timeout")

def mark_sequence_dropped(session_id: str, seq: int) -> None:
    """Mark a sequence as dropped/skipped to advance the ordering pointer."""
    if not SEGMENT_RETRY_ENABLED:
        return  # No ordering system to notify

    lock_to_use = session_locks.get(session_id, threading.Lock())

    with lock_to_use:
        sess = active_sessions.get(session_id)
        if not sess:
            logger.warning(f"Session {session_id} not found when marking seq={seq} as dropped")
            return

        # Initialize ordering state if not exists
        if "expected_seq" not in sess:
            sess["expected_seq"] = 1
        if "pending_ordered_seqs" not in sess:
            sess["pending_ordered_seqs"] = {}

        expected_seq = sess["expected_seq"]
        pending_seqs = sess["pending_ordered_seqs"]

        if seq == expected_seq:
            # This was the next expected sequence - advance the pointer
            sess["expected_seq"] = seq + 1
            logger.debug(f"Session {session_id}: Advanced expected_seq to {seq + 1} (dropped seq={seq})")

            # Deliver any consecutive pending sequences that are now ready
            while sess["expected_seq"] in pending_seqs:
                next_seq = sess["expected_seq"]
                pending_data = pending_seqs.pop(next_seq)
                _deliver_to_client_and_s3(session_id, next_seq,
                                        pending_data["text"],
                                        pending_data["timestamp_str"],
                                        pending_data.get("metadata"))
                sess["expected_seq"] = next_seq + 1
                sess["max_delivered_seq"] = next_seq
                logger.debug(f"Session {session_id}: Released pending seq={next_seq} after drop advancement")

        elif seq in pending_seqs:
            # This sequence was queued but now dropped - remove it from pending
            pending_seqs.pop(seq, None)
            logger.debug(f"Session {session_id}: Removed dropped seq={seq} from pending queue")

        # If seq < expected_seq, it was already processed or dropped - no action needed


def pop_ready_ordered_segments(session_id: str) -> List[Tuple[int, str]]:
    """Pop all contiguous, ready-to-flush segments from the ordering buffer."""
    if not SEGMENT_RETRY_ENABLED:
        return []

    lock_to_use = session_locks.get(session_id, threading.Lock())
    ready_items = []

    with lock_to_use:
        sess = active_sessions.get(session_id)
        if not sess:
            return []

        expected_seq = sess.get("expected_seq", 1)
        pending_seqs = sess.get("pending_ordered_seqs", {})

        # Find all contiguous sequences starting from expected_seq
        current_seq = expected_seq
        while current_seq in pending_seqs:
            pending_data = pending_seqs.pop(current_seq)
            ready_items.append((current_seq, pending_data["text"]))
            sess["expected_seq"] = current_seq + 1
            sess["max_delivered_seq"] = current_seq
            current_seq += 1

        if ready_items:
            logger.debug(f"Session {session_id}: Popped {len(ready_items)} ready segments (seqs {ready_items[0][0]}-{ready_items[-1][0]})")

    return ready_items


def flush_all_ordered_segments(session_id: str) -> List[Tuple[int, str]]:
    """Flush all remaining buffered segments for a session (used during finalization)."""
    if not SEGMENT_RETRY_ENABLED:
        return []  # No ordering system to flush

    lock_to_use = session_locks.get(session_id, threading.Lock())
    flushed_items = []

    with lock_to_use:
        sess = active_sessions.get(session_id)
        if not sess:
            logger.warning(f"Session {session_id} not found when flushing ordered segments")
            return []

        pending_seqs = sess.get("pending_ordered_seqs", {})
        if not pending_seqs:
            logger.debug(f"Session {session_id}: No pending sequences to flush")
            return []

        logger.info(f"Session {session_id}: Flushing {len(pending_seqs)} remaining ordered segments")

        # Sort pending sequences and prepare them for S3 writing
        for seq in sorted(pending_seqs.keys()):
            pending_data = pending_seqs[seq]
            gap_marker = f"[Final flush seq={seq}] "
            full_text = gap_marker + pending_data["text"]
            flushed_items.append((seq, full_text))
            logger.debug(f"Session {session_id}: Prepared final flush seq={seq}")

        # Clear the pending queue
        sess["pending_ordered_seqs"] = {}
        logger.info(f"Session {session_id}: All pending segments prepared for flush, queue cleared")

    return flushed_items


def _deliver_to_client_and_s3(session_id: str, seq: int, transcript_text: str, timestamp_str: str, metadata: Optional[dict] = None) -> None:
    """Deliver transcript to both client and S3."""
    try:
        sess = active_sessions.get(session_id)
        if not sess:
            return

        # Append to S3 (reuse existing S3 logic)
        s3_transcript_key = sess.get('s3_transcript_key')
        metadata = metadata or {}

        if s3_transcript_key:
            from utils.s3_utils import get_s3_client
            s3 = get_s3_client()
            aws_s3_bucket = os.getenv('AWS_S3_BUCKET')

            if s3 and aws_s3_bucket:
                try:
                    # Read existing content
                    try:
                        obj = s3.get_object(Bucket=aws_s3_bucket, Key=s3_transcript_key)
                        existing_content = obj['Body'].read().decode('utf-8')
                    except s3.exceptions.NoSuchKey:
                        # Create header for new transcript
                        session_start_time_utc = sess.get('session_start_time_utc')
                        agent_name = sess.get('agent_name', 'N/A')
                        event_id = sess.get('event_id', 'N/A')
                        language_setting = sess.get('language_setting_from_client', 'any')

                        existing_content = f"# Transcript - Session {session_id}\n"
                        existing_content += f"Agent: {agent_name}, Event: {event_id}\n"
                        existing_content += f"Language: {language_setting}\n"
                        if session_start_time_utc:
                            existing_content += f"Session Started (UTC): {session_start_time_utc.isoformat()}\n"
                        existing_content += "\n"

                    # Append new line
                    formatted_line = format_transcript_line(timestamp_str, transcript_text, metadata)
                    new_line = f"{formatted_line}\n"
                    updated_content = existing_content + new_line

                    # Write back to S3
                    s3.put_object(Bucket=aws_s3_bucket, Key=s3_transcript_key, Body=updated_content.encode('utf-8'))
                    logger.debug(f"Session {session_id}: Appended seq={seq} to S3")

                except Exception as s3_e:
                    logger.error(f"Session {session_id}: Failed to append seq={seq} to S3: {s3_e}")

        # Send to WebSocket client (reuse existing ring buffer logic)
        recent_results = sess.get("recent_results")
        next_ws_seq = sess.get("next_seq", 1)

        if recent_results is not None:
            result_data = {
                "seq": next_ws_seq,
                "text": transcript_text,
                "ts": datetime.now(timezone.utc).isoformat(),
                "timestamp_str": timestamp_str,
                "segment_seq": seq  # Include segment sequence for debugging
            }
            result_data["low_confidence"] = bool(metadata.get("low_confidence"))
            result_data["used_fallback"] = bool(metadata.get("used_fallback"))
            result_data["drop_reason"] = metadata.get("drop_reason", "NONE")
            if metadata.get("fallback_reason"):
                result_data["fallback_reason"] = metadata["fallback_reason"]
            if metadata.get("stats"):
                result_data["postprocess_stats"] = metadata["stats"]
            if metadata.get("pii_pass") is not None:
                result_data["pii_pass"] = metadata["pii_pass"]
            recent_results.append(result_data)
            sess["next_seq"] = next_ws_seq + 1

            # Send to WebSocket if connected
            ws = sess.get("websocket_connection")
            if ws:
                try:
                    ws.send(json.dumps({
                        "type": "transcript",
                        "seq": next_ws_seq,
                        "text": transcript_text,
                        "ts": result_data["ts"],
                        "timestamp": timestamp_str,
                        "segment_seq": seq
                    }))
                    logger.debug(f"Session {session_id}: Sent seq={seq} to WebSocket")
                except Exception as ws_e:
                    logger.warning(f"Session {session_id}: Failed to send seq={seq} to WebSocket: {ws_e}")

    except Exception as e:
        logger.error(f"Session {session_id}: Error delivering seq={seq}: {e}")
    else:
        try:
            push_session_event(session_id, "segment_processed", {"seq": seq})
        except Exception as event_err:
            logger.debug(f"Session {session_id}: Unable to record segment_processed event: {event_err}")

def _heartbeat_loop():
    """Monitor WebSocket connections and handle grace period expiry."""
    while True:
        time.sleep(10)  # Check every 10 seconds
        current_time = time.time()
        sessions_to_finalize = []

        # Create a snapshot of sessions to avoid modification during iteration
        session_snapshots = list(active_sessions.items())

        for session_id, sess in session_snapshots:
            try:
                with session_locks.get(session_id, threading.Lock()):
                    # Skip if session was removed
                    if session_id not in active_sessions:
                        continue

                    sess = active_sessions[session_id]
                    ws = sess.get("websocket_connection")
                    grace_deadline = sess.get("grace_deadline")

                    if ws:
                        # WebSocket is connected, send ping
                        try:
                            ws.send(json.dumps({"type": "ping", "t": int(current_time)}))
                            logger.debug(f"Heartbeat ping sent to session {session_id}")
                        except Exception as e:
                            logger.warning(f"Heartbeat ping failed for session {session_id}: {e}")
                            # Mark as disconnected
                            _mark_ws_disconnected(session_id)
                    else:
                        # No WebSocket connection, check grace expiry
                        if grace_deadline and current_time > grace_deadline:
                            # Skip if already being finalized
                            if sess.get("is_finalizing", False):
                                logger.debug(f"Heartbeat: Session {session_id} is already being finalized, skipping")
                                continue
                            logger.info(f"Heartbeat: Grace period expired for session {session_id}")
                            sessions_to_finalize.append(session_id)

            except Exception as e:
                logger.error(f"Error in heartbeat loop for session {session_id}: {e}")

        # Finalize expired sessions
        for session_id in sessions_to_finalize:
            try:
                logger.info(f"Heartbeat: Finalizing expired session {session_id}")
                _finalize_session(session_id)
            except Exception as e:
                logger.error(f"Error finalizing expired session {session_id}: {e}")

def _retry_worker():
    """Retry queue disabled in append-only pipeline."""
    logger.info('Retry worker disabled; append-only pipeline does not resubmit segments.')
    return

# Start background threads
_heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
_heartbeat_thread.start()
logger.info("Heartbeat monitoring thread started")

if SEGMENT_RETRY_ENABLED:
    _retry_worker_thread = threading.Thread(target=_retry_worker, daemon=True)
    _retry_worker_thread.start()
    logger.info("Segment retry worker thread started")
else:
    logger.info("Segment retry worker disabled")

@sock.route('/ws/audio_stream/<session_id>')
def audio_stream_socket(ws, session_id: str):
    token = request.args.get('token')
    client_id = request.args.get('client_id') or "unknown"
    resume = (request.args.get('resume') == '1')
    user = verify_user(token) 
    
    if not user:
        logger.warning(f"WebSocket Auth Failed: Invalid token for session {session_id}. Closing.")
        # Close with RFC6455 code + reason; guard send errors
        try:
            ws.close(code=1008, reason='Authentication failed')  # Policy violation/unauthorized
        except Exception as _:
            pass
        return

    logger.info(f"WebSocket connection attempt for session {session_id}, user {user.id}, client_id={client_id}, resume={resume}")

    with session_locks[session_id]:
        if session_id not in active_sessions:
            logger.warning(f"WebSocket: Session {session_id} not found after acquiring lock. Closing.")
            ws.close(1011, "Session not found")
            return
        
        existing_ws = active_sessions[session_id].get("websocket_connection")
        existing_user_id = active_sessions[session_id].get("user_id")
        if existing_ws is not None:
            if resume and existing_user_id == user.id:
                # Replace ownership: close old socket and accept new
                try:
                    try:
                        existing_ws.send(json.dumps({"type": "info", "reason": "replaced", "by": client_id}))
                    except Exception:
                        pass
                    try:
                        existing_ws.close(1012, f"Replaced by client {client_id}")
                    except Exception:
                        pass
                finally:
                    active_sessions[session_id]["websocket_connection"] = ws
                    active_sessions[session_id]["last_activity_timestamp"] = time.time()
                    active_sessions[session_id]["is_active"] = True
                    active_sessions[session_id]["ws_disconnected"] = False
                    logger.info(f"WebSocket takeover: session {session_id} user {user.id} client {client_id} replaced previous connection.")
            else:
                logger.warning(f"WebSocket: Duplicate connection for session {session_id}. resume={resume}, same_user={existing_user_id == user.id}. Closing new one.")
                ws.close(1008, "Connection already exists")
                return
        else:
            # Check if this is a reattachment scenario
            session_state = active_sessions[session_id].get("session_state")
            if session_state and session_state.reattach_deadline and not session_state.is_reattach_expired():
                # Successful reattachment within grace period
                session_state.mark_ws_reconnected()
                session_state.last_status = {"type": "status", "state": "RESUMED"}
                active_sessions[session_id]["websocket_connection"] = ws
                active_sessions[session_id]["last_activity_timestamp"] = time.time()
                active_sessions[session_id]["is_active"] = True
                active_sessions[session_id]["ws_disconnected"] = False
                logger.info(f"WebSocket for session {session_id} (user {user.id}) successfully reattached within grace period.")
                # Send RESUMED status to client
                try:
                    ws.send(json.dumps({"type": "status", "state": "RESUMED"}))
                except Exception as e:
                    logger.warning(f"Failed to send RESUMED status to reattached client {session_id}: {e}")
            else:
                # New connection or reattachment expired
                if session_state:
                    session_state.mark_ws_reconnected()  # Clear any stale deadline
                    session_state.last_status = {"type": "status", "state": "RESUMED"}
                active_sessions[session_id]["websocket_connection"] = ws
                active_sessions[session_id]["last_activity_timestamp"] = time.time()
                active_sessions[session_id]["is_active"] = True
                active_sessions[session_id]["ws_disconnected"] = False
                logger.info(f"WebSocket for session {session_id} (user {user.id}) connected and registered.")

    # Server-side keepalive using threading
    def _server_keepalive_thread(ws, session_id, interval=15, log=logger):
        # app-level keepalive to satisfy proxies; pairs with client's 'pong' handler
        # Modified to continue running during WebSocket disconnection for reattachment
        try:
            while True:
                time.sleep(interval)
                # Check if session is still active
                sess = active_sessions.get(session_id)
                if not sess or not sess.get("is_active"):
                    log.info(f"WS {session_id}: Session no longer active, stopping keepalive")
                    break

                # Check for reattachment expiry
                session_state = sess.get("session_state")
                if session_state and session_state.reattach_deadline and session_state.is_reattach_expired():
                    log.info(f"WS {session_id}: Reattachment grace period expired, stopping keepalive")
                    break

                # Try to send keepalive if WebSocket is available
                current_ws = sess.get("websocket_connection")
                if current_ws:
                    try:
                        current_ws.send(json.dumps({"type": "ping"}))
                        log.debug(f"WS {session_id}: Server keepalive ping sent")
                    except Exception as e:
                        log.warning(f"WS {session_id}: keepalive send failed: {e}")
                        # Don't break - WebSocket might have disconnected, continue for reattachment
                else:
                    # No WebSocket, but session might be waiting for reattachment
                    if session_state and session_state.reattach_deadline:
                        log.debug(f"WS {session_id}: No WebSocket connection, continuing keepalive for reattachment")
                    else:
                        log.info(f"WS {session_id}: No WebSocket and no reattachment expected, stopping keepalive")
                        break
        except Exception as e:
            log.error(f"WS {session_id}: keepalive thread error: {e}")

    # Start keepalive thread
    keepalive_thread = threading.Thread(
        target=_server_keepalive_thread, 
        args=(ws, session_id), 
        daemon=True
    )
    keepalive_thread.start()
    active_sessions[session_id]["keepalive_thread"] = keepalive_thread

    try:
        ws.send(json.dumps({"type": "hello", "session_id": session_id}))
    except Exception as e:
        logger.debug(f"Session {session_id}: Failed to send hello on WebSocket connect: {e}")

    PING_INTERVAL_SECONDS = 5
    last_ping_sent = time.time()

    AUDIO_SEGMENT_DURATION_SECONDS_TARGET = 15 

    try:
        while True:
            now = time.time()
            if now - last_ping_sent >= PING_INTERVAL_SECONDS:
                try:
                    ws.send(json.dumps({"type": "ping", "ts": int(now)}))
                    last_ping_sent = now
                except ConnectionClosed:
                    raise
                except Exception as ping_error:
                    logger.warning(f"Session {session_id}: Failed to send ping: {ping_error}")
                    break

            try:
                message = ws.receive(timeout=1)
            except TimeoutError:
                continue
            except ConnectionClosed:
                raise
            except Exception as receive_error:
                logger.warning(f"Session {session_id}: WebSocket receive error: {receive_error}")
                break

            if message is None:
                sess = active_sessions.get(session_id)
                if sess:
                    last_ts = sess.get("last_activity_timestamp", 0)
                    paused = bool(sess.get("is_backend_processing_paused", False))
                    if not paused and (time.time() - last_ts > 70):
                        logger.warning(f"WebSocket for session {session_id} timed out due to inactivity (loop check). Closing.")
                        break
                continue

            if session_id not in active_sessions or not active_sessions[session_id].get("is_active"):
                logger.info(f"WebSocket session {session_id}: Session seems to have been stopped/cleaned up. Closing WS.")
                break

            active_sessions[session_id]["last_activity_timestamp"] = time.time()

            if isinstance(message, str):
                try:
                    control_msg = json.loads(message)
                    action = control_msg.get("action")
                    msg_type = control_msg.get("type")

                    if msg_type == "pong" or action == "pong":
                        continue

                    if msg_type == "ping" and action is None:
                        ws.send(json.dumps({"type": "pong"}))
                        continue

                    # Check if this is an audio header (mobile recording)
                    if not action and all(key in control_msg for key in ['contentType', 'rate', 'channels']):
                        # This is an audio header from mobile recording
                        session_data = active_sessions.get(session_id, {})
                        session_data["mobile_audio_header"] = control_msg
                        session_data["content_type"] = control_msg.get("contentType", "audio/webm")
                        session_data["sample_rate"] = control_msg.get("rate", 48000)
                        session_data["channels"] = control_msg.get("channels", 1)
                        session_data["bit_depth"] = control_msg.get("bitDepth", 16)

                        logger.info(f"WebSocket session {session_id}: Received mobile audio header: {control_msg}")

                        # Initialize mobile recording telemetry
                        session_data["mobile_telemetry"] = {
                            "codec_detected": control_msg.get("contentType", "unknown"),
                            "header_received_at": time.time(),
                            "transcode_attempts": 0,
                            "transcode_successes": 0,
                            "stt_requests": 0,
                            "error_count": 0
                        }
                        continue

                    logger.debug(f"WebSocket session {session_id}: Received control message: {control_msg}")

                    if action == "ping":
                        ws.send(json.dumps({"type": "pong"}))
                        continue # Don't log further for pings

                    logger.info(f"WebSocket session {session_id}: Received control message: {control_msg}")
                    if action == "set_processing_state": 
                        is_paused_by_client = control_msg.get("paused", False)
                        active_sessions[session_id]["is_backend_processing_paused"] = is_paused_by_client
                        logger.info(f"Session {session_id}: Backend audio processing "
                                    f"{'paused' if is_paused_by_client else 'resumed'} based on client state.")
                        session_data_ref = active_sessions[session_id]
                        current_offset = session_data_ref.get('current_total_audio_duration_processed_seconds', 0.0)
                        if is_paused_by_client:
                            session_data_ref["pause_marker_to_write"] = "<<REC PAUSED>>"
                            session_data_ref["pause_event_timestamp_offset"] = current_offset
                            logger.info(f"Session {session_id}: Queued '<<REC PAUSED>>' marker at offset {current_offset:.2f}s.")
                        else: 
                            session_data_ref["pause_marker_to_write"] = "<<REC RESUMED>>"
                            session_data_ref["pause_event_timestamp_offset"] = current_offset
                            logger.info(f"Session {session_id}: Queued '<<REC RESUMED>>' marker at offset {current_offset:.2f}s.")
                    elif action == "stop_stream": 
                        logger.info(f"WebSocket session {session_id}: Received 'stop_stream'. Initiating finalization.")
                        _finalize_session(session_id) 
                        break 
                except json.JSONDecodeError:
                    logger.warning(f"WebSocket session {session_id}: Received invalid JSON control message: {message}")
                except Exception as e:
                    logger.error(f"WebSocket session {session_id}: Error processing control message '{message}': {e}")

            elif isinstance(message, bytes):
                backlog_message = None
                backlog_ws = None
                drop_chunk = False
                with session_locks[session_id]:
                    if session_id not in active_sessions:
                        logger.warning(f"WebSocket {session_id}: Session disappeared after acquiring lock. Aborting message processing.")
                        break
                    session_data = active_sessions[session_id]

                    pending = int(session_data.get("pending_segments", 0))
                    max_pending = int(session_data.get("max_pending_segments", MAX_PENDING_SEGMENTS))
                    if pending >= max_pending:
                        # HARD ADMISSION CONTROL: do not enqueue, tell client to retry later
                        event_payload = {"pending": pending, "max": max_pending, "retry_after_ms": 1500}
                        event_ts = datetime.now(timezone.utc).isoformat()
                        session_data.setdefault("recent_events", deque(maxlen=200)).append({
                            "type": "backpressure_reject",
                            "payload": event_payload,
                            "ts": event_ts,
                        })
                        backlog_message = {"type": "event", "event": "backpressure_reject", "ts": event_ts, **event_payload}
                        backlog_ws = session_data.get("websocket_connection")
                        # Drop this audio chunk (no enqueue)
                        drop_chunk = True

                    if session_data.get("is_backend_processing_paused", False):
                        logger.debug(f"Session {session_id} is paused. Discarding {len(message)} audio bytes.")
                        continue

                    # Detect codec if not already set by header
                    if "content_type" not in session_data:
                        detected_codec = _detect_audio_codec(message)
                        session_data["content_type"] = detected_codec
                        session_data["codec_detected_from_bytes"] = True
                        logger.info(f"WebSocket session {session_id}: Detected codec from binary data: {detected_codec}")

                        # Update telemetry
                        if "mobile_telemetry" not in session_data:
                            session_data["mobile_telemetry"] = {
                                "codec_detected": detected_codec,
                                "header_received_at": None,
                                "transcode_attempts": 0,
                                "transcode_successes": 0,
                                "stt_requests": 0,
                                "error_count": 0
                            }
                    
                    # Log mobile audio header for telemetry but skip normalization
                    # VAD → WAV pipeline already handles all audio formats correctly
                    has_mobile_header = "mobile_audio_header" in session_data
                    if has_mobile_header:
                        mobile_header = session_data["mobile_audio_header"]
                        logger.info(f"Session {session_id}: Mobile audio detected ({mobile_header.get('contentType')}) - using VAD pipeline")
                        # Update telemetry
                        if "mobile_telemetry" not in session_data:
                            session_data["mobile_telemetry"] = {
                                "codec_detected": mobile_header.get("contentType", "unknown"),
                                "header_received_at": time.time(),
                                "vad_pipeline_used": True,
                                "normalization_bypassed": True
                            }

                    # Check if VAD is enabled for this session
                    vad_enabled = session_data.get("vad_enabled", False)

                    if vad_enabled and VAD_IMPORT_SUCCESS and vad_bridge:
                        # VAD Processing Path with robust header and data accumulation
                        if not session_data.get("is_first_blob_received", False):
                            first_blob = bytes(message)
                            session_data["is_first_blob_received"] = True
                            logger.info(
                                f"Session {session_id}: Captured first blob as WebM header candidate ({len(first_blob)} bytes)."
                            )
                            _store_first_webm_blob(session_id, session_data, first_blob)
                            # The first blob is the first segment.
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(first_blob)
                        else:
                            # Subsequent blobs are just audio data fragments, append them.
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                        
                        logger.debug(f"Session {session_id}: Appended {len(message)} bytes. Total buffer: {len(session_data['current_segment_raw_bytes'])}")
                        
                        session_data["accumulated_audio_duration_for_current_segment_seconds"] += 3.0

                        if not session_data["is_backend_processing_paused"] and \
                           session_data["accumulated_audio_duration_for_current_segment_seconds"] >= AUDIO_SEGMENT_DURATION_SECONDS_TARGET:
                            
                            logger.info(f"Session {session_id}: Accumulated enough audio ({session_data['accumulated_audio_duration_for_current_segment_seconds']:.2f}s est.). Processing segment.")

                            current_fragment_bytes = bytes(session_data["current_segment_raw_bytes"])
                            global_header_bytes, effective_header_mode = _resolve_webm_header(session_id, session_data)

                            if global_header_bytes and not current_fragment_bytes.startswith(global_header_bytes):
                                bytes_to_process = global_header_bytes + current_fragment_bytes
                                logger.debug(
                                    f"Session {session_id}: Prepended {len(global_header_bytes)} header bytes (mode={effective_header_mode}) before VAD dispatch."
                                )
                            else:
                                bytes_to_process = current_fragment_bytes

                            # Get the duration of this segment for the atomic update.
                            duration_of_this_segment = session_data["accumulated_audio_duration_for_current_segment_seconds"]

                            # CRITICAL FIX: Reset the buffer to be truly empty.
                            session_data["current_segment_raw_bytes"] = bytearray()
                            session_data["accumulated_audio_duration_for_current_segment_seconds"] = 0.0
                            
                            if not bytes_to_process:
                                logger.warning(f"Session {session_id}: Combined byte stream is empty, skipping VAD processing call.")
                                continue

                            try:
                                # The bridge will now receive a complete, valid WebM blob
                                vad_bridge.process_audio_blob(session_id, bytes_to_process)
                                logger.debug(f"Session {session_id}: Dispatched {len(bytes_to_process)} bytes to VAD bridge.")
                                

                            except Exception as e:
                                logger.error(f"Session {session_id}: Error dispatching audio to VAD bridge: {e}", exc_info=True)
                        
                        continue
                    
                    # Original Processing Path (fallback or when VAD not enabled)
                    if not vad_enabled:
                        logger.debug(f"Session {session_id}: Using original processing path (VAD disabled or failed for chunk).")
                        if not session_data.get("is_first_blob_received", False):
                            first_blob = bytes(message)
                            session_data["is_first_blob_received"] = True
                            logger.info(f"Session {session_id}: Captured first blob as global WebM header ({len(first_blob)} bytes).")
                            _store_first_webm_blob(session_id, session_data, first_blob)
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(first_blob)
                        else:
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                        
                        logger.debug(f"Session {session_id}: Appended {len(message)} bytes to raw_bytes buffer. Total buffer: {len(session_data['current_segment_raw_bytes'])}")

                        session_data["accumulated_audio_duration_for_current_segment_seconds"] += 3.0 

                        if not session_data["is_backend_processing_paused"] and \
                           session_data["accumulated_audio_duration_for_current_segment_seconds"] >= AUDIO_SEGMENT_DURATION_SECONDS_TARGET:
                            
                            logger.info(f"Session {session_id}: Accumulated enough audio ({session_data['accumulated_audio_duration_for_current_segment_seconds']:.2f}s est.). Processing segment from raw bytes.")
                            
                            current_fragment_bytes = bytes(session_data["current_segment_raw_bytes"])
                            global_header_bytes, effective_header_mode = _resolve_webm_header(session_id, session_data)

                            if not global_header_bytes and current_fragment_bytes:
                                logger.warning(f"Session {session_id}: Global header not captured, but processing fragments. This might fail if not the very first segment.")
                                all_segment_bytes = current_fragment_bytes
                            elif global_header_bytes and current_fragment_bytes:
                                if current_fragment_bytes.startswith(global_header_bytes) and len(global_header_bytes) > 0:
                                    all_segment_bytes = current_fragment_bytes
                                    logger.debug(f"Session {session_id}: Processing first segment data which includes its own header.")
                                else:
                                    all_segment_bytes = global_header_bytes + current_fragment_bytes
                                    logger.debug(
                                        f"Session {session_id}: Prepended header ({len(global_header_bytes)} bytes, mode={effective_header_mode}) to current fragments ({len(current_fragment_bytes)} bytes)."
                                    )
                            elif global_header_bytes and not current_fragment_bytes:
                                logger.warning(f"Session {session_id}: Global header exists but no current fragments. Skipping empty segment.")
                                session_data["current_segment_raw_bytes"] = bytearray() 
                                session_data["accumulated_audio_duration_for_current_segment_seconds"] = 0.0
                                session_data["actual_segment_duration_seconds"] = 0.0
                                continue
                            
                            bytes_to_process = bytes(session_data["current_segment_raw_bytes"])
                            global_header_for_thread, _ = _resolve_webm_header(session_id, session_data)

                            session_data["current_segment_raw_bytes"] = bytearray()
                            session_data["accumulated_audio_duration_for_current_segment_seconds"] = 0.0
                            
                            if not bytes_to_process: 
                                logger.warning(f"Session {session_id}: Raw byte buffer for current fragments is empty, though duration target met. Skipping processing.")
                                continue

                            all_segment_bytes_for_ffmpeg_thread = b''
                            if not global_header_for_thread and bytes_to_process:
                                all_segment_bytes_for_ffmpeg_thread = bytes_to_process
                            elif global_header_for_thread and bytes_to_process:
                                if bytes_to_process.startswith(global_header_for_thread) and len(global_header_for_thread) > 0:
                                    all_segment_bytes_for_ffmpeg_thread = bytes_to_process
                                else:
                                    all_segment_bytes_for_ffmpeg_thread = global_header_for_thread + bytes_to_process
                            elif global_header_for_thread and not bytes_to_process:
                                 logger.warning(f"Session {session_id}: Global header exists but no current fragments to process. Skipping empty segment.")
                                 continue
                            else: 
                                logger.warning(f"Session {session_id}: No global header and no current fragments to process. Skipping.")
                                continue
                            
                            if not all_segment_bytes_for_ffmpeg_thread:
                                logger.warning(f"Session {session_id}: Combined bytes for FFmpeg is empty. Skipping.")
                                continue

                            temp_processing_dir_thread = os.path.join(session_data['temp_audio_session_dir'], "segments_processing")
                            os.makedirs(temp_processing_dir_thread, exist_ok=True)
                            segment_uuid_thread = uuid.uuid4().hex
                            final_output_wav_path_thread = os.path.join(temp_processing_dir_thread, f"final_audio_{segment_uuid_thread}.wav")

                            def process_segment_in_thread(
                                s_id, s_data_ref, lock_ref,
                                audio_bytes, wav_path,
                                openai_key, anthropic_key,
                                vad_level
                            ):
                                try:
                                    ffmpeg_command = [
                                        'ffmpeg',
                                        '-loglevel', 'warning',
                                        '-fflags', '+discardcorrupt',
                                        '-err_detect', 'ignore_err',
                                        '-y',
                                        '-i', 'pipe:0',
                                        '-ar', '16000',
                                        '-ac', '1',
                                        '-acodec', 'pcm_s16le',
                                        wav_path
                                    ]
                                    logger.info(f"Thread Session {s_id}: Executing ffmpeg: {' '.join(ffmpeg_command)}")
                                    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    _, stderr_ffmpeg = process.communicate(input=audio_bytes)

                                    if process.returncode != 0:
                                        logger.error(f"Thread Session {s_id}: ffmpeg failed. RC: {process.returncode}, Err: {stderr_ffmpeg.decode('utf-8','ignore')}")
                                        return 
                                    logger.info(f"Thread Session {s_id}: Successfully converted to {wav_path}")

                                    ffprobe_command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', wav_path]
                                    actual_segment_dur = 0.0
                                    try:
                                        duration_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=True)
                                        actual_segment_dur = float(duration_result.stdout.strip())
                                        logger.info(f"Thread Session {s_id}: Actual duration of WAV {wav_path} is {actual_segment_dur:.2f}s")
                                    except (subprocess.CalledProcessError, ValueError) as ffprobe_err:
                                        logger.error(f"Thread Session {s_id}: ffprobe failed for {wav_path}: {ffprobe_err}. Estimating.")
                                        bytes_per_second = 16000 * 2
                                        estimated_dur = len(audio_bytes) / bytes_per_second if bytes_per_second > 0 else 0
                                        actual_segment_dur = estimated_dur
                                        logger.warning(f"Thread Session {s_id}: Using rough estimated duration: {actual_segment_dur:.2f}s")
                                    
                                    with lock_ref:
                                        if s_id in active_sessions: 
                                            active_sessions[s_id]['actual_segment_duration_seconds'] = actual_segment_dur
                                        else:
                                            logger.warning(f"Thread Session {s_id}: Session data missing when trying to update actual_segment_duration.")
                                            s_data_ref['actual_segment_duration_seconds'] = actual_segment_dur

                                    try:
                                        SESSION_ADAPTER.on_segment(
                                            session_id=s_id,
                                            raw_path=wav_path,
                                            captured_ts=time.time(),
                                            duration_s=actual_segment_dur,
                                            language=s_data_ref.get('language_setting_from_client'),
                                            vad_aggressiveness=vad_level
                                        )
                                    except Exception as adapter_err:
                                        logger.error(f"Thread Session {s_id}: Adapter processing failed for segment {wav_path}: {adapter_err}", exc_info=True)
                                    finally:
                                        try:
                                            with lock_ref:
                                                s_data_ref['pending_segments'] = max(0, int(s_data_ref.get('pending_segments', 1)) - 1)
                                        except Exception as pending_err:
                                            logger.debug(f"Thread Session {s_id}: Failed to decrement pending segments: {pending_err}")

                                except Exception as thread_e:
                                    logger.error(f"Thread Session {s_id}: Error during threaded segment processing: {thread_e}", exc_info=True)
                                finally:
                                    pass
                            
                            try:
                                session_data["pending_segments"] = int(session_data.get("pending_segments", 0)) + 1
                            except Exception as pending_err:
                                logger.debug(f"Session {session_id}: Unable to increment pending segments before transcription: {pending_err}")

                            processing_thread = threading.Thread(
                                target=process_segment_in_thread,
                                args=(
                                    session_id,
                                    session_data, 
                                    session_locks[session_id], 
                                    all_segment_bytes_for_ffmpeg_thread,
                                    final_output_wav_path_thread,
                                    session_data.get("openai_api_key"),
                                    session_data.get("anthropic_api_key"),
                                    session_data.get("vad_aggressiveness_from_client")
                                )
                            )
                            processing_thread.start()
                            logger.info(f"Session {session_id}: Started processing thread for segment {segment_uuid_thread}")
                if 'drop_chunk' in locals() and drop_chunk:
                    try:
                        backlog_ws and backlog_ws.send(json.dumps(backlog_message))
                    except Exception as backlog_err:
                        logger.debug(f"Session {session_id}: Failed to send backpressure reject: {backlog_err}")
                    continue
                if backlog_message and backlog_ws:
                    try:
                        backlog_ws.send(json.dumps(backlog_message))
                    except Exception as backlog_err:
                        logger.debug(f"Session {session_id}: Failed to send backpressure event: {backlog_err}")

            if session_id not in active_sessions:
                logger.info(f"WebSocket session {session_id}: Session was externally stopped during message processing. Closing connection.")
                if ws.fileno() != -1: 
                    try: ws.close(1000, "Session stopped externally")
                    except Exception as e_close: logger.error(f"Error closing WebSocket for externally stopped session {session_id}: {e_close}")
                break

    except ConnectionClosed as e:
        logger.info(f"WebSocket for session {session_id} (user {user.id}): Connection closed by client: {e}")
    except ConnectionResetError:
        logger.warning(f"WebSocket for session {session_id} (user {user.id}): Connection reset by client.")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id} (user {user.id}): {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket for session {session_id} (user {user.id}) disconnected. Marking for reattachment grace period.")
        _mark_ws_disconnected(session_id)

        try:
            ws.close()
        except Exception:
            pass
        

@app.route('/api/recording/start', methods=['POST'])
@supabase_auth_required(agent_required=False)
def start_recording():
    """Start or get existing recording session with reconnection token."""
    user = g.user
    session_id = request.args.get("session_id") or str(uuid4())

    with session_locks[session_id]:
        if session_id not in active_sessions:
            # Session doesn't exist, return error - sessions must be created via existing transcript endpoint
            return jsonify({"error": "Session not found. Create session via /api/transcribe/start first."}), 404

        sess = active_sessions[session_id]

        # Check if user owns this session
        if sess.get("user_id") != user.id:
            return jsonify({"error": "Access denied"}), 403

        # Return existing session info with reconnect token
        return jsonify({
            "session_id": session_id,
            "reconnect_token": sess.get("reconnect_token", ""),
            "next_seq": sess.get("next_seq", 1)
        }), 200

@app.route('/api/recording/status', methods=['GET'])
def get_recording_status_route():
    session_id_param = request.args.get('session_id')
    status_data = _get_current_recording_status_snapshot(session_id_param)
    return jsonify(status_data), 200

@app.route('/api/index/<string:index_name>/stats', methods=['GET'])
def get_pinecone_index_stats(index_name: str):
    logger.info(f"Request received for stats of index: {index_name}")
    try:
        pc = init_pinecone()
        if not pc: return jsonify({"error": "Pinecone client initialization failed"}), 500
        try: index = pc.Index(index_name)
        except NotFoundException: return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e: logger.error(f"Error accessing index '{index_name}': {e}", exc_info=True); return jsonify({"error": f"Failed to access index '{index_name}'"}), 500
        stats = index.describe_index_stats(); stats_dict = {};
        if hasattr(stats, 'to_dict'): stats_dict = stats.to_dict()
        elif isinstance(stats, dict): stats_dict = stats
        else: stats_dict = {"raw_stats": str(stats)}
        logger.info(f"Successfully retrieved stats for index '{index_name}'.")
        return jsonify(stats_dict), 200
    except Exception as e: 
        logger.error(f"Error getting stats for index '{index_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/list_ids', methods=['GET'])
def list_vector_ids_in_namespace(index_name: str, namespace_name: str):
    limit = request.args.get('limit', default=1000, type=int); next_token = request.args.get('next_token', default=None, type=str)
    logger.info(f"Request list IDs index '{index_name}', ns '{namespace_name}' (limit: {limit}, token: {next_token})")
    if not 1 <= limit <= 1000: limit = 1000
    try:
        pc = init_pinecone();
        if not pc: return jsonify({"error": "Pinecone client initialization failed"}), 500
        try: index = pc.Index(index_name)
        except NotFoundException: return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e: logger.error(f"Error access index '{index_name}': {e}", exc_info=True); return jsonify({"error": f"Failed access index '{index_name}'"}), 500
        query_namespace = "" if namespace_name == "_default" else namespace_name
        logger.info(f"Querying Pinecone ns: '{query_namespace}' (URL requested: '{namespace_name}')")
        list_response = index.list(namespace=query_namespace, limit=limit)
        vector_ids = []; next_page_token = None
        try:
            if hasattr(list_response, '__iter__'): 
                 for id_batch in list_response: 
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch): vector_ids.extend(id_batch);
                     elif hasattr(id_batch, 'ids') and isinstance(id_batch.ids, list): vector_ids.extend(id_batch.ids); 
                     else: logger.warning(f"Unexpected item type from list generator: {type(id_batch)}");
                     if len(vector_ids) >= limit: vector_ids = vector_ids[:limit]; break 
            else: 
                logger.warning(f"index.list() did not return a generator. Type: {type(list_response)}")
                if hasattr(list_response, 'ids') and isinstance(list_response.ids, list): vector_ids = list_response.ids[:limit]
                elif hasattr(list_response, 'vectors') and isinstance(list_response.vectors, list): vector_ids = [v.id for v in list_response.vectors][:limit]
                if hasattr(list_response, 'pagination') and list_response.pagination and hasattr(list_response.pagination, 'next'):
                    next_page_token = list_response.pagination.next
        except Exception as proc_e: logger.error(f"Error processing list response: {proc_e}", exc_info=True)
        logger.info(f"Found {len(vector_ids)} vector IDs in '{query_namespace}'. Next Token: {next_page_token}")
        return jsonify({"namespace": namespace_name, "vector_ids": vector_ids, "next_token": next_page_token}), 200
    except Exception as e: 
        logger.error(f"Error listing vector IDs for index '{index_name}', ns '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "Unexpected error listing vector IDs"}), 500

@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/exists', methods=['GET'])
def namespace_exists(index_name: str, namespace_name: str):
    """Check if a namespace exists and is not empty."""
    logger.info(f"Request to check existence of namespace '{namespace_name}' in index '{index_name}'")
    try:
        pc = init_pinecone()
        if not pc:
            return jsonify({"error": "Pinecone client initialization failed"}), 500
        
        index = pc.Index(index_name)
        # Query with a dummy vector to check for the presence of any vectors in the namespace.
        # This is the most reliable way to check for a non-empty namespace.
        query_response = index.query(
            namespace=namespace_name,
            top_k=1,
            include_values=False,
            include_metadata=False,
            vector=[0] * 1536 # Assuming 1536 dimensions for the dummy vector
        )
        
        exists = len(query_response['matches']) > 0
        logger.info(f"Namespace '{namespace_name}' in index '{index_name}' {'exists and is not empty' if exists else 'does not exist or is empty'}.")
        return jsonify({"exists": exists}), 200

    except NotFoundException:
        return jsonify({"error": f"Index '{index_name}' not found"}), 404
    except Exception as e:
        logger.error(f"Error checking namespace '{namespace_name}' in index '{index_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/list_docs', methods=['GET'])
def list_unique_docs_in_namespace(index_name: str, namespace_name: str):
    limit = request.args.get('limit', default=100, type=int);
    if not 1 <= limit <= 100: limit = 100
    logger.info(f"Request list unique docs index '{index_name}', ns '{namespace_name}' (fetch {limit} IDs)")
    try:
        pc = init_pinecone();
        if not pc: return jsonify({"error": "Pinecone client initialization failed"}), 500
        try: index = pc.Index(index_name)
        except NotFoundException: return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e: logger.error(f"Error access index '{index_name}': {e}", exc_info=True); return jsonify({"error": f"Failed access index '{index_name}'"}), 500
        query_namespace = "" if namespace_name == "_default" else namespace_name
        vector_ids_to_parse = []
        try:
            list_response = index.list(namespace=query_namespace, limit=limit); 
            count = 0
            if hasattr(list_response, '__iter__'): 
                 for id_batch in list_response:
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch): vector_ids_to_parse.extend(id_batch); count += len(id_batch)
                     else: logger.warning(f"Unexpected item type from list gen: {type(id_batch)}")
                     if count >= limit: break 
            elif hasattr(list_response, 'ids') and isinstance(list_response.ids, list): 
                vector_ids_to_parse = list_response.ids
            else: logger.warning(f"index.list() returned unexpected type: {type(list_response)}")

            logger.info(f"Retrieved {len(vector_ids_to_parse)} vector IDs for parsing document names.")
        except Exception as list_e: 
            logger.error(f"Error listing vector IDs: {list_e}", exc_info=True)
            return jsonify({"error": "Failed list vector IDs from Pinecone"}), 500
        
        unique_doc_names = set()
        for vec_id in vector_ids_to_parse:
            try:
                last_underscore_index = vec_id.rfind('_')
                if last_underscore_index != -1: 
                    doc_name_part = vec_id[:last_underscore_index]
                    if vec_id[last_underscore_index+1:].isdigit():
                         unique_doc_names.add(urllib.parse.unquote_plus(doc_name_part))
                    else: 
                         unique_doc_names.add(urllib.parse.unquote_plus(vec_id)) 
                else: 
                    unique_doc_names.add(urllib.parse.unquote_plus(vec_id))
            except Exception as parse_e: 
                logger.error(f"Error parsing vector ID '{vec_id}': {parse_e}")
        
        sorted_doc_names = sorted(list(unique_doc_names))
        logger.info(f"Found {len(sorted_doc_names)} unique doc names in ns '{query_namespace}'.")
        return jsonify({"index": index_name, "namespace": namespace_name, "unique_document_names": sorted_doc_names, "vector_ids_checked": len(vector_ids_to_parse)}), 200
    except Exception as e: 
        logger.error(f"Error listing unique docs index '{index_name}', ns '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "Unexpected error listing documents"}), 500

@app.route('/api/transcription/status/<job_id>', methods=['GET'])
@supabase_auth_required(agent_required=False)
def get_transcription_job_status(user: SupabaseUser, job_id: str):
    """Get the status of a transcription job."""
    if job_id not in transcription_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = transcription_jobs[job_id]
    
    # Verify user owns this job
    if job_data.get('user_id') != user.id:
        return jsonify({"error": "Access denied"}), 403
    
    # Clean response - don't send internal data
    response_data = {
        'job_id': job_id,
        'status': job_data['status'],
        'progress': job_data['progress'],
        'current_step': job_data['current_step'],
        'total_chunks': job_data['total_chunks'],
        'completed_chunks': job_data['completed_chunks'],
        'created_at': job_data['created_at'],
        'updated_at': job_data['updated_at'],
    }
    
    # Include result if completed
    if job_data['status'] == 'completed' and job_data['result']:
        response_data['result'] = job_data['result']
    
    # Include error if failed
    if job_data['status'] == 'failed' and job_data['error']:
        response_data['error'] = job_data['error']
    
    return jsonify(response_data), 200

@app.route('/api/transcription/cancel/<job_id>', methods=['POST'])
@supabase_auth_required(agent_required=False)
def cancel_transcription_job(user: SupabaseUser, job_id: str):
    """Cancel a running transcription job."""
    if job_id not in transcription_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = transcription_jobs[job_id]
    
    # Verify user owns this job
    if job_data.get('user_id') != user.id:
        return jsonify({"error": "Access denied"}), 403
    
    # Check if job can be cancelled
    current_status = job_data.get('status', 'unknown')
    if current_status in ['completed', 'failed', 'cancelled']:
        return jsonify({"error": f"Cannot cancel job with status: {current_status}"}), 400
    
    # Mark job as cancelled
    update_job_progress(job_id, status='cancelled', current_step='Cancelled by user')
    logger.info(f"Job {job_id} cancelled by user {user.id}")
    
    return jsonify({"message": "Job cancelled successfully", "job_id": job_id}), 200

def process_transcription_job_async(job_id: str, agent_name: str, s3_key: str, original_filename: str, transcription_language: str, user: SupabaseUser):
    """Process transcription job in background thread with progress tracking."""
    temp_filepath = None
    transcription_successful = False
    
    try:
        # Check for cancellation before starting
        if transcription_jobs.get(job_id, {}).get('status') == 'cancelled':
            logger.info(f"Job {job_id} was cancelled before processing started")
            return
            
        update_job_progress(job_id, status='processing', current_step='Downloading file from S3...')
        
        # 1. Download file from S3 to a temporary local path
        s3 = get_s3_client()
        aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
        if not s3 or not aws_s3_bucket:
            update_job_progress(job_id, status='failed', error='S3 service not configured')
            return

        unique_id = uuid.uuid4().hex
        temp_filename = f"{unique_id}_{secure_filename(original_filename)}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        logger.info(f"Job {job_id}: Downloading s3://{aws_s3_bucket}/{s3_key} to {temp_filepath}")
        s3.download_file(aws_s3_bucket, s3_key, temp_filepath)
        logger.info(f"Job {job_id}: Download complete.")

        update_job_progress(job_id, current_step='Preparing transcription...', progress=0.1)

        # 2. Get file info for chunking
        file_size = os.path.getsize(temp_filepath)
        max_chunk_size = 20 * 1024 * 1024  # 20MB chunks for safety
        
        # Estimate chunks (rough calculation)
        estimated_chunks = max(1, math.ceil(file_size / max_chunk_size))
        update_job_progress(job_id, total_chunks=estimated_chunks, current_step=f'Processing {estimated_chunks} chunks...')

        # 3. Transcribe with progress callback
        openai_api_key = get_api_key(agent_name, 'openai')
        if not openai_api_key:
            logger.error(f"Job {job_id}: OpenAI API key not found for agent '{agent_name}'.")
            update_job_progress(job_id, status='failed', error='Transcription service not configured (missing API key)')
            return
        

        # Check for cancellation before starting transcription
        if transcription_jobs.get(job_id, {}).get('status') == 'cancelled':
            logger.info(f"Job {job_id} was cancelled before transcription started")
            return
            
        logger.info(f"Job {job_id}: Starting transcription for {temp_filepath} with language: {transcription_language}...")
        
        def progress_callback(completed_chunks: int, total_chunks: int, current_step: str):
            # Check for cancellation during progress updates
            if transcription_jobs.get(job_id, {}).get('status') == 'cancelled':
                logger.info(f"Job {job_id} cancelled during chunk processing")
                raise Exception("Job cancelled by user")
                
            progress = 0.1 + (completed_chunks / total_chunks) * 0.8  # 10% for download, 80% for transcription
            update_job_progress(job_id, 
                               progress=progress,
                               completed_chunks=completed_chunks, 
                               total_chunks=total_chunks,
                               current_step=current_step)
        
        transcription_data = transcribe_large_audio_file_with_progress(
            audio_file_path=temp_filepath,
            openai_api_key=openai_api_key,
            language_setting_from_client=transcription_language,
            progress_callback=progress_callback,
            chunk_size_mb=20  # Use 20MB chunks
        )

        # 4. Process result with intelligent fallback handling
        if transcription_data and 'text' in transcription_data and 'segments' in transcription_data:
            transcription_successful = True
            full_transcript_text = transcription_data['text']
            
            # Check if this is a partial result
            if transcription_data.get('partial', False):
                success_rate = transcription_data.get('success_rate', 1.0)
                warning_msg = transcription_data.get('warning', 'Partial transcription completed')
                logger.warning(f"Job {job_id}: Partial success with {success_rate:.1%} completion rate")
                
                # Update job with partial success indicator
                result_data = {
                    'transcript': full_transcript_text,
                    'segments': transcription_data['segments'],
                    'partial': True,
                    'success_rate': success_rate,
                    'warning': warning_msg
                }
                
                update_job_progress(job_id, 
                                   status='completed',
                                   progress=1.0,  # Always 100% when completed
                                   current_step=f'Completed with {success_rate:.0%} success rate',
                                   result=result_data)
            else:
                # Complete success
                result_data = {
                    'transcript': full_transcript_text,
                    'segments': transcription_data['segments']
                }
            segments = transcription_data['segments']

            user_name = user.user_metadata.get('full_name', user.email if user.email else 'UnknownUser')
            upload_timestamp_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            
            header = (
                f"# Transcript - Uploaded\n"
                f"Agent: {agent_name}\n"
                f"User: {user_name}\n"
                f"Language: {format_language_for_header(transcription_language)}\n"
                f"Provider: {get_current_transcription_provider()}\n"
                f"Transcript Uploaded (UTC): {upload_timestamp_utc}\n\n"
            )
            
            final_transcript_with_header = header + full_transcript_text
            
            # Store result and mark as completed
            result = {"transcript": final_transcript_with_header, "segments": segments}
            update_job_progress(job_id, 
                               status='completed', 
                               progress=1.0,
                               current_step='Transcription completed successfully!',
                               result=result)
            
            logger.info(f"Job {job_id}: S3-based transcription successful for {original_filename}. Total Text Length (with header): {len(final_transcript_with_header)}, Segments: {len(segments)}")
        else:
            error_msg = "Transcription failed or returned incomplete data."
            logger.error(f"Job {job_id}: {error_msg} File: {original_filename} (S3: {s3_key}). API Result (if any): {str(transcription_data)[:500]}...")
            update_job_progress(job_id, status='failed', error=error_msg)

    except Exception as e:
        logger.error(f"Job {job_id}: Error processing transcription job: {e}", exc_info=True)
        update_job_progress(job_id, status='failed', error=f"Internal server error: {str(e)}")
    finally:
        # 5. Clean up the temporary local file
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logger.info(f"Job {job_id}: Cleaned up temporary file: {temp_filepath}")
            except Exception as e_clean:
                logger.error(f"Job {job_id}: Error cleaning up temporary file {temp_filepath}: {e_clean}")
        
        # 6. Clean up the original file from S3 if transcription was successful
        if transcription_successful:
            try:
                logger.info(f"Job {job_id}: Transcription successful. Deleting original file from S3: {s3_key}")
                s3.delete_object(Bucket=aws_s3_bucket, Key=s3_key)
                logger.info(f"Job {job_id}: Successfully deleted {s3_key} from S3.")
            except Exception as e_s3_delete:
                logger.error(f"Job {job_id}: Error deleting original file {s3_key} from S3: {e_s3_delete}")

@app.route('/api/transcription/start-job-from-s3', methods=['POST'])
@supabase_auth_required(agent_required=True)
def start_transcription_from_s3(user: SupabaseUser):
    """Start an async transcription job and return job ID immediately."""
    data = g.get('json_data', {})
    agent_name = data.get('agentName')
    if not agent_name: agent_name = data.get('agent')
    s3_key = data.get('s3Key')
    original_filename = data.get('originalFilename')
    transcription_language = data.get('transcriptionLanguage', 'any')

    if not all([agent_name, s3_key, original_filename]):
        return jsonify({"error": "Missing agentName, s3Key, or originalFilename"}), 400

    # Basic validation
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        return jsonify({"error": "S3 service not configured"}), 503

    # Create job and start processing in background
    job_id = create_transcription_job(agent_name, s3_key, original_filename, user.id)
    
    # Submit job to thread pool
    app.executor.submit(
        process_transcription_job_async,
        job_id, agent_name, s3_key, original_filename, transcription_language, user
    )
    
    logger.info(f"Started async transcription job {job_id} for {original_filename}")
    return jsonify({"job_id": job_id, "message": "Transcription job started"}), 202


@app.route('/internal_api/transcribe_file', methods=['POST'])
@supabase_auth_required(agent_required=False)
def transcribe_uploaded_file(user: SupabaseUser):
    logger.info(f"Received request /internal_api/transcribe_file from user: {user.id}")
    if 'audio_file' not in request.files:
        logger.warning("No audio_file part in request files.")
        return jsonify({"error": "No audio_file part in the request"}), 400
    
    file = request.files['audio_file']
    transcriptionLanguage = request.form.get('transcription_language', 'any') # Get language from form
    
    if file.filename == '':
        logger.warning("No selected file provided in audio_file part.")
        return jsonify({"error": "No selected file"}), 400

    if file:
        temp_filepath = None # Ensure temp_filepath is defined for cleanup
        try:
            filename = secure_filename(file.filename if file.filename else "audio.tmp")
            unique_id = uuid.uuid4().hex
            temp_filename = f"{unique_id}_{filename}"
            temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
            
            logger.info(f"Saving uploaded file temporarily to: {temp_filepath}")
            file.save(temp_filepath)

            # Get agent_name from form data and fetch the corresponding API key
            agent_name_from_form = request.form.get('agent_name', 'UnknownAgent')
            logger.info(f"Agent name from form for header: {agent_name_from_form}")

            openai_api_key = get_api_key(agent_name_from_form, 'openai')
            if not openai_api_key:
                logger.error(f"OpenAI API key not found for agent '{agent_name_from_form}' or globally.")
                return jsonify({"error": "Transcription service not configured (missing API key)"}), 500
            
            
            logger.info(f"Starting transcription for {temp_filepath} with language: {transcriptionLanguage}...")
            transcription_data = transcribe_large_audio_file(
                audio_file_path=temp_filepath,
                openai_api_key=openai_api_key,
                language_setting_from_client=transcriptionLanguage # Pass the language setting
            )

            if transcription_data and 'text' in transcription_data and 'segments' in transcription_data:
                full_transcript_text = transcription_data['text']
                segments = transcription_data['segments']

                # Construct header
                user_name = user.user_metadata.get('full_name', user.email if user.email else 'UnknownUser')
                upload_timestamp_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
                
                header = (
                    f"# Transcript - Uploaded\n"
                    f"Agent: {agent_name_from_form}\n"
                    f"User: {user_name}\n"
                    f"Language: {format_language_for_header(transcriptionLanguage)}\n"
                    f"Provider: {get_current_transcription_provider()}\n"
                    f"Transcript Uploaded (UTC): {upload_timestamp_utc}\n\n"
                )
                
                final_transcript_with_header = header + full_transcript_text
                
                logger.info(f"Transcription successful for {temp_filepath}. Header added. Total Text Length (with header): {len(final_transcript_with_header)}, Segments: {len(segments)}")
                return jsonify({"transcript": final_transcript_with_header, "segments": segments}), 200
            else:
                error_msg = "Transcription failed or returned incomplete data (missing text or segments)."
                detailed_error_info = "No specific error details from service."
                if transcription_data:
                    if 'text' not in transcription_data:
                        error_msg += " Full transcript text missing."
                    if 'segments' not in transcription_data:
                        error_msg += " Segment data missing."
                    if transcription_data.get('error'):
                         detailed_error_info = f"Service error: {transcription_data['error']}"
                    elif not transcription_data.get('text') and transcription_data.get('segments') is not None : # segments exist but text is empty
                         error_msg = "Transcription successful but no speech detected in the audio."

                logger.error(f"{error_msg} File: {temp_filepath}. API Result (if any): {str(transcription_data)[:500]}...")
                return jsonify({"error": error_msg, "details": detailed_error_info}), 500

        except Exception as e:
            logger.error(f"Error processing uploaded file {file.filename if file.filename else 'unknown'}: {e}", exc_info=True)
            return jsonify({"error": "Internal server error during file processing"}), 500
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    logger.info(f"Cleaned up temporary file: {temp_filepath}")
                except Exception as e_clean:
                    logger.error(f"Error cleaning up temporary file {temp_filepath}: {e_clean}")
    
    return jsonify({"error": "File not processed correctly"}), 400

@app.route('/api/recordings/list', methods=['POST'])
@supabase_auth_required(agent_required=True)
def list_recordings(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agentName') or data.get('agent')
    if not agent_name:
        return jsonify({"error": "agentName or agent is required"}), 400

    logger.info(f"Listing recordings for agent: {agent_name}")
    
    # Construct the S3 prefix for the agent's recordings
    s3_prefix = f"organizations/river/agents/{agent_name}/recordings/"
    
    try:
        s3_objects = list_s3_objects_metadata(s3_prefix)
        
        # The list_s3_objects_metadata function returns a list of dicts with 'Key', 'Size', 'LastModified'
        # We need to format this into the structure the frontend expects.
        recordings = [
            {
                "s3Key": obj['Key'],
                "filename": os.path.basename(obj['Key']),
                "timestamp": obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                "size": obj['Size']
            }
            for obj in s3_objects
            if not obj['Key'].endswith('/') # Filter out folder markers
        ]
        
        # Sort by timestamp descending
        recordings.sort(key=lambda r: r['timestamp'] or '', reverse=True)
        
        logger.info(f"Found {len(recordings)} recordings for agent '{agent_name}' at prefix '{s3_prefix}'.")
        return jsonify(recordings), 200
    except Exception as e:
        logger.error(f"Error listing recordings from S3 for prefix '{s3_prefix}': {e}", exc_info=True)
        return jsonify({"error": "Failed to list recordings from S3"}), 500

@app.route('/api/s3/generate-presigned-url', methods=['POST'])
@supabase_auth_required(agent_required=False) # Agent name is in payload but doesn't need DB access check for this
def generate_presigned_url(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agentName')
    filename = data.get('filename')
    file_type = data.get('fileType')

    if not all([agent_name, filename, file_type]):
        return jsonify({"error": "Missing agentName, filename, or fileType"}), 400

    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        return jsonify({"error": "S3 service not configured"}), 503

    secure_name = secure_filename(filename)
    s3_key = f"organizations/river/agents/{agent_name}/uploads/{uuid.uuid4().hex}/{secure_name}"

    try:
        # Generate a presigned URL for a POST request
        presigned_post = s3.generate_presigned_post(
            Bucket=aws_s3_bucket,
            Key=s3_key,
            Fields={"Content-Type": file_type},
            Conditions=[
                {"Content-Type": file_type},
                ["content-length-range", 1, 500000000]  # 1 byte to 500 MB
            ],
            ExpiresIn=900  # 15 minutes
        )
        
        response_data = {
            **presigned_post,
            "s3Key": s3_key
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}", exc_info=True)
        return jsonify({"error": "Could not generate upload URL"}), 500

@app.route('/api/s3/list', methods=['GET'])
@supabase_auth_required(agent_required=False)
def list_s3_documents(user: SupabaseUser):
    logger.info(f"Received request /api/s3/list from user: {user.id}") 
    s3_prefix = request.args.get('prefix')
    if not s3_prefix: return jsonify({"error": "Missing 'prefix' query parameter"}), 400
    try:
        s3_objects = list_s3_objects_metadata(s3_prefix)
        formatted_files = []
        for obj in s3_objects:
            if obj['Key'].endswith('/') and obj['Size'] == 0: continue
            filename = os.path.basename(obj['Key']); file_type = "text/plain"
            if '.' in filename:
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in ['txt', 'md', 'log']: file_type = "text/plain"
                elif ext == 'json': file_type = "application/json"
                elif ext == 'xml': file_type = "application/xml"
            formatted_files.append({
                "name": filename, "size": obj['Size'],
                "lastModified": obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                "s3Key": obj['Key'], "type": file_type
            })
        if "transcripts/" in s3_prefix: formatted_files = [f for f in formatted_files if not f['name'].startswith('rolling-')]
        return jsonify(formatted_files), 200
    except Exception as e: 
        logger.error(f"Error listing S3 objects for prefix '{s3_prefix}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error listing S3 objects"}), 500

@app.route('/api/s3/view', methods=['GET'])
@supabase_auth_required(agent_required=False)
def view_s3_document(user: SupabaseUser):
    logger.info(f"Received request /api/s3/view from user: {user.id}") 
    s3_key = request.args.get('s3Key')
    if not s3_key: return jsonify({"error": "Missing 's3Key' query parameter"}), 400

    # IDOR Prevention: Verify ownership of the S3 key
    if not verify_s3_key_ownership(s3_key, user):
        return jsonify({"error": "Access denied to this resource"}), 403

    try:
        from utils.s3_utils import read_file_content as s3_read_content 
        content = s3_read_content(s3_key, f"S3 file for viewing ({s3_key})")
        if content is None: return jsonify({"error": "File not found or could not be read"}), 404
        return jsonify({"content": content}), 200
    except Exception as e: 
        logger.error(f"Error viewing S3 object '{s3_key}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error viewing S3 object"}), 500

@app.route('/api/s3/list-events', methods=['GET'])
@supabase_auth_required(agent_required=True)
def list_s3_events(user: SupabaseUser):
    agent_name = request.args.get('agentName') or request.args.get('agent')
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not agent_name:
        return jsonify({"error": "Missing agentName"}), 400
    if not s3 or not aws_s3_bucket:
        return jsonify({"error": "S3 service not configured"}), 503
    prefix = f"organizations/river/agents/{agent_name}/events/"
    try:
        paginator = s3.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=aws_s3_bucket, Prefix=prefix, Delimiter='/')
        event_ids = set()
        for page in result:
            for cp in page.get('CommonPrefixes', []):
                full = cp.get('Prefix', '')
                if full.startswith(prefix):
                    tail = full[len(prefix):].strip('/')
                    if tail:
                        event_ids.add(tail)
        events = sorted(event_ids)
        return jsonify({"events": events}), 200
    except Exception as e:
        logger.error(f"List S3 events error for agent '{agent_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to list events"}), 500

@app.route('/api/s3/manage-file', methods=['POST'])
@supabase_auth_required(agent_required=False) # Agent name from payload is used for path construction
def manage_s3_file(user: SupabaseUser):
    logger.info(f"Received request /api/s3/manage-file from user: {user.id}")
    data = g.get('json_data', {})
    s3_key_to_manage = data.get('s3Key')
    action_to_perform = data.get('action')

    # IDOR Prevention: Verify ownership of the S3 key before any action
    if not verify_s3_key_ownership(s3_key_to_manage, user):
        return jsonify({"error": "Access denied to this resource"}), 403
        
    agent_name_param = data.get('agentName')
    event_id_param = data.get('eventId')

    if not all([s3_key_to_manage, action_to_perform, agent_name_param, event_id_param]):
        logger.error(f"ManageFile: Missing parameters. Received: s3Key={s3_key_to_manage}, action={action_to_perform}, agentName={agent_name_param}, eventId={event_id_param}")
        return jsonify({"error": "Missing s3Key, action, agentName, or eventId in request"}), 400

    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        logger.error("ManageFile: S3 client or bucket not configured.")
        return jsonify({"error": "S3 service not available"}), 503

    if action_to_perform == "archive":
        logger.info(f"Archive action requested for {s3_key_to_manage}. Agent: {agent_name_param}, Event: {event_id_param}")
        try:
            original_filename = os.path.basename(s3_key_to_manage)
            # Construct destination key in the archive subfolder
            # Path: organizations/river/agents/{agentName}/events/{eventId}/transcripts/archive/{original_filename}
            destination_key = f"organizations/river/agents/{agent_name_param}/events/{event_id_param}/transcripts/archive/{original_filename}"
            
            logger.info(f"Archiving: Copying '{s3_key_to_manage}' to '{destination_key}' in bucket '{aws_s3_bucket}'")
            
            copy_source = {'Bucket': aws_s3_bucket, 'Key': s3_key_to_manage}
            s3.copy_object(CopySource=copy_source, Bucket=aws_s3_bucket, Key=destination_key)
            logger.info(f"Archiving: Successfully copied to '{destination_key}'.")

            s3.delete_object(Bucket=aws_s3_bucket, Key=s3_key_to_manage)
            logger.info(f"Archiving: Successfully deleted original file '{s3_key_to_manage}'.")
            
            return jsonify({"message": f"File '{original_filename}' successfully archived to '{destination_key}'."}), 200

        except s3.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'NoSuchKey':
                logger.error(f"Archive Error: Original file not found at {s3_key_to_manage}. Error: {e}")
                return jsonify({"error": f"File not found at {s3_key_to_manage}"}), 404
            else:
                logger.error(f"Archive Error: S3 ClientError while managing file {s3_key_to_manage}. Error: {e}", exc_info=True)
                return jsonify({"error": f"S3 operation failed: {error_code}"}), 500
        except Exception as e:
            logger.error(f"Archive Error: Unexpected error while archiving {s3_key_to_manage}. Error: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred during archiving."}), 500
            
    elif action_to_perform == "save_as_memory":
        logger.info(f"Save as Memory action requested for {s3_key_to_manage}. Agent: {agent_name_param}, Event: {event_id_param}")
        # Placeholder for future implementation
        # 1. Call summarization service (utils/transcript_summarizer.py -> generate_transcript_summary)
        # 2. If summarization successful and JSON summary saved to S3:
        #    Move original transcript to `.../transcripts/saved/{original_filename}`
        #    (similar S3 copy & delete logic as "archive")
        return jsonify({"message": f"Save as Memory action for {s3_key_to_manage} received, summarization logic pending."}), 501 # Not Implemented
    
    return jsonify({"error": f"Unsupported action: {action_to_perform}"}), 400

@app.route('/api/agent/docs/update', methods=['POST'])
@supabase_auth_required(agent_required=True)
def update_agent_doc(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    doc_name = data.get('doc_name') # e.g., "api_specifications.md"
    content = data.get('content')

    if not all([doc_name, content is not None]):
        return jsonify({"error": "Missing 'doc_name' or 'content' in request body"}), 400

    # The user has already been authorized for this agent by the decorator.
    # Now we call our new, specific utility function.
    success = write_agent_doc(agent_name, doc_name, content)

    if success:
        return jsonify({"status": "success", "message": f"Documentation '{doc_name}' for agent '{agent_name}' updated successfully."}), 200
    else:
        return jsonify({"error": "Failed to update documentation file in S3"}), 500


@app.route('/api/agent/list-managed', methods=['GET'])
@supabase_auth_required(agent_required=False) # The inner decorator handles role check.
@admin_or_super_user_required
def list_managed_agents(user: SupabaseUser):
    """Lists all agents for management purposes."""
    client = get_supabase_client()
    if not client:
        return jsonify({"error": "Database service unavailable"}), 503
    try:
        response = client.table("agents").select("id, name, description, created_at, workspace_id, workspaces(name)").or_("is_hidden.is.null,is_hidden.eq.false").order("name", desc=False).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Admin Dashboard: Error querying agents: {response.error}")
            return jsonify({"error": "Database error querying agents"}), 500

        # Process the data to flatten workspace name
        processed_data = []
        for agent in response.data:
            processed_agent = {
                "id": agent["id"],
                "name": agent["name"],
                "description": agent["description"],
                "created_at": agent["created_at"],
                "workspace_id": agent["workspace_id"],
                "workspace_name": agent["workspaces"]["name"] if agent.get("workspaces") and agent["workspaces"] else None
            }
            processed_data.append(processed_agent)

        return jsonify(processed_data), 200
    except Exception as e:
        logger.error(f"Admin Dashboard: Unexpected error listing agents: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/agent/name-exists', methods=['GET'])
@supabase_auth_required(agent_required=False) # Requires auth, but not a specific agent
@admin_or_super_user_required
def agent_name_exists(user: SupabaseUser):
    agent_name = request.args.get('name')
    if not agent_name:
        return jsonify({"error": "Missing 'name' query parameter"}), 400

    client = get_supabase_client()
    if not client:
        return jsonify({"error": "Database service unavailable"}), 503

    try:
        # Use count='exact' to get the total count of matching rows efficiently.
        res = client.table("agents").select("id", count='exact').eq("name", agent_name).limit(1).execute()
        
        # The Supabase-py v2 client returns the count in the `count` attribute of the response
        exists = res.count > 0 if hasattr(res, 'count') and res.count is not None else False
        
        logger.info(f"Checking for agent name '{agent_name}'. Exists: {exists}")
        return jsonify({"exists": exists}), 200
    except Exception as e:
        logger.error(f"Error checking if agent name '{agent_name}' exists: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during name check"}), 500

@app.route('/api/agent/create', methods=['POST'])
@supabase_auth_required(agent_required=False) # Role check is handled by the inner decorator.
@admin_or_super_user_required
def create_agent(user: SupabaseUser):
    """Creates a new agent from multipart/form-data, including S3 structure, DB entries, documents, and Pinecone namespace."""
    if 'agent_name' not in request.form:
        return jsonify({"error": "Missing required field: agent_name"}), 400

    agent_name = request.form['agent_name']
    description = request.form.get('description', '')
    system_prompt_content = request.form.get('system_prompt_content')
    api_keys_json = request.form.get('api_keys')
    user_ids_to_grant_access_json = request.form.get('user_ids_to_grant_access', '[]')
    new_users_to_create_json = request.form.get('new_users_to_create', '[]')

    client = get_supabase_client()
    if not client: return jsonify({"error": "Database service unavailable"}), 503

    try:
        user_ids_to_grant_access = json.loads(user_ids_to_grant_access_json)
        new_users_to_create = json.loads(new_users_to_create_json)
        if not isinstance(user_ids_to_grant_access, list) or not isinstance(new_users_to_create, list):
            raise TypeError("User access data must be lists.")
    except (json.JSONDecodeError, TypeError) as e:
        return jsonify({"error": f"Invalid format for user access data: {e}"}), 400

    agent_id_for_cleanup = None
    try:
        # === Step 1: Pre-flight checks & New User Creation ===
        existing_agent = client.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if existing_agent.data:
            return jsonify({"error": f"Agent with name '{agent_name}' already exists."}), 409

        created_user_ids = []
        if new_users_to_create:
            logger.info(f"Creating {len(new_users_to_create)} new users as part of agent creation.")
            for new_user_data in new_users_to_create:
                email = new_user_data.get('email')
                password = new_user_data.get('password')
                if not email or not password:
                    raise Exception(f"Missing email or password for new user: {new_user_data}")
                
                new_user_res = client.auth.admin.create_user({
                    "email": email, "password": password, "email_confirm": True,
                })
                new_user_obj = new_user_res.user
                if not new_user_obj:
                    raise Exception(f"Failed to create new user with email {email}.")
                created_user_ids.append(new_user_obj.id)
                logger.info(f"Successfully created new user with ID: {new_user_obj.id}")

        # === Step 2: Foundational Setup (S3, DB, Pinecone) ===
        if not create_agent_structure(agent_name):
            return jsonify({"error": "Failed to create agent folder structure in S3."}), 500
        if not create_namespace("river", agent_name):
            logger.warning(f"Could not initialize Pinecone namespace for '{agent_name}'. This can be fixed later.")

        agent_res = client.table("agents").insert({"name": agent_name, "description": description, "created_by": user.id}).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            raise Exception(f"DB error creating agent record: {agent_res.error}")
        agent_id = agent_res.data[0]['id']
        agent_id_for_cleanup = agent_id # For cleanup on failure
        logger.info(f"Created agent '{agent_name}' (ID: {agent_id}) in database.")

        # === Step 3: Grant Access ===
        all_user_ids_to_grant = set(user_ids_to_grant_access)
        all_user_ids_to_grant.update(created_user_ids)
        all_user_ids_to_grant.add(str(user.id))

        if all_user_ids_to_grant:
            access_records = [{"user_id": uid, "agent_id": agent_id} for uid in all_user_ids_to_grant]
            access_res = client.table("user_agent_access").insert(access_records).execute()
            if hasattr(access_res, 'error') and access_res.error:
                 logger.error(f"Error granting user access for agent {agent_id}: {access_res.error}")
            else:
                 logger.info(f"Successfully granted access to {len(access_res.data)} users for agent {agent_id}.")

        # === Step 4: Process Uploaded Files ===
        s3_docs = request.files.getlist('s3_docs')
        for doc in s3_docs:
            filename = secure_filename(doc.filename)
            content = doc.read().decode('utf-8')
            write_agent_doc(agent_name, filename, content)
            logger.info(f"Uploaded S3 doc '{filename}' for agent '{agent_name}'.")

        pinecone_docs = request.files.getlist('pinecone_docs')
        if pinecone_docs:
            embed_handler = EmbeddingHandler(index_name="river", namespace=agent_name)
            temp_dir = os.path.join('tmp', 'embedding_uploads', str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            try:
                for doc in pinecone_docs:
                    filename = secure_filename(doc.filename); temp_path = os.path.join(temp_dir, filename)
                    doc.save(temp_path)
                    with open(temp_path, 'r', encoding='utf-8') as f: content = f.read()
                    metadata = {
                        "source": "agent_creation_upload",
                        "source_type": "doc",
                        "file_name": filename,
                        "agent_name": agent_name,
                        "event_id": "0000",
                        "is_core_memory": True,  # Wizard uploads seed the agent's foundational knowledge
                    }
                    embed_handler.embed_and_upsert(content, metadata)
                    logger.info(f"Embedded Pinecone doc '{filename}' for agent '{agent_name}'.")
            finally:
                import shutil
                shutil.rmtree(temp_dir)

        # === Step 5: Save System Prompt ===
        if system_prompt_content:
            prompt_filename = f"systemprompt_aID-{agent_name}.md"
            prompt_s3_key = f"organizations/river/agents/{agent_name}/_config/{prompt_filename}"
            s3 = get_s3_client(); aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
            if s3 and aws_s3_bucket:
                s3.put_object(Bucket=aws_s3_bucket, Key=prompt_s3_key, Body=system_prompt_content.encode('utf-8'), ContentType='text/markdown; charset=utf-8')
                logger.info(f"Saved system prompt for agent '{agent_name}' to '{prompt_s3_key}'.")

        # === Step 6: Save API Keys ===
        if api_keys_json:
            api_keys = json.loads(api_keys_json)
            keys_to_insert = [{"agent_id": agent_id, "service_name": s, "api_key": k} for s, k in api_keys.items() if k]
            if keys_to_insert:
                client.table("agent_api_keys").insert(keys_to_insert).execute()
                logger.info(f"Saved {len(keys_to_insert)} API keys for agent '{agent_name}'.")
        
        return jsonify({"status": "success", "message": f"Agent '{agent_name}' created successfully.", "agent": agent_res.data[0]}), 201

    except Exception as e:
        logger.error(f"Unexpected error creating agent '{agent_name}': {e}", exc_info=True)
        if agent_id_for_cleanup:
            client.table("agents").delete().eq("id", agent_id_for_cleanup).execute()
            logger.info(f"Rolled back agent creation for ID {agent_id_for_cleanup}.")
        logger.error(f"Unexpected error creating agent '{agent_name}': {e}", exc_info=True)
        # Attempt to clean up failed agent creation from DB
        if 'agent_id' in locals():
            client.table("agents").delete().eq("id", agent_id).execute()
        return jsonify({"error": "An internal server error occurred during agent creation. The operation was rolled back."}), 500

@app.route('/api/agent/<agent_id>/update', methods=['PATCH'])
@supabase_auth_required(agent_required=False) # Role check is handled by the inner decorator.
@admin_or_super_user_required
def update_agent(user: SupabaseUser, agent_id: str):
    """Updates an agent's basic information (description, etc.)"""
    data = g.get('json_data', {})
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    client = get_supabase_client()
    if not client:
        return jsonify({"error": "Database service unavailable"}), 503

    try:
        # Verify agent exists and user has access
        agent_check = client.table("agents").select("id, name").eq("id", agent_id).execute()
        if not agent_check.data:
            return jsonify({"error": "Agent not found"}), 404

        # Build update data from allowed fields
        update_data = {}
        if 'description' in data:
            update_data['description'] = data['description']

        # Add more updateable fields here as needed
        # if 'name' in data:
        #     update_data['name'] = data['name']

        if not update_data:
            return jsonify({"error": "No valid fields to update"}), 400

        # Perform the update
        result = client.table("agents").update(update_data).eq("id", agent_id).execute()

        if not result.data:
            return jsonify({"error": "Failed to update agent"}), 500

        logger.info(f"Agent '{agent_id}' updated successfully by user '{user.id}'")
        return jsonify({
            "status": "success",
            "message": "Agent updated successfully",
            "agent": result.data[0]
        }), 200

    except Exception as e:
        logger.error(f"Error updating agent '{agent_id}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred"}), 500

_WARMUP_INFLIGHT = set()
_WARMUP_LOCK = threading.Lock()

@app.route('/api/agent/warm-up', methods=['POST'])
@supabase_auth_required(agent_required=True)
def warm_up_agent_cache(user: SupabaseUser):
    """
    Proactively loads an agent's essential data (prompts, frameworks, etc.)
    into the in-memory cache to speed up the first chat interaction.
    This endpoint returns immediately while caching happens in the background.
    """
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    event_id = data.get('event', '0000')

    logger.info(f"Received cache warm-up request for agent: '{agent_name}', event: '{event_id}' from user: {user.id}")

    singleflight_key = f"{agent_name}:{event_id}"
    feature_event_docs_enabled = os.getenv('FEATURE_EVENT_DOCS', 'false').lower() == 'true'
    with _WARMUP_LOCK:
        if singleflight_key in _WARMUP_INFLIGHT:
            logger.info(f"Warm-up singleflight: already in progress for {singleflight_key}; skipping duplicate.")
            return jsonify({"status": "success", "message": "Agent pre-caching already running"}), 202
        _WARMUP_INFLIGHT.add(singleflight_key)

    def cache_worker(agent, event):
        with app.app_context(): # Ensure background thread has app context if needed
            logger.info(f"Background cache worker started for agent: '{agent}', event: '{event}'")
            try:
                get_objective_function(agent)
                # Calling these functions will populate the cache due to their internal logic
                get_latest_system_prompt(agent)
                get_latest_frameworks(agent)
                get_latest_context(agent, event)
                get_agent_docs(agent)
                if feature_event_docs_enabled:
                    get_event_docs(agent, event)
                get_transcript_summaries(agent, event) # Also warm up summaries
                logger.info(f"Background cache worker finished for agent: '{agent}', event: '{event}'")
            except Exception as e:
                logger.error(f"Error in background cache worker for agent '{agent}': {e}", exc_info=True)
            finally:
                with _WARMUP_LOCK:
                    _WARMUP_INFLIGHT.discard(f"{agent}:{event}")

    # Submit the caching tasks to the background executor
    app.executor.submit(cache_worker, agent_name, event_id)

    return jsonify({"status": "success", "message": "Agent pre-caching initiated"}), 202


@app.route('/api/admin/clear-cache', methods=['POST'])
@supabase_auth_required(agent_required=False)
def clear_s3_cache(user: SupabaseUser):
    data = g.get('json_data', {})
    scope = data.get('scope') # e.g., 'all' or an agent_name
    
    logger.info(f"User {user.id} requested cache clear for scope: '{scope}'")
    
    keys_to_delete = []
    with S3_CACHE_LOCK:
        if scope == 'all':
            keys_to_delete = list(S3_FILE_CACHE.keys())
        elif scope: # Assume scope is an agent_name
            for key in S3_FILE_CACHE.keys():
                if scope in key:
                    keys_to_delete.append(key)
        
        if not keys_to_delete:
             logger.info(f"No matching cache keys found to clear for scope '{scope}'.")
             return jsonify({"status": "noop", "message": "No matching cache keys found to clear."}), 200

        deleted_count = 0
        for key in keys_to_delete:
            if key in S3_FILE_CACHE:
                del S3_FILE_CACHE[key]
                deleted_count += 1
        
    logger.info(f"CACHE INVALIDATED: Cleared {deleted_count} cache entries for scope '{scope}'.")
    return jsonify({
        "status": "success",
        "message": f"Successfully cleared {deleted_count} cache entries for agent '{scope}'.",
        "cleared_keys": keys_to_delete
    }), 200


@app.route('/api/s3/download', methods=['GET'])
@supabase_auth_required(agent_required=False)
def download_s3_document(user: SupabaseUser):
    logger.info(f"Received request /api/s3/download from user: {user.id}") 
    s3_key = request.args.get('s3Key'); filename_param = request.args.get('filename')
    if not s3_key: return jsonify({"error": "Missing 's3Key' query parameter"}), 400

    # IDOR Prevention: Verify ownership of the S3 key
    if not verify_s3_key_ownership(s3_key, user):
        return jsonify({"error": "Access denied to this resource"}), 403

    s3 = get_s3_client(); aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket: return jsonify({"error": "S3 client or bucket not configured"}), 500
    try:
        s3_object = s3.get_object(Bucket=aws_s3_bucket, Key=s3_key)
        download_filename = filename_param or os.path.basename(s3_key)
        return Response(
            s3_object['Body'].iter_chunks(),
            mimetype=s3_object.get('ContentType', 'application/octet-stream'),
            headers={"Content-Disposition": f"attachment;filename=\"{download_filename}\""} 
        )
    except s3.exceptions.NoSuchKey: 
        logger.warning(f"S3 Download: File not found at key: {s3_key}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e: 
        logger.error(f"Error downloading S3 object '{s3_key}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error downloading S3 object"}), 500

@app.route('/api/chat', methods=['POST'])
@supabase_auth_required(agent_required=True)
def handle_chat(user: SupabaseUser):
    """
    Handles chat requests by streaming responses from the Anthropic API.
    This function extracts request data and then uses a generator to stream the response,
    ensuring the request context is not lost.
    """
    request_start_time = time.time()
    logger.info(f"Received POST request to /api/chat from user: {user.id}")

    if not anthropic_client:
        logger.error("Chat fail: Anthropic client not initialized.")
        return jsonify({"error": "AI service unavailable"}), 503

    try:
        data = g.get('json_data', {}) # Use g instead of request.json
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing 'messages' in request body"}), 400
    except Exception as e:
        logger.error(f"Error accessing request data: {e}")
        return jsonify({"error": "Invalid request data"}), 400

    # Extract all necessary data from the request *before* starting the stream
    agent_name = data.get('agent')
    event_id = data.get('event', '0000')
    transcript_listen_mode = data.get('transcriptListenMode', 'latest')
    saved_transcript_memory_mode = data.get('savedTranscriptMemoryMode', 'disabled')
    transcription_language_setting = data.get('transcriptionLanguage', 'any')
    # Get default model from the existing LLM_MODEL_NAME env var, with a hardcoded fallback.
    default_model = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-20250514")
    model_selection = data.get('model', default_model)
    temperature = data.get('temperature', 0.7) # Default temperature
    incoming_messages = data.get('messages', [])
    chat_session_id_log = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S'))
    # Extract new individual memory toggle data
    individual_memory_toggle_states = data.get('individualMemoryToggleStates', {})
    saved_transcript_summaries = data.get('savedTranscriptSummaries', [])
    individual_raw_transcript_toggle_states = data.get('individualRawTranscriptToggleStates', {})
    raw_transcript_files = data.get('rawTranscriptFiles', [])
    initial_context_for_aicreator = data.get('initialContext')
    current_draft_content_for_aicreator = data.get('currentDraftContent')
    disable_retrieval = data.get('disableRetrieval', False) # New
    signal_bias = data.get('signalBias', 'medium') # 'low' | 'medium' | 'high'

    def generate_stream():
        """
        A generator function that prepares data, calls the LLM, and yields the response stream.
        This runs within the `stream_with_context` to handle the response generation.
        """
        try:
            # WIZARD DETECTION: A wizard session is defined by disabling retrieval and providing initial context.
            is_wizard = disable_retrieval and initial_context_for_aicreator is not None
            
            effective_transcript_listen_mode = "none" if is_wizard else transcript_listen_mode

            logger.info(f"Chat stream started for Agent: {agent_name}, Event: {event_id}, User: {user.id}, Wizard Mode: {is_wizard}")
            logger.info(f"Stream settings - Listen: {effective_transcript_listen_mode}, Memory: {saved_transcript_memory_mode}, Language: {transcription_language_setting}, Individual toggles: {len(individual_memory_toggle_states)} items")

            # --- System Prompt and Context Assembly (UNIFIED & RE-ORDERED) ---
            s3_load_start_time = time.time()
            final_system_prompt = ""

            if agent_name == '_aicreator':
                # Special instructions for the agent creator wizard
                # ... (this logic remains the same)
                ai_creator_instructions = """
\n\n## Core Mission: AI Agent Prompt Creator
You are an expert AI assistant who helps users draft high-quality system prompts for other AI agents. Engage in a collaborative conversation to refine the user's ideas into a precise and effective prompt.

## Critical Instructions & Rules
1.  **Prioritize the User's Draft:** The user's current work-in-progress is provided in a `<current_draft>` block. This is your **single source of truth** for the prompt's content. When the user says "my edits", "this version", or "the draft", they are referring to the content of this block. Your primary goal is to refine THIS DRAFT.
2.  **Output Format Mandate:** When you generate a new version of the system prompt, you MUST follow this format precisely:
    a.  Provide a brief, conversational message explaining your changes.
    b.  Immediately after your message, provide a markdown code block containing the new system prompt.

**EXAMPLE OUTPUT:**
I've updated the prompt to include the dragon's personality as you requested. It now has a more defined, wise character.

```
You are a wise and ancient dragon. You have seen empires rise and fall. You speak in a measured, calm tone, offering cryptic but helpful advice.
```

**FORMATTING RULES:**
- The code block MUST start with ``` and end with ```.
- Inside the code block, provide ONLY the complete system prompt content.
- Do NOT add any extra text, comments, or formatting inside the code block.
- The content inside the code block should be clean, well-formatted markdown that can be used directly as a system prompt.
"""
                core_directive_content = (get_latest_system_prompt(agent_name) or "You are a helpful assistant.") + ai_creator_instructions
                if initial_context_for_aicreator:
                    core_directive_content += f"\n\n<document_context>\n{initial_context_for_aicreator}\n</document_context>"
                if current_draft_content_for_aicreator:
                    core_directive_content += f"\n\n## User's Current Draft (Authoritative)\n<current_draft>\n{current_draft_content_for_aicreator}\n</current_draft>"
                
                final_system_prompt += f"=== CORE DIRECTIVE ===\n{core_directive_content}\n=== END CORE DIRECTIVE ==="
            else:
                # Unified prompt builder path for regular agents (incl. event layer)
                feature_event_prompts = os.getenv('FEATURE_EVENT_PROMPTS', 'true').lower() == 'true'
                feature_event_docs = os.getenv('FEATURE_EVENT_DOCS', 'false').lower() == 'true'
                try:
                    final_system_prompt = prompt_builder(
                        agent=agent_name,
                        event=event_id or os.getenv('DEFAULT_EVENT', '0000'),
                        user_context=None,
                        retrieval_hints=None,
                        feature_event_prompts=feature_event_prompts,
                        feature_event_docs=feature_event_docs,
                    )
                except Exception as e:
                    logger.error(f"PromptBuilder error; falling back: {e}", exc_info=True)
                    base_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
                    final_system_prompt = f"=== CORE DIRECTIVE ===\n{base_system_prompt}\n=== END CORE DIRECTIVE ==="

            # --- Part 2: Dynamic Task-Specific Context (RAG) ---
            if not is_wizard:
                rag_usage_instructions = ("\n\n=== INSTRUCTIONS FOR USING RETRIEVED CONTEXT ===\n"
                                          "1. **Prioritize Info Within `[Retrieved Context]`:** Base answer primarily on info in `[Retrieved Context]` block below, if relevant. \n"
                                          "2. **Assess Timeliness:** Each source has an `(Age: ...)` tag. Use this to assess relevance. More recent information is generally more reliable, unless it's a 'Core Memory' which is timeless. \n"
                                          "3. **Direct Extraction for Lists/Facts:** If user asks for list/definition/specific info explicit in `[Retrieved Context]`, present that info directly. Do *not* state info missing if clearly provided. \n"
                                          "4. **Cite Sources:** Remember cite source file name using Markdown footnotes (e.g., `[^1]`) for info from context, list sources under `### Sources`. \n"
                                          "5. **Synthesize When Necessary:** If query requires combining info or summarizing, do so, but ground answer in provided context. \n"
                                          "6. **Acknowledge Missing Info Appropriately:** Only state info missing if truly absent from context and relevant.\n"
                                          "=== END INSTRUCTIONS FOR USING RETRIEVED CONTEXT ===")
                final_system_prompt += rag_usage_instructions

                # Provide a natural index of available meetings (saved/) for the LLM
                try:
                    saved_items = list_saved_transcripts(agent_name, event_id)
                except Exception as _e_idx:
                    saved_items = []
                if saved_items:
                    max_index_items = int(os.getenv('MEETINGS_INDEX_LIMIT', '40'))
                    lines = []
                    for itm in saved_items[:max_index_items]:
                        date_str = itm.get('meeting_date') or 'unknown-date'
                        fname = itm.get('filename') or 'unknown'
                        summary_id = f"summary_{fname}"
                        lines.append(f"- {date_str} — {fname} (id: {summary_id})")
                    final_system_prompt += "\n\n=== AVAILABLE MEETINGS ===\n" + "\n".join(lines) + "\n=== END AVAILABLE MEETINGS ==="

                last_user_message_obj = next((msg for msg in reversed(incoming_messages) if msg.get("role") == "user"), None)
                last_actual_user_message_for_rag = last_user_message_obj.get("content") if last_user_message_obj else None
                
                retrieved_docs_for_reinforcement = []
                rag_start_time = time.time()
                try:
                    normalized_query = (last_actual_user_message_for_rag or "").strip().lower().rstrip('.!?')
                    if normalized_query and normalized_query not in SIMPLE_QUERIES_TO_BYPASS_RAG:
                        retriever = RetrievalHandler(
                            index_name="river", agent_name=agent_name, session_id=chat_session_id_log,
                            event_id=event_id, anthropic_api_key=get_api_key(agent_name, 'anthropic'),
                            openai_api_key=get_api_key(agent_name, 'openai')
                        )
                        # Event-scoped tiered retrieval with caps and MMR
                        tier_caps_env = os.getenv('RETRIEVAL_TIER_CAPS', '7,6,6,4').split(',')
                        try:
                            tier_caps = [int(x.strip()) for x in tier_caps_env]
                        except Exception:
                            tier_caps = [7, 6, 6, 4]
                        mmr_lambda = float(os.getenv('RETRIEVAL_MMR_LAMBDA', '0.6'))
                        mmr_k = int(os.getenv('RETRIEVAL_MMR_K', '23'))
                        allow_t3_low = os.getenv('ALLOW_T3_ON_LOW', 'false').lower() == 'true'
                        include_t3 = True
                        if (signal_bias or 'medium') == 'low' and not allow_t3_low:
                            include_t3 = False
                        # Optional: if user references a specific meeting (by filename/id or unique date), restrict retrieval
                        metadata_filter = None
                        try:
                            mf = None
                            if saved_items and last_actual_user_message_for_rag:
                                msg_l = last_actual_user_message_for_rag.lower()
                                # 1) Filename/id match
                                for itm in saved_items:
                                    fname = (itm.get('filename') or '').lower()
                                    summary_id = f"summary_{fname}"
                                    if fname and (fname in msg_l or summary_id.lower() in msg_l):
                                        mf = {"file_name": {"$eq": summary_id}}
                                        break
                                # 2) Unique date match (YYYY-MM-DD)
                                if not mf:
                                    import re
                                    dates = re.findall(r"(\d{4}-\d{2}-\d{2})", msg_l)
                                    for d in dates:
                                        matches = [itm for itm in saved_items if itm.get('meeting_date') == d]
                                        if len(matches) == 1:
                                            fname = matches[0].get('filename')
                                            if fname:
                                                mf = {"file_name": {"$eq": f"summary_{fname}"}}
                                                break
                            metadata_filter = mf
                            if mf:
                                logger.info(f"Applying metadata filter for exact meeting retrieval: {mf}")
                        except Exception as _e_mf:
                            metadata_filter = None

                        retrieved_docs = retriever.get_relevant_context_tiered(
                            query=last_actual_user_message_for_rag,
                            tier_caps=tier_caps,
                            mmr_k=mmr_k,
                            mmr_lambda=mmr_lambda,
                            include_t3=include_t3,
                            metadata_filter=metadata_filter,
                        )
                        retrieved_docs_for_reinforcement = retrieved_docs
                        if retrieved_docs:
                            # Add disclosure if non-T1 sources present
                            non_t1_events = sorted({
                                str(d.metadata.get('event_id'))
                                for d in retrieved_docs
                                if str(d.metadata.get('event_id', '0000')) not in [str(event_id or '0000')]
                            })
                            if non_t1_events:
                                disclosure = (
                                    f"Context includes shared {agent_name}/0000 and/or other sub-teams: {', '.join(non_t1_events)}. "
                                    f"Treat as background unless it directly affects {event_id or '0000'}."
                                )
                                final_system_prompt += f"\n\n[DISCLOSURE] {disclosure}"

                            def src_label(md: dict) -> str:
                                return md.get('source_label') or ''

                            items = [
                                (
                                    f"--- START Context Source: {d.metadata.get('file_name','Unknown')} {src_label(d.metadata)} "
                                    f"(Age: {d.metadata.get('age_display', 'Unknown')}, Score: {d.metadata.get('score',0):.2f}) ---\n"
                                    f"{d.page_content}\n--- END Context Source: {d.metadata.get('file_name','Unknown')} ---"
                                )
                                for d in retrieved_docs
                            ]
                            final_system_prompt += "\n\n=== RETRIEVED CONTEXT ===\n" + "\n\n".join(items) + "\n=== END RETRIEVED CONTEXT ==="
                except Exception as e:
                    logger.error(f"Unexpected RAG error: {e}", exc_info=True)
                    final_system_prompt += "\n\n=== RETRIEVED CONTEXT ===\n[Note: Error retrieving documents via RAG]\n=== END RETRIEVED CONTEXT ==="
                    # Ensure the variable is initialized even if an error occurred
                    if 'retrieved_docs_for_reinforcement' not in locals():
                        retrieved_docs_for_reinforcement = []
                logger.info(f"[PERF] RAG processing took {time.time() - rag_start_time:.4f}s")
            
            # --- Part 3: Static Knowledge Base & Historical Context ---
            historical_context_parts = []
            if not is_wizard:
                event_context = get_latest_context(agent_name, event_id)
                if event_context:
                    historical_context_parts.append(f"=== HISTORICAL CONTEXT ===\n{event_context}\n=== END HISTORICAL CONTEXT ===")
            
            memory_update_instructions = """=== INSTRUCTIONS FOR MEMORY UPDATES ===
When you identify information that should be permanently stored in your agent documentation, use this exact format:

[DOC_UPDATE_PROPOSAL]
{
  "doc_name": "filename.md",
  "content": "Complete document content here",
  "justification": "Brief explanation why this update is needed"
}

**JSON Schema (Draft 7):**```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["doc_name", "content"],
  "properties": {
    "doc_name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+\\.md$"},
    "content": {"type": "string"},
    "justification": {"type": "string"}
  }
}
```

**Requirements:**
- Use valid JSON after [DOC_UPDATE_PROPOSAL] prefix
- Include complete document content, not just additions
- Do not add content unless specifically asked to by user
- Choose descriptive filenames ending in .md
=== END INSTRUCTIONS FOR MEMORY UPDATES ==="""
            historical_context_parts.append(memory_update_instructions)

            feature_event_docs = os.getenv('FEATURE_EVENT_DOCS', 'false').lower() == 'true'
            if not is_wizard:
                agent_docs = get_agent_docs(agent_name)
                if agent_docs:
                    historical_context_parts.append(f"=== AGENT DOCUMENTATION ===\n{agent_docs}\n=== END AGENT DOCUMENTATION ===")
                if feature_event_docs and event_id:
                    event_docs = get_event_docs(agent_name, event_id)
                    if event_docs:
                        historical_context_parts.append(
                            f"=== EVENT DOCUMENTATION ===\n{event_docs}\n=== END EVENT DOCUMENTATION ==="
                        )
            
                meeting_content_instructions = (
                    "\n\n=== INSTRUCTIONS FOR USING MEETING CONTENT ===\n"
                    "You have access to meeting information in multiple formats, prioritize as follows:\n\n"
                    "**1. ANALYZED MEETING SUMMARIES** (Highest Priority):\n"
                    "- Multi-layered AI-generated analysis with Business Context, Organizational Dynamics, Strategic Implications, and Wisdom Learning\n"
                    "- Marked with **[Meeting Analysis - {event} - {date}]** headers\n"
                    "- Include meeting date/time for temporal context\n"
                    "- These are analyses generated by AI from the transcript; they are NOT verbatim participant statements and must not be attributed to speakers\n"
                    "- These provide the richest insights and should be your primary source\n\n"
                    "**2. SAVED TRANSCRIPT SUMMARIES** (Secondary):\n"
                    "- Legacy flat summaries for validation or when analyzed summaries unavailable\n"
                    "- These are AI-generated summaries derived from transcripts; they are NOT verbatim participant statements and must not be attributed to speakers\n"
                    "- Use to cross-reference or fill gaps in analyzed summaries\n\n"
                    "**3. RAW MEETING TRANSCRIPTS** (Detailed Reference):\n"
                    "- `=== HISTORICAL MEETING TRANSCRIPT ===`: Full transcript before most recent updates\n"
                    "- `=== LATEST MEETING TRANSCRIPT ===`: Most recent, real-time additions\n"
                    "- Use for specific quotes or when summaries lack detail\n\n"
                    "**SYNTHESIS RULE:** When multiple formats exist for the same meeting, use analyzed summaries as primary source, validate with other formats, and note any significant discrepancies.\n"
                    "=== END INSTRUCTIONS FOR USING MEETING CONTENT ==="
                )
                historical_context_parts.append(meeting_content_instructions)

                # Add summaries to historical context
                if saved_transcript_memory_mode == 'all' or (saved_transcript_memory_mode == 'some' and individual_memory_toggle_states):
                    summaries_to_add = []
                    if saved_transcript_memory_mode == 'all':
                        summaries_to_add = get_transcript_summaries(agent_name, event_id)
                    else: # 'some'
                        all_summaries = get_transcript_summaries(agent_name, event_id)
                        summaries_to_add = []
                        logger.info(f"Filtering summaries in 'some' mode. Available summaries: {len(all_summaries)}, Toggle states: {individual_memory_toggle_states}")
                        for s in all_summaries:
                            # Construct the summary S3 key that matches what frontend sends
                            summary_filename = s.get("metadata", {}).get("summary_filename", "")
                            if summary_filename:
                                summary_s3_key = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/summarized/{summary_filename}"
                                is_selected = individual_memory_toggle_states.get(summary_s3_key, False)
                                logger.info(f"Summary check: filename='{summary_filename}', constructed_key='{summary_s3_key}', selected={is_selected}")
                                if is_selected:
                                    summaries_to_add.append(s)
                    
                    # Add ANALYZED MEETING SUMMARIES (new multi-agent summaries from vector DB)
                    analyzed_summaries_context = ""
                    try:
                        # Multi-pass retrieval: Phase 1 - Metadata filtering + Phase 2 - Semantic search
                        meeting_retriever = RetrievalHandler(
                            index_name="river", 
                            agent_name=agent_name,
                            event_id=event_id,
                            anthropic_api_key=get_api_key(agent_name, 'anthropic'),
                            openai_api_key=get_api_key(agent_name, 'openai')
                        )
                        
                        # Build metadata filter for meeting summaries from this event
                        metadata_filter = {
                            "content_category": "meeting_summary",
                            "event_id": event_id,
                            "analysis_type": "multi_agent"
                        }
                        
                        # Retrieve meeting summaries with contextual ranking (recent first)
                        meeting_docs = meeting_retriever.get_relevant_context_tiered(
                            query="recent meeting summaries and decisions", 
                            tier_caps=[5, 0, 0],  # Focus on event-specific content only
                            mmr_lambda=0.3,  # Lower diversity, higher relevance for meetings
                            mmr_k=8,         # Get up to 8 meeting summary chunks
                            include_t3=False,  # No cross-event contamination
                            metadata_filter=metadata_filter  # Apply our content category filter
                        )
                        
                        if meeting_docs:
                            analyzed_summaries_context = (
                                "=== ANALYZED MEETING SUMMARIES ===\n"
                                "Note: The following sections are AI-generated analyses of the meetings.\n"
                                "They are not quotes or statements from participants and should not be attributed to any speaker.\n\n"
                            )
                            
                            # Group and sort by meeting date (recent first)
                            meeting_summaries = {}
                            for doc in meeting_docs:
                                meeting_date = doc.metadata.get('meeting_date', 'unknown')
                                file_name = doc.metadata.get('file_name', 'unknown')
                                
                                if meeting_date not in meeting_summaries:
                                    meeting_summaries[meeting_date] = {
                                        'date': meeting_date,
                                        'file_name': file_name,
                                        'content_chunks': []
                                    }
                                meeting_summaries[meeting_date]['content_chunks'].append(doc.page_content)
                            
                            # Sort by date (recent first) and format
                            for meeting_date in sorted(meeting_summaries.keys(), reverse=True):
                                summary_data = meeting_summaries[meeting_date]
                                analyzed_summaries_context += f"**[Meeting Analysis - {event_id} - {summary_data['date']}]**\n"
                                analyzed_summaries_context += f"**Source:** {summary_data['file_name']}\n\n"
                                
                                # Combine content chunks
                                combined_content = "\n\n---\n\n".join(summary_data['content_chunks'])
                                analyzed_summaries_context += f"{combined_content}\n\n"
                            
                            analyzed_summaries_context += "=== END ANALYZED MEETING SUMMARIES ==="
                            historical_context_parts.append(analyzed_summaries_context)
                            
                            logger.info(f"Retrieved {len(meeting_docs)} meeting summary chunks for {event_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to retrieve meeting summaries for {event_id}: {e}")
                        # Fallback to empty - don't break the flow
                    
                    if summaries_to_add:
                        summaries_context_str = (
                            "=== SAVED TRANSCRIPT SUMMARIES ===\n"
                            "Note: The following are AI-generated summaries derived from meeting transcripts.\n"
                            "They are not quotes or statements from participants and should not be attributed to any speaker.\n\n"
                        )
                        for summary_doc in summaries_to_add:
                            summary_filename = summary_doc.get("metadata", {}).get("summary_filename", "unknown_summary.json")
                            summaries_context_str += f"### Summary: {summary_filename}\n{json.dumps(summary_doc, indent=2, ensure_ascii=False)}\n\n"
                        summaries_context_str += "=== END SAVED TRANSCRIPT SUMMARIES ==="
                        historical_context_parts.append(summaries_context_str)

            if historical_context_parts:
                final_system_prompt += "\n\n" + "\n\n".join(historical_context_parts)
            
                        # --- Prepare Messages Array & Historical Transcripts (UNIFIED LOGIC) ---
            final_llm_messages = []
            if not is_wizard and effective_transcript_listen_mode != 'none':
                from utils.s3_utils import read_file_content
                
                relevant_transcripts_meta = []
                if effective_transcript_listen_mode == 'latest':
                    from utils.transcript_utils import get_latest_transcript_file
                    latest_key = get_latest_transcript_file(agent_name, event_id)
                    if latest_key:
                        s3 = get_s3_client(); aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
                        if s3 and aws_s3_bucket:
                            try:
                                head_obj = s3.head_object(Bucket=aws_s3_bucket, Key=latest_key)
                                relevant_transcripts_meta.append({'Key': latest_key, 'LastModified': head_obj['LastModified']})
                            except Exception: pass
                
                elif effective_transcript_listen_mode == 'all':
                    from utils.s3_utils import list_s3_objects_metadata
                    transcript_prefix = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/"
                    all_files_meta = list_s3_objects_metadata(transcript_prefix)
                    relevant_transcripts_meta = [f for f in all_files_meta if not os.path.basename(f['Key']).startswith('rolling-') and f['Key'].endswith('.txt')]
                
                elif effective_transcript_listen_mode == 'some':
                    # Filter the frontend-provided file list based on the toggle states
                    relevant_transcripts_meta = [
                        f for f in raw_transcript_files 
                        if f.get('s3Key') and individual_raw_transcript_toggle_states.get(f['s3Key'], False)
                    ]

                if relevant_transcripts_meta:
                    # Sort by date to handle both datetime objects from S3 and string ISO timestamps from frontend
                    def get_sort_key(meta_dict):
                        return meta_dict.get('LastModified') or meta_dict.get('lastModified') or ''
                    relevant_transcripts_meta.sort(key=get_sort_key)
                    
                    # Pop the latest for the user message
                    latest_transcript_meta = relevant_transcripts_meta.pop()
                    latest_key = latest_transcript_meta.get('s3Key') or latest_transcript_meta.get('Key')
                    if latest_key:
                        latest_content = read_file_content(latest_key, "latest transcript")
                        if latest_content:
                            latest_block = f"=== LATEST MEETING TRANSCRIPT ===\n{latest_content}\n=== END LATEST MEETING TRANSCRIPT ==="
                            final_llm_messages.append({'role': 'user', 'content': latest_block})

                    # The rest are historical, add them to the system prompt
                    if relevant_transcripts_meta:
                        historical_contents = []
                        for old_meta in relevant_transcripts_meta:
                            old_key = old_meta.get('s3Key') or old_meta.get('Key')
                            old_name = old_meta.get('name') or os.path.basename(old_key or '')
                            if old_key:
                                content = read_file_content(old_key, "historical transcript")
                                if content:
                                    historical_contents.append(f"--- START Transcript Source: {old_name} ---\n{content}\n--- END Transcript Source: {old_name} ---")
                        if historical_contents:
                            historical_block = "\n\n".join(historical_contents)
                            final_system_prompt += f"\n\n=== HISTORICAL MEETING TRANSCRIPT (OLDEST FIRST) ===\n{historical_block}\n=== END HISTORICAL MEETING TRANSCRIPT ==="

            # --- Final State & Timestamped History ---
            if not is_wizard:
                current_utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                final_system_prompt += f"\n\n=== CURRENT TIME ===\nYour internal clock shows the current date and time is: **{current_utc_time}**.\n=== END CURRENT TIME ==="

            s3_load_time = time.time() - s3_load_start_time
            logger.info(f"[PERF] S3 Prompts/Context loaded in {s3_load_time:.4f}s. Final prompt length: {len(final_system_prompt)}")
            
            if agent_name == '_aicreator':
                llm_messages_from_client = [{"role": msg["role"], "content": msg["content"]} for msg in incoming_messages if msg.get("id") != 'initial-wizard-prompt' and "role" in msg and "content" in msg]
                final_llm_messages.extend(llm_messages_from_client)
            else:
                last_user_message_obj = next((msg for msg in reversed(incoming_messages) if msg.get("role") == "user"), None)
                last_actual_user_message_for_rag = last_user_message_obj.get("content") if last_user_message_obj else ""
                timestamped_history_lines = ["This is the conversation history with timestamps for your reference. Do not replicate this format in your responses."]
                for msg in incoming_messages:
                    if msg is last_user_message_obj: continue
                    if msg.get("role") in ["user", "assistant"]:
                        timestamp = msg.get("createdAt", datetime.now(timezone.utc).isoformat())
                        role = msg.get("role"); content = msg.get("content")
                        timestamped_history_lines.append(f"[{timestamp}] {role}: {content}")
                history_context_block = "\n".join(timestamped_history_lines)
                final_llm_messages.append({"role": "user", "content": f"=== CHAT HISTORY ===\n{history_context_block}\n=== END CHAT HISTORY ==="})
                final_llm_messages.append({"role": "user", "content": last_actual_user_message_for_rag})

            # --- Call LLM and Stream ---
            max_tokens_for_call = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))
            llm_provider = 'anthropic' # Default
            if model_selection == 'gpt-oss-120b' or model_selection == 'gpt-oss-20b': llm_provider = 'groq'
            elif model_selection.startswith('gemini'): llm_provider = 'google'
            elif model_selection.startswith('gpt-'): llm_provider = 'openai'
            llm_api_key = get_api_key(agent_name, llm_provider)

            try:
                # PERF marker: log immediately before sending to LLM
                ctx_count = len(retrieved_docs_for_reinforcement) if 'retrieved_docs_for_reinforcement' in locals() and retrieved_docs_for_reinforcement else 0
                logger.info(f"[PERF] LLM dispatch: provider={llm_provider}, model={model_selection}, ctx_count={ctx_count}, msgs={len(final_llm_messages)}, max_tokens={max_tokens_for_call}")
                _llm_t0 = time.perf_counter()
                _first_token_logged = False
                if llm_provider == 'groq':
                    api_model_name = "openai/gpt-oss-120b" if model_selection == 'gpt-oss-120b' else "openai/gpt-oss-20b"
                    groq_stream = _call_groq_stream_with_retry(model_name=api_model_name, max_tokens=max_tokens_for_call, system_instruction=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key, temperature=temperature)
                    for chunk in groq_stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            if not _first_token_logged:
                                logger.info(f"[PERF] LLM first_token: {time.perf_counter() - _llm_t0:.3f}s provider=groq model={api_model_name}")
                                _first_token_logged = True
                            yield f"data: {json.dumps({'delta': chunk.choices[0].delta.content})}\n\n"
                elif llm_provider == 'google':
                    gemini_stream = _call_gemini_stream_with_retry(model_name=model_selection, max_tokens=max_tokens_for_call, system_instruction=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key, temperature=temperature)
                    for chunk in gemini_stream:
                        if chunk.parts:
                            if not _first_token_logged:
                                logger.info(f"[PERF] LLM first_token: {time.perf_counter() - _llm_t0:.3f}s provider=google model={model_selection}")
                                _first_token_logged = True
                            yield f"data: {json.dumps({'delta': chunk.text})}\n\n"
                elif llm_provider == 'openai':
                    openai_stream = _call_openai_stream_with_retry(model_name=model_selection, max_tokens=max_tokens_for_call, system_instruction=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key, temperature=temperature)
                    with openai_stream as stream:
                        for event in stream:
                            if getattr(event, "type", "") == "response.output_text.delta":
                                if not _first_token_logged:
                                    logger.info(f"[PERF] LLM first_token: {time.perf_counter() - _llm_t0:.3f}s provider=openai model={model_selection}")
                                    _first_token_logged = True
                                yield f"data: {json.dumps({'delta': event.delta})}\n\n"
                else: # Anthropic
                    stream_manager = _call_anthropic_stream_with_retry(model=model_selection, max_tokens=max_tokens_for_call, system=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key)
                    with stream_manager as stream:
                        for chunk in stream:
                            if chunk.type == "content_block_delta":
                                if not _first_token_logged:
                                    logger.info(f"[PERF] LLM first_token: {time.perf_counter() - _llm_t0:.3f}s provider=anthropic model={model_selection}")
                                    _first_token_logged = True
                                yield f"data: {json.dumps({'delta': chunk.delta.text})}\n\n"
                
                # Ensure retrieved_docs_for_reinforcement is defined
                if 'retrieved_docs_for_reinforcement' not in locals():
                    retrieved_docs_for_reinforcement = []
                doc_ids_for_reinforcement = [doc.metadata.get('vector_id') for doc in retrieved_docs_for_reinforcement if doc.metadata.get('vector_id')]
                sse_done_data = {'done': True, 'retrieved_doc_ids': doc_ids_for_reinforcement}
                yield f"data: {json.dumps(sse_done_data)}\n\n"
                logger.info(f"Stream for chat with agent {agent_name} (model: {model_selection}) completed successfully.")

            except CircuitBreakerOpen as e:
                logger.error(f"Circuit breaker is open. Aborting stream. Error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except (google_exceptions.GoogleAPICallError, OpenAI_APIError, GroqAPIError, AnthropicError, RetryError) as e:
                logger.error(f"LLM API error after retries: {e}", exc_info=True)
                if isinstance(e, (google_exceptions.GoogleAPICallError, RetryError)): gemini_circuit_breaker.record_failure()
                if isinstance(e, (OpenAI_APIError, RetryError)): openai_circuit_breaker.record_failure()
                if isinstance(e, (GroqAPIError, RetryError)): groq_circuit_breaker.record_failure()
                if isinstance(e, (AnthropicError, RetryError)): anthropic_circuit_breaker.record_failure()
                yield f"data: {json.dumps({'error': f'Assistant ({llm_provider.capitalize()}) API Error: {str(e)}'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'An internal server error occurred during the stream.'})}\n\n"
    
    # Start the streaming response
    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

def sync_agents_from_s3_to_supabase():
    client = get_supabase_client()
    if not client:
        logger.warning("Agent Sync: Supabase client not available. Skipping.")
        return
    logger.info("Agent Sync: Starting synchronization from S3 to Supabase...")
    s3_agent_names = list_agent_names_from_s3()
    if s3_agent_names is None: 
        logger.error("Agent Sync: Failed to list agent names from S3. Aborting.")
        return
    if not s3_agent_names: 
        logger.info("Agent Sync: No agent directories found in S3.")
        return
    try:
        response = client.table("agents").select("name").execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Agent Sync: Error querying agents from Supabase: {response.error}")
            return
        db_agent_names = {agent['name'] for agent in response.data}
        logger.info(f"Agent Sync: Found {len(db_agent_names)} agents in Supabase.")
    except Exception as e: 
        logger.error(f"Agent Sync: Unexpected error querying Supabase agents: {e}", exc_info=True)
        return
    missing_agents = [name for name in s3_agent_names if name not in db_agent_names]
    if not missing_agents: 
        logger.info("Agent Sync: Supabase 'agents' table is up-to-date with S3 directories.")
        return
    logger.info(f"Agent Sync: Found {len(missing_agents)} agents in S3 to add to Supabase: {missing_agents}")
    agents_to_insert = [{'name': name, 'description': f'Agent discovered from S3 path: {name}'} for name in missing_agents]
    try:
        insert_response = client.table("agents").insert(agents_to_insert).execute()
        if hasattr(insert_response, 'error') and insert_response.error:
            logger.error(f"Agent Sync: Error inserting agents into Supabase: {insert_response.error}")
        elif insert_response.data: 
            logger.info(f"Agent Sync: Successfully inserted {len(insert_response.data)} new agents into Supabase.")
        else: 
            logger.warning(f"Agent Sync: Insert call succeeded but reported 0 rows inserted. Check response: {insert_response}")
    except Exception as e: 
        logger.error(f"Agent Sync: Unexpected error inserting agents: {e}", exc_info=True)
    logger.info("Agent Sync: Synchronization finished.")

# Chat History Management Endpoints
@app.route('/api/memory/reinforce', methods=['POST'])
@supabase_auth_required(agent_required=True)
def reinforce_memory_route(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    doc_ids = data.get('doc_ids')

    if not all([agent_name, doc_ids]):
        return jsonify({"error": "agent and doc_ids are required"}), 400
    
    if not isinstance(doc_ids, list):
        return jsonify({"error": "doc_ids must be a list"}), 400

    logger.info(f"Reinforcement request for agent '{agent_name}', docs: {doc_ids}")

    try:
        # We need to initialize a retriever to get access to the reinforce_memories method
        # This is lightweight as it doesn't perform a query.
        retriever = RetrievalHandler(
            index_name="river",
            agent_name=agent_name,
            anthropic_api_key=get_api_key(agent_name, 'anthropic'),
            openai_api_key=get_api_key(agent_name, 'openai')
        )
        
        # Create dummy Document objects with just the necessary metadata
        docs_to_reinforce = [Document(page_content="", metadata={"vector_id": doc_id}) for doc_id in doc_ids]
        
        retriever.reinforce_memories(docs_to_reinforce)
        
        return jsonify({"status": "success", "message": f"Reinforced {len(doc_ids)} memories."}), 200

    except Exception as e:
        logger.error(f"Error during memory reinforcement for agent '{agent_name}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during reinforcement."}), 500

# --- New "Intelligent Log" Memory Endpoints ---
from utils.log_enricher import enrich_chat_log
from utils.embedding_handler import EmbeddingHandler

@app.route('/api/recordings/embed', methods=['POST'])
@supabase_auth_required(agent_required=True)
def embed_recording_route(user: SupabaseUser):
    data = g.get('json_data', {})
    s3_key = data.get('s3Key')
    agent_name = data.get('agentName')
    event_id_param = data.get('eventId') or data.get('event') or '0000'

    if not all([s3_key, agent_name]):
        return jsonify({"error": "s3Key and agentName are required"}), 400

    logger.info(f"Embedding request for s3Key: {s3_key}, agent: {agent_name}")

    try:
        # Use the new function to get content AND metadata
        from utils.s3_utils import read_file_content_with_metadata
        s3_data = read_file_content_with_metadata(s3_key, f"S3 file for embedding ({s3_key})")
        if not s3_data or 'content' not in s3_data:
            return jsonify({"error": "File not found or could not be read from S3"}), 404

        content = s3_data['content']
        last_modified_timestamp = s3_data.get('LastModified')

        # MODIFIED: Use a shared index name and the agent_name as the namespace
        embedding_handler = EmbeddingHandler( 
            index_name="river",
            namespace=agent_name
        )
        
        # A virtual filename for metadata purposes
        virtual_filename = os.path.basename(s3_key)
        
        metadata_for_embedding = {
            "agent_name": agent_name,
            "source": "recording",
            "source_type": "doc",
            "file_name": virtual_filename,
            "s3_key": s3_key,
            "event_id": event_id_param,
        }

        # Add the 'created_at' timestamp from S3's LastModified metadata
        if last_modified_timestamp:
            # Convert datetime object to ISO 8601 string format, which the re-ranker expects
            metadata_for_embedding['created_at'] = last_modified_timestamp.isoformat()
            logger.info(f"Using S3 LastModified as 'created_at' timestamp for embedding: {metadata_for_embedding['created_at']}")

        upsert_success = embedding_handler.embed_and_upsert(content, metadata_for_embedding)

        if not upsert_success:
            logger.error(f"Failed to embed and upsert recording {s3_key} to Pinecone.")
            return jsonify({"error": "Failed to index recording for retrieval"}), 500

        logger.info(f"Successfully embedded recording from {s3_key}.")
        return jsonify({"status": "success", "message": "Recording embedded successfully."}), 200

    except Exception as e:
        logger.error(f"Error during recording embedding for s3Key {s3_key}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during embedding."}), 500


@app.route('/api/memory/save-chat', methods=['POST'])
@supabase_auth_required(agent_required=True)
def save_chat_memory_log(user: SupabaseUser):
    """
    Creates or updates an "Intelligent Log" for a chat session.
    This is the new primary pipeline for saving memories.
    """
    data = g.get('json_data', {})
    agent_name = data.get('agentName') # Note: Frontend might send agentName
    if not agent_name: agent_name = data.get('agent') # Fallback to 'agent'
    
    messages = data.get('messages', [])
    session_id = data.get('sessionId')
    event_id_param = data.get('eventId') or data.get('event') or '0000'

    client = get_supabase_client()
    if not all([agent_name, messages, session_id, client]):
        return jsonify({"error": "agentName, messages, and sessionId are required, or DB is unavailable"}), 400

    logger.info(f"Save Memory Log: Received request for agent '{agent_name}', session '{session_id}'.")

    try:
        # Step 1: Enrich the chat log
        google_api_key = get_api_key(agent_name, 'google')
        if not google_api_key:
            return jsonify({"error": "Enrichment service not configured (missing API key)"}), 500
        
        structured_content, summary, triplets = enrich_chat_log(messages, google_api_key)
        
        # Step 2: Upsert the log into Supabase
        upsert_data = {
            "agent_name": agent_name,
            "source_identifier": session_id,
            "structured_content": structured_content,
            "summary": summary,
        }
        
        # Use on_conflict to handle both insert and update in one call
        upsert_res = client.table("agent_memory_logs") \
            .upsert(upsert_data, on_conflict="source_identifier") \
            .execute()
        
        if not upsert_res.data:
            logger.error(f"Save Memory Log: Supabase upsert failed for session '{session_id}'. Error: {upsert_res.error}")
            return jsonify({"error": "Failed to save memory log to database", "details": str(upsert_res.error)}), 500
        
        supabase_log_id = upsert_res.data[0]['id']
        created_at_timestamp = upsert_res.data[0]['created_at']
        logger.info(f"Save Memory Log: Successfully upserted log to Supabase. ID: {supabase_log_id}, Created At: {created_at_timestamp}")

        # Step 3 & 4: Re-index in Pinecone (Delete then Upsert)
        # MODIFIED: Use a shared index name and the agent_name as the namespace
        embedding_handler = EmbeddingHandler( 
            index_name="river",
            namespace=agent_name
        )
        
        # Delete old vectors for this session to prevent stale data
        logger.info(f"Save Memory Log: Deleting old vectors from Pinecone for session '{session_id}'...")
        delete_success = embedding_handler.delete_document(source_identifier=session_id)
        if not delete_success:
            logger.warning(f"Save Memory Log: Could not delete old vectors for session '{session_id}'. This may result in duplicate data.")

        # Upsert new vectors
        logger.info(f"Save Memory Log: Upserting new vectors to Pinecone for session '{session_id}'...")
        metadata_for_embedding = {
            "agent_name": agent_name,
            "source_identifier": session_id,
            "supabase_log_id": supabase_log_id,
            "file_name": f"chat_memory_{session_id}.md", # A virtual filename
            "created_at": created_at_timestamp,
            "triplets": triplets, # Pass the triplets to the embedding handler
            "source_type": "chat",
            "event_id": event_id_param,
        }
        
        try:
            upsert_success = embedding_handler.embed_and_upsert(structured_content, metadata_for_embedding)
        except Exception as e:
            upsert_success = False
            logger.warning(f"Save Memory Log: Pinecone embedding step failed with exception for session '{session_id}': {e}")

        if not upsert_success:
            # Degrade gracefully: Supabase log saved, but retrieval indexing unavailable.
            logger.warning(f"Save Memory Log: Indexing unavailable; returning success with DB-only save for session '{session_id}'.")
            return jsonify({"status": "success", "message": "Chat memory saved (indexing unavailable).", "log_id": supabase_log_id, "indexed": False}), 200

        logger.info(f"Save Memory Log: Pipeline completed successfully for session '{session_id}'.")
        return jsonify({"status": "success", "message": "Chat memory saved and indexed.", "log_id": supabase_log_id, "indexed": True}), 200

    except Exception as e:
        logger.error(f"Save Memory Log: Unexpected error in pipeline for agent '{agent_name}', session '{session_id}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while saving memory."}), 500

@app.route('/api/memory/list-saved-chats', methods=['GET'])
@supabase_auth_required(agent_required=True)
def list_saved_chat_logs(user: SupabaseUser):
    """Lists all saved memory logs for a given agent that the user can access."""
    agent_name = request.args.get('agentName') # Corrected from 'agent'
    client = get_supabase_client()
    if not agent_name or not client:
        return jsonify({"error": "Agent name is required or DB is unavailable"}), 400

    try:
        query = client.table("agent_memory_logs") \
            .select("id, created_at, summary") \
            .eq("agent_name", agent_name) \
            .order("created_at", desc=True)
        
        response = query.execute()

        if response.data:
            return jsonify(response.data), 200
        else:
            logger.warning(f"List Saved Chats: No saved logs found for agent '{agent_name}'.")
            return jsonify([]), 200
    except Exception as e:
        logger.error(f"List Saved Chats: Error fetching logs for agent '{agent_name}': {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve saved memories."}), 500

@app.route('/api/memory/forget-chat', methods=['POST'])
@supabase_auth_required(agent_required=True)
def forget_chat_memory_log(user: SupabaseUser):
    """Permanently deletes a memory log from Supabase and Pinecone."""
    data = g.get('json_data', {})
    agent_name = data.get('agentName')
    memory_id = data.get('memoryId')

    client = get_supabase_client()
    if not all([agent_name, memory_id, client]):
        return jsonify({"error": "agentName and memoryId are required, or DB is unavailable"}), 400
    
    logger.info(f"Forget Memory: Request received for agent '{agent_name}', memory ID '{memory_id}'.")

    try:
        # Step 1: Get the source_identifier from Supabase before deleting the row
        select_res = client.table("agent_memory_logs") \
            .select("source_identifier") \
            .eq("id", memory_id) \
            .eq("agent_name", agent_name) \
            .single() \
            .execute()
        
        if not select_res.data:
            logger.warning(f"Forget Memory: Memory log with ID '{memory_id}' not found for agent '{agent_name}'.")
            return jsonify({"error": "Memory not found"}), 404
        
        source_identifier = select_res.data['source_identifier']

        # Step 2: Delete vectors from Pinecone
        # MODIFIED: Use a shared index name and the agent_name as the namespace
        embedding_handler = EmbeddingHandler(index_name="river", namespace=agent_name)
        delete_success = embedding_handler.delete_document(source_identifier=source_identifier)
        if not delete_success:
            # Log a warning but proceed with DB deletion. The user wants the memory gone.
            logger.warning(f"Forget Memory: Failed to delete vectors from Pinecone for source '{source_identifier}'. The database entry will still be removed.")

        # Step 3: Delete from Supabase
        delete_res = client.table("agent_memory_logs") \
            .delete() \
            .eq("id", memory_id) \
            .execute()

        if not delete_res.data:
            logger.error(f"Forget Memory: Supabase delete failed for ID '{memory_id}'. Error: {delete_res.error}")
            return jsonify({"error": "Failed to delete memory from database"}), 500
        
        logger.info(f"Forget Memory: Successfully deleted memory ID '{memory_id}' (source: '{source_identifier}') for agent '{agent_name}'.")
        return jsonify({"status": "success", "message": "Memory forgotten."}), 200

    except Exception as e:
        logger.error(f"Forget Memory: Unexpected error for agent '{agent_name}', memory ID '{memory_id}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while forgetting memory."}), 500

@app.route('/internal_api/pinecone/list-indexes', methods=['GET'])
@supabase_auth_required(agent_required=False) # Internal, but still requires a valid user token
def list_pinecone_indexes(user: SupabaseUser):
    """Internal endpoint to list all available Pinecone indexes."""
    from utils.pinecone_utils import list_indexes as list_pinecone_indexes_util
    
    logger.info(f"Internal API: User {user.id} requested to list Pinecone indexes.")
    try:
        indexes = list_pinecone_indexes_util()
        return jsonify({"indexes": indexes}), 200
    except Exception as e:
        logger.error(f"Internal API: Error listing Pinecone indexes: {e}", exc_info=True)
        return jsonify({"error": "Failed to list Pinecone indexes"}), 500

# --- Pinecone Proxy Endpoints (ACL + Metadata Enforcement) ---

@app.route('/api/pinecone/query', methods=['POST'])
@supabase_auth_required(agent_required=True)
def pinecone_query_proxy(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    event_id = data.get('event', '0000')
    query = data.get('query')
    signal_bias = data.get('signalBias', 'medium')
    top_k = int(data.get('topK', 15))

    if not all([agent_name, query]):
        return jsonify({"error": "agent and query are required"}), 400

    try:
        retriever = RetrievalHandler(
            index_name="river",
            agent_name=agent_name,
            event_id=event_id,
            anthropic_api_key=get_api_key(agent_name, 'anthropic'),
            openai_api_key=get_api_key(agent_name, 'openai')
        )
        tier_caps_env = os.getenv('RETRIEVAL_TIER_CAPS', '7,6,6,4').split(',')
        try:
            tier_caps = [int(x.strip()) for x in tier_caps_env]
        except Exception:
            tier_caps = [7, 6, 6, 4]
        mmr_lambda = float(os.getenv('RETRIEVAL_MMR_LAMBDA', '0.6'))
        mmr_k = min(int(os.getenv('RETRIEVAL_MMR_K', '23')), top_k)
        allow_t3_low = os.getenv('ALLOW_T3_ON_LOW', 'false').lower() == 'true'
        include_t3 = not ((signal_bias or 'medium') == 'low' and not allow_t3_low)

        docs = retriever.get_relevant_context_tiered(
            query=query,
            tier_caps=tier_caps,
            mmr_k=mmr_k,
            mmr_lambda=mmr_lambda,
            include_t3=include_t3,
        )
        items = [
            {
                'content': d.page_content,
                'metadata': d.metadata,
            }
            for d in docs
        ]
        return jsonify({'results': items}), 200
    except Exception as e:
        logger.error(f"Pinecone proxy query error: {e}", exc_info=True)
        return jsonify({"error": "Failed to query memory"}), 500


@app.route('/api/pinecone/upsert', methods=['POST'])
@supabase_auth_required(agent_required=True)
def pinecone_upsert_proxy(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    event_id = data.get('event', '0000')
    content = data.get('content')
    metadata = data.get('metadata', {}) or {}
    if not all([agent_name, content]):
        return jsonify({"error": "agent and content are required"}), 400

    try:
        embedding_handler = EmbeddingHandler(index_name="river", namespace=agent_name)
        # Enforce required metadata fields
        metadata.setdefault('agent_name', agent_name)
        metadata.setdefault('event_id', event_id)
        upsert_success = embedding_handler.embed_and_upsert(content, metadata)
        if not upsert_success:
            return jsonify({"error": "Failed to upsert memory"}), 500
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Pinecone proxy upsert error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during upsert"}), 500

# --- Restored User Chat History Endpoints ---

def generate_chat_title(first_user_message: str) -> str:
    """Generate a concise title for a chat using Groq GPT-OSS-20B"""
    logger.info(f"generate_chat_title called with message: {first_user_message[:100] if first_user_message else 'None'}")
    try:
        logger.info("Calling Groq API for title generation...")
        logger.info(f"Request details - Model: openai/gpt-oss-20b, Max tokens: 50, Temperature: 0.9")
        logger.info(f"System instruction: 'Generate a concise, descriptive title (max 4 words) for this chat based on the user's first message. Return only the title, no quotes or extra text.'")
        logger.info(f"User message: '{first_user_message}'")

        title = _call_groq_non_stream_with_retry(
            model_name="openai/gpt-oss-20b",
            max_tokens=50,
            system_instruction="Generate only a 3-4 word title using short words based on the first user message. No explanation, no quotes, just the title.",
            messages=[{"role": "user", "content": f"Create a title for: {first_user_message}"}],
            api_key=os.getenv('GROQ_API_KEY'), # Use global key for this utility
            temperature=0.3,
            reasoning_effort="low"
        )
        logger.info(f"Raw Groq response: '{title}'")
        final_title = title.strip().strip('"')[:100] if title else "Untitled Chat"
        logger.info(f"Final processed title: '{final_title}'")
        return final_title
    except Exception as e:
        logger.error(f"Error generating chat title: {e}", exc_info=True)
        fallback = first_user_message[:50] + "..." if len(first_user_message) > 50 else first_user_message
        logger.info(f"Using fallback title: '{fallback}'")
        return fallback

@app.route('/api/chat/history/save', methods=['POST'])
@supabase_auth_required(agent_required=True)
@retry_strategy_supabase
def save_chat_history(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_name = data.get('agent')
    messages = data.get('messages', [])
    chat_id = data.get('chatId')
    title = data.get('title')
    last_message_id = data.get('lastMessageId') # New field
    client_session_id = data.get('clientSessionId') or data.get('client_session_id')
    if not agent_name or not messages:
        return jsonify({'error': 'Agent name and messages are required'}), 400
    
    client = get_supabase_client()
    if not client: return jsonify({'error': 'Database not available'}), 503
    agent_result = client.table('agents').select('id').eq('name', agent_name).single().execute()
    if not agent_result.data:
        return jsonify({'error': 'Agent not found'}), 404
    agent_id = agent_result.data['id']

    if not title and not chat_id and messages:
        first_user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
        logger.info(f"Generating title for new chat. First user message: {first_user_message[:50] if first_user_message else 'None'}...")
        title = generate_chat_title(first_user_message) if first_user_message else "New Chat"
        logger.info(f"Generated title: '{title}'")
    else:
        logger.info(f"Skipping title generation - title: {bool(title)}, chat_id: {bool(chat_id)}, messages: {len(messages) if messages else 0}")

    try:
        def _merge_messages(existing_list, incoming_list):
            try:
                if not isinstance(existing_list, list) or not existing_list:
                    return incoming_list or []
                if not isinstance(incoming_list, list) or not incoming_list:
                    return existing_list or []
                by_id = {}
                order = []
                # preserve existing first
                for m in existing_list:
                    mid = m.get('id') if isinstance(m, dict) else None
                    if mid is None:
                        order.append((None, m))
                        continue
                    if mid not in by_id:
                        by_id[mid] = m
                        order.append((mid, None))
                # overlay incoming (new/edited)
                for m in incoming_list:
                    mid = m.get('id') if isinstance(m, dict) else None
                    if mid is None:
                        order.append((None, m))
                        continue
                    by_id[mid] = m
                    if (mid, None) not in order and all(x[0] != mid for x in order):
                        order.append((mid, None))
                # rebuild in recorded order; for None ids include raw entries
                merged = []
                for mid, placeholder in order:
                    if mid is None and placeholder is not None:
                        merged.append(placeholder)
                    elif mid is not None and mid in by_id:
                        merged.append(by_id[mid])
                # fallback: if merged is unexpectedly empty, return incoming
                return merged or incoming_list
            except Exception:
                return incoming_list or existing_list or []
        if chat_id:
            update_payload = {
                'title': title,
                'messages': messages,
                'updated_at': 'now()',
                'last_message_at': 'now()',
            }
            if last_message_id:
                update_payload['last_message_id_at_save'] = last_message_id
            # Merge with existing to avoid losing messages on partial saves
            try:
                existing_res = client.table('chat_history').select('messages').eq('id', chat_id).eq('user_id', user.id).single().execute()
                existing_msgs = existing_res.data.get('messages') if existing_res and existing_res.data else []
                merged_msgs = _merge_messages(existing_msgs, messages)
                update_payload['messages'] = merged_msgs
            except Exception:
                pass
            # Attempt to include event_id if column exists
            try:
                update_payload['event_id'] = request.args.get('event') or g.json_data.get('event') or '0000'
                client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
            except Exception:
                # retry without event_id if migration not applied
                update_payload.pop('event_id', None)
                result = client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
        else:
            # Idempotent creation by client_session_id, if provided
            if client_session_id:
                try:
                    existing = client.table('chat_history') \
                        .select('id, title') \
                        .eq('user_id', user.id) \
                        .eq('agent_id', agent_id) \
                        .eq('client_session_id', client_session_id) \
                        .single().execute()
                except Exception:
                    existing = None
                if existing and existing.data:
                    chat_id = existing.data['id']
                    update_payload = {
                        'title': title,
                        'messages': messages,
                        'updated_at': 'now()',
                        'last_message_at': 'now()',
                    }
                    if last_message_id:
                        update_payload['last_message_id_at_save'] = last_message_id
                    try:
                        existing_res = client.table('chat_history').select('messages').eq('id', chat_id).eq('user_id', user.id).single().execute()
                        existing_msgs = existing_res.data.get('messages') if existing_res and existing_res.data else []
                        merged_msgs = _merge_messages(existing_msgs, messages)
                        update_payload['messages'] = merged_msgs
                    except Exception:
                        pass
                    client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
                    return jsonify({'success': True, 'chatId': chat_id, 'title': existing.data.get('title', title)})

            # Before creating a new chat, check for recent duplicates to prevent race condition artifacts
            # Look for chats created in the last 60 seconds with the same title and agent (increased from 30s)
            from datetime import datetime, timedelta
            cutoff_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
            
            recent_chats = client.table('chat_history').select('id, title, messages, created_at').eq('user_id', user.id).eq('agent_id', agent_id).eq('title', title).gte('created_at', cutoff_time).order('created_at', desc=True).execute()
            
            # If we find a recent chat with the same title, check if it's likely a duplicate
            if recent_chats.data:
                for recent_chat in recent_chats.data:
                    recent_messages = recent_chat.get('messages', [])
                    
                    # Improved duplicate detection: Check message IDs for exact matches
                    if len(recent_messages) <= len(messages):
                        recent_msg_ids = {msg.get('id') for msg in recent_messages if msg.get('id')}
                        current_msg_ids = {msg.get('id') for msg in messages if msg.get('id')}
                        
                        # If recent messages are a subset of current messages (by ID), it's likely a race condition
                        if recent_msg_ids and recent_msg_ids.issubset(current_msg_ids):
                            chat_id = recent_chat['id']
                            update_payload = {
                                'title': title,
                                'messages': messages,
                            }
                            if last_message_id:
                                update_payload['last_message_id_at_save'] = last_message_id
                            try:
                                existing_res = client.table('chat_history').select('messages').eq('id', chat_id).eq('user_id', user.id).single().execute()
                                existing_msgs = existing_res.data.get('messages') if existing_res and existing_res.data else []
                                merged_msgs = _merge_messages(existing_msgs, messages)
                                update_payload['messages'] = merged_msgs
                            except Exception:
                                pass
                            
                            result = client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
                            break
                        
                        # Fallback: Check by content if IDs don't match
                        recent_msg_content = [msg.get('content', '') for msg in recent_messages if 'content' in msg]
                        current_msg_content = [msg.get('content', '') for msg in messages if 'content' in msg]
                        
                        # Check if all recent messages are contained in current messages (subset check)
                        if recent_msg_content and all(msg in current_msg_content for msg in recent_msg_content):
                            chat_id = recent_chat['id']
                            update_payload = {
                                'title': title,
                                'messages': messages,
                            }
                            if last_message_id:
                                update_payload['last_message_id_at_save'] = last_message_id
                            
                            result = client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
                            break
                else:
                    # No duplicate found, proceed with insert
                    insert_payload = {
                        'user_id': user.id,
                        'agent_id': agent_id,
                        'title': title,
                        'messages': messages,
                        'client_session_id': client_session_id,
                        'updated_at': 'now()',
                        'last_message_at': 'now()',
                    }
                    if last_message_id:
                        insert_payload['last_message_id_at_save'] = last_message_id
                    
                    # Try with event_id; on failure, retry without
                    try:
                        insert_payload['event_id'] = request.args.get('event') or g.json_data.get('event') or '0000'
                        result = client.table('chat_history').insert(insert_payload).execute()
                    except Exception:
                        insert_payload.pop('event_id', None)
                        result = client.table('chat_history').insert(insert_payload).execute()
                    chat_id = result.data[0]['id'] if result.data else None
                    title = result.data[0]['title'] if result.data else title
            else:
                # No recent chats, proceed with insert
                insert_payload = {
                    'user_id': user.id,
                    'agent_id': agent_id,
                    'title': title,
                    'messages': messages,
                    'client_session_id': client_session_id,
                    'updated_at': 'now()',
                    'last_message_at': 'now()',
                }
                if last_message_id:
                    insert_payload['last_message_id_at_save'] = last_message_id
                try:
                    insert_payload['event_id'] = request.args.get('event') or g.json_data.get('event') or '0000'
                    result = client.table('chat_history').insert(insert_payload).execute()
                except Exception:
                    insert_payload.pop('event_id', None)
                    result = client.table('chat_history').insert(insert_payload).execute()
                chat_id = result.data[0]['id'] if result.data else None
                title = result.data[0]['title'] if result.data else title
        
        return jsonify({'success': True, 'chatId': chat_id, 'title': title})
    except Exception as e:
        logger.error(f"Error saving chat history: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save chat history'}), 500

@app.route('/api/chat/history/list', methods=['GET'])
@supabase_auth_required(agent_required=False)
@retry_strategy_supabase
def list_chat_history(user: SupabaseUser):
    agent_name = request.args.get('agentName')
    event_id = request.args.get('event')  # optional filter
    client = get_supabase_client()
    if not agent_name or not client:
        return jsonify({'error': 'agentName parameter is required or DB is unavailable'}), 400

    try:
        # Robust agent lookup with small retry to handle transient HTTP/2 disconnects
        agent_result = None
        last_err = None
        for i in range(3):
            try:
                agent_result = client.table('agents').select('id').eq('name', agent_name).single().execute()
                break
            except (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_err = e
                time.sleep(min(1.5, 0.2 * (2 ** i)))
        if agent_result is None:
            raise last_err or Exception('Unknown error fetching agent id')
        if not agent_result.data:
            return jsonify([])
        agent_id = agent_result.data['id']

        try:
            # Try selecting with event_id (new schema)
            base_query = client.table('chat_history') \
                .select('id, title, last_message_at, agent_id, messages, event_id') \
                .eq('user_id', user.id) \
                .eq('agent_id', agent_id)
            if event_id:
                base_query = base_query.eq('event_id', event_id)
            history_result = base_query.order('last_message_at', desc=True).limit(100).execute()

            # If no rows and event filter was applied, retry without the event filter (legacy rows)
            if event_id and (not history_result.data or len(history_result.data) == 0):
                history_result = client.table('chat_history') \
                    .select('id, title, last_message_at, agent_id, messages') \
                    .eq('user_id', user.id) \
                    .eq('agent_id', agent_id) \
                    .order('last_message_at', desc=True).limit(100).execute()
                logger.info(f"Chat history fallback (no event filter) returned {len(history_result.data or [])} rows.")
        except Exception as e:
            # Schema likely missing event_id; select without it
            logger.info(f"Chat history: event_id column not available or query failed ({e}); retrying without event column.")
            history_result = client.table('chat_history') \
                .select('id, title, last_message_at, agent_id, messages') \
                .eq('user_id', user.id) \
                .eq('agent_id', agent_id) \
                .order('last_message_at', desc=True).limit(100).execute()

        if not history_result.data:
            return jsonify([])

        chat_ids = [chat['id'] for chat in history_result.data]
        
        memory_logs_result = client.table('agent_memory_logs') \
            .select('source_identifier') \
            .eq('agent_name', agent_name) \
            .execute()

        saved_conversation_ids = set()
        saved_message_ids = set()
        if memory_logs_result.data:
            for log in memory_logs_result.data:
                source_id = log['source_identifier']
                if source_id in chat_ids:
                    saved_conversation_ids.add(source_id)
                elif source_id.startswith('message_'):
                    parts = source_id.split('_')
                    if len(parts) > 1:
                        saved_message_ids.add(parts[1])

        message_to_chat_map = {}
        for chat in history_result.data:
            if isinstance(chat.get('messages'), list):
                for message in chat['messages']:
                    if isinstance(message, dict) and 'id' in message:
                        message_to_chat_map[message['id']] = chat['id']

        chats_with_saved_messages = set()
        for msg_id in saved_message_ids:
            if msg_id in message_to_chat_map:
                chats_with_saved_messages.add(message_to_chat_map[msg_id])

        formatted_history = []
        for chat in history_result.data:
            chat_id = chat['id']
            is_convo_saved = chat_id in saved_conversation_ids
            has_saved_messages = chat_id in chats_with_saved_messages

            # The 'messages' field can be large, so we don't return it in the list view.
            # The 'get' endpoint will return the full message payload.
            formatted_history.append({
                'id': chat_id,
                'title': chat['title'],
                'updatedAt': chat['last_message_at'],
                'agentId': chat['agent_id'],
                'agentName': agent_name,
                'eventId': chat.get('event_id', None),
                'isConversationSaved': is_convo_saved,
                'hasSavedMessages': has_saved_messages
            })

        return jsonify(formatted_history)
    except Exception as e:
        logger.error(f"Error listing chat history for agent '{agent_name}': {e}", exc_info=True)
        return jsonify({'error': 'Failed to list chat history'}), 500

@app.route('/api/chat/history/get', methods=['GET'])
@supabase_auth_required(agent_required=False)
@retry_strategy_supabase
def get_chat_history(user: SupabaseUser):
    chat_id = request.args.get('chatId')
    client = get_supabase_client()
    if not chat_id or not client:
        return jsonify({'error': 'Chat ID is required or DB is unavailable'}), 400

    try:
        chat_result = client.table('chat_history').select('*, agents(name)').eq('id', chat_id).eq('user_id', user.id).single().execute()
        
        if not chat_result.data:
            return jsonify({'error': 'Chat not found or access denied'}), 404

        chat_data = chat_result.data
        agent_name = chat_data.get('agents', {}).get('name')
        if not agent_name:
            logger.warning(f"Could not determine agent name for chat ID {chat_id}. Memory info will be incomplete.")
            return jsonify(chat_data)

        saved_message_ids = {}
        is_conversation_saved = False
        last_conversation_save_time = None
        conversation_memory_id = None

        # Check for full conversation save first
        convo_memory_log_res = client.table('agent_memory_logs') \
            .select('id, created_at') \
            .eq('agent_name', agent_name) \
            .eq('source_identifier', chat_id) \
            .order('created_at', desc=True) \
            .limit(1).execute()

        if convo_memory_log_res.data:
            is_conversation_saved = True
            last_conversation_save_time = convo_memory_log_res.data[0]['created_at']
            conversation_memory_id = convo_memory_log_res.data[0]['id']

        # Check for individual message saves within this chat
        message_ids_in_chat = [msg['id'] for msg in chat_data.get('messages', []) if 'id' in msg]
        if message_ids_in_chat:
            # Create a list of 'like' patterns for the query
            like_patterns = [f"message_{msg_id}_%" for msg_id in message_ids_in_chat]
            
            # Use 'or' filter to match any of the patterns
            message_memory_logs_res = client.table('agent_memory_logs') \
                .select('id, source_identifier, created_at') \
                .eq('agent_name', agent_name) \
                .or_(','.join([f'source_identifier.like.{p}' for p in like_patterns])) \
                .execute()

            if message_memory_logs_res.data:
                for log in message_memory_logs_res.data:
                    source_id = log['source_identifier']
                    created_at = log['created_at']
                    memory_id = log['id']
                    if source_id.startswith('message_'):
                        parts = source_id.split('_')
                        if len(parts) > 1 and parts[1] in message_ids_in_chat:
                            message_id = parts[1]
                            saved_message_ids[message_id] = {
                                "savedAt": created_at,
                                "memoryId": memory_id
                            }
        
        chat_data['savedMessageIds'] = saved_message_ids
        chat_data['isConversationSaved'] = is_conversation_saved
        chat_data['lastConversationSaveTime'] = last_conversation_save_time
        chat_data['conversationMemoryId'] = conversation_memory_id
        # The last_message_id_at_save is already in chat_data from the initial query

        return jsonify(chat_data), 200

    except Exception as e:
        logger.error(f"Error getting chat history for ID '{chat_id}': {e}", exc_info=True)
        return jsonify({'error': 'Failed to get chat history'}), 500

@app.route('/delete_message', methods=['POST'])
@supabase_auth_required(agent_required=False)
def delete_message(user: SupabaseUser):
    logger.info("--- DELETE MESSAGE REQUEST START ---")
    data = request.get_json()
    chat_id = data.get('chat_id')
    message_id_to_delete = data.get('message_id')
    requesting_user_id = data.get('user_id')
    logger.info(f"Delete request for user '{requesting_user_id}', chat '{chat_id}', message '{message_id_to_delete}'")

    if not all([chat_id, message_id_to_delete, requesting_user_id]):
        logger.error("Delete request missing required fields.")
        return jsonify({'error': 'chat_id, message_id, and user_id are required'}), 400

    if user.id != requesting_user_id:
        logger.error(f"User ID mismatch: token user '{user.id}' vs request user '{requesting_user_id}'")
        return jsonify({'error': 'User ID mismatch'}), 403

    client = get_supabase_client()
    if not client:
        logger.error("Supabase client not available for delete operation.")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        # --- Step 1: Fetch chat and agent info ---
        logger.info(f"Fetching chat_history for chat_id: {chat_id}")
        chat_res = client.table('chat_history').select('messages, agent_id, last_message_id_at_save').eq('id', chat_id).eq('user_id', user.id).single().execute()
        if not chat_res.data:
            logger.error(f"Chat not found or access denied for chat_id: {chat_id}")
            return jsonify({'error': 'Chat not found or access denied'}), 404

        messages = chat_res.data.get('messages', [])
        agent_id = chat_res.data.get('agent_id')
        last_save_marker_id = chat_res.data.get('last_message_id_at_save')
        logger.info(f"Found chat with {len(messages)} messages. Agent ID: {agent_id}, Save Marker: {last_save_marker_id}")
        
        agent_res = client.table('agents').select('name').eq('id', agent_id).single().execute()
        if not agent_res.data:
            logger.error(f"Agent not found for agent_id: {agent_id}")
            return jsonify({'error': 'Agent not found for this chat'}), 404
        agent_name = agent_res.data['name']
        logger.info(f"Agent name: {agent_name}")

        # --- Step 2: Find and remove the message from the list ---
        updated_messages = [msg for msg in messages if msg.get('id') != message_id_to_delete]

        if len(updated_messages) == len(messages):
            logger.error(f"Message {message_id_to_delete} not found in chat history for chat {chat_id}")
            return jsonify({'error': 'Message not found in chat history'}), 404
        logger.info(f"Message {message_id_to_delete} removed. New message count: {len(updated_messages)}")

        # --- Step 3: Determine the new save marker ---
        new_last_save_marker_id = last_save_marker_id
        if last_save_marker_id == message_id_to_delete:
            new_last_save_marker_id = updated_messages[-1]['id'] if updated_messages else None
            logger.info(f"Save marker was on deleted message. New marker: {new_last_save_marker_id}")
        
        # --- Step 4: Update chat_history table ---
        update_payload = {
            'messages': updated_messages,
            'last_message_id_at_save': new_last_save_marker_id
        }
        logger.info(f"Updating chat_history for {chat_id} with {len(updated_messages)} messages.")
        update_res = client.table('chat_history').update(update_payload).eq('id', chat_id).execute()
        if hasattr(update_res, 'error') and update_res.error:
            logger.error(f"Failed to update chat_history for chat {chat_id}: {update_res.error}")
            return jsonify({'error': 'Failed to save updated chat history'}), 500

        logger.info(f"Successfully deleted message {message_id_to_delete} from chat_history {chat_id}")

        # --- Step 5: Synchronously update memory logs and Pinecone ---
        logger.info("Starting memory and vector DB update process...")
        # MODIFIED: Use a shared index name and the agent_name as the namespace
        embedding_handler = EmbeddingHandler(
            index_name="river",
                namespace=agent_name)
        
        # Case A: The message was part of a full conversation save
        logger.info(f"Checking for full conversation memory log with source_identifier: {chat_id}")
        memory_log_res = client.table('agent_memory_logs').select('id').eq('source_identifier', chat_id).execute()
        if memory_log_res.data:
            logger.info(f"Found full conversation memory log for chat {chat_id}. Processing update.")
            if not updated_messages:
                # If the last message was deleted, remove the memory log entirely
                logger.info(f"Conversation {chat_id} is now empty. Deleting memory log and vectors.")
                embedding_handler.delete_document(source_identifier=chat_id)
                client.table('agent_memory_logs').delete().eq('source_identifier', chat_id).execute()
                logger.info(f"Deleted memory log and vectors for chat {chat_id}.")
            else:
                # Re-enrich and re-save the conversation memory
                logger.info(f"Re-enriching and re-indexing memory for chat {chat_id}.")
                google_api_key = get_api_key(agent_name, 'google')
                if not google_api_key:
                    logger.error("Cannot re-enrich memory: Google API key not found.")
                else:
                    structured_content, summary, triplets = enrich_chat_log(updated_messages, google_api_key)
                    client.table("agent_memory_logs").update({
                        "structured_content": structured_content,
                        "summary": summary
                    }).eq("source_identifier", chat_id).execute()
                    logger.info(f"Updated agent_memory_logs for {chat_id} in Supabase.")
                    
                    embedding_handler.delete_document(source_identifier=chat_id)
                    logger.info(f"Deleted old vectors for {chat_id}.")
                    embedding_handler.embed_and_upsert(structured_content, {
                        "agent_name": agent_name,
                        "source_identifier": chat_id,
                        "file_name": f"chat_memory_{chat_id}.md",
                        "saved_at": datetime.now(timezone.utc).timestamp(),
                        "triplets": triplets,
                        "source_type": "chat",
                        "event_id": event_id_param,
                    })
                    logger.info(f"Successfully re-indexed conversation memory for chat {chat_id}.")
        else:
            logger.info(f"No full conversation memory log found for chat {chat_id}.")

        # Case B: The message was saved individually
        logger.info(f"Checking for individually saved message memory for message_id: {message_id_to_delete}")
        individual_log_res = client.table('agent_memory_logs').select('id, source_identifier').like('source_identifier', f'message_{message_id_to_delete}%').execute()
        if individual_log_res.data:
            logger.info(f"Found {len(individual_log_res.data)} individually saved memory logs for message {message_id_to_delete}.")
            for log in individual_log_res.data:
                log_source_id = log['source_identifier']
                logger.info(f"Deleting individually saved message memory: {log_source_id}")
                embedding_handler.delete_document(source_identifier=log_source_id)
                client.table('agent_memory_logs').delete().eq('id', log['id']).execute()
                logger.info(f"Deleted log and vectors for {log_source_id}.")
        else:
            logger.info(f"No individually saved memory logs found for message {message_id_to_delete}.")

        logger.info("--- DELETE MESSAGE REQUEST END ---")
        return jsonify({
            'success': True, 
            'message': 'Message deleted successfully',
            'new_last_message_id_at_save': new_last_save_marker_id
        })

    except Exception as e:
        logger.error(f"--- DELETE MESSAGE REQUEST FAILED ---: Error deleting message {message_id_to_delete} from chat {chat_id}: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred'}), 500


@app.route('/api/chat/history/delete_conversation', methods=['POST'])
@supabase_auth_required(agent_required=False)
def delete_conversation(user: SupabaseUser):
    logger.info("--- DELETE CONVERSATION REQUEST START ---")
    data = request.get_json()
    chat_id = data.get('chat_id')
    # The user_id from the request body is for confirmation, but the primary user ID is from the token.
    requesting_user_id = data.get('user_id')
    logger.info(f"Delete conversation request for user '{user.id}', chat '{chat_id}'")

    if not chat_id:
        logger.error("Delete conversation request missing chat_id.")
        return jsonify({'error': 'chat_id is required'}), 400

    # Optional: Verify that the user_id in the body matches the authenticated user
    if requesting_user_id and user.id != requesting_user_id:
        logger.error(f"User ID mismatch: token user '{user.id}' vs request user '{requesting_user_id}'")
        return jsonify({'error': 'User ID mismatch'}), 403

    client = get_supabase_client()
    if not client:
        logger.error("Supabase client not available for delete operation.")
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        # --- Step 1: Fetch chat and agent info ---
        logger.info(f"Fetching chat_history for chat_id: {chat_id}")
        chat_res = client.table('chat_history').select('messages, agent_id').eq('id', chat_id).eq('user_id', user.id).single().execute()
        if not chat_res.data:
            logger.error(f"Chat not found or access denied for chat_id: {chat_id}")
            return jsonify({'error': 'Chat not found or access denied'}), 404

        messages = chat_res.data.get('messages', [])
        agent_id = chat_res.data.get('agent_id')
        
        agent_name = None
        if agent_id:
            agent_res = client.table('agents').select('name').eq('id', agent_id).single().execute()
            if agent_res.data:
                agent_name = agent_res.data['name']
                logger.info(f"Agent name: {agent_name}")
            else:
                logger.warning(f"Could not find agent for agent_id {agent_id}. Memory logs may not be fully cleaned up.")
        
            # --- Step 2: Clean up memory logs and Pinecone vectors if agent is known ---
            if agent_name:
                try:
                    # MODIFIED: Use a shared index name and the agent_name as the namespace
                    embedding_handler = EmbeddingHandler(
                        index_name="river",
                         namespace=agent_name)
                    if embedding_handler.index:
                        # Delete full conversation memory
                        logger.info(f"Deleting full conversation memory for source_identifier: {chat_id}")
                        embedding_handler.delete_document(source_identifier=chat_id)
                    
                        # Delete individual message memories from this chat
                        message_ids_in_chat = [msg['id'] for msg in messages if 'id' in msg]
                        if message_ids_in_chat:
                            like_patterns = [f"message_{msg_id}_%" for msg_id in message_ids_in_chat]
                            individual_log_res = client.table('agent_memory_logs').select('id, source_identifier').eq('agent_name', agent_name).or_(','.join([f'source_identifier.like.{p}' for p in like_patterns])).execute()
                            if individual_log_res.data:
                                logger.info(f"Found {len(individual_log_res.data)} individual message memories to delete for chat {chat_id}.")
                                for log in individual_log_res.data:
                                    log_source_id = log['source_identifier']
                                    embedding_handler.delete_document(source_identifier=log_source_id)
                                    logger.info(f"Deleted vectors for individual memory log: {log_source_id}")
                    else:
                        logger.warning(f"Agent '{agent_name}' does not have a Pinecone index. Skipping vector deletion.")
                except Exception as e:
                    logger.error(f"Error during Pinecone cleanup for agent '{agent_name}': {e}", exc_info=True)
                    # Do not re-raise, allow Supabase deletion to proceed

            # Always delete from Supabase regardless of Pinecone status
            client.table('agent_memory_logs').delete().eq('source_identifier', chat_id).eq('agent_name', agent_name).execute()
            
            message_ids_in_chat = [msg['id'] for msg in messages if 'id' in msg]
            if message_ids_in_chat:
                logger.info(f"Deleting {len(message_ids_in_chat)} individual message memories in batches.")
                batch_size = 50  # Process 50 message deletions at a time
                for i in range(0, len(message_ids_in_chat), batch_size):
                    batch_ids = message_ids_in_chat[i:i + batch_size]
                    like_patterns = [f'message_{msg_id}_%' for msg_id in batch_ids]
                    logger.debug(f"Deleting batch {i//batch_size + 1} with {len(batch_ids)} message IDs.")
                    try:
                        client.table('agent_memory_logs').delete().eq('agent_name', agent_name).or_(','.join([f'source_identifier.like.{p}' for p in like_patterns])).execute()
                    except httpx.RemoteProtocolError as e:
                        logger.error(f"Server disconnected during batch deletion of message memories. Batch {i//batch_size + 1} may have failed. Error: {e}")
                        # Continue to the next batch, as this is not a fatal error for the overall deletion process.
                        continue
            logger.info(f"Deleted all associated agent_memory_logs from Supabase for chat {chat_id}.")

        # --- Step 3: Delete the chat history record ---
        logger.info(f"Deleting chat_history record for id: {chat_id}")
        delete_res = client.table('chat_history').delete().eq('id', chat_id).eq('user_id', user.id).execute()
        if hasattr(delete_res, 'error') and delete_res.error:
            logger.error(f"Failed to delete chat_history for chat {chat_id}: {delete_res.error}")
            return jsonify({'error': 'Failed to delete chat history record'}), 500

        logger.info(f"--- DELETE CONVERSATION REQUEST END ---")
        return jsonify({'success': True, 'message': 'Conversation deleted successfully'})

    except Exception as e:
        logger.error(f"--- DELETE CONVERSATION REQUEST FAILED ---: Error deleting conversation {chat_id}: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred'}), 500


@app.route('/api/agents/capabilities', methods=['POST'])
@supabase_auth_required(agent_required=False) # Requires auth, but not a specific agent
def get_agents_capabilities(user: SupabaseUser):
    data = g.get('json_data', {})
    agent_names = data.get('agent_names')

    if not isinstance(agent_names, list):
        return jsonify({"error": "agent_names must be a list"}), 400

    logger.info(f"Checking capabilities for {len(agent_names)} agents for user {user.id}")
    capabilities = {}
    try:
        # Determine index name from env (fallback to default used in embedding handler)
        import os
        from utils.pinecone_utils import get_index_stats
        index_name = os.getenv('PINECONE_INDEX', 'river')
        stats = get_index_stats(index_name)
        namespaces = {}
        if stats:
            # pinecone SDK returns namespaces as dict mapping namespace->meta
            namespaces = stats.get('namespaces') or {}
        else:
            # If stats not available, treat capabilities as unknown/false
            namespaces = {}

        for name in agent_names:
            exists = False
            if isinstance(namespaces, dict):
                exists = name in namespaces.keys()
            elif isinstance(namespaces, list):
                exists = name in namespaces
            capabilities[name] = {"pinecone_index_exists": exists}

        return jsonify(capabilities), 200
    except Exception as e:
        logger.error(f"Error checking agent capabilities: {e}", exc_info=True)
        # Return a failure state for all agents if the check fails
        for name in agent_names:
            capabilities[name] = {"pinecone_index_exists": False}
        return jsonify(capabilities), 500


def cleanup_idle_sessions():
    while True:
        time.sleep(30)
        now = time.time()
        sessions_to_finalize = []
        sessions_to_delete = []

        # Create a copy of items to avoid issues with modifying the dictionary while iterating
        current_sessions = list(active_sessions.items())

        for session_id, session_data in current_sessions:
            with session_locks[session_id]:
                # Case 1: Session is active but has lost its WebSocket connection.
                # Check for reattachment grace period using both session state and grace_deadline
                session_state = session_data.get("session_state")
                grace_deadline = session_data.get("grace_deadline")
                is_finalizing = session_data.get("is_finalizing")
                # Avoid re-triggering finalization on sessions already in-flight

                # Check session state reattachment expiry
                if (not is_finalizing) and session_state and session_state.reattach_deadline and session_state.is_reattach_expired():
                    logger.warning(f"Idle session cleanup: Session {session_id} reattachment grace period expired (session_state). Marking for finalization.")
                    sessions_to_finalize.append(session_id)
                # Check new grace_deadline field
                elif (not is_finalizing) and grace_deadline and now > grace_deadline:
                    logger.warning(f"Idle session cleanup: Session {session_id} grace period expired (grace_deadline). Marking for finalization.")
                    sessions_to_finalize.append(session_id)
                # Fallback for sessions without session_state or grace_deadline (legacy behavior)
                elif (not is_finalizing) and \
                     not session_state and not grace_deadline and \
                     session_data.get("is_active") and \
                     session_data.get("websocket_connection") is None and \
                     now - session_data.get("last_activity_timestamp", 0) > REATTACH_GRACE_SECONDS:
                    logger.warning(f"Idle session cleanup: Session {session_id} (legacy mode) has been active without a WebSocket for >{REATTACH_GRACE_SECONDS}s. Marking for finalization.")
                    sessions_to_finalize.append(session_id)
                
                # Case 2: Session has been finalized and is waiting for garbage collection.
                # This handles gracefully stopped sessions after a delay.
                elif not session_data.get("is_active") and \
                     session_data.get("is_finalizing") and \
                     now - session_data.get("finalization_timestamp", 0) > 120: # 2-minute grace period
                    logger.info(f"Idle session cleanup: Session {session_id} was finalized >120s ago. Marking for deletion.")
                    sessions_to_delete.append(session_id)

        # Finalize sessions that need it
        for session_id_to_finalize in sessions_to_finalize:
            logger.info(f"Idle session cleanup: Finalizing disconnected session {session_id_to_finalize}")
            _finalize_session(session_id_to_finalize)

        # Delete sessions that have been finalized for a while
        for session_id_to_delete in sessions_to_delete:
            logger.info(f"Idle session cleanup: Deleting garbage-collected session {session_id_to_delete}")
            with session_locks[session_id_to_delete]:
                if session_id_to_delete in active_sessions:
                    session_data = active_sessions[session_id_to_delete]
                    temp_session_audio_dir = session_data.get('temp_audio_session_dir')
                    
                    if temp_session_audio_dir and os.path.exists(temp_session_audio_dir):
                        try:
                            # Using shutil.rmtree is more robust for non-empty directories
                            import shutil
                            shutil.rmtree(temp_session_audio_dir)
                            logger.info(f"Cleaned up temporary directory: {temp_session_audio_dir}")
                        except Exception as e:
                            logger.error(f"Error cleaning up temp directory {temp_session_audio_dir}: {e}")
                    
                    del active_sessions[session_id_to_delete]
                
                if session_id_to_delete in session_locks:
                    del session_locks[session_id_to_delete]
            logger.info(f"Session {session_id_to_delete} fully cleaned up and removed.")

if __name__ == '__main__':
    if get_supabase_client():
        sync_agents_from_s3_to_supabase()
    else:
        logger.warning("Skipping agent sync because Supabase client failed to initialize.")
    
    idle_cleanup_thread = threading.Thread(target=cleanup_idle_sessions, daemon=True)
    idle_cleanup_thread.start()
    logger.info("Idle session cleanup thread started.")

    

    port = int(os.getenv('PORT', 5001)); debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    
    logger.info(f"PII Filtering Enabled for Transcripts: {os.getenv('ENABLE_TRANSCRIPT_PII_FILTERING', 'false')}")
    logger.info(f"PII Redaction LLM Model: {os.getenv('PII_REDACTION_MODEL_NAME', 'claude-3-haiku-20240307')}")
    logger.info(f"PII Redaction Fallback Behavior: {os.getenv('PII_REDACTION_FALLBACK_BEHAVIOR', 'regex_only')}")

    use_reloader_env = os.getenv('FLASK_USE_RELOADER', 'False').lower() == 'true'
    
    if not os.getenv("GUNICORN_CMD"):
        effective_reloader = use_reloader_env if debug_mode else False
        logger.info(f"Running with Flask dev server. Debug: {debug_mode}, Reloader: {effective_reloader}")
        app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=effective_reloader)
    else:
        logger.info("Gunicorn is expected to manage the application. Flask's app.run() will not be called directly.")
