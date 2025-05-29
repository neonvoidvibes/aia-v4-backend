import os
import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from dotenv import load_dotenv
import threading 
import time
import json
from datetime import datetime, timezone, timedelta # Ensure datetime, timezone, timedelta are imported
import urllib.parse 
from functools import wraps 
from typing import Optional, List, Dict, Any, Tuple, Union 
import uuid 
from collections import defaultdict
import subprocess 
from werkzeug.utils import secure_filename # Added for file uploads

from flask_sock import Sock 

from supabase import create_client, Client
from gotrue.errors import AuthApiError
from gotrue.types import User as SupabaseUser 

from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content 
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history, get_s3_client,
    list_agent_names_from_s3, list_s3_objects_metadata, get_transcript_summaries # Added get_transcript_summaries
)
from utils.transcript_summarizer import generate_transcript_summary # Added import
from utils.pinecone_utils import init_pinecone
from pinecone.exceptions import NotFoundException
from anthropic import Anthropic, APIStatusError, AnthropicError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
from flask_cors import CORS

from transcription_service import process_audio_segment_and_update_s3

load_dotenv()

def setup_logging(debug=False):
    log_filename = 'api_server.log'; root_logger = logging.getLogger(); log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8'); fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'); fh.setFormatter(ff); root_logger.addHandler(fh)
    except Exception as e: print(f"Error setting up file logger: {e}", file=sys.stderr)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level)
    cf = logging.Formatter('[%(levelname)-8s] %(name)s: %(message)s'); ch.setFormatter(cf); root_logger.addHandler(ch)
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone', 'werkzeug', 'flask_sock']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('utils').setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('transcription_service').setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger

s3_bucket_on_startup = os.getenv('AWS_S3_BUCKET')
startup_logger = logging.getLogger(__name__ + ".startup") 
if s3_bucket_on_startup:
    startup_logger.info(f"AWS_S3_BUCKET on startup: '{s3_bucket_on_startup}'")
else:
    startup_logger.error("AWS_S3_BUCKET environment variable is NOT SET at startup!")

logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

app = Flask(__name__)
CORS(app) 
sock = Sock(app) 
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

UPLOAD_FOLDER = 'tmp/uploaded_transcriptions/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
try: 
    init_pinecone()
    logger.info("Pinecone initialized (or skipped).")
except Exception as e: 
    logger.warning(f"Pinecone initialization failed: {e}")

supabase: Optional[Client] = None
try:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("Supabase environment variables (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) not found. Supabase features disabled.")
    else:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)

transcript_state_cache: Dict[Tuple[str, str], TranscriptState] = {} 
transcript_state_lock = threading.Lock() 

try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not found")
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized.")
except Exception as e: 
    logger.critical(f"Failed Anthropic client init: {e}", exc_info=True)
    anthropic_client = None # Keep it None on failure so other parts can check

def log_retry_error(retry_state): 
    logger.warning(f"Retrying API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")
retry_strategy = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry_error_callback=log_retry_error, retry=(retry_if_exception_type(APIStatusError)))

active_sessions: Dict[str, Dict[str, Any]] = {}
session_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock) 

def verify_user(token: Optional[str]) -> Optional[SupabaseUser]: 
    if not supabase:
        logger.error("Auth check failed: Supabase client not initialized.")
        return None

    if not token:
        logger.warning("Auth check failed: No token provided.")
        return None

    try:
        user_resp = supabase.auth.get_user(token)
        if user_resp and user_resp.user:
            logger.debug(f"Token verified for user ID: {user_resp.user.id}")
            return user_resp.user 
        else:
            logger.warning("Auth check failed: Invalid token or user not found in response.")
            return None
    except AuthApiError as e:
        logger.warning(f"Auth API Error during token verification: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
        return None

def verify_user_agent_access(token: Optional[str], agent_name: Optional[str]) -> Tuple[Optional[SupabaseUser], Optional[Response]]: 
    if not supabase:
        logger.error("Auth check failed: Supabase client not initialized.")
        return None, jsonify({"error": "Auth service unavailable"}, 503)

    user = verify_user(token)
    if not user: 
        return None, jsonify({"error": "Unauthorized: Invalid or missing token"}), 401
    
    if not agent_name:
        logger.debug(f"Authorization check: User {user.id} authenticated. No specific agent access check required for this route.")
        return user, None

    try:
        agent_res = supabase.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            logger.error(f"Database error finding agent_id for name '{agent_name}': {agent_res.error}")
            return None, jsonify({"error": "Database error checking agent"}), 500
        if not agent_res.data:
            logger.warning(f"Authorization check failed: Agent with name '{agent_name}' not found in DB.")
            return None, jsonify({"error": "Forbidden: Agent not found"}), 403
        
        agent_id = agent_res.data[0]['id']
        logger.debug(f"Found agent_id '{agent_id}' for name '{agent_name}'.")

        access_res = supabase.table("user_agent_access") \
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

    except Exception as e:
        logger.error(f"Unexpected error during agent access check for user {user.id} / agent {agent_name}: {e}", exc_info=True)
        return None, jsonify({"error": "Internal server error during authorization"}), 500

def supabase_auth_required(agent_required: bool = True):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            token = None
            if auth_header and auth_header.startswith("Bearer "): 
                token = auth_header.split(" ", 1)[1]
            
            agent_name_from_payload = None
            if agent_required:
                if request.is_json and request.json and 'agent' in request.json:
                    agent_name_from_payload = request.json.get('agent')
                elif 'agent' in request.args: 
                    agent_name_from_payload = request.args.get('agent')
                # For manage-file, agentName is expected in payload
                elif request.is_json and request.json and 'agentName' in request.json and request.path.endswith('/manage-file'):
                    agent_name_from_payload = request.json.get('agentName')


            user, error_response = verify_user_agent_access(token, agent_name_from_payload if agent_required else None)
            if error_response: 
                status_code = error_response.status_code if hasattr(error_response, 'status_code') else 500
                error_json = error_response.get_json() if hasattr(error_response, 'get_json') else {"error": "Unknown authorization error"}
                return jsonify(error_json), status_code
            return f(user=user, *args, **kwargs) 
        return decorated_function
    return decorator

@retry_strategy
def _call_anthropic_stream_with_retry(model, max_tokens, system, messages):
    if not anthropic_client: raise RuntimeError("Anthropic client not initialized.")
    # Enhanced logging for API call payload
    logger.info("="*15 + " ANTHROPIC API CALL (STREAM) " + "="*15)
    logger.info(f"Model: {model}, MaxTokens: {max_tokens}")
    logger.info(f"System Prompt Length: {len(system)}")
    if logger.isEnabledFor(logging.DEBUG): # Only log full system prompt in debug
        logger.debug(f"System Prompt (first 500 chars): {system[:500]}...")
        logger.debug(f"System Prompt (last 500 chars): ...{system[-500:]}")
    
    logger.info(f"Number of Messages for API: {len(messages)}")
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content_len = len(msg.get("content", ""))
        # Log more for the potentially large initial transcript message
        if msg.get("content", "").startswith("IMPORTANT CONTEXT: The following is the full meeting transcript"):
            content_snippet = msg.get("content", "")[:300] + " ... [TRUNCATED INITIAL TRANSCRIPT] ... " + msg.get("content", "")[-300:]
        else:
            content_snippet = msg.get("content", "")[:150] # Shorter snippet for other messages
        logger.info(f"  LLM Message [{i}]: Role='{role}', Length={content_len}, Content Snippet='{content_snippet}...'")
    logger.info("="*15 + " END ANTHROPIC API CALL " + "="*15)
    
    return anthropic_client.messages.stream(model=model, max_tokens=max_tokens, system=system, messages=messages)

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "ok", "message": "Backend is running", "anthropic_client_initialized": anthropic_client is not None, "s3_client_initialized": get_s3_client() is not None}), 200

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
    data = request.json
    agent_name = data.get('agent') 
    event_id = data.get('event')
    language = data.get('language') 
    
    if not event_id: 
        return jsonify({"status": "error", "message": "Missing event ID"}), 400

    session_id = uuid.uuid4().hex
    session_start_time_utc = datetime.now(timezone.utc)
    
    s3_transcript_base_filename = f"transcript_D{session_start_time_utc.strftime('%Y%m%d')}-T{session_start_time_utc.strftime('%H%M%S')}_uID-{user.id}_oID-river_aID-{agent_name}_eID-{event_id}_sID-{session_id}.txt"
    s3_transcript_key = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/{s3_transcript_base_filename}"
    
    temp_audio_base_dir = os.path.join('tmp', 'audio_sessions', session_id)
    os.makedirs(temp_audio_base_dir, exist_ok=True)
    
    active_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user.id,
        "agent_name": agent_name,
        "event_id": event_id,
        "language": language, 
        "session_start_time_utc": session_start_time_utc,
        "s3_transcript_key": s3_transcript_key,
        "temp_audio_session_dir": temp_audio_base_dir, 
        "is_backend_processing_paused": False,
        "current_total_audio_duration_processed_seconds": 0.0,
        "websocket_connection": None,
        "last_activity_timestamp": time.time(),
        "is_active": True, 
        "is_finalizing": False, 
        "current_segment_raw_bytes": bytearray(),
        "accumulated_audio_duration_for_current_segment_seconds": 0.0,
        "actual_segment_duration_seconds": 0.0,
        "webm_global_header_bytes": None, 
        "is_first_blob_received": False,   
    }
    logger.info(f"Recording session {session_id} started for agent {agent_name}, event {event_id} by user {user.id}.")
    logger.info(f"Session temp audio dir: {temp_audio_base_dir}, S3 transcript key: {s3_transcript_key}")
    
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if s3 and aws_s3_bucket:
        header = f"# Transcript - Session {session_id}\nAgent: {agent_name}, Event: {event_id}\nUser: {user.id}\nSession Started (UTC): {session_start_time_utc.isoformat()}\n\n"
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
        logger.info(f"Finalizing session {session_id} (marked is_finalizing=True, is_active=False).")
        
        current_fragment_bytes_final = bytes(session_data.get("current_segment_raw_bytes", bytearray()))
        global_header_bytes_final = session_data.get("webm_global_header_bytes", b'')
        
        all_final_segment_bytes = b''
        if global_header_bytes_final and current_fragment_bytes_final:
            if current_fragment_bytes_final.startswith(global_header_bytes_final): 
                all_final_segment_bytes = current_fragment_bytes_final
            else:
                all_final_segment_bytes = global_header_bytes_final + current_fragment_bytes_final
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
                ffmpeg_command = ['ffmpeg', '-y', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', final_output_wav_path]
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

                    process_audio_segment_and_update_s3(final_output_wav_path, session_data, session_locks[session_id])
                else:
                    logger.error(f"Session {session_id} Finalize: ffmpeg direct WAV conversion failed. RC: {process.returncode}, Err: {stderr.decode('utf-8', 'ignore')}")
            except Exception as e:
                logger.error(f"Error processing final audio segment (piped) for session {session_id}: {e}", exc_info=True)
            finally:
                pass
        else: 
            logger.info(f"Session {session_id} Finalize: No remaining audio bytes to process.")
        
        ws = session_data.get("websocket_connection")
        if ws:
            logger.info(f"Attempting to close WebSocket for session {session_id} during finalization.")
            try:
                ws.close(1000, "Session stopped by server (finalize)")
                logger.info(f"WebSocket for session {session_id} close() called.")
            except Exception as e: 
                logger.warning(f"Error closing WebSocket for session {session_id} during finalization: {e}", exc_info=True)
    
    if session_id in active_sessions: 
        temp_session_audio_dir = session_data['temp_audio_session_dir']
        if os.path.exists(temp_session_audio_dir):
            try:
                for root_dir, dirs, files in os.walk(temp_session_audio_dir, topdown=False):
                    for name in files:
                        try: os.remove(os.path.join(root_dir, name))
                        except OSError as e_file: logger.warning(f"Error deleting file {os.path.join(root_dir, name)}: {e_file}")
                    for name in dirs:
                        try: os.rmdir(os.path.join(root_dir, name))
                        except OSError as e_dir: logger.warning(f"Error deleting dir {os.path.join(root_dir, name)}: {e_dir}")
                try: os.rmdir(temp_session_audio_dir) 
                except OSError as e_main_dir: logger.warning(f"Error deleting main session dir {temp_session_audio_dir}: {e_main_dir}")
                logger.info(f"Cleaned up temporary directory: {temp_session_audio_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {temp_session_audio_dir}: {e}")

        if session_id in active_sessions: 
            del active_sessions[session_id]
        if session_id in session_locks: 
            del session_locks[session_id] 
        logger.info(f"Session {session_id} finalized and removed from active sessions.")
    else:
        logger.warning(f"Session {session_id} was already removed from active_sessions before final cleanup phase.")

@app.route('/api/s3/summarize-transcript', methods=['POST'])
@supabase_auth_required(agent_required=False) # agentName will be in payload
def summarize_transcript_route(user: SupabaseUser):
    logger.info(f"Received request /api/s3/summarize-transcript from user: {user.id}")
    data = request.json
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
    
    if not anthropic_client:
        logger.error("SummarizeTranscript: Anthropic client not initialized.")
        return jsonify({"error": "LLM service for summarization not available"}), 503

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
            llm_client=anthropic_client 
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
                    # For now, return success for summary creation but log the move error. The client won't know about this specific error.
                    # A more robust solution might involve a multi-step transaction or a cleanup mechanism.
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
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Missing session_id"}), 400

    logger.info(f"Stop recording request for session {session_id} by user {user.id}") 
    
    if session_id not in active_sessions:
        logger.warning(f"Stop request for session {session_id}, but session not found in active_sessions. It might have already been finalized or never started.")
        return jsonify({"status": "success", "message": "Session already stopped or not found", "session_id": session_id}), 200 
    
    _finalize_session(session_id)
    
    return jsonify({
        "status": "success", 
        "message": "Recording session stopped", 
        "session_id": session_id,
        "recording_status": {"is_recording": False, "is_backend_processing_paused": False} 
    })

@sock.route('/ws/audio_stream/<session_id>')
def audio_stream_socket(ws, session_id: str):
    token = request.args.get('token')
    user = verify_user(token) 
    
    if not user:
        logger.warning(f"WebSocket Auth Failed: Invalid token for session {session_id}. Closing.")
        ws.close(reason='Authentication failed')
        return

    logger.info(f"WebSocket connection attempt for session {session_id}, user {user.id}")

    if session_id not in active_sessions:
        logger.warning(f"WebSocket: Session {session_id} not found. Closing.")
        ws.close(reason='Session not found or not initialized')
        return
    
    if active_sessions[session_id].get("websocket_connection") is not None:
        logger.warning(f"WebSocket: Session {session_id} already has a WebSocket connection. Closing new one.")
        ws.close(reason='Session already has an active stream')
        return

    active_sessions[session_id]["websocket_connection"] = ws
    active_sessions[session_id]["last_activity_timestamp"] = time.time()
    active_sessions[session_id]["is_active"] = True 
    logger.info(f"WebSocket for session {session_id} (user {user.id}) connected and registered.")

    AUDIO_SEGMENT_DURATION_SECONDS_TARGET = 15 

    try:
        while True:
            message = ws.receive(timeout=1) 

            if message is None: 
                if active_sessions.get(session_id) and (time.time() - active_sessions[session_id]["last_activity_timestamp"] > 60): 
                    logger.warning(f"WebSocket for session {session_id} timed out due to inactivity. Closing.")
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
                with session_locks[session_id]: 
                    if session_id not in active_sessions: 
                        logger.warning(f"WebSocket {session_id}: Session disappeared after acquiring lock. Aborting message processing.")
                        break
                    session_data = active_sessions[session_id]
                    
                    if not session_data.get("is_first_blob_received", False):
                        session_data["webm_global_header_bytes"] = bytes(message) 
                        session_data["is_first_blob_received"] = True
                        logger.info(f"Session {session_id}: Captured first blob as global WebM header ({len(message)} bytes).")
                        session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                    else:
                        session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                    
                    logger.debug(f"Session {session_id}: Appended {len(message)} bytes to raw_bytes buffer. Total buffer: {len(session_data['current_segment_raw_bytes'])}")

                    session_data["accumulated_audio_duration_for_current_segment_seconds"] += 3.0 

                    if not session_data["is_backend_processing_paused"] and \
                       session_data["accumulated_audio_duration_for_current_segment_seconds"] >= AUDIO_SEGMENT_DURATION_SECONDS_TARGET:
                        
                        logger.info(f"Session {session_id}: Accumulated enough audio ({session_data['accumulated_audio_duration_for_current_segment_seconds']:.2f}s est.). Processing segment from raw bytes.")
                        
                        current_fragment_bytes = bytes(session_data["current_segment_raw_bytes"])
                        global_header_bytes = session_data.get("webm_global_header_bytes", b'')

                        if not global_header_bytes and current_fragment_bytes:
                            logger.warning(f"Session {session_id}: Global header not captured, but processing fragments. This might fail if not the very first segment.")
                            all_segment_bytes = current_fragment_bytes
                        elif global_header_bytes and current_fragment_bytes:
                            if current_fragment_bytes.startswith(global_header_bytes) and len(global_header_bytes) > 0:
                                all_segment_bytes = current_fragment_bytes
                                logger.debug(f"Session {session_id}: Processing first segment data which includes its own header.")
                            else:
                                all_segment_bytes = global_header_bytes + current_fragment_bytes
                                logger.debug(f"Session {session_id}: Prepended global header ({len(global_header_bytes)} bytes) to current fragments ({len(current_fragment_bytes)} bytes).")
                        elif global_header_bytes and not current_fragment_bytes:
                            logger.warning(f"Session {session_id}: Global header exists but no current fragments. Skipping empty segment.")
                            session_data["current_segment_raw_bytes"] = bytearray() 
                            session_data["accumulated_audio_duration_for_current_segment_seconds"] = 0.0
                            session_data["actual_segment_duration_seconds"] = 0.0
                            continue
                        logger.info(f"Session {session_id}: Accumulated enough audio ({session_data['accumulated_audio_duration_for_current_segment_seconds']:.2f}s est.). Processing segment from raw bytes.")
                        
                        bytes_to_process = bytes(session_data["current_segment_raw_bytes"])
                        global_header_for_thread = session_data.get("webm_global_header_bytes", b'')
                        
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
                            audio_bytes, wav_path, current_offset
                        ):
                            try:
                                ffmpeg_command = ['ffmpeg', '-y', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', wav_path]
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
                                    estimated_dur = len(audio_bytes) / (16000 * 2 * 0.2) 
                                    actual_segment_dur = estimated_dur
                                    logger.warning(f"Thread Session {s_id}: Using rough estimated duration: {actual_segment_dur:.2f}s")
                                
                                with lock_ref:
                                    if s_id in active_sessions: 
                                        active_sessions[s_id]['actual_segment_duration_seconds'] = actual_segment_dur
                                    else:
                                        logger.warning(f"Thread Session {s_id}: Session data missing when trying to update actual_segment_duration. Transcription might use old offset.")
                                        s_data_ref['actual_segment_duration_seconds'] = actual_segment_dur

                                success_transcribe = process_audio_segment_and_update_s3(wav_path, s_data_ref, lock_ref)
                                if not success_transcribe:
                                     logger.error(f"Thread Session {s_id}: Transcription or S3 update failed for segment {wav_path}")

                            except Exception as thread_e:
                                logger.error(f"Thread Session {s_id}: Error during threaded segment processing: {thread_e}", exc_info=True)
                            finally:
                                pass
                        
                        processing_thread = threading.Thread(
                            target=process_segment_in_thread,
                            args=(
                                session_id,
                                session_data, 
                                session_locks[session_id], 
                                all_segment_bytes_for_ffmpeg_thread,
                                final_output_wav_path_thread,
                                session_data['current_total_audio_duration_processed_seconds'] 
                            )
                        )
                        processing_thread.start()
                        logger.info(f"Session {session_id}: Started processing thread for segment {segment_uuid_thread}")
            
            if session_id not in active_sessions:
                logger.info(f"WebSocket session {session_id}: Session was externally stopped during message processing. Closing connection.")
                if ws.fileno() != -1: 
                    try: ws.close(1000, "Session stopped externally")
                    except Exception as e_close: logger.error(f"Error closing WebSocket for externally stopped session {session_id}: {e_close}")
                break

    except ConnectionResetError:
        logger.warning(f"WebSocket for session {session_id} (user {user.id}): Connection reset by client.")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id} (user {user.id}): {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket for session {session_id} (user {user.id}) disconnected.")
        if session_id in active_sessions and active_sessions[session_id].get("websocket_connection") == ws:
            with session_locks[session_id]: 
                 if session_id in active_sessions: 
                    active_sessions[session_id]["websocket_connection"] = None
                    logger.info(f"WebSocket for session {session_id} (user {user.id}) deregistered in finally block.")
        

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

@app.route('/internal_api/transcribe_file', methods=['POST'])
@supabase_auth_required(agent_required=False)
def transcribe_uploaded_file(user: SupabaseUser):
    logger.info(f"Received request /internal_api/transcribe_file from user: {user.id}")
    if 'audio_file' not in request.files:
        logger.warning("No audio_file part in request files.")
        return jsonify({"error": "No audio_file part in the request"}), 400
    
    file = request.files['audio_file']
    
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

            # Get agent_name from form data
            agent_name_from_form = request.form.get('agent_name', 'UnknownAgent')
            logger.info(f"Agent name from form for header: {agent_name_from_form}")

            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error("OpenAI API key not found for transcription service.")
                return jsonify({"error": "Transcription service not configured (missing API key)"}), 500
            
            from transcription_service import _transcribe_audio_segment_openai as transcribe_whisper_file
            
            logger.info(f"Starting transcription for {temp_filepath}...")
            transcription_data = transcribe_whisper_file(
                audio_file_path=temp_filepath,
                openai_api_key=openai_api_key,
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
    try:
        from utils.s3_utils import read_file_content as s3_read_content 
        content = s3_read_content(s3_key, f"S3 file for viewing ({s3_key})")
        if content is None: return jsonify({"error": "File not found or could not be read"}), 404
        return jsonify({"content": content}), 200
    except Exception as e: 
        logger.error(f"Error viewing S3 object '{s3_key}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error viewing S3 object"}), 500

@app.route('/api/s3/manage-file', methods=['POST'])
@supabase_auth_required(agent_required=False) # Agent name from payload is used for path construction
def manage_s3_file(user: SupabaseUser):
    logger.info(f"Received request /api/s3/manage-file from user: {user.id}")
    data = request.json
    s3_key_to_manage = data.get('s3Key')
    action_to_perform = data.get('action')
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

@app.route('/api/s3/download', methods=['GET'])
@supabase_auth_required(agent_required=False)
def download_s3_document(user: SupabaseUser):
    logger.info(f"Received request /api/s3/download from user: {user.id}") 
    s3_key = request.args.get('s3Key'); filename_param = request.args.get('filename')
    if not s3_key: return jsonify({"error": "Missing 's3Key' query parameter"}), 400
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
    logger.info(f"Received request /api/chat method: {request.method} from user: {user.id}") 
    global transcript_state_cache
    if not anthropic_client: 
        logger.error("Chat fail: Anthropic client not init.")
        return jsonify({"error": "AI service unavailable"}), 503
    try:
        data = request.json
        if not data or 'messages' not in data: 
            return jsonify({"error": "Missing 'messages'"}), 400
        agent_name = data.get('agent') 
        event_id = data.get('event', '0000')
        chat_session_id_log = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S')) 
        logger.info(f"Chat request for Agent: {agent_name}, Event: {event_id} by User: {user.id}")
        
        incoming_messages = data['messages']
        llm_messages_from_client = [{"role": msg["role"], "content": msg["content"]} 
                                    for msg in incoming_messages 
                                    if msg.get("role") in ["user", "assistant"]]
        
        if not llm_messages_from_client: 
            logger.warning("Received chat request with no user/assistant messages from client.")

        final_llm_messages: List[Dict[str, str]] = []

        try:
            base_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
            frameworks = get_latest_frameworks(agent_name); 
            event_context = get_latest_context(agent_name, event_id); 
            agent_docs = get_agent_docs(agent_name)
            logger.debug("Loaded base system prompt, frameworks, context, docs from S3.")
        except Exception as e: 
            logger.error(f"Error loading prompts/context from S3: {e}", exc_info=True); 
            base_system_prompt = "Error: Could not load system configuration."; frameworks = event_context = agent_docs = None
        
        source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify exact source file name (e.g., `frameworks_base.md`, `context_aID-river_eID-20240116.txt`, `transcript_...txt`, `doc_XYZ.pdf`) for info using Markdown footnotes like `[^1]`. \n2. Place footnote directly after sentence/paragraph. \n3. List all cited sources at end under `### Sources` as `[^1]: source_file_name.ext`."
        
        transcript_handling_instructions = (
            "\n\n## IMPORTANT INSTRUCTIONS FOR USING TRANSCRIPTS:\n"
            "1.  **Initial Full Transcript:** The very first user message may contain a block labeled '=== BEGIN FULL MEETING TRANSCRIPT ===' to '=== END FULL MEETING TRANSCRIPT ==='. This is the complete historical context of the meeting up to the start of our current conversation. You MUST refer to this entire block for any questions about past events, overall context, or specific details mentioned earlier in the meeting.\n"
            "2.  **Live Updates:** Subsequent user messages starting with '[Meeting Transcript Update (from S3)]' provide new, live additions to the transcript. These updates are chronological and should be considered the most current information.\n"
            "3.  **Comprehensive Awareness:** When asked about the transcript (e.g., 'what can you see?', 'what's the first/last timestamp?'), your answer must be based on ALL transcript information you have received, including the initial full load AND all subsequent delta updates. Do not only refer to the latest delta.\n"
            "4.  **Source Attribution:** When referencing information from any transcript segment (initial or delta), cite the (Source File: ...) provided within that segment if possible, or generally mention it's from the meeting transcript."
        )
        
        final_system_prompt = base_system_prompt + source_instr + transcript_handling_instructions
        
        if frameworks: final_system_prompt += "\n\n## Frameworks\n" + frameworks
        if event_context: final_system_prompt += "\n\n## Context\n" + event_context
        if agent_docs: final_system_prompt += "\n\n## Agent Documentation\n" + agent_docs
        
        rag_usage_instructions = "\n\n## Using Retrieved Context\n1. **Prioritize Info Within `[Retrieved Context]`:** Base answer primarily on info in `[Retrieved Context]` block below, if relevant. \n2. **Direct Extraction for Lists/Facts:** If user asks for list/definition/specific info explicit in `[Retrieved Context]`, present that info directly. Do *not* state info missing if clearly provided. \n3. **Cite Sources:** Remember cite source file name using Markdown footnotes (e.g., `[^1]`) for info from context, list sources under `### Sources`. \n4. **Synthesize When Necessary:** If query requires combining info or summarizing, do so, but ground answer in provided context. \n5. **Acknowledge Missing Info Appropriately:** Only state info missing if truly absent from context and relevant."
        final_system_prompt += rag_usage_instructions
        
        rag_context_block = ""; 
        last_actual_user_message_for_rag = next((msg['content'] for msg in reversed(llm_messages_from_client) if msg['role'] == 'user'), None)

        if last_actual_user_message_for_rag:
            normalized_query = last_actual_user_message_for_rag.strip().lower().rstrip('.!?')
            is_simple_query = normalized_query in SIMPLE_QUERIES_TO_BYPASS_RAG
            if not is_simple_query:
                logger.info(f"Complex query ('{normalized_query[:50]}...'), attempting RAG.")
                try: 
                    retriever = RetrievalHandler(index_name=agent_name, agent_name=agent_name, session_id=chat_session_id_log, event_id=event_id, anthropic_client=anthropic_client)
                    retrieved_docs = retriever.get_relevant_context(query=last_actual_user_message_for_rag, top_k=5)
                    if retrieved_docs:
                        items = [f"--- START Context Source: {d.metadata.get('file_name','Unknown')} (Score: {d.metadata.get('score',0):.2f}) ---\n{d.page_content}\n--- END Context Source: {d.metadata.get('file_name','Unknown')} ---" for i, d in enumerate(retrieved_docs)]
                        rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n" + "\n\n".join(items) + "\n=== END RETRIEVED CONTEXT ==="
                    else: rag_context_block = "\n\n[Note: No relevant documents found for this query via RAG.]"
                except RuntimeError as e: 
                    logger.warning(f"RAG skipped: {e}")
                    rag_context_block = f"\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval failed for index '{agent_name}']\n=== END RETRIEVED CONTEXT ==="
                except Exception as e: 
                    logger.error(f"Unexpected RAG error: {e}", exc_info=True)
                    rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Error retrieving documents via RAG]\n=== END RETRIEVED CONTEXT ==="
            elif is_simple_query: 
                rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval skipped for simple query.]\n=== END RETRIEVED CONTEXT ==="
        else: 
            rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval not applicable for this turn.]\n=== END RETRIEVED CONTEXT ==="
        final_system_prompt += rag_context_block 
        
        # Fetch and prepend saved transcript summaries
        try:
            summaries = get_transcript_summaries(agent_name, event_id)
            if summaries:
                logger.info(f"Found {len(summaries)} saved transcript summaries for agent '{agent_name}', event '{event_id}'.")
                summaries_context_str = "\n\n## Saved Transcript Summaries (Historical Context)\n"
                for summary_doc in summaries:
                    # Assuming summary_doc is the parsed JSON content
                    summary_filename = summary_doc.get("metadata", {}).get("summary_filename", "unknown_summary.json")
                    summaries_context_str += f"### Summary: {summary_filename}\n"
                    summaries_context_str += json.dumps(summary_doc, indent=2, ensure_ascii=False)
                    summaries_context_str += "\n\n"
                
                logger.debug(f"Constructed summaries_context_str (first 1000 chars): {summaries_context_str[:1000]}...")
                logger.debug(f"Constructed summaries_context_str (last 500 chars): ...{summaries_context_str[-500:]}")
                logger.info(f"Total length of summaries_context_str: {len(summaries_context_str)}")

                final_system_prompt = summaries_context_str + final_system_prompt
                logger.info(f"Prepended {len(summaries)} summaries to system prompt. New total length: {len(final_system_prompt)}") # Changed from debug to info
            else:
                logger.info(f"No saved transcript summaries found for agent '{agent_name}', event '{event_id}'.")
        except Exception as e_sum:
            logger.error(f"Error fetching or processing transcript summaries: {e_sum}", exc_info=True)
            # Continue without summaries if there's an error
            
        state_key = (agent_name, event_id)
        with transcript_state_lock:
            if state_key not in transcript_state_cache: 
                transcript_state_cache[state_key] = TranscriptState()
                logger.info(f"New TranscriptState created for {agent_name}/{event_id}")
            current_transcript_state = transcript_state_cache[state_key]
            
            logger.debug(f"Resetting cached initial_full_transcript_content and current_latest_key for {agent_name}/{event_id} before S3 read.")
            current_transcript_state.initial_full_transcript_content = None
            current_transcript_state.current_latest_key = None 
        
        try:
            transcript_data_from_s3, was_initial_load_attempt = read_new_transcript_content(
                current_transcript_state, agent_name, event_id
            )
            
            if current_transcript_state.initial_full_transcript_content and \
               not current_transcript_state.initial_full_transcript_content.startswith("__LOAD_ATTEMPTED"): 
                initial_transcript_message_content = (
                    f"IMPORTANT CONTEXT: The following is the full meeting transcript up to this point. "
                    f"Refer to this as the primary source for historical information throughout our conversation.\n\n"
                    f"=== BEGIN FULL MEETING TRANSCRIPT ===\n"
                    f"{current_transcript_state.initial_full_transcript_content}\n"
                    f"=== END FULL MEETING TRANSCRIPT ==="
                )
                final_llm_messages.append({'role': 'user', 'content': initial_transcript_message_content})
                logger.info(f"Prepended freshly loaded full transcript (length {len(current_transcript_state.initial_full_transcript_content)}) to current API call's messages.")

            final_llm_messages.extend(llm_messages_from_client)

            if transcript_data_from_s3 and not was_initial_load_attempt: 
                label = "[Meeting Transcript Update (from S3)]" 
                transcript_delta_to_add = f"{label}\n{transcript_data_from_s3}"
                
                insert_idx_for_delta = len(final_llm_messages) 
                for i in range(len(final_llm_messages) - 1, -1, -1):
                    msg_content_iter = final_llm_messages[i]['content']
                    is_prog_initial_load_msg = msg_content_iter.startswith("IMPORTANT CONTEXT: The following is the full meeting transcript")
                    is_prog_delta_update_msg = msg_content_iter.startswith("[Meeting Transcript Update (from S3)]")
                    
                    if final_llm_messages[i]['role'] == 'user' and not is_prog_initial_load_msg and not is_prog_delta_update_msg:
                        insert_idx_for_delta = i
                        break
                
                if insert_idx_for_delta == len(final_llm_messages) and final_llm_messages and \
                   final_llm_messages[0]['content'].startswith("IMPORTANT CONTEXT:"):
                    insert_idx_for_delta = 1

                final_llm_messages.insert(insert_idx_for_delta, {'role': 'user', 'content': transcript_delta_to_add})
                logger.info(f"Inserted transcript DELTA (length {len(transcript_data_from_s3)}) into LLM messages at effective index {insert_idx_for_delta} (this indicates growth during API call processing).")
            
            elif not transcript_data_from_s3 and was_initial_load_attempt:
                 logger.info(f"Initial transcript load for {agent_name}/{event_id} resulted in no data, or was already processed as initial.")
            elif not transcript_data_from_s3 and not was_initial_load_attempt:
                 logger.info(f"No new transcript delta to add for {agent_name}/{event_id} for this turn (after initial load).")

        except Exception as e: 
            logger.error(f"Error reading/processing transcript updates from S3: {e}", exc_info=True)

        now_utc = datetime.now(timezone.utc); time_str = now_utc.strftime('%A, %Y-%m-%d %H:%M:%S %Z')
        final_system_prompt += f"\nCurrent Time Context: {time_str}" 
        llm_model_name = os.getenv("LLM_MODEL_NAME", "claude-sonnet-4-20250514"); llm_max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))
        
        if not final_llm_messages: 
            logger.error("No messages to send to LLM after all processing. Aborting call.")
            return jsonify({"error": "No content to process for LLM."}), 400

        def generate_stream():
            response_content = ""; stream_error = None
            try:
                with _call_anthropic_stream_with_retry(model=llm_model_name, max_tokens=llm_max_tokens, system=final_system_prompt, messages=final_llm_messages) as stream:
                    for text in stream.text_stream: 
                        response_content += text
                        sse_data = json.dumps({'delta': text})
                        yield f"data: {sse_data}\n\n"
            except RetryError as e: 
                stream_error = "Assistant unavailable after retries."
                logger.error(f"RetryError during Anthropic stream: {e}")
            except APIStatusError as e:
                error_body_str = "Unknown API error structure"
                try:
                    error_detail_json = e.response.json()
                    error_body_str = json.dumps(error_detail_json) 
                except json.JSONDecodeError:
                    error_body_str = e.response.text if hasattr(e.response, 'text') else "Non-JSON error response"
                except Exception:
                    pass 
                stream_error = f"API Error: Status {e.status_code} - {error_body_str}"
                logger.error(f"APIStatusError during Anthropic stream: Status {e.status_code}, Body: {error_body_str}", exc_info=True)
            except AnthropicError as e: 
                stream_error = f"Anthropic Error: {str(e)}"
                logger.error(f"AnthropicError during Anthropic stream: {e}", exc_info=True)
            except Exception as e: 
                stream_error = f"Unexpected error during stream: {str(e)}"
                logger.error(f"Generic Exception during Anthropic stream: {e}", exc_info=True)
            
            if stream_error: 
                sse_error_data = json.dumps({'error': stream_error}); 
                yield f"data: {sse_error_data}\n\n"
            
            sse_done_data = json.dumps({'done': True}); 
            yield f"data: {sse_done_data}\n\n"
            
        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
    except Exception as e: 
        logger.error(f"Error in /api/chat: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

def sync_agents_from_s3_to_supabase():
    if not supabase: 
        logger.warning("Agent Sync: Supabase client not initialized. Skipping.")
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
        response = supabase.table("agents").select("name").execute()
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
        insert_response = supabase.table("agents").insert(agents_to_insert).execute()
        if hasattr(insert_response, 'error') and insert_response.error: 
            logger.error(f"Agent Sync: Error inserting agents into Supabase: {insert_response.error}")
        elif insert_response.data: 
            logger.info(f"Agent Sync: Successfully inserted {len(insert_response.data)} new agents into Supabase.")
        else: 
            logger.warning(f"Agent Sync: Insert call succeeded but reported 0 rows inserted. Check response: {insert_response}")
    except Exception as e: 
        logger.error(f"Agent Sync: Unexpected error inserting agents: {e}", exc_info=True)
    logger.info("Agent Sync: Synchronization finished.")

def cleanup_idle_sessions():
    while True:
        time.sleep(5 * 60) 
        now = time.time()
        sessions_to_cleanup = []
        for session_id, session_data in list(active_sessions.items()): 
            if not session_data.get("is_active", False) and not session_data.get("is_finalizing", False): 
                 if now - session_data.get("last_activity_timestamp", 0) > (15 * 60): 
                      logger.warning(f"Found stale inactive session {session_id}. Adding to cleanup queue.")
                      sessions_to_cleanup.append(session_id)
                 continue

            if session_data.get("websocket_connection") is None and \
               now - session_data.get("last_activity_timestamp", 0) > (10 * 60): 
                logger.warning(f"Session {session_id} has no WebSocket and is idle. Marking for cleanup.")
                sessions_to_cleanup.append(session_id)
            elif session_data.get("websocket_connection") is not None and \
                 now - session_data.get("last_activity_timestamp", 0) > (30 * 60): 
                logger.warning(f"Session {session_id} has an active WebSocket but is idle. Marking for cleanup.")
                sessions_to_cleanup.append(session_id)
        
        for session_id_to_clean in sessions_to_cleanup:
            logger.info(f"Idle session cleanup: Finalizing session {session_id_to_clean}")
            _finalize_session(session_id_to_clean)

if __name__ == '__main__':
    if supabase: 
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