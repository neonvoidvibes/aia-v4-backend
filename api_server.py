import os
import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from dotenv import load_dotenv
import threading
import time
import json
from datetime import datetime, timezone, timedelta # Added timedelta
import urllib.parse # For decoding filenames in IDs
from functools import wraps # For decorator
from typing import Optional, List, Dict, Any, Tuple # Added Tuple

# Supabase Imports
from supabase import create_client, Client
from gotrue.errors import AuthApiError

# Import necessary modules from our project
from magic_audio import MagicAudio
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history, get_s3_client,
    list_agent_names_from_s3, list_s3_objects_metadata # Import the new functions
)
from utils.pinecone_utils import init_pinecone
from pinecone.exceptions import NotFoundException

from anthropic import Anthropic, APIStatusError, AnthropicError
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
from flask_cors import CORS

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
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone', 'werkzeug']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('utils').setLevel(logging.DEBUG if debug else logging.INFO)
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger

# Log S3 bucket name on startup
s3_bucket_on_startup = os.getenv('AWS_S3_BUCKET')
startup_logger = logging.getLogger(__name__ + ".startup") # Use a distinct logger for startup messages
if s3_bucket_on_startup:
    startup_logger.info(f"AWS_S3_BUCKET on startup: '{s3_bucket_on_startup}'")
else:
    startup_logger.error("AWS_S3_BUCKET environment variable is NOT SET at startup!")


logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

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

magic_audio_instance: MagicAudio | None = None
magic_audio_lock = threading.Lock()
recording_status = {
    "is_recording": False, "is_paused": False, "start_time": None,
    "pause_start_time": None, # Tracks start of current pause period
    "last_pause_timestamp": None, # Tracks the absolute time the last pause began, used for auto-stop
    "elapsed_time": 0, # Stores cumulative elapsed time *excluding* pauses
    "agent": None, "event": None
}

anthropic_client: Anthropic | None = None
try: init_pinecone(); logger.info("Pinecone initialized (or skipped).")
except Exception as e: logger.warning(f"Pinecone initialization failed: {e}")

# Initialize Supabase Client
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

transcript_state_cache = {}
transcript_state_lock = threading.Lock()

try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not found")
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized.")
except Exception as e: logger.critical(f"Failed Anthropic client init: {e}", exc_info=True); anthropic_client = None

def log_retry_error(retry_state): logger.warning(f"Retrying API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")
retry_strategy = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry_error_callback=log_retry_error, retry=(retry_if_exception_type(APIStatusError)))

# --- Supabase Auth Helper Functions ---

def verify_user(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Verifies the JWT and returns the user object or None."""
    if not supabase:
        logger.error("Auth check failed: Supabase client not initialized.")
        return None # Or raise an internal server error

    if not token:
        logger.warning("Auth check failed: No token provided.")
        return None

    try:
        # Verify the token and get user data
        user_resp = supabase.auth.get_user(token)
        # Check if user data exists and is valid
        if user_resp and hasattr(user_resp, 'user') and user_resp.user:
            logger.debug(f"Token verified for user ID: {user_resp.user.id}")
            return user_resp.user
        else:
            logger.warning("Auth check failed: Invalid token or user not found.")
            return None
    except AuthApiError as e:
        logger.warning(f"Auth API Error during token verification: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
        return None

def verify_user_agent_access(token: Optional[str], agent_name: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Response]]:
    """Verifies JWT, checks user access to the agent by name, returns (user, error_response)."""
    if not supabase:
        logger.error("Auth check failed: Supabase client not initialized.")
        return None, jsonify({"error": "Auth service unavailable"}), 503

    user = verify_user(token)
    if not user:
        return None, jsonify({"error": "Unauthorized: Invalid or missing token"}), 401

    if not agent_name:
        logger.warning(f"Authorization check skipped for user {user.id}: No agent_name provided.")
        return user, None # Assume agent check is optional if not provided

    try:
        # 1. Find the agent_id from the agent_name
        agent_res = supabase.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            logger.error(f"Database error finding agent_id for name '{agent_name}': {agent_res.error}")
            return None, jsonify({"error": "Database error checking agent"}), 500
        if not agent_res.data:
            logger.warning(f"Authorization check failed: Agent with name '{agent_name}' not found in DB.")
            # Treat as forbidden - agent doesn't exist for permission check
            return None, jsonify({"error": "Forbidden: Agent not found"}), 403
        agent_id = agent_res.data[0]['id']
        logger.debug(f"Found agent_id '{agent_id}' for name '{agent_name}'.")

        # 2. Check user access using user_id and agent_id
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
        return user, None # User is authenticated and authorized

    except Exception as e:
        logger.error(f"Unexpected error during agent access check for user {user.id} / agent {agent_name}: {e}", exc_info=True)
        return None, jsonify({"error": "Internal server error during authorization"}), 500

# Decorator for simplified route protection
def supabase_auth_required(agent_required: bool = True):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            token = None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]

            agent_name = None # Use agent name now
            if agent_required:
                # Attempt to get agent name from request JSON body first
                if request.is_json and request.json and 'agent' in request.json:
                    agent_name = request.json.get('agent')
                # Or from Flask route parameters if not in body (adapt as needed)
                # Example: if agent_name is part of the URL like /api/agent/<agent_name>
                # agent_name = kwargs.get('agent_name') # This depends on your route definition

            # Pass agent_name to the verification function
            user, error_response = verify_user_agent_access(token, agent_name if agent_required else None)

            if error_response:
                # Return the error response directly from the helper
                status_code = error_response.status_code
                error_json = error_response.get_json()
                return jsonify(error_json), status_code

            # Inject user object into the request context or pass as argument if needed
            # For simplicity, we'll just proceed if auth is successful.
            # You could use Flask's 'g' object: g.user = user
            return f(user=user, *args, **kwargs) # Pass user object to the route function
        return decorated_function
    return decorator

# --- End Supabase Auth Helper Functions ---


@retry_strategy
def _call_anthropic_stream_with_retry(model, max_tokens, system, messages):
    if not anthropic_client: raise RuntimeError("Anthropic client not initialized.")
    logger.debug(f"Anthropic API: Model={model}, MaxTokens={max_tokens}, SystemPromptLen={len(system)}, NumMessages={len(messages)}")
    return anthropic_client.messages.stream(model=model, max_tokens=max_tokens, system=system, messages=messages)

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "ok", "message": "Backend is running", "anthropic_client": anthropic_client is not None, "s3_client": get_s3_client() is not None}), 200

@app.route('/api/status', methods=['GET'])
def get_app_status():
    with magic_audio_lock: rec_status = recording_status.copy()
    status_data = {
        'agent_name': rec_status.get("agent", "N/A"), 'event_id': rec_status.get("event", "N/A"),
        'listen_transcript': False, 'memory_enabled': True,
        'is_recording': rec_status.get("is_recording", False), 'is_paused': rec_status.get("is_paused", False),
    }
    logger.debug(f"Reporting app status: {status_data}")
    return jsonify(status_data), 200

def _get_current_recording_status_snapshot():
    """Helper to get a snapshot of the current recording status for responses.
       Calculates current elapsed time dynamically if recording and not paused.
    """
    status_copy = recording_status.copy() # Work with a copy
    calculated_elapsed = status_copy["elapsed_time"] # Start with stored cumulative time

    if status_copy["is_recording"] and not status_copy["is_paused"] and status_copy["start_time"]:
        # If actively recording, add time since last resume/start
        calculated_elapsed = time.time() - status_copy["start_time"]

    return {
        "is_recording": status_copy["is_recording"],
        "is_paused": status_copy["is_paused"],
        "elapsed_time": int(calculated_elapsed),
        "agent": status_copy.get("agent"),
        "event": status_copy.get("event"),
        "last_pause_timestamp": status_copy.get("last_pause_timestamp") # Include this field
    }

@app.route('/api/recording/start', methods=['POST'])
@supabase_auth_required(agent_required=True) # Requires agent in payload
def start_recording_route(user): # User object passed by decorator
    global magic_audio_instance, recording_status
    logger.info(f"Start recording request from user: {user.id}")
    with magic_audio_lock:
        if recording_status["is_recording"]:
            return jsonify({"status": "error", "message": "Already recording", "recording_status": _get_current_recording_status_snapshot()}), 400
        data = request.json; agent = data.get('agent'); event = data.get('event'); language = data.get('language')
        # Agent already verified by decorator, no need to check existence here
        if not event: # Check event explicitly
            return jsonify({"status": "error", "message": "Missing event", "recording_status": _get_current_recording_status_snapshot()}), 400
        logger.info(f"Starting recording for Agent: {agent}, Event: {event}, Lang: {language}")
        try:
            if magic_audio_instance:
                try: magic_audio_instance.stop() # Ensure previous stops completely first
                except Exception as e: logger.warning(f"Error stopping previous audio instance: {e}")
            magic_audio_instance = MagicAudio(agent=agent, event=event, language=language)
            magic_audio_instance.start()
            recording_status.update({
                "is_recording": True, "is_paused": False, "start_time": time.time(),
                "pause_start_time": None, # Reset pause specific fields
                "last_pause_timestamp": None, # Reset this field too
                "elapsed_time": 0, # Reset elapsed time
                "agent": agent, "event": event
            })
            logger.info("Recording started successfully.")
            return jsonify({"status": "success", "message": "Recording started", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True); magic_audio_instance = None
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "elapsed_time": 0, "agent": None, "event": None})
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

def _stop_magic_audio_async(instance_to_stop):
    """Function to run magic_audio.stop() in a separate thread."""
    global magic_audio_instance # Need global to potentially clear the instance ref
    if instance_to_stop:
        logger.info("Background stop thread started.")
        try:
            instance_to_stop.stop() # This might block
            logger.info("Background stop thread: magic_audio.stop() completed.")
            # Clear global instance only if it hasn't been replaced by a new one
            with magic_audio_lock:
                if magic_audio_instance == instance_to_stop:
                    magic_audio_instance = None
                    logger.info("Background stop thread: Cleared global magic_audio_instance.")
        except Exception as e:
            logger.error(f"Background stop thread error: {e}", exc_info=True)
        logger.info("Background stop thread finished.")
    else:
        logger.warning("Background stop thread: No instance provided to stop.")

@app.route('/api/recording/stop', methods=['POST'])
@supabase_auth_required(agent_required=False) # Just need authentication
def stop_recording_route(user): # User object passed by decorator
    global magic_audio_instance, recording_status
    logger.info(f"Stop recording request from user: {user.id}")
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "last_pause_timestamp": None, "elapsed_time": 0})
            logger.info("Stop called but not recording or instance missing. Status reset.")
            return jsonify({"status": "success", "message": "Not recording or instance missing", "recording_status": _get_current_recording_status_snapshot()}), 200

        logger.info("Stop recording requested. Initiating background stop...")

        # --- Optimistic Update & Async Stop ---
        # 1. Capture the instance to stop
        instance_to_stop = magic_audio_instance
        magic_audio_instance = None # Immediately nullify global ref to prevent reuse

        # 2. Calculate final elapsed time *before* resetting status
        final_elapsed_time = recording_status["elapsed_time"] # Start with stored cumulative
        if recording_status["is_recording"] and not recording_status["is_paused"] and recording_status["start_time"]:
             # If it was running when stop was called, add the last running duration
             final_elapsed_time = time.time() - recording_status["start_time"]

        # 3. Update global status optimistically (reflecting stopped state)
        recording_status.update({
            "is_recording": False, "is_paused": False, "start_time": None,
            "pause_start_time": None, "last_pause_timestamp": None,
            "elapsed_time": int(final_elapsed_time) # Store calculated time
        })

        # 4. Start background thread to call the blocking stop method
        stop_thread = threading.Thread(target=_stop_magic_audio_async, args=(instance_to_stop,))
        stop_thread.daemon = True # Allow app to exit even if this thread hangs (though stop should eventually finish)
        stop_thread.start()

        # 5. Return success response immediately
        logger.info("Stop recording request acknowledged. Background cleanup initiated.")
        # Return the status reflecting the *optimistic* stopped state
        final_status = _get_current_recording_status_snapshot()
        final_status["elapsed_time"] = 0 # Explicitly set 0 for UI stop
        return jsonify({"status": "success", "message": "Stop initiated", "recording_status": final_status})

@app.route('/api/recording/pause', methods=['POST'])
@supabase_auth_required(agent_required=False) # Just need authentication
def pause_recording_route(user): # User object passed by decorator
    global magic_audio_instance, recording_status
    logger.info(f"Pause recording request from user: {user.id}")
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            return jsonify({"status": "error", "message": "Not recording", "recording_status": _get_current_recording_status_snapshot()}), 400
        if recording_status["is_paused"]:
            return jsonify({"status": "success", "message": "Already paused", "recording_status": _get_current_recording_status_snapshot()}), 200
        logger.info("Pausing recording...")
        try:
            magic_audio_instance.pause()
            pause_time = time.time()
            recording_status["is_paused"] = True
            recording_status["pause_start_time"] = pause_time
            recording_status["last_pause_timestamp"] = pause_time # Record the absolute time pause began
            if recording_status["start_time"]:
                # Update the stored elapsed time up to the pause point
                recording_status["elapsed_time"] = pause_time - recording_status["start_time"]
            logger.info("Recording paused.")
            return jsonify({"status": "success", "message": "Recording paused", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Error pausing recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/resume', methods=['POST'])
@supabase_auth_required(agent_required=False) # Just need authentication
def resume_recording_route(user): # User object passed by decorator
    global magic_audio_instance, recording_status
    logger.info(f"Resume recording request from user: {user.id}")
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            return jsonify({"status": "error", "message": "Not recording", "recording_status": _get_current_recording_status_snapshot()}), 400
        if not recording_status["is_paused"]:
            return jsonify({"status": "success", "message": "Not paused", "recording_status": _get_current_recording_status_snapshot()}), 200
        logger.info("Resuming recording...")
        try:
            # Adjust start_time based on how long it was paused
            if recording_status["pause_start_time"] and recording_status["start_time"]:
                pause_duration = time.time() - recording_status["pause_start_time"]
                # Effectively shift the start time forward by the pause duration
                # This means elapsed time = current_time - adjusted_start_time
                recording_status["start_time"] += pause_duration
                logger.debug(f"Resuming after {pause_duration:.2f}s pause. Adjusted start time.")
            else: logger.warning("Could not calculate pause duration accurately on resume.")

            magic_audio_instance.resume()
            recording_status["is_paused"] = False
            recording_status["pause_start_time"] = None # Clear current pause start
            recording_status["last_pause_timestamp"] = None # Clear absolute pause timestamp
            logger.info("Recording resumed.")
            return jsonify({"status": "success", "message": "Recording resumed", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Error resuming recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/status', methods=['GET'])
def get_recording_status_route():
    with magic_audio_lock:
        status_data = _get_current_recording_status_snapshot()
    return jsonify(status_data), 200

# --- Pinecone Routes (Unchanged) ---
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
    except Exception as e: logger.error(f"Error getting stats for index '{index_name}': {e}", exc_info=True); return jsonify({"error": "An unexpected error occurred"}), 500

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
                     else: logger.warning(f"Unexpected item type from list gen: {type(id_batch)}");
                     if len(vector_ids) >= limit: vector_ids = vector_ids[:limit]; break
            else:
                logger.warning(f"index.list() not iterable. Type: {type(list_response)}")
                if hasattr(list_response, 'ids') and isinstance(list_response.ids, list): vector_ids = list_response.ids[:limit]
                elif hasattr(list_response, 'vectors') and isinstance(list_response.vectors, list): vector_ids = [v.id for v in list_response.vectors][:limit]
                if hasattr(list_response, 'pagination') and list_response.pagination and hasattr(list_response.pagination, 'next'): next_page_token = list_response.pagination.next
        except Exception as proc_e: logger.error(f"Error processing list response: {proc_e}", exc_info=True)
        logger.info(f"Found {len(vector_ids)} vector IDs in '{query_namespace}'. Next Token: {next_page_token}")
        return jsonify({"namespace": namespace_name, "vector_ids": vector_ids, "next_token": next_page_token}), 200
    except Exception as e: logger.error(f"Error listing vector IDs for index '{index_name}', ns '{namespace_name}': {e}", exc_info=True); return jsonify({"error": "Unexpected error listing vector IDs"}), 500

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
            list_response = index.list(namespace=query_namespace, limit=limit); count = 0
            if hasattr(list_response, '__iter__'):
                 for id_batch in list_response:
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch): vector_ids_to_parse.extend(id_batch); count += len(id_batch)
                     else: logger.warning(f"Unexpected item type from list gen: {type(id_batch)}")
                     if count >= limit: break
            else: logger.warning(f"index.list() not iterable. Type: {type(list_response)}")
            logger.info(f"Retrieved {len(vector_ids_to_parse)} vector IDs for parsing.")
        except Exception as list_e: logger.error(f"Error listing vector IDs: {list_e}", exc_info=True); return jsonify({"error": "Failed list vector IDs from Pinecone"}), 500
        unique_doc_names = set()
        for vec_id in vector_ids_to_parse:
            try:
                last_underscore_index = vec_id.rfind('_')
                if last_underscore_index != -1: unique_doc_names.add(urllib.parse.unquote_plus(vec_id[:last_underscore_index]))
                else: logger.warning(f"Vector ID '{vec_id}' no expected suffix.")
            except Exception as parse_e: logger.error(f"Error parsing vector ID '{vec_id}': {parse_e}")
        sorted_doc_names = sorted(list(unique_doc_names))
        logger.info(f"Found {len(sorted_doc_names)} unique doc names in ns '{query_namespace}'.")
        return jsonify({"index": index_name, "namespace": namespace_name, "unique_document_names": sorted_doc_names, "vector_ids_checked": len(vector_ids_to_parse)}), 200
    except Exception as e: logger.error(f"Error listing unique docs index '{index_name}', ns '{namespace_name}': {e}", exc_info=True); return jsonify({"error": "Unexpected error listing documents"}), 500

# --- S3 Document Management API Routes ---
@app.route('/api/s3/list', methods=['GET'])
@supabase_auth_required(agent_required=False) # No specific agent context needed, just auth
def list_s3_documents(user):
    logger.info(f"Received request /api/s3/list from user: {user.id}")
    s3_prefix = request.args.get('prefix')
    if not s3_prefix:
        return jsonify({"error": "Missing 'prefix' query parameter"}), 400

    try:
        s3_objects = list_s3_objects_metadata(s3_prefix)
        # Transform to frontend-friendly format
        formatted_files = []
        for obj in s3_objects:
            # Skip folder objects that might be returned by list_objects_v2 if prefix doesn't end with /
            if obj['Key'].endswith('/') and obj['Size'] == 0:
                continue

            filename = os.path.basename(obj['Key'])
            # Basic type detection from extension
            file_type = "text/plain" # Default
            if '.' in filename:
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in ['txt', 'md', 'log']: file_type = "text/plain"
                elif ext == 'json': file_type = "application/json"
                elif ext == 'xml': file_type = "application/xml"
                # Add more types as needed

            formatted_files.append({
                "name": filename,
                "size": obj['Size'],
                "lastModified": obj['LastModified'].isoformat() if obj.get('LastModified') else None,
                "s3Key": obj['Key'],
                "type": file_type
            })
        
        # Filter out files starting with 'rolling-' for transcription lists, if they are not desired.
        # This specific filtering is for the 'transcripts' prefix use case.
        # A more generic API wouldn't hardcode this. For now, it's fine as the frontend will primarily use it for transcripts.
        if "transcripts/" in s3_prefix:
            formatted_files = [f for f in formatted_files if not f['name'].startswith('rolling-')]

        return jsonify(formatted_files), 200
    except Exception as e:
        logger.error(f"Error listing S3 objects for prefix '{s3_prefix}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error listing S3 objects"}), 500

@app.route('/api/s3/view', methods=['GET'])
@supabase_auth_required(agent_required=False)
def view_s3_document(user):
    logger.info(f"Received request /api/s3/view from user: {user.id}")
    s3_key = request.args.get('s3Key')
    if not s3_key:
        return jsonify({"error": "Missing 's3Key' query parameter"}), 400

    try:
        # Re-use existing s3_utils.read_file_content
        from utils.s3_utils import read_file_content as s3_read_content # Local import to avoid name clash
        content = s3_read_content(s3_key, f"S3 file for viewing ({s3_key})")
        if content is None:
            return jsonify({"error": "File not found or could not be read"}), 404
        return jsonify({"content": content}), 200
    except Exception as e:
        logger.error(f"Error viewing S3 object '{s3_key}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error viewing S3 object"}), 500

@app.route('/api/s3/download', methods=['GET'])
@supabase_auth_required(agent_required=False)
def download_s3_document(user):
    logger.info(f"Received request /api/s3/download from user: {user.id}")
    s3_key = request.args.get('s3Key')
    filename_param = request.args.get('filename') # Optional desired filename

    if not s3_key:
        return jsonify({"error": "Missing 's3Key' query parameter"}), 400

    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if not s3 or not aws_s3_bucket:
        return jsonify({"error": "S3 client or bucket not configured"}), 500

    try:
        s3_object = s3.get_object(Bucket=aws_s3_bucket, Key=s3_key)
        
        # Determine filename for download
        download_filename = filename_param or os.path.basename(s3_key)
        
        return Response(
            s3_object['Body'].iter_chunks(),
            mimetype=s3_object.get('ContentType', 'application/octet-stream'),
            headers={"Content-Disposition": f"attachment;filename={download_filename}"}
        )
    except s3.exceptions.NoSuchKey:
        logger.warning(f"S3 Download: File not found at key: {s3_key}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error downloading S3 object '{s3_key}': {e}", exc_info=True)
        return jsonify({"error": "Internal server error downloading S3 object"}), 500

# --- Chat API Route ---
@app.route('/api/chat', methods=['POST'])
@supabase_auth_required(agent_required=True)
def handle_chat(user): # User object is now passed by the decorator
    logger.info(f"Received request /api/chat method: {request.method} from user: {user.id}")
    global transcript_state_cache
    # Anthropic client check can remain, but auth is handled by decorator
    if not anthropic_client: logger.error("Chat fail: Anthropic client not init."); return jsonify({"error": "AI service unavailable"}), 503
    try:
        data = request.json
        if not data or 'messages' not in data: return jsonify({"error": "Missing 'messages'"}), 400
        agent_name = data.get('agent') # Agent existence already verified by decorator if agent_required=True
        event_id = data.get('event', '0000')
        session_id = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S'))
        logger.info(f"Chat request for Agent: {agent_name}, Event: {event_id} by User: {user.id}")
        incoming_messages = data['messages']
        llm_messages = [{"role": msg["role"], "content": msg["content"]} for msg in incoming_messages if msg.get("role") in ["user", "assistant"]]
        if not llm_messages: return jsonify({"error": "No valid user/assistant messages"}), 400
        try:
            base_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
            frameworks = get_latest_frameworks(agent_name); event_context = get_latest_context(agent_name, event_id); agent_docs = get_agent_docs(agent_name)
            logger.debug("Loaded prompts/context/docs from S3.")
        except Exception as e: logger.error(f"Error loading prompts/context from S3: {e}", exc_info=True); base_system_prompt = "Error: Could not load config."; frameworks = event_context = agent_docs = None
        source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify exact source file name (e.g., `frameworks_base.md`, `context_aID-river_eID-20240116.txt`, `transcript_...txt`, `doc_XYZ.pdf`) for info using Markdown footnotes like `[^1]`. \n2. Place footnote directly after sentence/paragraph. \n3. List all cited sources at end under `### Sources` as `[^1]: source_file_name.ext`."
        realtime_instr = "\n\nIMPORTANT: Prioritize [REAL-TIME Meeting Transcript Update] content for 'current' or 'latest' state queries."
        final_system_prompt = base_system_prompt + source_instr + realtime_instr
        if frameworks: final_system_prompt += "\n\n## Frameworks\n" + frameworks
        if event_context: final_system_prompt += "\n\n## Context\n" + event_context
        if agent_docs: final_system_prompt += "\n\n## Agent Documentation\n" + agent_docs
        rag_usage_instructions = "\n\n## Using Retrieved Context\n1. **Prioritize Info Within `[Retrieved Context]`:** Base answer primarily on info in `[Retrieved Context]` block below, if relevant. \n2. **Direct Extraction for Lists/Facts:** If user asks for list/definition/specific info explicit in `[Retrieved Context]`, present that info directly. Do *not* state info missing if clearly provided. \n3. **Cite Sources:** Remember cite source file name using Markdown footnotes (e.g., `[^1]`) for info from context, list sources under `### Sources`. \n4. **Synthesize When Necessary:** If query requires combining info or summarizing, do so, but ground answer in provided context. \n5. **Acknowledge Missing Info Appropriately:** Only state info missing if truly absent from context and relevant."
        final_system_prompt += rag_usage_instructions
        rag_context_block = ""; last_user_message_content = next((msg['content'] for msg in reversed(llm_messages) if msg['role'] == 'user'), None)
        normalized_query = "";
        if last_user_message_content: normalized_query = last_user_message_content.strip().lower().rstrip('.!?')
        is_simple_query = normalized_query in SIMPLE_QUERIES_TO_BYPASS_RAG
        if not is_simple_query and last_user_message_content:
            logger.info(f"Complex query ('{normalized_query[:50]}...'), RAG.")
            try:
                retriever = RetrievalHandler(index_name=agent_name, agent_name=agent_name, session_id=session_id, event_id=event_id, anthropic_client=anthropic_client)
                retrieved_docs = retriever.get_relevant_context(query=last_user_message_content, top_k=5)
                if retrieved_docs:
                    items = [f"--- START Context Source: {d.metadata.get('file_name','Unknown')} (Score: {d.metadata.get('score',0):.2f}) ---\n{d.page_content}\n--- END Context Source: {d.metadata.get('file_name','Unknown')} ---" for i, d in enumerate(retrieved_docs)]
                    rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n" + "\n\n".join(items) + "\n=== END RETRIEVED CONTEXT ==="
                else: rag_context_block = "\n\n[Note: No relevant documents found for this query.]"
            except RuntimeError as e: logger.warning(f"RAG skipped: {e}"); rag_context_block = f"\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval failed for index '{agent_name}']\n=== END RETRIEVED CONTEXT ==="
            except Exception as e: logger.error(f"Unexpected RAG error: {e}", exc_info=True); rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Error retrieving documents]\n=== END RETRIEVED CONTEXT ==="
        elif is_simple_query: rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval skipped for simple query.]\n=== END RETRIEVED CONTEXT ==="
        else: rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval not applicable.]\n=== END RETRIEVED CONTEXT ==="
        final_system_prompt += rag_context_block
        transcript_content_to_add = ""; state_key = (agent_name, event_id)
        with transcript_state_lock:
            if state_key not in transcript_state_cache: transcript_state_cache[state_key] = TranscriptState(); logger.info(f"New TranscriptState for {agent_name}/{event_id}")
            current_transcript_state = transcript_state_cache[state_key]
        try:
            new_transcript = read_new_transcript_content(current_transcript_state, agent_name, event_id)
            if new_transcript: label = "[REAL-TIME Meeting Transcript Update]"; transcript_content_to_add = f"{label}\n{new_transcript}"; llm_messages.insert(-1, {'role': 'user', 'content': transcript_content_to_add})
        except Exception as e: logger.error(f"Error reading tx updates: {e}", exc_info=True)
        now_utc = datetime.now(timezone.utc); time_str = now_utc.strftime('%A, %Y-%m-%d %H:%M:%S %Z')
        final_system_prompt += f"\nCurrent Time Context: {time_str}"
        llm_model_name = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20240620"); llm_max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))
        def generate_stream():
            response_content = ""; stream_error = None
            try:
                with _call_anthropic_stream_with_retry(model=llm_model_name, max_tokens=llm_max_tokens, system=final_system_prompt, messages=llm_messages) as stream:
                    for text in stream.text_stream: response_content += text; sse_data = json.dumps({'delta': text}); yield f"data: {sse_data}\n\n"
            except RetryError as e: stream_error = "Assistant unavailable after retries."
            except APIStatusError as e: stream_error = f"API Error: {e.message}" if hasattr(e, 'message') else str(e);
            except AnthropicError as e: stream_error = f"Anthropic Error: {str(e)}"
            except Exception as e: stream_error = f"Unexpected error: {str(e)}"
            if stream_error: sse_error_data = json.dumps({'error': stream_error}); yield f"data: {sse_error_data}\n\n"
            sse_done_data = json.dumps({'done': True}); yield f"data: {sse_done_data}\n\n"
        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
    except Exception as e: logger.error(f"Error in /api/chat: {e}", exc_info=True); return jsonify({"error": "Internal server error"}), 500

# --- Auto-Stop Background Task (Updated) ---
def auto_stop_long_paused_recordings():
    global magic_audio_instance, recording_status, magic_audio_lock
    pause_timeout_seconds = 2 * 60 * 60; check_interval_seconds = 5 * 60
    logger.info(f"Auto-stop thread: Check every {check_interval_seconds}s for pauses > {pause_timeout_seconds}s.")
    while True:
        time.sleep(check_interval_seconds)
        with magic_audio_lock:
            if (recording_status["is_recording"] and recording_status["is_paused"] and recording_status["last_pause_timestamp"] is not None):
                pause_duration = time.time() - recording_status["last_pause_timestamp"]
                if pause_duration > pause_timeout_seconds:
                    logger.warning(f"Auto-stopping recording for {recording_status.get('agent')}/{recording_status.get('event')} after {pause_duration:.0f}s pause.")
                    try:
                        if magic_audio_instance: _stop_magic_audio_async(magic_audio_instance) # Use async helper
                        # Update status optimistically here as well
                        # Use the stored elapsed_time which reflects the time *before* the pause started
                        elapsed = recording_status["elapsed_time"]
                        recording_status.update({
                            "is_recording": False, "is_paused": False, "start_time": None,
                            "pause_start_time": None, "last_pause_timestamp": None,
                            "elapsed_time": int(elapsed) # Keep the elapsed time up to the pause point
                        })
                        magic_audio_instance = None # Clear ref
                        logger.info("Recording auto-stopped.")
                    except Exception as e: logger.error(f"Error during auto-stop: {e}", exc_info=True)
                else:
                    # logger.debug(f"Auto-stop check: Recording not paused or pause duration ({pause_duration:.0f}s) < timeout ({pause_timeout_seconds}s).")
                    pass # Reduce log noise

# --- Agent Sync Function ---
def sync_agents_from_s3_to_supabase():
    """Fetches agent names from S3 and inserts missing ones into Supabase 'agents' table."""
    if not supabase:
        logger.warning("Agent Sync: Supabase client not initialized. Skipping.")
        return

    logger.info("Agent Sync: Starting synchronization from S3 to Supabase...")

    # 1. Get agents from S3
    s3_agent_names = list_agent_names_from_s3()
    if s3_agent_names is None:
        logger.error("Agent Sync: Failed to list agent names from S3. Aborting.")
        return
    if not s3_agent_names:
        logger.info("Agent Sync: No agent directories found in S3.")
        # Decide if we should proceed or stop if S3 is empty. Let's proceed for now.
        # return

    # 2. Get existing agents from Supabase
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

    # 3. Find missing agents
    missing_agents = [name for name in s3_agent_names if name not in db_agent_names]

    if not missing_agents:
        logger.info("Agent Sync: Supabase 'agents' table is up-to-date with S3 directories.")
        return

    logger.info(f"Agent Sync: Found {len(missing_agents)} agents in S3 to add to Supabase: {missing_agents}")

    # 4. Prepare data for insertion
    agents_to_insert = [{'name': name, 'description': f'Agent discovered from S3 path: {name}'} for name in missing_agents]

    # 5. Insert missing agents
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


# --- Main Execution ---
if __name__ == '__main__':
    # Perform Agent Sync *after* Supabase client init
    if supabase:
        sync_agents_from_s3_to_supabase()
    else:
        logger.warning("Skipping agent sync because Supabase client failed to initialize.")

    auto_stop_thread = threading.Thread(target=auto_stop_long_paused_recordings, daemon=True); auto_stop_thread.start()
    port = int(os.getenv('PORT', 5001)); debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    use_reloader = False # Disable reloader if debug is on, as sync should run once
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=use_reloader)