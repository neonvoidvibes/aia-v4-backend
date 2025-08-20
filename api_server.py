import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables at the very top

import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context, g
from uuid import uuid4
import threading 
from concurrent.futures import ThreadPoolExecutor
import time
import json
from datetime import datetime, timezone, timedelta # Ensure datetime, timezone, timedelta are imported
import urllib.parse 
from functools import wraps 
from typing import Optional, List, Dict, Any, Tuple, Union 
import uuid 
from collections import defaultdict
import subprocess 
import re
from werkzeug.utils import secure_filename # Added for file uploads

from utils.api_key_manager import get_api_key # Import the new key manager
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
from utils.transcript_utils import read_new_transcript_content, read_all_transcripts_in_folder
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history, get_s3_client,
    list_agent_names_from_s3, list_s3_objects_metadata, get_transcript_summaries, get_objective_function,
    write_agent_doc, S3_CACHE_LOCK, S3_FILE_CACHE, create_agent_structure
)
from utils.transcript_summarizer import generate_transcript_summary # Added import
from utils.pinecone_utils import init_pinecone, create_namespace
from utils.embedding_handler import EmbeddingHandler
from pinecone.exceptions import NotFoundException
from anthropic import Anthropic, APIStatusError, AnthropicError, APIConnectionError
import anthropic # Need the module itself for type hints
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type
from flask_cors import CORS

from transcription_service import process_audio_segment_and_update_s3

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
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone', 'werkzeug', 'flask_sock', 'google.generativeai', 'google.api_core']:
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

# Instantiate it globally
anthropic_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
gemini_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
openai_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

gemini_config_lock = threading.Lock()

# Custom Exception for Circuit Breaker
class CircuitBreakerOpen(Exception):
    pass

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

    try:
        user_resp = client.auth.get_user(token)
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
        logger.error(f"Unexpected error fetching role for user {user_id}: {e}", exc_info=True)
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

from utils.llm_api_utils import (
    _call_anthropic_stream_with_retry, _call_gemini_stream_with_retry,
    _call_openai_stream_with_retry, _call_gemini_non_stream_with_retry,
    anthropic_circuit_breaker, gemini_circuit_breaker, openai_circuit_breaker,
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
    
    logger.info(f"Start recording: agent='{agent_name}', event='{event_id}', language_setting='{language_setting}'")

    if not event_id: 
        return jsonify({"status": "error", "message": "Missing event ID"}), 400

    session_id = uuid.uuid4().hex
    session_start_time_utc = datetime.now(timezone.utc)
    # Fetch agent-specific API keys or fall back to globals
    agent_openai_key = get_api_key(agent_name, 'openai')
    agent_anthropic_key = get_api_key(agent_name, 'anthropic')
    
    s3_transcript_base_filename = f"transcript_D{session_start_time_utc.strftime('%Y%m%d')}-T{session_start_time_utc.strftime('%H%M%S')}_uID-{user.id}_oID-river_aID-{agent_name}_eID-{event_id}_sID-{session_id}.txt"
    s3_transcript_key = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/{s3_transcript_base_filename}"
    
    temp_audio_base_dir = os.path.join('tmp', 'audio_sessions', session_id)
    os.makedirs(temp_audio_base_dir, exist_ok=True)
    
    active_sessions[session_id] = {
        "session_id": session_id,
        "user_id": user.id,
        "agent_name": agent_name,
        "event_id": event_id,
        "session_type": "transcript", # Differentiate session type
        "language_setting_from_client": language_setting, # Store the new setting
        "session_start_time_utc": session_start_time_utc,
        "s3_transcript_key": s3_transcript_key,
        "temp_audio_session_dir": temp_audio_base_dir, 
        "openai_api_key": agent_openai_key,       # Store the potentially agent-specific key
        "anthropic_api_key": agent_anthropic_key, # Store the potentially agent-specific key
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
        "vad_enabled": False,  # Will be set to True if VAD session is created successfully
        "last_successful_transcript": "", # For providing rolling context to Whisper
        "actual_segment_duration_seconds": 0.0, # To track duration of processed segments
    }
    logger.info(f"Transcript session {session_id} started for agent {agent_name}, event {event_id} by user {user.id}.")
    logger.info(f"Session temp audio dir: {temp_audio_base_dir}, S3 transcript key: {s3_transcript_key}")
    
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
    
    logger.info(f"Start audio recording: agent='{agent_name}', language='{language_setting}'")

    session_id = uuid.uuid4().hex
    session_start_time_utc = datetime.now(timezone.utc)
    
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
        "session_start_time_utc": session_start_time_utc,
        "s3_transcript_key": s3_recording_key, # Re-use the same key name for compatibility
        "temp_audio_session_dir": temp_audio_base_dir,
        "openai_api_key": agent_openai_key,
        "anthropic_api_key": agent_anthropic_key,
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
        "vad_enabled": False,
        "last_successful_transcript": "",
    }
    logger.info(f"Audio recording session {session_id} started for agent {agent_name} by user {user.id}.")
    
    s3 = get_s3_client()
    aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
    if s3 and aws_s3_bucket:
        header = f"# Recording - Session {session_id}\nAgent: {agent_name}\nUser: {user.id}\nSession Started (UTC): {session_start_time_utc.isoformat()}\n\n"
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

                    process_audio_segment_and_update_s3(
                        temp_segment_wav_path=final_output_wav_path,
                        session_data=session_data,
                        session_lock=session_locks[session_id],
                        openai_api_key=session_data.get("openai_api_key"),
                        anthropic_api_key=session_data.get("anthropic_api_key")
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
            except Exception as e: 
                logger.warning(f"Error closing WebSocket for session {session_id} during finalization: {e}", exc_info=True)
        
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

@sock.route('/ws/audio_stream/<session_id>')
def audio_stream_socket(ws, session_id: str):
    token = request.args.get('token')
    user = verify_user(token) 
    
    if not user:
        logger.warning(f"WebSocket Auth Failed: Invalid token for session {session_id}. Closing.")
        ws.close(reason='Authentication failed')
        return

    logger.info(f"WebSocket connection attempt for session {session_id}, user {user.id}")

    with session_locks[session_id]:
        if session_id not in active_sessions:
            logger.warning(f"WebSocket: Session {session_id} not found after acquiring lock. Closing.")
            ws.close(code=1011, reason="Session not found")
            return
        
        if active_sessions[session_id].get("websocket_connection") is not None:
            logger.warning(f"WebSocket: Session {session_id} already has a WebSocket connection. Closing new one.")
            ws.close(code=1008, reason="Connection already exists")
            return

        active_sessions[session_id]["websocket_connection"] = ws
        active_sessions[session_id]["last_activity_timestamp"] = time.time()
        active_sessions[session_id]["is_active"] = True 
        logger.info(f"WebSocket for session {session_id} (user {user.id}) connected and registered.")

    # Server-side keepalive using threading
    def _server_keepalive_thread(ws, session_id, interval=15, log=logger):
        # app-level keepalive to satisfy proxies; pairs with client's 'pong' handler
        try:
            while True:
                time.sleep(interval)
                # Check if session is still active
                if session_id not in active_sessions or not active_sessions[session_id].get("is_active"):
                    log.info(f"WS {session_id}: Session no longer active, stopping keepalive")
                    break
                try:
                    ws.send(json.dumps({"type": "ping"}))
                    log.debug(f"WS {session_id}: Server keepalive ping sent")
                except Exception as e:
                    log.warning(f"WS {session_id}: keepalive send failed: {e}")
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

    AUDIO_SEGMENT_DURATION_SECONDS_TARGET = 15 

    try:
        while True:
            # The timeout here is for receiving a message. It doesn't keep the connection alive by itself.
            # The main timeout logic is now handled in the finally block with a grace period.
            message = ws.receive(timeout=5) # Increased timeout slightly

            if message is None: 
                # This block is now less critical as the main timeout is handled in `finally`.
                # We can keep it as a secondary check.
                if active_sessions.get(session_id) and (time.time() - active_sessions[session_id].get("last_activity_timestamp", 0) > 70):
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
                with session_locks[session_id]:
                    if session_id not in active_sessions:
                        logger.warning(f"WebSocket {session_id}: Session disappeared after acquiring lock. Aborting message processing.")
                        break
                    session_data = active_sessions[session_id]

                    if session_data.get("is_backend_processing_paused", False):
                        logger.debug(f"Session {session_id} is paused. Discarding {len(message)} audio bytes.")
                        continue
                    
                    # Check if VAD is enabled for this session
                    vad_enabled = session_data.get("vad_enabled", False)
                    
                    if vad_enabled and VAD_IMPORT_SUCCESS and vad_bridge:
                        # VAD Processing Path with robust header and data accumulation
                        if not session_data.get("is_first_blob_received", False):
                            # The first blob contains the header AND the first audio chunk.
                            session_data["webm_global_header_bytes"] = bytes(message)
                            session_data["is_first_blob_received"] = True
                            logger.info(f"Session {session_id}: Captured first blob as global WebM header and initial data ({len(message)} bytes).")
                            # The first blob is the first segment.
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                        else:
                            # Subsequent blobs are just audio data fragments, append them.
                            session_data.setdefault("current_segment_raw_bytes", bytearray()).extend(message)
                        
                        logger.debug(f"Session {session_id}: Appended {len(message)} bytes. Total buffer: {len(session_data['current_segment_raw_bytes'])}")
                        
                        session_data["accumulated_audio_duration_for_current_segment_seconds"] += 3.0

                        if not session_data["is_backend_processing_paused"] and \
                           session_data["accumulated_audio_duration_for_current_segment_seconds"] >= AUDIO_SEGMENT_DURATION_SECONDS_TARGET:
                            
                            logger.info(f"Session {session_id}: Accumulated enough audio ({session_data['accumulated_audio_duration_for_current_segment_seconds']:.2f}s est.). Processing segment.")

                            current_fragment_bytes = bytes(session_data["current_segment_raw_bytes"])
                            global_header_bytes = session_data.get("webm_global_header_bytes", b"")
                            
                            # The first segment already has its header. For subsequent segments, prepend the stored header.
                            if not current_fragment_bytes.startswith(global_header_bytes):
                                bytes_to_process = global_header_bytes + current_fragment_bytes
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
                                
                                # NOTE: Duration update moved to transcription_service after actual measurement
                                # This fixes the timestamp issue where estimated duration (3s) != actual duration (~1.5s)
                                logger.debug(f"Session {session_id}: Duration update deferred to transcription_service for accuracy")

                            except Exception as e:
                                logger.error(f"Session {session_id}: Error dispatching audio to VAD bridge: {e}", exc_info=True)
                        
                        continue
                    
                    # Original Processing Path (fallback or when VAD not enabled)
                    if not vad_enabled:
                        logger.debug(f"Session {session_id}: Using original processing path (VAD disabled or failed for chunk).")
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
                                audio_bytes, wav_path,
                                openai_key, anthropic_key
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
                                    
                                    success_transcribe = process_audio_segment_and_update_s3(
                                        temp_segment_wav_path=wav_path,
                                        session_data=s_data_ref,
                                        session_lock=lock_ref,
                                        openai_api_key=openai_key,
                                        anthropic_api_key=anthropic_key
                                    )
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
                                    session_data.get("openai_api_key"),
                                    session_data.get("anthropic_api_key")
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

    except ConnectionClosed as e:
        logger.info(f"WebSocket for session {session_id} (user {user.id}): Connection closed by client: {e}")
    except ConnectionResetError:
        logger.warning(f"WebSocket for session {session_id} (user {user.id}): Connection reset by client.")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id} (user {user.id}): {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket for session {session_id} (user {user.id}) disconnected. Deregistering from session.")
        with session_locks[session_id]:
            sess = active_sessions.get(session_id)
            if sess:
                # Stop server keepalive thread by marking session as inactive
                # The thread will check this and exit naturally
                sess["is_active"] = False
                keepalive_thread = sess.pop("keepalive_thread", None)
                if keepalive_thread and keepalive_thread.is_alive():
                    logger.debug(f"Session {session_id}: Keepalive thread will stop naturally")
                
                # clear WS pointer so a new client can connect
                if sess.get("websocket_connection") == ws:
                    sess["websocket_connection"] = None
                    sess["last_activity_timestamp"] = time.time() # Start the idle timer
                    logger.debug(f"Session {session_id} WebSocket instance deregistered. Idle cleanup thread will handle finalization if needed.")
        try:
            ws.close()
        except Exception:
            pass
        

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
            
            from transcription_service import _transcribe_audio_segment_openai as transcribe_whisper_file
            
            logger.info(f"Starting transcription for {temp_filepath} with language: {transcriptionLanguage}...")
            transcription_data = transcribe_whisper_file(
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
        response = client.table("agents").select("id, name, description, created_at").order("name", desc=False).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Admin Dashboard: Error querying agents: {response.error}")
            return jsonify({"error": "Database error querying agents"}), 500
        
        return jsonify(response.data), 200
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
    api_keys_json = request.form.get('api_keys') # Expecting a JSON string

    client = get_supabase_client()
    if not client: return jsonify({"error": "Database service unavailable"}), 503

    try:
        # === Step 1: Pre-flight checks ===
        existing_agent = client.table("agents").select("id").eq("name", agent_name).limit(1).execute()
        if existing_agent.data:
            return jsonify({"error": f"Agent with name '{agent_name}' already exists."}), 409

        # === Step 2: Foundational Setup (S3, DB, Pinecone) ===
        if not create_agent_structure(agent_name):
            return jsonify({"error": "Failed to create agent folder structure in S3."}), 500
        if not create_namespace("river", agent_name):
            logger.warning(f"Could not initialize Pinecone namespace for '{agent_name}'. This can be fixed later.")

        agent_res = client.table("agents").insert({"name": agent_name, "description": description, "created_by": user.id}).execute()
        if hasattr(agent_res, 'error') and agent_res.error:
            raise Exception(f"DB error creating agent record: {agent_res.error}")
        agent_id = agent_res.data[0]['id']
        logger.info(f"Created agent '{agent_name}' (ID: {agent_id}) in database.")
        
        client.table("user_agent_access").insert({"user_id": user.id, "agent_id": agent_id}).execute()

        # === Step 3: Process Uploaded Files ===
        # S3 Docs (Core Knowledge)
        s3_docs = request.files.getlist('s3_docs')
        for doc in s3_docs:
            filename = secure_filename(doc.filename)
            content = doc.read().decode('utf-8')
            write_agent_doc(agent_name, filename, content)
            logger.info(f"Uploaded S3 doc '{filename}' for agent '{agent_name}'.")

        # Pinecone Docs (Vector Memory)
        pinecone_docs = request.files.getlist('pinecone_docs')
        if pinecone_docs:
            embed_handler = EmbeddingHandler(index_name="river", namespace=agent_name)
            temp_dir = os.path.join('tmp', 'embedding_uploads', str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            try:
                for doc in pinecone_docs:
                    filename = secure_filename(doc.filename)
                    temp_path = os.path.join(temp_dir, filename)
                    doc.save(temp_path)
                    
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    metadata = {"source": "agent_creation_upload", "file_name": filename, "agent_name": agent_name}
                    embed_handler.embed_and_upsert(content, metadata)
                    logger.info(f"Embedded Pinecone doc '{filename}' for agent '{agent_name}'.")
            finally:
                import shutil
                shutil.rmtree(temp_dir)

        # === Step 4: Save System Prompt ===
        if system_prompt_content:
            # Construct the full S3 key for the system prompt to ensure it's in the _config folder
            prompt_filename = f"systemprompt_aID-{agent_name}.md"
            prompt_s3_key = f"organizations/river/agents/{agent_name}/_config/{prompt_filename}"
            
            s3 = get_s3_client()
            aws_s3_bucket = os.getenv('AWS_S3_BUCKET')
            if s3 and aws_s3_bucket:
                try:
                    s3.put_object(Bucket=aws_s3_bucket, Key=prompt_s3_key, Body=system_prompt_content.encode('utf-8'), ContentType='text/markdown; charset=utf-8')
                    logger.info(f"Saved system prompt for agent '{agent_name}' to '{prompt_s3_key}'.")
                except Exception as e:
                    logger.error(f"Failed to save system prompt to S3 for agent '{agent_name}': {e}")
                    # Decide if this should be a fatal error for the creation process
            else:
                logger.error(f"S3 client not available, cannot save system prompt for agent '{agent_name}'.")

        # === Step 5: Save API Keys ===
        if api_keys_json:
            try:
                api_keys = json.loads(api_keys_json)
                keys_to_insert = []
                for service, key in api_keys.items():
                    if key: # Only insert if a key was provided
                        keys_to_insert.append({
                            "agent_id": agent_id,
                            "service_name": service,
                            "api_key": key
                        })
                if keys_to_insert:
                    client.table("agent_api_keys").insert(keys_to_insert).execute()
                    logger.info(f"Saved {len(keys_to_insert)} API keys for agent '{agent_name}'.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse or process API keys for agent '{agent_name}': {e}")
        
        return jsonify({"status": "success", "message": f"Agent '{agent_name}' created successfully.", "agent": agent_res.data[0]}), 201

    except Exception as e:
        logger.error(f"Unexpected error creating agent '{agent_name}': {e}", exc_info=True)
        # Attempt to clean up failed agent creation from DB
        if 'agent_id' in locals():
            client.table("agents").delete().eq("id", agent_id).execute()
        return jsonify({"error": "An internal server error occurred during agent creation. The operation was rolled back."}), 500

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
                get_transcript_summaries(agent, event) # Also warm up summaries
                logger.info(f"Background cache worker finished for agent: '{agent}', event: '{event}'")
            except Exception as e:
                logger.error(f"Error in background cache worker for agent '{agent}': {e}", exc_info=True)

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
    initial_context_for_aicreator = data.get('initialContext')
    current_draft_content_for_aicreator = data.get('currentDraftContent')
    disable_retrieval = data.get('disableRetrieval', False) # New

    def generate_stream():
        """
        A generator function that prepares data, calls the LLM, and yields the response stream.
        This runs within the `stream_with_context` to handle the response generation.
        """
        try:
            # WIZARD DETECTION: A wizard session is defined by disabling retrieval and providing initial context.
            is_wizard = disable_retrieval and initial_context_for_aicreator is not None
            
            # For wizard mode, enforce no transcript listening to save resources and avoid irrelevant I/O.
            effective_transcript_listen_mode = "none" if is_wizard else transcript_listen_mode

            logger.info(f"Chat stream started for Agent: {agent_name}, Event: {event_id}, User: {user.id}, Wizard Mode: {is_wizard}")
            logger.info(f"Stream settings - Listen: {effective_transcript_listen_mode}, Memory: {saved_transcript_memory_mode}, Language: {transcription_language_setting}, Individual toggles: {len(individual_memory_toggle_states)} items")

            # --- System Prompt and Context Assembly ---
            s3_load_start_time = time.time()
            
            # Base prompt and frameworks are always loaded for both agent and wizard.
            base_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
            frameworks = get_latest_frameworks(agent_name)
            
            # Conditional loading for non-wizard sessions
            objective_function = None
            event_context = None
            agent_docs = None
            if not is_wizard:
                objective_function = get_objective_function(agent_name)
                event_context = get_latest_context(agent_name, event_id)
                agent_docs = get_agent_docs(agent_name)

            s3_load_time = time.time() - s3_load_start_time
            logger.info(f"[PERF] S3 Prompts/Context loaded in {s3_load_time:.4f}s")
            
            # CONSTRUCT PROMPT WITH HIERARCHY
            final_system_prompt = base_system_prompt
            
            # Special handling for the _aicreator agent to inject context (this is the wizard)
            if agent_name == '_aicreator':
                ai_creator_instructions = """
\n\n## Core Mission: AI Agent Prompt Creator
You are an expert AI assistant who helps users draft high-quality system prompts for other AI agents. Engage in a collaborative conversation to refine the user's ideas into a precise and effective prompt.

## Critical Instructions & Rules
1.  **Prioritize the User's Draft:** The user's current work-in-progress is provided in a `<current_draft>` block. This is your **single source of truth** for the prompt's content. When the user says "my edits", "this version", or "the draft", they are referring to the content of this block. Your primary goal is to refine THIS DRAFT.
2.  **Output Format Mandate:** When you generate a new version of the system prompt, you MUST follow this format precisely:
    a.  Provide a brief, conversational message explaining your changes.
    b.  Immediately after your message, provide a JSON code block containing the new system prompt.

**EXAMPLE OUTPUT:**
I've updated the prompt to include the dragon's personality as you requested. It now has a more defined, wise character.
```json
{
  "system_prompt": "You are a wise and ancient dragon. You have seen empires rise and fall. You speak in a measured, calm tone, offering cryptic but helpful advice."
}
```

**JSON SCHEMA RULES:**
- The JSON block MUST start with ```json and end with ```.
- The root MUST be a JSON object (`{}`).
- It MUST contain one and only one key: `"system_prompt"`.
- The value of `"system_prompt"` MUST be a string containing the complete, new prompt.
- Do NOT add any text, comments, or trailing commas inside or after the JSON block.
"""
                final_system_prompt += ai_creator_instructions

                if initial_context_for_aicreator:
                    logger.info(f"Injecting initial document context for _aicreator agent (length: {len(initial_context_for_aicreator)}).")
                    final_system_prompt += f"\n\n<document_context>\n{initial_context_for_aicreator}\n</document_context>"

                if current_draft_content_for_aicreator:
                    logger.info(f"Injecting current draft context for _aicreator agent (length: {len(current_draft_content_for_aicreator)}).")
                    final_system_prompt += (
                        "\n\n## User's Current Draft (Authoritative)\n"
                        "<current_draft>\n" + current_draft_content_for_aicreator + "\n</current_draft>"
                    )
            
            # Add other prompt components only if NOT in wizard mode
            if not is_wizard:
                if objective_function:
                    final_system_prompt += "\n\n## Core Directive (Objective Function)\n"
                    final_system_prompt += "The following is your highest-priority, foundational objective. It is stable and rarely changes. It MUST be followed and overrides any conflicting instructions found in the 'Frameworks' or other contextual documents.\n\n"
                    final_system_prompt += objective_function
                
                transcript_handling_instructions = (
                    "\n\n## IMPORTANT INSTRUCTIONS FOR USING TRANSCRIPTS:\n"
                    "1.  **Initial Full Transcript:** The very first user message may contain a block labeled '=== BEGIN FULL MEETING TRANSCRIPT ===' to '=== END FULL MEETING TRANSCRIPT ==='. This is the complete historical context of the meeting up to the start of our current conversation. You MUST refer to this entire block for any questions about past events, overall context, or specific details mentioned earlier in the meeting. Very important: DO NOT summarize or analyze its content unless specifically asked by the user.\n"
                    "2.  **Live Updates:** Subsequent user messages starting with '[Meeting Transcript Update (from S3)]' provide new, live additions to the transcript. These are chronological and should be considered the most current information.\n"
                    "3.  **Comprehensive Awareness:** When asked about the transcript (e.g., 'what can you see?', 'what's the first/last timestamp?'), your answer must be based on ALL transcript information you have received, including the initial full load AND all subsequent delta updates. DO NOT summarize or analyze its content unless specifically asked by the user."
                )
                final_system_prompt += transcript_handling_instructions
            
            if frameworks: final_system_prompt += "\n\n## Frameworks\nThe following are operational frameworks and models for how to approach tasks. They are very important but secondary to your Core Directive.\n" + frameworks
            
            if not is_wizard:
                if event_context: final_system_prompt += "\n\n## Context\n" + event_context
                if agent_docs: final_system_prompt += "\n\n## Agent Documentation\n" + agent_docs
                rag_usage_instructions = "\n\n## Using Retrieved Context\n1. **Prioritize Info Within `[Retrieved Context]`:** Base answer primarily on info in `[Retrieved Context]` block below, if relevant. \n2. **Assess Timeliness:** Each source has an `(Age: ...)` tag. Use this to assess relevance. More recent information is generally more reliable, unless it's a 'Core Memory' which is timeless. \n3. **Direct Extraction for Lists/Facts:** If user asks for list/definition/specific info explicit in `[Retrieved Context]`, present that info directly. Do *not* state info missing if clearly provided. \n4. **Cite Sources:** Remember cite source file name using Markdown footnotes (e.g., `[^1]`) for info from context, list sources under `### Sources`. \n5. **Synthesize When Necessary:** If query requires combining info or summarizing, do so, but ground answer in provided context. \n6. **Acknowledge Missing Info Appropriately:** Only state info missing if truly absent from context and relevant."
                final_system_prompt += rag_usage_instructions
            
            memory_update_instructions = """

## Memory Update Protocol

When you identify information that should be permanently stored in your agent documentation, use this exact format:

[DOC_UPDATE_PROPOSAL]
{
  "doc_name": "filename.md",
  "content": "Complete document content here",
  "justification": "Brief explanation why this update is needed"
}

**JSON Schema (Draft 7):**
```json
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
- Choose descriptive filenames ending in .md"""
            final_system_prompt += memory_update_instructions

            # --- RAG ---
            rag_context_block = ""
            retrieved_docs_for_reinforcement = []
            
            # RAG is completely skipped for wizard mode.
            if not is_wizard:
                last_user_message_obj = next((msg for msg in reversed(incoming_messages) if msg.get("role") == "user"), None)
                last_actual_user_message_for_rag = last_user_message_obj.get("content") if last_user_message_obj else None
                
                openai_key_for_rag = get_api_key(agent_name, 'openai')
                anthropic_key_for_rag = get_api_key(agent_name, 'anthropic')

                if last_actual_user_message_for_rag:
                    rag_start_time = time.time()
                    normalized_query = last_actual_user_message_for_rag.strip().lower().rstrip('.!?')
                    is_simple_query = normalized_query in SIMPLE_QUERIES_TO_BYPASS_RAG
                    if not is_simple_query:
                        logger.info(f"Complex query ('{normalized_query[:50]}...'), attempting RAG.")
                        try:
                            retriever = RetrievalHandler(
                                index_name="river", agent_name=agent_name, session_id=chat_session_id_log,
                                event_id=event_id, anthropic_api_key=anthropic_key_for_rag,
                                openai_api_key=openai_key_for_rag
                            )
                            retrieved_docs = retriever.get_relevant_context(query=last_actual_user_message_for_rag, top_k=10)
                            retrieved_docs_for_reinforcement = retrieved_docs
                            if retrieved_docs:
                                items = [f"--- START Context Source: {d.metadata.get('file_name','Unknown')} (Age: {d.metadata.get('age_display', 'Unknown')}, Score: {d.metadata.get('score',0):.2f}) ---\n{d.page_content}\n--- END Context Source: {d.metadata.get('file_name','Unknown')} ---" for d in retrieved_docs]
                                rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n" + "\n\n".join(items) + "\n=== END RETRIEVED CONTEXT ==="
                            else:
                                rag_context_block = "\n\n[Note: No relevant documents found for this query via RAG.]"
                        except Exception as e:
                            logger.error(f"Unexpected RAG error: {e}", exc_info=True)
                            rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Error retrieving documents via RAG]\n=== END RETRIEVED CONTEXT ==="
                    else:
                        rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval skipped for simple query.]\n=== END RETRIEVED CONTEXT ==="
                    rag_time = time.time() - rag_start_time
                    logger.info(f"[PERF] RAG processing took {rag_time:.4f}s")
            
            final_system_prompt += rag_context_block

            # --- Summary & Transcript Loading (non-wizard only) ---
            final_llm_messages = []
            if not is_wizard:
                if individualMemoryToggleStates and saved_transcript_summaries:
                    summaries_context_str = "\n\n## Saved Transcript Summaries (Historical Context)\n"
                    enabled_summaries_count = 0
                    for summary_data in saved_transcript_summaries:
                        summary_key = summary_data.get('s3Key', summary_data.get('name', ''))
                        if individual_memory_toggle_states.get(summary_key, False):
                            summary_filename = summary_data.get('name', 'unknown_summary.json')
                            if summary_key:
                                from utils.s3_utils import read_file_content
                                summary_content = read_file_content(summary_key, f"Individual summary {summary_filename}")
                                if summary_content:
                                    try:
                                        summary_doc = json.loads(summary_content)
                                        summaries_context_str += f"### Summary: {summary_filename}\n{json.dumps(summary_doc, indent=2, ensure_ascii=False)}\n\n"
                                        enabled_summaries_count += 1
                                    except json.JSONDecodeError: logger.warning(f"Failed to parse JSON content for summary {summary_filename}")
                    if enabled_summaries_count > 0:
                        final_system_prompt = summaries_context_str + final_system_prompt
                        logger.info(f"Added {enabled_summaries_count} individual transcript summaries to context")
                    else: logger.info("No individual transcript summaries enabled")
                elif saved_transcript_memory_mode == 'enabled':
                    summaries = get_transcript_summaries(agent_name, event_id)
                    if summaries:
                        summaries_context_str = "\n\n## Saved Transcript Summaries (Historical Context)\n"
                        for summary_doc in summaries:
                            summary_filename = summary_doc.get("metadata", {}).get("summary_filename", "unknown_summary.json")
                            summaries_context_str += f"### Summary: {summary_filename}\n{json.dumps(summary_doc, indent=2, ensure_ascii=False)}\n\n"
                        final_system_prompt = summaries_context_str + final_system_prompt
                        logger.info(f"Added all transcript summaries to context (legacy mode)")
                else:
                    logger.info("No transcript summaries to include (main toggle disabled, no individual toggles enabled)")

                if effective_transcript_listen_mode == 'all':
                    all_transcripts_content = read_all_transcripts_in_folder(agent_name, event_id)
                    if all_transcripts_content: final_llm_messages.append({'role': 'user', 'content': f"IMPORTANT CONTEXT: The following is the FULL transcript history for this meeting, provided because 'Listen: All' mode is active.\n\n=== BEGIN ALL TRANSCRIPTS FOR THIS TURN ===\n{all_transcripts_content}\n=== END ALL TRANSCRIPTS FOR THIS TURN ==="})
                elif effective_transcript_listen_mode == 'latest':
                    transcript_content, success = read_new_transcript_content(agent_name, event_id)
                    if transcript_content: final_llm_messages.append({'role': 'user', 'content': f"IMPORTANT CONTEXT: The following is the content of the latest transcript file. Refer to this as the most current information available.\n\n=== BEGIN LATEST TRANSCRIPT ===\n{transcript_content}\n=== END LATEST TRANSCRIPT ==="})

            # --- Add Current Time to System Prompt (non-wizard only) ---
            if not is_wizard:
                current_utc_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                final_system_prompt += f"\n\n## Current Time\nYour internal clock shows the current date and time is: **{current_utc_time}**."

            # --- Timestamped History Injection ---
            if agent_name == '_aicreator':
                llm_messages_from_client = [{"role": msg["role"], "content": msg["content"]} for msg in incoming_messages if msg.get("id") != 'initial-wizard-prompt' and "role" in msg and "content" in msg]
                final_llm_messages.extend(llm_messages_from_client)
            else:
                last_user_message_obj = next((msg for msg in reversed(incoming_messages) if msg.get("role") == "user"), None)
                last_actual_user_message_for_rag = last_user_message_obj.get("content") if last_user_message_obj else None
                timestamped_history_lines = ["This is the conversation history with timestamps for your reference. Do not replicate this format in your responses."]
                for msg in incoming_messages:
                    if msg is last_user_message_obj: continue
                    if msg.get("role") in ["user", "assistant"]:
                        timestamp = msg.get("createdAt", datetime.now(timezone.utc).isoformat())
                        role = msg.get("role"); content = msg.get("content")
                        timestamped_history_lines.append(f"[{timestamp}] {role}: {content}")
                history_context_block = "\n".join(timestamped_history_lines)
                llm_messages_from_client = [
                    {"role": "user", "content": f"=== CURRENT CHAT HISTORY ===\n{history_context_block}\n=== END CURRENT CHAT HISTORY ==="},
                    {"role": "user", "content": last_actual_user_message_for_rag if last_actual_user_message_for_rag else ""}
                ]
                final_llm_messages.extend(llm_messages_from_client)

            # --- Call LLM and Stream ---
            max_tokens_for_call = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))

            # Determine which LLM provider key to get for the main chat
            if model_selection.startswith('gemini'): llm_provider = 'google'
            elif model_selection.startswith('gpt-'): llm_provider = 'openai'
            else: llm_provider = 'anthropic'
            llm_api_key = get_api_key(agent_name, llm_provider)

            try:
                if model_selection.startswith('gemini'):
                    logger.info(f"Dispatching chat request to Gemini model: {model_selection}")
                    gemini_stream = _call_gemini_stream_with_retry(model_name=model_selection, max_tokens=max_tokens_for_call, system_instruction=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key, temperature=temperature)
                    for chunk in gemini_stream:
                        if chunk.parts: yield f"data: {json.dumps({'delta': chunk.text})}\n\n"
                elif model_selection.startswith('gpt-'):
                    logger.info(f"Dispatching chat request to OpenAI model: {model_selection}")
                    openai_stream = _call_openai_stream_with_retry(model_name=model_selection, max_tokens=max_tokens_for_call, system_instruction=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key, temperature=temperature)
                    with openai_stream as stream:
                        for event in stream:
                            if getattr(event, "type", "") == "response.output_text.delta": yield f"data: {json.dumps({'delta': event.delta})}\n\n"
                else: # Default to Anthropic
                    stream_manager = _call_anthropic_stream_with_retry(model=model_selection, max_tokens=max_tokens_for_call, system=final_system_prompt, messages=final_llm_messages, api_key=llm_api_key)
                    with stream_manager as stream:
                        for chunk in stream:
                            if chunk.type == "content_block_delta": yield f"data: {json.dumps({'delta': chunk.delta.text})}\n\n"
                
                doc_ids_for_reinforcement = [doc.metadata.get('vector_id') for doc in retrieved_docs_for_reinforcement if doc.metadata.get('vector_id')]
                sse_done_data = {'done': True, 'retrieved_doc_ids': doc_ids_for_reinforcement}
                yield f"data: {json.dumps(sse_done_data)}\n\n"
                logger.info(f"Stream for chat with agent {agent_name} (model: {model_selection}) completed successfully. Sent {len(doc_ids_for_reinforcement)} doc IDs for reinforcement.")

            except CircuitBreakerOpen as e:
                logger.error(f"Circuit breaker is open. Aborting stream. Error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except (google_exceptions.GoogleAPICallError, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError, RetryError) as e:
                logger.error(f"Gemini API error after retries: {e}", exc_info=True); gemini_circuit_breaker.record_failure()
                yield f"data: {json.dumps({'error': f'Assistant (Gemini) API Error: {str(e)}'})}\n\n"
            except (OpenAI_APIError, RetryError) as e:
                logger.error(f"OpenAI API error after retries: {e}", exc_info=True); openai_circuit_breaker.record_failure()
                yield f"data: {json.dumps({'error': f'Assistant (OpenAI) API Error: {str(e)}'})}\n\n"
            except (AnthropicError, RetryError) as e:
                logger.error(f"Anthropic API error after retries: {e}", exc_info=True); anthropic_circuit_breaker.record_failure()
                yield f"data: {json.dumps({'error': f'Assistant (Anthropic) API Error: {str(e)}'})}\n\n"
            
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
            "source": "recording", # As specified in the plan
            "file_name": virtual_filename,
            "s3_key": s3_key,
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
            "triplets": triplets # Pass the triplets to the embedding handler
        }
        
        upsert_success = embedding_handler.embed_and_upsert(structured_content, metadata_for_embedding)
        if not upsert_success:
            # This is a critical failure as the memory won't be retrievable.
            logger.error(f"Save Memory Log: CRITICAL - Failed to upsert new vectors to Pinecone for session '{session_id}'.")
            # We might want to rollback the Supabase entry or flag it for re-indexing here.
            # For now, we return an error to the user.
            return jsonify({"error": "Failed to index memory for retrieval"}), 500

        logger.info(f"Save Memory Log: Pipeline completed successfully for session '{session_id}'.")
        return jsonify({"status": "success", "message": "Chat memory saved and indexed.", "log_id": supabase_log_id}), 200

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

# --- Restored User Chat History Endpoints ---

def generate_chat_title(first_user_message: str) -> str:
    """Generate a concise title for a chat using Gemini 1.5 Flash"""
    try:
        title = _call_gemini_non_stream_with_retry(
            model_name="gemini-1.5-flash-latest",
            max_tokens=50,
            system_instruction="Generate a concise, descriptive title (max 6 words) for this chat based on the user's first message. Return only the title, no quotes or extra text.",
            messages=[{"role": "user", "content": first_user_message}],
            api_key=os.getenv('GOOGLE_API_KEY'), # Use global key for this utility
            temperature=0.9
        )
        return title.strip().strip('"')[:100]
    except Exception as e:
        logger.error(f"Error generating chat title: {e}")
        return first_user_message[:50] + "..." if len(first_user_message) > 50 else first_user_message

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
        title = generate_chat_title(first_user_message) if first_user_message else "New Chat"

    try:
        if chat_id:
            update_payload = {
                'title': title,
                'messages': messages,
                'updated_at': 'now()',
                'last_message_at': 'now()',
            }
            if last_message_id:
                update_payload['last_message_id_at_save'] = last_message_id

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
                    client.table('chat_history').update(update_payload).eq('id', chat_id).eq('user_id', user.id).execute()
                    return jsonify({'success': True, 'chatId': chat_id, 'title': existing.data.get('title', title)})

            # Before creating a new chat, check for recent duplicates to prevent race condition artifacts
            # Look for chats created in the last 60 seconds with the same title and agent (increased from 30s)
            from datetime import datetime, timedelta
            cutoff_time = (datetime.utcnow() - timedelta(seconds=60)).isoformat()
            
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
    agent_name = request.args.get('agent')
    client = get_supabase_client()
    if not agent_name or not client:
        return jsonify({'error': 'Agent parameter is required or DB is unavailable'}), 400

    try:
        agent_result = client.table('agents').select('id').eq('name', agent_name).single().execute()
        if not agent_result.data:
            return jsonify([])
        agent_id = agent_result.data['id']

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
                    structured_content, summary = enrich_chat_log(updated_messages, google_api_key)
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
                        "saved_at": datetime.now(timezone.utc).timestamp()
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
        pc = init_pinecone()
        if not pc:
            raise Exception("Pinecone client not initialized")
        
        index = pc.Index("river")
        index_stats = index.describe_index_stats()
        
        for name in agent_names:
            # Check from the already fetched stats to avoid multiple API calls to Pinecone
            exists = name in index_stats.namespaces
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
                # This is for unexpected disconnects.
                if session_data.get("is_active") and \
                   not session_data.get("is_finalizing") and \
                   session_data.get("websocket_connection") is None and \
                   now - session_data.get("last_activity_timestamp", 0) > 45: # 45s grace period for reconnect
                    logger.warning(f"Idle session cleanup: Session {session_id} has been active without a WebSocket for >45s. Marking for finalization.")
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
