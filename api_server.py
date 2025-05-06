import os
import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from dotenv import load_dotenv
import threading
import time
import json
from datetime import datetime, timezone
import urllib.parse # For decoding filenames in IDs

# Import necessary modules from our project
from magic_audio import MagicAudio # Import the copied module
# Import necessary modules from our project
from magic_audio import MagicAudio # Import the copied module
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history, get_s3_client # Added get_s3_client
)
# Import specific Pinecone exception
from utils.pinecone_utils import init_pinecone
from pinecone.exceptions import NotFoundException

# LLM Client
from anthropic import Anthropic, APIStatusError, AnthropicError

# Retry mechanism
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type

# CORS
from flask_cors import CORS # Import CORS

# --- Load environment variables ---
load_dotenv()

# --- Logging Setup ---
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
    # Be more specific about silencing libraries
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone', 'werkzeug']: # Added werkzeug
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('utils').setLevel(logging.DEBUG if debug else logging.INFO) # Log our utils
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger
logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # <-- Enable CORS for all origins
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- Define Simple Queries to Bypass RAG ---
# This set contains normalized (lowercase, trimmed, trailing punctuation removed)
# queries that should bypass Pinecone retrieval.
SIMPLE_QUERIES_TO_BYPASS_RAG = frozenset([
    # Greetings
    "hello", "hello :)" "hi", "hi :)" "hey", "hey :)" "yo", "greetings", "good morning", "good afternoon",
    "good evening", "morning", "afternoon", "evening",
    # Farewells
    "bye", "goodbye", "see you", "see ya", "take care", "farewell", "bye bye",
    "later", "cheers", "good night",
    # Thanks
    "thanks", "thanks :)" "thank you", "thank you :)" "ty", "thx", "appreciate it", "thanks so much",
    "thank you very much", "much appreciated", "thanks a lot",
    # Agreement / Acknowledgment
    "ok", "ok :)" "okay", "okay :)" "k", "kk", "got it", "understood", "sounds good", "perfect",
    "great", "cool", "alright", "roger", "fine", "sure", "yes", "yes :)", "yes please", "yes please :)",
    "yep", "yeah", "yeah :)", "indeed", "affirmative", "certainly", "absolutely", "definitely",
    "exactly", "right", "correct", "i see", "makes sense", "fair enough", "will do",
    "you bet", "of course", "agreed", "true",
    # Disagreement / Negative (Standalone)
    "no", "nope", "nah", "negative", "not really", "i disagree", "disagree",
    "false", "incorrect",
    # Politeness / Fillers
    "please", "pardon", "excuse me", "you're welcome", "yw", "no problem", "np",
    "my pleasure", "don't worry", "it's ok", "it's okay", "no worries",
    # Exclamations (Standalone)
    "wow", "oops", "nice", "awesome", "excellent", "fantastic", "amazing",
    "brilliant", "sweet",
    # Apologies (Standalone)
    "sorry", "my apologies", "apologies", "my bad",
    # Uncertainty (Standalone)
    "maybe", "perhaps", "i don't know", "idk", "not sure", "hard to say",
    # Simple Confirmation Requests (Standalone)
    "really", "are you sure", "is that right", "correct",
    # Swedish
    "hej", "hejsan", "god morgon", "god kväll", "hej då", "vi ses", "ha det bra",
    "tack", "tack så mycket", "okej", "perfekt", "toppen", "ja", "nej", "jepp",
    "inga problem", "kanske", "jag vet inte", "är du säker", "precis"
    # Swedish :)
    "hej :)", "hejsan :)", "god morgon :)", "god kväll :)", "hej då :)", "vi ses :)",
    "tack :)", "tack så mycket :)", "okej :)", "perfekt :)", "toppen :)", "ja :)", "ja tack :)",
    "jepp :)", "inga problem :)", "precis :)"
])


# --- Global State (Simplified for POC) ---
# Transcription
magic_audio_instance: MagicAudio | None = None
magic_audio_lock = threading.Lock()
recording_status = { "is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "elapsed_time": 0, "agent": None, "event": None }

# Chat (Anthropic Client is global)
anthropic_client: Anthropic | None = None
# Initialize Pinecone globally if RetrievalHandler needs it readily available
# This might still raise an error if PINECONE_API_KEY is missing, but won't block server start
try:
    init_pinecone()
    logger.info("Pinecone initialized (or skipped if keys missing).")
except Exception as e:
    logger.warning(f"Pinecone initialization failed during startup: {e}")


# Chat History & State (Potentially per-request or per-session later)
# For POC, let's manage history conceptually per API call for now
# We'll need TranscriptState instances per active agent/event combo eventually
transcript_state_cache = {} # Cache: key=(agent, event), value=TranscriptState()
transcript_state_lock = threading.Lock()

# --- Initialize LLM Client ---
try:
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key: raise ValueError("ANTHROPIC_API_KEY not found in environment")
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic client initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Anthropic client: {e}", exc_info=True)
    anthropic_client = None # Ensure it's None if failed

# --- Tenacity Retry Configuration ---
def log_retry_error(retry_state):
    logger.warning(f"Retrying Anthropic API call (attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}")

retry_strategy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=log_retry_error,
    retry=(retry_if_exception_type(APIStatusError)) # Retry only on specific API status errors (like overload)
)

@retry_strategy
def _call_anthropic_stream_with_retry(model, max_tokens, system, messages):
    """Calls Anthropic stream API with configured retry logic."""
    if not anthropic_client:
         raise RuntimeError("Anthropic client is not initialized.")
    # The actual API call is now wrapped by the tenacity decorator
    logger.debug(f"Making Anthropic API call: Model={model}, MaxTokens={max_tokens}, SystemPromptLen={len(system)}, NumMessages={len(messages)}")
    return anthropic_client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages
    )
# --- End Retry Config ---

# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    anthropic_ok = anthropic_client is not None
    # Add checks for S3, Pinecone if needed
    s3_ok = get_s3_client() is not None
    # Pinecone check is tricky as init might fail silently if keys missing
    return jsonify({"status": "ok", "message": "Backend is running", "anthropic_client": anthropic_ok, "s3_client": s3_ok}), 200

@app.route('/api/status', methods=['GET'])
def get_app_status():
    """Returns the status of backend features like memory and transcript listening."""
    # For POC, these are based on initial config flags or assumptions.
    # TODO: Integrate config loading properly to reflect actual runtime state.
    is_memory_enabled = True # Assume memory is intended to be used based on old logic
    is_transcript_listening = False # Assume false unless explicitly started/configured

    with magic_audio_lock:
         rec_status = recording_status.copy() # Get current recording status

    status_data = {
        'agent_name': rec_status.get("agent", "N/A"),
        'event_id': rec_status.get("event", "N/A"),
        'listen_transcript': is_transcript_listening, # Placeholder
        'memory_enabled': is_memory_enabled,         # Placeholder
        'is_recording': rec_status.get("is_recording", False),
        'is_paused': rec_status.get("is_paused", False),
    }
    logger.debug(f"Reporting app status: {status_data}")
    return jsonify(status_data), 200


# --- Transcription Control API Routes ---
@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if recording_status["is_recording"]: return jsonify({"status": "error", "message": "Already recording"}), 400
        data = request.json; agent = data.get('agent'); event = data.get('event'); language = data.get('language')
        if not agent or not event: return jsonify({"status": "error", "message": "Missing agent or event"}), 400
        logger.info(f"Starting recording for Agent: {agent}, Event: {event}, Lang: {language}")
        try:
            if magic_audio_instance:
                try: magic_audio_instance.stop()
                except Exception as e: logger.warning(f"Error stopping previous audio instance: {e}")
            magic_audio_instance = MagicAudio(agent=agent, event=event, language=language)
            magic_audio_instance.start()
            recording_status.update({"is_recording": True, "is_paused": False, "start_time": time.time(), "pause_start_time": None, "elapsed_time": 0, "agent": agent, "event": event})
            logger.info("Recording started successfully.")
            return jsonify({"status": "success", "message": "Recording started"})
        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True); magic_audio_instance = None
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "elapsed_time": 0, "agent": None, "event": None})
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance: return jsonify({"status": "success", "message": "Not recording"}), 200
        logger.info("Stopping recording...")
        try:
            magic_audio_instance.stop()
            if recording_status["start_time"]:
                 now = time.time()
                 if recording_status["is_paused"] and recording_status["pause_start_time"]: recording_status["elapsed_time"] = recording_status["pause_start_time"] - recording_status["start_time"]
                 else: recording_status["elapsed_time"] = now - recording_status["start_time"]
            else: recording_status["elapsed_time"] = 0
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None})
            magic_audio_instance = None
            logger.info("Recording stopped and transcript saved.")
            return jsonify({"status": "success", "message": "Recording stopped and transcript saved."})
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            recording_status.update({"is_recording": False, "is_paused": False})
            magic_audio_instance = None
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/pause', methods=['POST'])
def pause_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance: return jsonify({"status": "error", "message": "Not recording"}), 400
        if recording_status["is_paused"]: return jsonify({"status": "success", "message": "Already paused"}), 200
        logger.info("Pausing recording...")
        try:
            magic_audio_instance.pause()
            recording_status["is_paused"] = True; recording_status["pause_start_time"] = time.time()
            if recording_status["start_time"]: recording_status["elapsed_time"] = recording_status["pause_start_time"] - recording_status["start_time"]
            logger.info("Recording paused.")
            return jsonify({"status": "success", "message": "Recording paused"})
        except Exception as e: logger.error(f"Error pausing recording: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/resume', methods=['POST'])
def resume_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance: return jsonify({"status": "error", "message": "Not recording"}), 400
        if not recording_status["is_paused"]: return jsonify({"status": "success", "message": "Not paused"}), 200
        logger.info("Resuming recording...")
        try:
            if recording_status["pause_start_time"] and recording_status["start_time"]:
                pause_duration = time.time() - recording_status["pause_start_time"]
                recording_status["start_time"] += pause_duration
                logger.debug(f"Resuming after {pause_duration:.2f}s pause. Adjusted start time.")
            else: logger.warning("Could not calculate pause duration accurately on resume.")
            magic_audio_instance.resume()
            recording_status["is_paused"] = False; recording_status["pause_start_time"] = None
            logger.info("Recording resumed.")
            return jsonify({"status": "success", "message": "Recording resumed"})
        except Exception as e: logger.error(f"Error resuming recording: {e}", exc_info=True); return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/status', methods=['GET'])
def get_recording_status():
    with magic_audio_lock:
        current_elapsed = recording_status["elapsed_time"]
        if recording_status["is_recording"] and not recording_status["is_paused"] and recording_status["start_time"]: current_elapsed = time.time() - recording_status["start_time"]
        status_data = {"is_recording": recording_status["is_recording"], "is_paused": recording_status["is_paused"], "elapsed_time": int(current_elapsed), "agent": recording_status.get("agent"), "event": recording_status.get("event")}
    return jsonify(status_data), 200 # Explicitly return 200 OK


# --- Pinecone Index Info Route ---
@app.route('/api/index/<string:index_name>/stats', methods=['GET'])
def get_pinecone_index_stats(index_name: str):
    """Retrieves statistics for the specified Pinecone index."""
    logger.info(f"Request received for stats of index: {index_name}")
    try:
        # Initialize Pinecone client (assuming keys are available)
        pc = init_pinecone()
        if not pc:
            logger.error(f"Failed to initialize Pinecone client for stats request.")
            return jsonify({"error": "Pinecone client initialization failed"}), 500

        # Check if index exists and get handle
        try:
            index = pc.Index(index_name)
            logger.info(f"Accessing Pinecone index '{index_name}'")
        except NotFoundException:
            logger.warning(f"Index '{index_name}' not found.")
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e:
             logger.error(f"Error accessing index '{index_name}': {e}", exc_info=True)
             return jsonify({"error": f"Failed to access index '{index_name}'"}), 500

        # Get statistics
        stats = index.describe_index_stats()
        # The stats object might have different structures, convert to dict safely
        stats_dict = {}
        if hasattr(stats, 'to_dict'):
            stats_dict = stats.to_dict()
        elif isinstance(stats, dict):
             stats_dict = stats
        else:
             # Fallback if the structure is unexpected
             stats_dict = {"raw_stats": str(stats)}

        logger.info(f"Successfully retrieved stats for index '{index_name}'.")
        # Return the full stats object for now, frontend can parse namespaces
        return jsonify(stats_dict), 200

    except Exception as e:
        logger.error(f"Error getting stats for index '{index_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500


# --- Pinecone Vector ID Listing Route ---
@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/list_ids', methods=['GET'])
def list_vector_ids_in_namespace(index_name: str, namespace_name: str):
    """Lists vector IDs within a specific namespace of a Pinecone index."""
    limit = request.args.get('limit', default=1000, type=int)
    next_token = request.args.get('next_token', default=None, type=str)

    logger.info(f"Request received to list IDs in index '{index_name}', namespace '{namespace_name}' (limit: {limit}, token: {next_token})")

    # Validate limit
    if not 1 <= limit <= 1000:
         logger.warning(f"Invalid limit requested: {limit}. Using default 1000.")
         limit = 1000 # Pinecone limit is 1000 per page

    try:
        pc = init_pinecone()
        if not pc:
            logger.error(f"Failed to initialize Pinecone client for listing IDs.")
            return jsonify({"error": "Pinecone client initialization failed"}), 500

        try:
            index = pc.Index(index_name)
            logger.info(f"Accessing Pinecone index '{index_name}'")
        except NotFoundException:
            logger.warning(f"Index '{index_name}' not found.")
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e:
             logger.error(f"Error accessing index '{index_name}': {e}", exc_info=True)
             return jsonify({"error": f"Failed to access index '{index_name}'"}), 500

        # List vector IDs from the specified namespace
        # Use empty string for the default namespace if requested via '_default'
        query_namespace = "" if namespace_name == "_default" else namespace_name
        logger.info(f"Querying Pinecone namespace: '{query_namespace}' (URL requested: '{namespace_name}')")

        list_response = index.list(
            namespace=query_namespace,
            limit=limit
            # pagination_token=next_token # Generators might not use token directly this way
            # For full pagination with generators, you'd typically loop and call list() again if needed.
            # Let's focus on getting the first batch for now.
        )

        vector_ids = []
        next_page_token = None # Initialize pagination token

        try:
            # Check if the response is directly iterable (likely a generator)
            if hasattr(list_response, '__iter__'):
                 logger.debug("Processing response as an iterator/generator")
                 for id_batch in list_response: # Iterate through the generator
                     # Check if the yielded item is a list of strings (IDs)
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch):
                         logger.debug(f"Processing batch of {len(id_batch)} IDs.")
                         vector_ids.extend(id_batch)
                         # Stop if we've reached the requested limit
                         if len(vector_ids) >= limit:
                             vector_ids = vector_ids[:limit] # Trim excess if batch pushed over
                             break
                     # Check if yielded item is an object with an 'ids' attribute (older client versions?)
                     elif hasattr(id_batch, 'ids') and isinstance(id_batch.ids, list):
                         logger.debug(f"Processing batch object with .ids attribute (length {len(id_batch.ids)})")
                         vector_ids.extend(id_batch.ids)
                         if len(vector_ids) >= limit:
                              vector_ids = vector_ids[:limit]
                              break
                     else:
                          # Log unexpected item structure from generator
                          logger.warning(f"Unexpected item type yielded by list generator: {type(id_batch)}. Content (sample): {str(id_batch)[:100]}")

                 # Pagination token handling might be different with generators.
                 # It might be an attribute of the generator object itself after iteration,
                 # or pagination might be handled by just calling list() again.
                 # For now, we assume no easily accessible token from the generator post-iteration.
                 next_page_token = None
                 # Example placeholder if token was found differently:
                 # if hasattr(list_response, 'next_page_token_attribute'):
                 #    next_page_token = list_response.next_page_token_attribute

            else:
                # Handle non-iterable response types if necessary (older clients?)
                logger.warning(f"index.list() did not return an iterable object. Type: {type(list_response)}")
                if hasattr(list_response, 'ids') and isinstance(list_response.ids, list):
                     vector_ids = list_response.ids[:limit]
                elif hasattr(list_response, 'vectors') and isinstance(list_response.vectors, list):
                     vector_ids = [v.id for v in list_response.vectors][:limit]

                # Check for pagination on the response object itself
                if hasattr(list_response, 'pagination') and list_response.pagination and hasattr(list_response.pagination, 'next'):
                    next_page_token = list_response.pagination.next

        except Exception as proc_e:
             logger.error(f"Error processing list response: {proc_e}", exc_info=True)

        logger.info(f"Found {len(vector_ids)} vector IDs in queried namespace '{query_namespace}'. Next Token: {next_page_token}")
        #     next_page_token = list_response.pagination.next

        logger.info(f"Found {len(vector_ids)} vector IDs in queried namespace '{query_namespace}'.")

        return jsonify({
            "namespace": namespace_name,
            "vector_ids": vector_ids,
            "next_token": next_page_token
        }), 200

    except Exception as e:
        logger.error(f"Error listing vector IDs for index '{index_name}', namespace '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while listing vector IDs"}), 500


# --- Pinecone Unique Document Listing Route ---
@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/list_docs', methods=['GET'])
def list_unique_docs_in_namespace(index_name: str, namespace_name: str):
    """
    Lists unique source document filenames by parsing vector IDs within a namespace.
    # Note: This fetches vector IDs and parses them. For very large indexes,
    # consider alternative methods if performance becomes an issue. Fetches up to 100 IDs per page.
    """
    # Use Pinecone's max limit as the default
    limit = request.args.get('limit', default=100, type=int)
    # For true completeness on huge indexes, pagination would be needed here,
    # calling index.list repeatedly with tokens. Simplified for now.
    # next_token = request.args.get('next_token', default=None, type=str)

    logger.info(f"Request received to list unique docs in index '{index_name}', namespace '{namespace_name}' (fetching up to {limit} IDs)")

    # Validate limit against Pinecone's actual constraint (1-100)
    if not 1 <= limit <= 100:
        logger.warning(f"Invalid limit requested: {limit}. Clamping to 100.")
        limit = 100

    try:
        pc = init_pinecone()
        if not pc:
            return jsonify({"error": "Pinecone client initialization failed"}), 500

        try:
            index = pc.Index(index_name)
        except NotFoundException:
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e:
             logger.error(f"Error accessing index '{index_name}': {e}", exc_info=True)
             return jsonify({"error": f"Failed to access index '{index_name}'"}), 500

        query_namespace = "" if namespace_name == "_default" else namespace_name
        logger.info(f"Querying Pinecone namespace: '{query_namespace}' for vector IDs to parse.")

        vector_ids_to_parse = []
        try:
            # Fetch IDs using the list generator
            list_response = index.list(namespace=query_namespace, limit=limit)
            if hasattr(list_response, '__iter__'):
                 logger.debug("Processing list response as an iterator...")
                 count = 0
                 for id_batch in list_response:
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch):
                         vector_ids_to_parse.extend(id_batch)
                         count += len(id_batch)
                         if count >= limit: # Should respect limit from call, but double-check
                             break
                     else:
                          logger.warning(f"Unexpected item type yielded by list generator: {type(id_batch)}")
            else:
                 logger.warning(f"index.list() did not return an iterable object. Type: {type(list_response)}")
                 # Add fallbacks for older response types if needed here

            logger.info(f"Retrieved {len(vector_ids_to_parse)} vector IDs for parsing.")

        except Exception as list_e:
            logger.error(f"Error listing vector IDs for parsing: {list_e}", exc_info=True)
            return jsonify({"error": "Failed to list vector IDs from Pinecone"}), 500

        # --- Parse IDs to get unique document names ---
        unique_doc_names = set()
        for vec_id in vector_ids_to_parse:
            try:
                # Find the last underscore
                last_underscore_index = vec_id.rfind('_')
                if last_underscore_index != -1:
                    # Get the part before the last underscore
                    sanitized_name_part = vec_id[:last_underscore_index]
                    # URL-decode the sanitized name part
                    original_name = urllib.parse.unquote_plus(sanitized_name_part)
                    unique_doc_names.add(original_name)
                else:
                    # If no underscore, maybe it's an old ID format? Log it.
                    logger.warning(f"Vector ID '{vec_id}' does not contain expected '_chunkindex' suffix.")
                    # Optionally add the whole ID if it might be a filename itself
                    # unique_doc_names.add(urllib.parse.unquote_plus(vec_id))
            except Exception as parse_e:
                logger.error(f"Error parsing vector ID '{vec_id}': {parse_e}")
                continue # Skip problematic IDs

        sorted_doc_names = sorted(list(unique_doc_names))
        logger.info(f"Found {len(sorted_doc_names)} unique document names in namespace '{query_namespace}'.")

        return jsonify({
            "index": index_name,
            "namespace": namespace_name, # Return the requested name
            "unique_document_names": sorted_doc_names,
            "vector_ids_checked": len(vector_ids_to_parse) # Info about how many IDs were checked
        }), 200

    except Exception as e:
        logger.error(f"Error listing unique docs for index '{index_name}', namespace '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while listing documents"}), 500


# --- Chat API Route ---
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    logger.info(f"Received request for /api/chat with method: {request.method}") # Log request method
    global transcript_state_cache # Access global cache

    if not anthropic_client:
        logger.error("Chat request failed: Anthropic client not initialized.")
        return jsonify({"error": "AI service unavailable"}), 503

    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing 'messages' in request body"}), 400

        # --- Get Agent & Event Context ---
        agent_name = data.get('agent')
        event_id = data.get('event', '0000') # Default to '0000' if not provided
        session_id = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S')) # Or generate if missing

        if not agent_name:
             return jsonify({"error": "Missing 'agent' in request body"}), 400

        logger.info(f"Chat request for Agent: {agent_name}, Event: {event_id}")

        # --- Prepare Messages & History ---
        incoming_messages = data['messages']
        llm_messages = [
             {"role": msg["role"], "content": msg["content"]}
             for msg in incoming_messages
             if msg.get("role") in ["user", "assistant"]
         ]
        if not llm_messages: return jsonify({"error": "No valid user/assistant messages found"}), 400

        # --- Load Prompts & Context ---
        try:
            base_system_prompt = get_latest_system_prompt(agent_name) or "You are a helpful assistant."
            frameworks = get_latest_frameworks(agent_name)
            event_context = get_latest_context(agent_name, event_id) # Get ORG + EVENT context
            agent_docs = get_agent_docs(agent_name)
            logger.debug("Loaded prompts/context/docs from S3.")
        except Exception as e:
             logger.error(f"Error loading prompts/context from S3: {e}", exc_info=True)
             base_system_prompt = "Error: Could not load configuration."
             frameworks = event_context = agent_docs = None

        # --- Assemble System Prompt ---
        source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file name (e.g., `frameworks_base.md`, `context_aID-river_eID-20240116.txt`, `transcript_...txt`, `doc_XYZ.pdf`) from which you derive information using Markdown footnotes like `[^1]`. \n2. Place the footnote marker directly after the sentence or paragraph containing the information. \n3. List all cited sources at the end under a `### Sources` heading, formatted as `[^1]: source_file_name.ext`."
        realtime_instr = "\n\nIMPORTANT: Prioritize [REAL-TIME Meeting Transcript Update] content for answering questions about the 'current' or 'latest' state of the conversation."
        final_system_prompt = base_system_prompt + source_instr + realtime_instr

        if frameworks: final_system_prompt += "\n\n## Frameworks\n" + frameworks
        if event_context: final_system_prompt += "\n\n## Context\n" + event_context
        if agent_docs: final_system_prompt += "\n\n## Agent Documentation\n" + agent_docs
        # TODO: Add Memory Loading Here

        # --- Add RAG Context (Conditional) ---
        # Add specific instructions on how to USE the context block that follows
        rag_usage_instructions = """

        ## Using Retrieved Context
        When responding to the user:
        1.  **Prioritize Information Within `[Retrieved Context]`:** Base your answer primarily on the information provided in the `[Retrieved Context]` block below, if relevant to the user's query.
        2.  **Direct Extraction for Lists/Facts:** If the user asks for a list, definition, or specific piece of information that is explicitly present in the `[Retrieved Context]`, present that information directly and accurately. Do *not* state that the information is missing if it is clearly provided in the context.
        3.  **Cite Sources:** Remember to cite the source file name using Markdown footnotes (e.g., `[^1]`) for information derived from the context, and list sources under `### Sources`.
        4.  **Synthesize When Necessary:** If the query requires combining information from multiple sources or summarizing, do so, but still ground your answer in the provided context.
        5.  **Acknowledge Missing Info Appropriately:** Only state that information is missing if it is truly absent from the provided context and relevant to the query.
        """
        final_system_prompt += rag_usage_instructions # Add instructions BEFORE the context block itself

        rag_context_block = "" # Initialize empty
        last_user_message_content = next((msg['content'] for msg in reversed(llm_messages) if msg['role'] == 'user'), None)

        # Normalize the last user message for the check
        normalized_query = ""
        if last_user_message_content:
            normalized_query = last_user_message_content.strip().lower()
            # Remove trailing punctuation for a more robust check
            normalized_query = normalized_query.rstrip('.!?')

        # Check if the normalized query is in our simple set
        is_simple_query = normalized_query in SIMPLE_QUERIES_TO_BYPASS_RAG

        if not is_simple_query and last_user_message_content:
            logger.info(f"Complex query detected ('{normalized_query[:50]}...'), proceeding with RAG.")
            try:
                # Initialize RetrievalHandler with agent_name as index_name
                retriever = RetrievalHandler(
                    index_name=agent_name, # <-- USE AGENT NAME FOR INDEX NAME
                    agent_name=agent_name,
                    session_id=session_id,
                    event_id=event_id,
                    anthropic_client=anthropic_client
                )
                retrieved_docs = retriever.get_relevant_context(query=last_user_message_content, top_k=5) # Use original query
                if retrieved_docs:
                    items = [f"--- START Context Source: {d.metadata.get('file_name','Unknown Source')} (Score: {d.metadata.get('score',0):.2f}) ---\n{d.page_content}\n--- END Context Source: {d.metadata.get('file_name','Unknown Source')} ---" for i, d in enumerate(retrieved_docs)]
                    # Use clearer markers for the entire block
                    rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n" + "\n\n".join(items) + "\n=== END RETRIEVED CONTEXT ==="
                    logger.debug(f"Added {len(retrieved_docs)} RAG context docs ({len(rag_context_block)} chars).")
                else:
                    logger.debug("No RAG context docs found for complex query.")
                    # Explicitly add a note if no context was found AFTER attempting retrieval
                    rag_context_block = "\n\n[Note: No relevant documents found in the knowledge base for this query.]"
            except RuntimeError as e:
                 # Catch specific "Failed connection" or other init errors
                 logger.warning(f"RAG context retrieval skipped: {e}")
                 # Use consistent marker format
                 rag_context_block = f"\n\n=== START RETRIEVED CONTEXT ===\n[Note: Document retrieval failed for index '{agent_name}']\n=== END RETRIEVED CONTEXT ==="
            except Exception as e:
                logger.error(f"Unexpected error during RAG context retrieval: {e}", exc_info=True)
                 # Use consistent marker format
                rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Error retrieving documents]\n=== END RETRIEVED CONTEXT ==="
        elif is_simple_query:
            logger.info(f"Simple query detected ('{normalized_query}'), bypassing RAG.")
            # Add a note if RAG was skipped due to simple query
            rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Document retrieval skipped for this simple query.]\n=== END RETRIEVED CONTEXT ==="
        else:
             logger.debug("No user message found or skipping RAG for other reasons (e.g., first message).")
             # Add a note if RAG was skipped for other reasons
             rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Document retrieval not applicable for this message.]\n=== END RETRIEVED CONTEXT ==="

        # Append the (potentially empty) RAG block to the system prompt
        final_system_prompt += rag_context_block

        # --- Add Transcript Context ---
        transcript_content_to_add = ""
        state_key = (agent_name, event_id)
        with transcript_state_lock:
            if state_key not in transcript_state_cache:
                transcript_state_cache[state_key] = TranscriptState()
                logger.info(f"Created new TranscriptState for {agent_name}/{event_id}")
            current_transcript_state = transcript_state_cache[state_key]

        try:
            new_transcript = read_new_transcript_content(current_transcript_state, agent_name, event_id)
            if new_transcript:
                label = "[REAL-TIME Meeting Transcript Update]"
                transcript_content_to_add = f"{label}\n{new_transcript}"
                logger.info(f"Adding recent transcript update ({len(new_transcript)} chars).")
                llm_messages.insert(-1, {'role': 'user', 'content': transcript_content_to_add})
            else:
                logger.debug("No new transcript updates found.")
        except Exception as e:
             logger.error(f"Error reading transcript updates: {e}", exc_info=True)


        # --- Add Time Context ---
        now_utc = datetime.now(timezone.utc)
        time_str = now_utc.strftime('%A, %Y-%m-%d %H:%M:%S %Z')
        time_context = f"\nCurrent Time Context: {time_str}"
        final_system_prompt += time_context

        # --- Prepare for LLM Call ---
        llm_model_name = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20240620")
        llm_max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))

        logger.debug(f"Final System Prompt Length: {len(final_system_prompt)}")
        logger.debug(f"Messages for API ({len(llm_messages)}):")
        if logger.isEnabledFor(logging.DEBUG):
             for i, msg in enumerate(llm_messages[-5:]):
                  logger.debug(f"  Msg [-{len(llm_messages)-i}]: Role={msg['role']}, Len={len(msg['content'])}, Content='{msg['content'][:100]}...'")

        # --- Streaming Response ---
        def generate_stream():
            response_content = ""; stream_error = None
            try:
                with _call_anthropic_stream_with_retry(
                    model=llm_model_name,
                    max_tokens=llm_max_tokens,
                    system=final_system_prompt,
                    messages=llm_messages
                ) as stream:
                    for text in stream.text_stream:
                        response_content += text
                        # Format as Server-Sent Event (SSE)
                        sse_data = json.dumps({'delta': text})
                        yield f"data: {sse_data}\n\n" # Ensure double newline separator
                logger.info(f"LLM stream completed successfully ({len(response_content)} chars).")

            except RetryError as e:
                 logger.error(f"Anthropic API call failed after multiple retries: {e}", exc_info=True)
                 stream_error = "Assistant is currently unavailable after multiple retries. Please try again later."
            except APIStatusError as e:
                logger.error(f"Anthropic API Status Error (non-retryable or final attempt): {e}", exc_info=True)
                stream_error = f"API Error: {e.message}" if hasattr(e, 'message') else str(e)
                if 'overloaded' in str(e).lower(): stream_error = "Assistant API is temporarily overloaded. Please try again."
            except AnthropicError as e:
                logger.error(f"Anthropic API Error: {e}", exc_info=True)
                stream_error = f"Anthropic Error: {str(e)}"
            except Exception as e:
                 if "aborted" in str(e).lower() or "cancel" in str(e).lower():
                      logger.warning(f"LLM stream aborted or cancelled: {e}")
                      stream_error="Stream stopped by client."
                 else:
                      logger.error(f"LLM stream error: {e}", exc_info=True)
                      stream_error = f"An unexpected error occurred: {str(e)}"

            if stream_error:
                sse_error_data = json.dumps({'error': stream_error})
                yield f"data: {sse_error_data}\n\n"

            # TODO: Add chat archiving logic here using s3_utils if needed

            sse_done_data = json.dumps({'done': True})
            yield f"data: {sse_done_data}\n\n"

        # Return the streaming response with correct MIME type for SSE
        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    # Use host='0.0.0.0' for Render/Docker
    # Turn off Flask's default reloader if debug is True, as it can interfere
    use_reloader = False if debug_mode else False
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=use_reloader)