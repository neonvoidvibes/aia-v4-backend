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
from magic_audio import MagicAudio
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history, get_s3_client
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
    "pause_start_time": None, "last_pause_timestamp": None,
    "elapsed_time": 0, "agent": None, "event": None
}

anthropic_client: Anthropic | None = None
try: init_pinecone(); logger.info("Pinecone initialized (or skipped).")
except Exception as e: logger.warning(f"Pinecone initialization failed: {e}")

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
    """Helper to get a snapshot of the current recording status for responses."""
    current_elapsed = recording_status["elapsed_time"]
    if recording_status["is_recording"] and not recording_status["is_paused"] and recording_status["start_time"]:
        current_elapsed = time.time() - recording_status["start_time"]
    return {
        "is_recording": recording_status["is_recording"],
        "is_paused": recording_status["is_paused"],
        "elapsed_time": int(current_elapsed),
        "agent": recording_status.get("agent"),
        "event": recording_status.get("event")
    }

@app.route('/api/recording/start', methods=['POST'])
def start_recording_route(): # Renamed to avoid conflict with function name
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if recording_status["is_recording"]:
            return jsonify({"status": "error", "message": "Already recording", "recording_status": _get_current_recording_status_snapshot()}), 400
        data = request.json; agent = data.get('agent'); event = data.get('event'); language = data.get('language')
        if not agent or not event:
            return jsonify({"status": "error", "message": "Missing agent or event", "recording_status": _get_current_recording_status_snapshot()}), 400
        logger.info(f"Starting recording for Agent: {agent}, Event: {event}, Lang: {language}")
        try:
            if magic_audio_instance:
                try: magic_audio_instance.stop()
                except Exception as e: logger.warning(f"Error stopping previous audio instance: {e}")
            magic_audio_instance = MagicAudio(agent=agent, event=event, language=language)
            magic_audio_instance.start()
            recording_status.update({
                "is_recording": True, "is_paused": False, "start_time": time.time(),
                "pause_start_time": None, "last_pause_timestamp": None, "elapsed_time": 0,
                "agent": agent, "event": event
            })
            logger.info("Recording started successfully.")
            return jsonify({"status": "success", "message": "Recording started", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True); magic_audio_instance = None
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "elapsed_time": 0, "agent": None, "event": None})
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording_route(): # Renamed
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            # Ensure status reflects not recording if called when already stopped
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "last_pause_timestamp": None, "elapsed_time": 0})
            return jsonify({"status": "success", "message": "Not recording or instance missing", "recording_status": _get_current_recording_status_snapshot()}), 200
        
        logger.info("Stopping recording...")
        try:
            magic_audio_instance.stop()
            if recording_status["start_time"]:
                now = time.time()
                if recording_status["is_paused"] and recording_status["pause_start_time"]:
                    recording_status["elapsed_time"] = recording_status["pause_start_time"] - recording_status["start_time"]
                else: recording_status["elapsed_time"] = now - recording_status["start_time"]
            else: recording_status["elapsed_time"] = 0 # Should not happen if start_time is set

            recording_status.update({
                "is_recording": False, "is_paused": False, "start_time": None,
                "pause_start_time": None, "last_pause_timestamp": None
                # elapsed_time is set above and retained
            })
            magic_audio_instance = None
            logger.info("Recording stopped and transcript saved.")
            # Capture final status after updates
            final_status = _get_current_recording_status_snapshot()
            final_status["elapsed_time"] = 0 # Explicitly set to 0 on stop for UI
            return jsonify({"status": "success", "message": "Recording stopped", "recording_status": final_status})
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            # Attempt to reset status even on error
            recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "last_pause_timestamp": None, "elapsed_time":0})
            magic_audio_instance = None
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/pause', methods=['POST'])
def pause_recording_route(): # Renamed
    global magic_audio_instance, recording_status
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
            recording_status["last_pause_timestamp"] = pause_time
            if recording_status["start_time"]: recording_status["elapsed_time"] = pause_time - recording_status["start_time"]
            logger.info("Recording paused.")
            return jsonify({"status": "success", "message": "Recording paused", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Error pausing recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/resume', methods=['POST'])
def resume_recording_route(): # Renamed
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            return jsonify({"status": "error", "message": "Not recording", "recording_status": _get_current_recording_status_snapshot()}), 400
        if not recording_status["is_paused"]:
            return jsonify({"status": "success", "message": "Not paused", "recording_status": _get_current_recording_status_snapshot()}), 200
        logger.info("Resuming recording...")
        try:
            if recording_status["pause_start_time"] and recording_status["start_time"]:
                pause_duration = time.time() - recording_status["pause_start_time"]
                recording_status["start_time"] += pause_duration
                logger.debug(f"Resuming after {pause_duration:.2f}s pause. Adjusted start time.")
            else: logger.warning("Could not calculate pause duration accurately on resume.")
            
            magic_audio_instance.resume()
            recording_status["is_paused"] = False
            recording_status["pause_start_time"] = None
            recording_status["last_pause_timestamp"] = None
            # elapsed_time will be recalculated based on new start_time by status getter or next action
            logger.info("Recording resumed.")
            return jsonify({"status": "success", "message": "Recording resumed", "recording_status": _get_current_recording_status_snapshot()})
        except Exception as e:
            logger.error(f"Error resuming recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e), "recording_status": _get_current_recording_status_snapshot()}), 500

@app.route('/api/recording/status', methods=['GET'])
def get_recording_status_route(): # Renamed
    with magic_audio_lock:
        status_data = _get_current_recording_status_snapshot()
    return jsonify(status_data), 200

# ... (rest of the file: /api/index routes, /api/chat, auto_stop_long_paused_recordings, main) ...
# The /api/chat, /api/index/*, and other utility functions remain unchanged from the provided codebase.
# I will only paste the changed sections for brevity unless you need the full file.
# For now, I'll assume the rest of api_server.py is the same and focus on the frontend.
# If you need the full api_server.py, let me know.

# --- Pinecone Index Info Route ---
@app.route('/api/index/<string:index_name>/stats', methods=['GET'])
def get_pinecone_index_stats(index_name: str):
    logger.info(f"Request received for stats of index: {index_name}")
    try:
        pc = init_pinecone()
        if not pc:
            logger.error(f"Failed to initialize Pinecone client for stats request.")
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
        stats = index.describe_index_stats()
        stats_dict = {}
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
    limit = request.args.get('limit', default=1000, type=int)
    next_token = request.args.get('next_token', default=None, type=str)
    logger.info(f"Request received to list IDs in index '{index_name}', namespace '{namespace_name}' (limit: {limit}, token: {next_token})")
    if not 1 <= limit <= 1000: logger.warning(f"Invalid limit requested: {limit}. Using default 1000."); limit = 1000
    try:
        pc = init_pinecone()
        if not pc: logger.error(f"Failed Pinecone client init."); return jsonify({"error": "Pinecone client initialization failed"}), 500
        try: index = pc.Index(index_name); logger.info(f"Accessing Pinecone index '{index_name}'")
        except NotFoundException: logger.warning(f"Index '{index_name}' not found."); return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e: logger.error(f"Error accessing index '{index_name}': {e}", exc_info=True); return jsonify({"error": f"Failed to access index '{index_name}'"}), 500
        query_namespace = "" if namespace_name == "_default" else namespace_name
        logger.info(f"Querying Pinecone namespace: '{query_namespace}' (URL requested: '{namespace_name}')")
        list_response = index.list(namespace=query_namespace, limit=limit)
        vector_ids = []; next_page_token = None
        try:
            if hasattr(list_response, '__iter__'):
                 logger.debug("Processing response as an iterator/generator")
                 for id_batch in list_response:
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch):
                         logger.debug(f"Processing batch of {len(id_batch)} IDs."); vector_ids.extend(id_batch)
                         if len(vector_ids) >= limit: vector_ids = vector_ids[:limit]; break
                     elif hasattr(id_batch, 'ids') and isinstance(id_batch.ids, list):
                         logger.debug(f"Processing batch object with .ids (len {len(id_batch.ids)})"); vector_ids.extend(id_batch.ids)
                         if len(vector_ids) >= limit: vector_ids = vector_ids[:limit]; break
                     else: logger.warning(f"Unexpected item type from list gen: {type(id_batch)}. Content (sample): {str(id_batch)[:100]}")
            else:
                logger.warning(f"index.list() did not return iterable. Type: {type(list_response)}")
                if hasattr(list_response, 'ids') and isinstance(list_response.ids, list): vector_ids = list_response.ids[:limit]
                elif hasattr(list_response, 'vectors') and isinstance(list_response.vectors, list): vector_ids = [v.id for v in list_response.vectors][:limit]
                if hasattr(list_response, 'pagination') and list_response.pagination and hasattr(list_response.pagination, 'next'): next_page_token = list_response.pagination.next
        except Exception as proc_e: logger.error(f"Error processing list response: {proc_e}", exc_info=True)
        logger.info(f"Found {len(vector_ids)} vector IDs in '{query_namespace}'. Next Token: {next_page_token}")
        return jsonify({"namespace": namespace_name, "vector_ids": vector_ids, "next_token": next_page_token}), 200
    except Exception as e:
        logger.error(f"Error listing vector IDs for index '{index_name}', ns '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "Unexpected error listing vector IDs"}), 500

@app.route('/api/index/<string:index_name>/namespace/<string:namespace_name>/list_docs', methods=['GET'])
def list_unique_docs_in_namespace(index_name: str, namespace_name: str):
    limit = request.args.get('limit', default=100, type=int)
    logger.info(f"Request to list unique docs in index '{index_name}', ns '{namespace_name}' (fetch up to {limit} IDs)")
    if not 1 <= limit <= 100: logger.warning(f"Invalid limit: {limit}. Clamping to 100."); limit = 100
    try:
        pc = init_pinecone()
        if not pc: return jsonify({"error": "Pinecone client initialization failed"}), 500
        try: index = pc.Index(index_name)
        except NotFoundException: return jsonify({"error": f"Index '{index_name}' not found"}), 404
        except Exception as e: logger.error(f"Error access index '{index_name}': {e}", exc_info=True); return jsonify({"error": f"Failed access index '{index_name}'"}), 500
        query_namespace = "" if namespace_name == "_default" else namespace_name
        logger.info(f"Querying Pinecone ns: '{query_namespace}' for vector IDs to parse.")
        vector_ids_to_parse = []
        try:
            list_response = index.list(namespace=query_namespace, limit=limit)
            if hasattr(list_response, '__iter__'):
                 logger.debug("Processing list response as iterator...")
                 count = 0
                 for id_batch in list_response:
                     if isinstance(id_batch, list) and all(isinstance(item, str) for item in id_batch):
                         vector_ids_to_parse.extend(id_batch); count += len(id_batch)
                         if count >= limit: break
                     else: logger.warning(f"Unexpected item type from list gen: {type(id_batch)}")
            else: logger.warning(f"index.list() not iterable. Type: {type(list_response)}")
            logger.info(f"Retrieved {len(vector_ids_to_parse)} vector IDs for parsing.")
        except Exception as list_e: logger.error(f"Error listing vector IDs for parsing: {list_e}", exc_info=True); return jsonify({"error": "Failed list vector IDs from Pinecone"}), 500
        unique_doc_names = set()
        for vec_id in vector_ids_to_parse:
            try:
                last_underscore_index = vec_id.rfind('_')
                if last_underscore_index != -1:
                    sanitized_name_part = vec_id[:last_underscore_index]
                    original_name = urllib.parse.unquote_plus(sanitized_name_part)
                    unique_doc_names.add(original_name)
                else: logger.warning(f"Vector ID '{vec_id}' no expected '_chunkindex' suffix.")
            except Exception as parse_e: logger.error(f"Error parsing vector ID '{vec_id}': {parse_e}")
        sorted_doc_names = sorted(list(unique_doc_names))
        logger.info(f"Found {len(sorted_doc_names)} unique doc names in ns '{query_namespace}'.")
        return jsonify({"index": index_name, "namespace": namespace_name, "unique_document_names": sorted_doc_names, "vector_ids_checked": len(vector_ids_to_parse)}), 200
    except Exception as e:
        logger.error(f"Error listing unique docs for index '{index_name}', ns '{namespace_name}': {e}", exc_info=True)
        return jsonify({"error": "Unexpected error listing documents"}), 500

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    logger.info(f"Received request /api/chat method: {request.method}")
    global transcript_state_cache
    if not anthropic_client: logger.error("Chat fail: Anthropic client not init."); return jsonify({"error": "AI service unavailable"}), 503
    try:
        data = request.json
        if not data or 'messages' not in data: return jsonify({"error": "Missing 'messages'"}), 400
        agent_name = data.get('agent'); event_id = data.get('event', '0000'); session_id = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S'))
        if not agent_name: return jsonify({"error": "Missing 'agent'"}), 400
        logger.info(f"Chat request for Agent: {agent_name}, Event: {event_id}")
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
                    logger.debug(f"Added {len(retrieved_docs)} RAG docs ({len(rag_context_block)} chars).")
                else: logger.debug("No RAG docs for complex query."); rag_context_block = "\n\n[Note: No relevant documents found for this query.]"
            except RuntimeError as e: logger.warning(f"RAG skipped: {e}"); rag_context_block = f"\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval failed for index '{agent_name}']\n=== END RETRIEVED CONTEXT ==="
            except Exception as e: logger.error(f"Unexpected RAG error: {e}", exc_info=True); rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Error retrieving documents]\n=== END RETRIEVED CONTEXT ==="
        elif is_simple_query: logger.info(f"Simple query ('{normalized_query}'), bypass RAG."); rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval skipped for simple query.]\n=== END RETRIEVED CONTEXT ==="
        else: logger.debug("No user msg or skip RAG."); rag_context_block = "\n\n=== START RETRIEVED CONTEXT ===\n[Note: Doc retrieval not applicable.]\n=== END RETRIEVED CONTEXT ==="
        final_system_prompt += rag_context_block
        transcript_content_to_add = ""; state_key = (agent_name, event_id)
        with transcript_state_lock:
            if state_key not in transcript_state_cache: transcript_state_cache[state_key] = TranscriptState(); logger.info(f"New TranscriptState for {agent_name}/{event_id}")
            current_transcript_state = transcript_state_cache[state_key]
        try:
            new_transcript = read_new_transcript_content(current_transcript_state, agent_name, event_id)
            if new_transcript:
                label = "[REAL-TIME Meeting Transcript Update]"; transcript_content_to_add = f"{label}\n{new_transcript}"
                logger.info(f"Adding recent tx update ({len(new_transcript)} chars)."); llm_messages.insert(-1, {'role': 'user', 'content': transcript_content_to_add})
            else: logger.debug("No new tx updates.")
        except Exception as e: logger.error(f"Error reading tx updates: {e}", exc_info=True)
        now_utc = datetime.now(timezone.utc); time_str = now_utc.strftime('%A, %Y-%m-%d %H:%M:%S %Z')
        final_system_prompt += f"\nCurrent Time Context: {time_str}"
        llm_model_name = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20240620")
        llm_max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))
        logger.debug(f"Final Sys Prompt Len: {len(final_system_prompt)}")
        logger.debug(f"Messages for API ({len(llm_messages)}):")
        if logger.isEnabledFor(logging.DEBUG):
             for i, msg in enumerate(llm_messages[-5:]): logger.debug(f"  Msg [-{len(llm_messages)-i}]: Role={msg['role']}, Len={len(msg['content'])}, Content='{msg['content'][:100]}...'")
        def generate_stream():
            response_content = ""; stream_error = None
            try:
                with _call_anthropic_stream_with_retry(model=llm_model_name, max_tokens=llm_max_tokens, system=final_system_prompt, messages=llm_messages) as stream:
                    for text in stream.text_stream: response_content += text; sse_data = json.dumps({'delta': text}); yield f"data: {sse_data}\n\n"
                logger.info(f"LLM stream completed ({len(response_content)} chars).")
            except RetryError as e: logger.error(f"Anthropic API fail after retries: {e}", exc_info=True); stream_error = "Assistant unavailable after retries. Try again."
            except APIStatusError as e: logger.error(f"Anthropic API Status Error: {e}", exc_info=True); stream_error = f"API Error: {e.message}" if hasattr(e, 'message') else str(e);
            except AnthropicError as e: logger.error(f"Anthropic API Error: {e}", exc_info=True); stream_error = f"Anthropic Error: {str(e)}"
            except Exception as e:
                 if "aborted" in str(e).lower() or "cancel" in str(e).lower(): logger.warning(f"LLM stream aborted/cancelled: {e}"); stream_error="Stream stopped by client."
                 else: logger.error(f"LLM stream error: {e}", exc_info=True); stream_error = f"Unexpected error: {str(e)}"
            if stream_error: sse_error_data = json.dumps({'error': stream_error}); yield f"data: {sse_error_data}\n\n"
            sse_done_data = json.dumps({'done': True}); yield f"data: {sse_done_data}\n\n"
        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
    except Exception as e: logger.error(f"Error in /api/chat: {e}", exc_info=True); return jsonify({"error": "Internal server error"}), 500

def auto_stop_long_paused_recordings():
    global magic_audio_instance, recording_status, magic_audio_lock
    pause_timeout_seconds = 2 * 60 * 60; check_interval_seconds = 5 * 60
    logger.info(f"Auto-stop thread: Check every {check_interval_seconds}s for pauses > {pause_timeout_seconds}s.")
    while True:
        time.sleep(check_interval_seconds)
        with magic_audio_lock:
            if (recording_status["is_recording"] and recording_status["is_paused"] and recording_status["last_pause_timestamp"] is not None):
                pause_duration = time.time() - recording_status["last_pause_timestamp"]
                logger.debug(f"Checking paused recording for {recording_status.get('agent')}/{recording_status.get('event')}. Pause duration: {pause_duration:.0f}s")
                if pause_duration > pause_timeout_seconds:
                    logger.warning(f"Auto-stopping recording for {recording_status.get('agent')}/{recording_status.get('event')} after {pause_duration:.0f}s pause (limit: {pause_timeout_seconds}s).")
                    try:
                        if magic_audio_instance: magic_audio_instance.stop()
                        if recording_status["start_time"]: recording_status["elapsed_time"] = recording_status["last_pause_timestamp"] - recording_status["start_time"]
                        else: recording_status["elapsed_time"] = 0
                        recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "last_pause_timestamp": None})
                        magic_audio_instance = None; logger.info("Recording auto-stopped.")
                    except Exception as e:
                        logger.error(f"Error during auto-stop: {e}", exc_info=True)
                        recording_status.update({"is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "last_pause_timestamp": None, "elapsed_time":0})
                        magic_audio_instance = None

if __name__ == '__main__':
    auto_stop_thread = threading.Thread(target=auto_stop_long_paused_recordings, daemon=True); auto_stop_thread.start()
    port = int(os.getenv('PORT', 5001)); debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    use_reloader = False if debug_mode else False
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=use_reloader)