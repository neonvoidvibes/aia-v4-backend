# /aia-v4-backend/api_server.py
import os
import sys
import logging
from flask import Flask, jsonify, request, Response, stream_with_context
from dotenv import load_dotenv
import threading
import time
import json
from datetime import datetime, timezone

# Import necessary modules from our project
from magic_audio import MagicAudio
from utils.retrieval_handler import RetrievalHandler
from utils.transcript_utils import TranscriptState, read_new_transcript_content, read_all_transcripts_in_folder, get_latest_transcript_file
from utils.s3_utils import (
    get_latest_system_prompt, get_latest_frameworks, get_latest_context,
    get_agent_docs, save_chat_to_s3, format_chat_history
)
from utils.pinecone_utils import init_pinecone # Ensure Pinecone is init'd if needed by RetrievalHandler

# LLM Client
from anthropic import Anthropic, APIStatusError, AnthropicError

# Retry mechanism
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, retry_if_exception_type

# --- Load environment variables ---
load_dotenv()

# --- Logging Setup (same as before) ---
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
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests', 'pinecone']: logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('utils').setLevel(logging.DEBUG if debug else logging.INFO) # Log our utils
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger
logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- Global State (Simplified for POC) ---
# Transcription
magic_audio_instance: MagicAudio | None = None
magic_audio_lock = threading.Lock()
recording_status = { "is_recording": False, "is_paused": False, "start_time": None, "pause_start_time": None, "elapsed_time": 0, "agent": None, "event": None }

# Chat (Anthropic Client is global)
anthropic_client: Anthropic | None = None
# Initialize Pinecone globally if RetrievalHandler needs it readily available
init_pinecone()

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
    return jsonify({"status": "ok", "message": "Backend is running", "anthropic_client": anthropic_ok}), 200

# --- Transcription Control API Routes (Keep as before) ---
@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    # ... (Keep existing implementation) ...
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
    # ... (Keep existing implementation) ...
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
    # ... (Keep existing implementation) ...
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
    # ... (Keep existing implementation) ...
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
    # ... (Keep existing implementation) ...
    with magic_audio_lock:
        current_elapsed = recording_status["elapsed_time"]
        if recording_status["is_recording"] and not recording_status["is_paused"] and recording_status["start_time"]: current_elapsed = time.time() - recording_status["start_time"]
        status_data = {"is_recording": recording_status["is_recording"], "is_paused": recording_status["is_paused"], "elapsed_time": int(current_elapsed), "agent": recording_status.get("agent"), "event": recording_status.get("event")}
    return jsonify(status_data)

# --- Chat API Route ---
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    global transcript_state_cache # Access global cache

    if not anthropic_client:
        logger.error("Chat request failed: Anthropic client not initialized.")
        return jsonify({"error": "AI service unavailable"}), 503

    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing 'messages' in request body"}), 400

        # --- Get Agent & Event Context ---
        # Assume these are passed in the request body or headers from the Next.js proxy
        # which gets them from URL params
        agent_name = data.get('agent')
        event_id = data.get('event', '0000') # Default to '0000' if not provided
        session_id = data.get('session_id', datetime.now().strftime('%Y%m%d-T%H%M%S')) # Or generate if missing

        if not agent_name:
             return jsonify({"error": "Missing 'agent' in request body"}), 400

        logger.info(f"Chat request for Agent: {agent_name}, Event: {event_id}")

        # --- Prepare Messages & History ---
        incoming_messages = data['messages']
        # For now, use only the messages sent in this request for the API call history
        # A more robust solution would involve session management or passing history
        llm_messages = [
             {"role": msg["role"], "content": msg["content"]}
             for msg in incoming_messages
             if msg.get("role") in ["user", "assistant"] # Filter out system/other roles if present
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
             # Use defaults or error message if loading fails critically
             base_system_prompt = "Error: Could not load configuration."
             frameworks = event_context = agent_docs = None

        # --- Assemble System Prompt ---
        # Combine instructions into the base prompt permanently during loading
        source_instr = "\n\n## Source Attribution Requirements\n1. ALWAYS specify the exact source file name (e.g., `frameworks_base.md`, `context_aID-river_eID-20240116.txt`, `transcript_...txt`, `doc_XYZ.pdf`) from which you derive information using Markdown footnotes like `[^1]`. \n2. Place the footnote marker directly after the sentence or paragraph containing the information. \n3. List all cited sources at the end under a `### Sources` heading, formatted as `[^1]: source_file_name.ext`."
        realtime_instr = "\n\nIMPORTANT: Prioritize [REAL-TIME Meeting Transcript Update] content for answering questions about the 'current' or 'latest' state of the conversation."
        # synth_instr = "\nSynthesizing from Context: When answering, combine related pieces of information from different context sources or transcript segments into a coherent response. Avoid simply listing chunks. Do not state 'incomplete context' if you have *some* relevant information; synthesize what you have."
        final_system_prompt = base_system_prompt + source_instr + realtime_instr #+ synth_instr

        if frameworks: final_system_prompt += "\n\n## Frameworks\n" + frameworks
        if event_context: final_system_prompt += "\n\n## Context\n" + event_context # Combined Org/Event Context
        if agent_docs: final_system_prompt += "\n\n## Agent Documentation\n" + agent_docs
        # TODO: Add Memory Loading Here if needed, similar to web_chat.py/magic_chat.py

        # --- Add RAG Context ---
        rag_context_block = ""
        try:
            retriever = RetrievalHandler(
                index_name=os.getenv('PINECONE_INDEX_NAME', 'magicchat'),
                agent_name=agent_name,
                session_id=session_id, # Pass session/event if needed by retriever logic
                event_id=event_id,
                anthropic_client=anthropic_client # Pass the client
            )
            # Use the last user message as the query for RAG
            last_user_message = next((msg['content'] for msg in reversed(llm_messages) if msg['role'] == 'user'), None)
            if last_user_message:
                retrieved_docs = retriever.get_relevant_context(query=last_user_message, top_k=5) # Get 5 docs for POC
                if retrieved_docs:
                    items = [f"[Ctx {i+1} from {d.metadata.get('file_name','?')}(Score:{d.metadata.get('score',0):.2f})]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
                    rag_context_block = "\n\n---\nRetrieved Context (for potential relevance):\n" + "\n\n".join(items)
                    logger.debug(f"Added {len(retrieved_docs)} RAG context docs ({len(rag_context_block)} chars).")
            else:
                logger.debug("No user message found to generate RAG context.")
        except Exception as e:
            logger.error(f"Error during RAG context retrieval: {e}", exc_info=True)
            rag_context_block = "\n\nError retrieving context."

        final_system_prompt += rag_context_block

        # --- Add Transcript Context ---
        transcript_content_to_add = ""
        # Get or create TranscriptState for this agent/event
        state_key = (agent_name, event_id)
        with transcript_state_lock:
            if state_key not in transcript_state_cache:
                transcript_state_cache[state_key] = TranscriptState()
                logger.info(f"Created new TranscriptState for {agent_name}/{event_id}")
            current_transcript_state = transcript_state_cache[state_key]

        try:
            # Fetch *new* transcript content since last check for this state
            new_transcript = read_new_transcript_content(current_transcript_state, agent_name, event_id)
            if new_transcript:
                # Simple approach for POC: just add the latest update block
                label = "[REAL-TIME Meeting Transcript Update]"
                transcript_content_to_add = f"{label}\n{new_transcript}"
                logger.info(f"Adding recent transcript update ({len(new_transcript)} chars).")
                # Inject as a user message *before* the actual current user message
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
        llm_model_name = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20240620") # Use 3.5 Sonnet as default
        llm_max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 4096))

        logger.debug(f"Final System Prompt Length: {len(final_system_prompt)}")
        logger.debug(f"Messages for API ({len(llm_messages)}):")
        if logger.isEnabledFor(logging.DEBUG):
             for i, msg in enumerate(llm_messages[-5:]): # Log last 5 messages
                  logger.debug(f"  Msg [-{len(llm_messages)-i}]: Role={msg['role']}, Len={len(msg['content'])}, Content='{msg['content'][:100]}...'")

        # --- Streaming Response ---
        def generate_stream():
            response_content = ""; stream_error = None
            try:
                # Use the retry-enabled helper method
                with _call_anthropic_stream_with_retry(
                    model=llm_model_name,
                    max_tokens=llm_max_tokens,
                    system=final_system_prompt,
                    messages=llm_messages
                ) as stream:
                    for text in stream.text_stream:
                        response_content += text
                        yield f"data: {json.dumps({'delta': text})}\n\n"
                logger.info(f"LLM stream completed successfully ({len(response_content)} chars).")

            except RetryError as e:
                 logger.error(f"Anthropic API call failed after multiple retries: {e}", exc_info=True)
                 stream_error = "Assistant is currently unavailable after multiple retries. Please try again later."
            except APIStatusError as e:
                logger.error(f"Anthropic API Status Error (non-retryable or final attempt): {e}", exc_info=True)
                stream_error = f"API Error: {e.message}" if hasattr(e, 'message') else str(e)
                if 'overloaded' in str(e).lower(): stream_error = "Assistant API is temporarily overloaded. Please try again."
            except AnthropicError as e: # Catch other Anthropic specific errors
                logger.error(f"Anthropic API Error: {e}", exc_info=True)
                stream_error = f"Anthropic Error: {str(e)}"
            except Exception as e:
                 if "aborted" in str(e).lower() or "cancel" in str(e).lower():
                      logger.warning(f"LLM stream aborted or cancelled: {e}")
                      stream_error="Stream stopped by client."
                 else:
                      logger.error(f"LLM stream error: {e}", exc_info=True)
                      stream_error = f"An unexpected error occurred: {str(e)}"

            # Send error message if any occurred
            if stream_error:
                yield f"data: {json.dumps({'error': stream_error})}\n\n"

            # TODO: Add chat archiving logic here using s3_utils if needed
            # If archiving: use the original incoming_messages + the response_content

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        # Return the streaming response
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
    app.run(host='0.0.0.0', port=port, debug=debug_mode)