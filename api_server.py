# /aia-v4-backend/api_server.py
import os
import sys
import logging
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import threading
import time
from magic_audio import MagicAudio # Import the copied module

# --- Load environment variables ---
load_dotenv()

# --- Logging Setup (same as before) ---
def setup_logging(debug=False):
    # ... (keep the existing setup_logging function) ...
    log_filename = 'api_server.log'
    root_logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8'); fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'); fh.setFormatter(ff); root_logger.addHandler(fh)
    except Exception as e: print(f"Error setting up file logger: {e}", file=sys.stderr)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level)
    cf = logging.Formatter('[%(levelname)-8s] %(name)s: %(message)s'); ch.setFormatter(cf); root_logger.addHandler(ch)
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests']: logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.INFO)
    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger
logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- Global State for Transcription (Simplified for POC) ---
# In a production scenario, you'd want session management or a more robust way
# to handle multiple users/recordings if needed. For now, one global instance.
magic_audio_instance: MagicAudio | None = None
magic_audio_lock = threading.Lock()
recording_status = {
    "is_recording": False,
    "is_paused": False,
    "start_time": None,
    "pause_start_time": None,
    "elapsed_time": 0,
    "agent": None,
    "event": None
}

# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

# --- Transcription Control API Routes ---
@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if recording_status["is_recording"]:
            logger.warning("Start recording request ignored: Already recording.")
            return jsonify({"status": "error", "message": "Already recording"}), 400

        data = request.json
        agent = data.get('agent')
        event = data.get('event')
        language = data.get('language') # Optional language from frontend

        if not agent or not event:
            logger.error("Start recording request failed: Missing agent or event.")
            return jsonify({"status": "error", "message": "Missing agent or event"}), 400

        logger.info(f"Starting recording for Agent: {agent}, Event: {event}, Lang: {language}")
        try:
            # Ensure previous instance is fully stopped if somehow lingering
            if magic_audio_instance:
                try: magic_audio_instance.stop()
                except Exception as e: logger.warning(f"Error stopping previous audio instance: {e}")

            magic_audio_instance = MagicAudio(
                agent=agent,
                event=event,
                language=language # Pass language if provided
            )
            magic_audio_instance.start()

            recording_status["is_recording"] = True
            recording_status["is_paused"] = False
            recording_status["start_time"] = time.time()
            recording_status["pause_start_time"] = None
            recording_status["elapsed_time"] = 0
            recording_status["agent"] = agent
            recording_status["event"] = event
            logger.info("Recording started successfully.")
            return jsonify({"status": "success", "message": "Recording started"})
        except Exception as e:
            logger.error(f"Failed to start recording: {e}", exc_info=True)
            magic_audio_instance = None # Clear instance on failure
            # Reset status
            recording_status["is_recording"] = False
            recording_status["is_paused"] = False
            recording_status["start_time"] = None
            recording_status["elapsed_time"] = 0
            recording_status["agent"] = None
            recording_status["event"] = None
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            logger.warning("Stop recording request ignored: Not recording.")
            return jsonify({"status": "success", "message": "Not recording"}), 200 # Be idempotent

        logger.info("Stopping recording...")
        try:
            magic_audio_instance.stop()
            # Update final elapsed time
            if recording_status["start_time"]:
                 now = time.time()
                 if recording_status["is_paused"] and recording_status["pause_start_time"]:
                     # If stopped while paused, calculate time up to pause point
                      recording_status["elapsed_time"] = recording_status["pause_start_time"] - recording_status["start_time"]
                 else:
                      recording_status["elapsed_time"] = now - recording_status["start_time"]
            else:
                recording_status["elapsed_time"] = 0 # Should not happen if start worked

            recording_status["is_recording"] = False
            recording_status["is_paused"] = False
            recording_status["start_time"] = None
            recording_status["pause_start_time"] = None

            magic_audio_instance = None # Release the instance
            logger.info("Recording stopped and transcript saved.")
            return jsonify({"status": "success", "message": "Recording stopped and transcript saved."})
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            # Attempt to reset state even on error
            recording_status["is_recording"] = False
            recording_status["is_paused"] = False
            magic_audio_instance = None
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/pause', methods=['POST'])
def pause_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            logger.warning("Pause request ignored: Not recording.")
            return jsonify({"status": "error", "message": "Not recording"}), 400
        if recording_status["is_paused"]:
            logger.warning("Pause request ignored: Already paused.")
            return jsonify({"status": "success", "message": "Already paused"}), 200

        logger.info("Pausing recording...")
        try:
            magic_audio_instance.pause()
            recording_status["is_paused"] = True
            recording_status["pause_start_time"] = time.time()
             # Update elapsed time up to the point of pausing
            if recording_status["start_time"]:
                 recording_status["elapsed_time"] = recording_status["pause_start_time"] - recording_status["start_time"]
            logger.info("Recording paused.")
            return jsonify({"status": "success", "message": "Recording paused"})
        except Exception as e:
            logger.error(f"Error pausing recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/resume', methods=['POST'])
def resume_recording():
    global magic_audio_instance, recording_status
    with magic_audio_lock:
        if not recording_status["is_recording"] or not magic_audio_instance:
            logger.warning("Resume request ignored: Not recording.")
            return jsonify({"status": "error", "message": "Not recording"}), 400
        if not recording_status["is_paused"]:
            logger.warning("Resume request ignored: Not paused.")
            return jsonify({"status": "success", "message": "Not paused"}), 200

        logger.info("Resuming recording...")
        try:
            # Adjust start time based on pause duration before resuming
            if recording_status["pause_start_time"] and recording_status["start_time"]:
                pause_duration = time.time() - recording_status["pause_start_time"]
                recording_status["start_time"] += pause_duration # Effectively shift start time forward
                logger.debug(f"Resuming after {pause_duration:.2f}s pause. Adjusted start time.")
            else:
                logger.warning("Could not calculate pause duration accurately on resume.")

            magic_audio_instance.resume()
            recording_status["is_paused"] = False
            recording_status["pause_start_time"] = None
            logger.info("Recording resumed.")
            return jsonify({"status": "success", "message": "Recording resumed"})
        except Exception as e:
            logger.error(f"Error resuming recording: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/recording/status', methods=['GET'])
def get_recording_status():
    with magic_audio_lock:
        # Calculate current elapsed time if recording is active and not paused
        current_elapsed = recording_status["elapsed_time"]
        if recording_status["is_recording"] and not recording_status["is_paused"] and recording_status["start_time"]:
            current_elapsed = time.time() - recording_status["start_time"]

        status_data = {
            "is_recording": recording_status["is_recording"],
            "is_paused": recording_status["is_paused"],
            "elapsed_time": int(current_elapsed),
             "agent": recording_status.get("agent"),
             "event": recording_status.get("event"),
        }
    # logger.debug(f"Reporting recording status: {status_data}") # Can be verbose
    return jsonify(status_data)

# --- Main Execution (modified) ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)