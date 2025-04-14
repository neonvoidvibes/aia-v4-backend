# /aia-v4-backend/api_server.py
import os
import sys
import logging
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# --- Logging Setup ---
def setup_logging(debug=False):
    log_filename = 'api_server.log'
    root_logger = logging.getLogger()
    log_level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(log_level)

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File Handler
    try:
        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setLevel(log_level)
        ff = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        fh.setFormatter(ff)
        root_logger.addHandler(fh)
    except Exception as e:
        print(f"Error setting up file logger: {e}", file=sys.stderr)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level) # Use same level for console for now
    cf = logging.Formatter('[%(levelname)-8s] %(name)s: %(message)s') # More informative console format
    ch.setFormatter(cf)
    root_logger.addHandler(ch)

    # Silence overly verbose libraries
    for lib in ['anthropic', 'httpx', 'boto3', 'botocore', 'urllib3', 's3transfer', 'openai', 'sounddevice', 'requests']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.INFO) # Keep Pinecone info

    logging.info(f"Logging setup complete. Level: {logging.getLevelName(log_level)}")
    return root_logger # Return the logger instance

logger = setup_logging(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')

# --- Flask App Initialization ---
app = Flask(__name__)
# TODO: Add Secret Key from environment for production sessions
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    logger.info("Health check requested")
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

# --- Main Execution ---
if __name__ == '__main__':
    # Configuration loading (Simplified for now, will integrate config.py later)
    port = int(os.getenv('PORT', 5001)) # Render uses PORT env var
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting API server on port {port} (Debug: {debug_mode})")
    # Use host='0.0.0.0' to be accessible externally (required by Render)
    app.run(host='0.0.0.0', port=port, debug=debug_mode)