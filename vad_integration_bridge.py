import os
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
from flask import current_app # Import to access the global executor

# Defer imports to fix circular dependency
# from vad_transcription_service import VADTranscriptionManager

# Import hallucination detection cleanup
from utils.hallucination_detector import cleanup_session_manager

# Configure logging
logger = logging.getLogger(__name__)

class VADIntegrationBridge:
    """
    Bridge class that integrates VAD transcription service with the existing API server.
    Manages the lifecycle of VAD sessions and coordinates with existing session data.
    This version is refactored to remove internal threading and use a global executor.
    """
    
    def __init__(self, openai_api_key: str, base_temp_dir: str = "tmp_vad_audio_sessions"):
        """
        Initialize the VAD integration bridge.
        
        Args:
            openai_api_key: OpenAI API key for Whisper transcription
            base_temp_dir: Base directory for VAD session files
        """
        from vad_transcription_service import VADTranscriptionManager # Import inside method

        self.openai_api_key = openai_api_key
        self.base_temp_dir = base_temp_dir
        
        # Initialize VAD transcription manager
        # Mode 3 is the most aggressive and recommended for filtering out non-speech to prevent hallucinations.
        # Use provider-specific defaults: Deepgram=1 (Quiet), Whisper=2 (Mid)
        from transcription_service import get_default_vad_aggressiveness
        vad_aggressiveness = int(os.getenv('VAD_AGGRESSIVENESS', str(get_default_vad_aggressiveness())))
        
        try:
            self.vad_manager = VADTranscriptionManager(
                openai_api_key=openai_api_key,
                base_temp_dir=base_temp_dir,
                vad_aggressiveness=vad_aggressiveness
            )
            logger.info(f"VAD Integration Bridge initialized with aggressiveness {vad_aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize VAD manager: {e}", exc_info=True)
            raise RuntimeError(f"VAD Integration Bridge initialization failed: {e}")
        
        # Integration state
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.bridge_lock = threading.RLock()
        
        # Statistics
        self.global_stats = {
            "sessions_created": 0,
            "sessions_destroyed": 0,
            "audio_blobs_processed": 0,
            "bridge_start_time": time.time()
        }
        
        logger.info("VAD Integration Bridge (Executor Model) ready for session management")

    def create_vad_session(self, session_id: str, existing_session_data: Dict[str, Any], 
                          main_session_lock: threading.RLock) -> bool:
        """
        Create (register) a new VAD session. This no longer starts threads.
        
        Args:
            session_id: Session identifier from existing system
            existing_session_data: Session data from active_sessions
            main_session_lock: Session lock from the main system
            
        Returns:
            True if session registered successfully, False otherwise
        """
        logger.info(f"Registering VAD session for {session_id}")
        
        with self.bridge_lock:
            if session_id in self.session_metadata:
                logger.warning(f"VAD session {session_id} already registered")
                return True
            
            try:
                # Create the session temporary directory
                session_temp_dir = self.vad_manager.get_session_temp_dir(session_id)
                os.makedirs(session_temp_dir, exist_ok=True)
                
                # Store session metadata
                self.session_metadata[session_id] = {
                    "created_at": time.time(),
                    "existing_session_data": existing_session_data,
                    "main_session_lock": main_session_lock,
                    "temp_dir": session_temp_dir,
                    "is_active": True
                }
                
                self.global_stats["sessions_created"] += 1
                logger.info(f"VAD session {session_id} registered successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register VAD session {session_id}: {e}", exc_info=True)
                return False

    def destroy_vad_session(self, session_id: str) -> bool:
        """
        Destroy a VAD session and clean up resources.

        Args:
            session_id: Session identifier

        Returns:
            True if session destroyed successfully, False otherwise
        """
        from session_cleanup import mark_session_terminated, maybe_cleanup_session

        logger.info(f"Destroying VAD session {session_id}")

        with self.bridge_lock:
            if session_id not in self.session_metadata:
                logger.warning(f"VAD session {session_id} not found for destruction")
                return False

            try:
                # Clean up hallucination manager for the session
                cleanup_session_manager(session_id)

                # Mark session as terminated, but don't immediately clean up files
                # This allows pending transcriptions to complete
                mark_session_terminated(session_id)

                # Try cleanup - this will only succeed if no pending transcriptions
                maybe_cleanup_session(session_id)

                try:
                    from pathlib import Path
                    retry_dir = Path("tmp/retry_segments") / session_id
                    if retry_dir.exists():
                        import shutil
                        shutil.rmtree(retry_dir)
                        logger.info(f"Session {session_id}: Cleaned retry queue artifacts during VAD teardown")
                except Exception as cleanup_err:
                    logger.warning(f"Session {session_id}: Failed to clean retry queue artifacts: {cleanup_err}")

                # Clean up local references
                del self.session_metadata[session_id]

                self.global_stats["sessions_destroyed"] += 1
                logger.info(f"VAD session {session_id} destroyed successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to destroy VAD session {session_id}: {e}", exc_info=True)
                return False

    def process_audio_blob(self, session_id: str, webm_blob_bytes: bytes) -> bool:
        """
        Process an audio blob through the VAD pipeline's fast path and submit
        the slow path (transcription) to the global executor.
        
        Args:
            session_id: Session identifier
            webm_blob_bytes: Raw WebM audio data
            
        Returns:
            True if processing initiated successfully, False otherwise
        """
        logger.debug(f"Processing audio blob for session {session_id} ({len(webm_blob_bytes)} bytes)")
        
        with self.bridge_lock:
            if session_id not in self.session_metadata:
                logger.error(f"VAD session {session_id} not found for audio processing")
                return False
            
            session_meta = self.session_metadata[session_id]
            if not session_meta.get("is_active", False):
                logger.warning(f"VAD session {session_id} is not active, skipping audio blob")
                return False
        
        try:
            # This is a direct, synchronous call to the VAD service's fast path.
            # It will perform VAD and submit the transcription task to the app's executor.
            self.vad_manager.vad_service.process_audio_segment(
                webm_blob_bytes=webm_blob_bytes,
                session_id=session_id,
                temp_dir=session_meta["temp_dir"],
                language_setting=session_meta["existing_session_data"].get('language_setting_from_client', 'any'),
                session_data=session_meta["existing_session_data"],
                session_lock=session_meta["main_session_lock"]
            )
            self.global_stats["audio_blobs_processed"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to process audio blob for session {session_id}: {e}", exc_info=True)
            return False

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global bridge statistics."""
        with self.bridge_lock:
            runtime = time.time() - self.global_stats["bridge_start_time"]
            
            return {
                **self.global_stats,
                "runtime_seconds": runtime,
                "active_sessions": len(self.session_metadata),
                "vad_manager_stats": {
                    "base_temp_dir": self.vad_manager.base_temp_dir,
                    "vad_aggressiveness": self.vad_manager.vad_aggressiveness
                }
            }

    def cleanup_all_sessions(self):
        """Clean up all VAD sessions."""
        logger.info("Cleaning up all VAD sessions")
        
        with self.bridge_lock:
            session_ids = list(self.session_metadata.keys())
        
        for session_id in session_ids:
            self.destroy_vad_session(session_id)
        
        logger.info(f"All {len(session_ids)} VAD sessions cleaned up")


# Global VAD bridge instance
_vad_bridge_instance: Optional[VADIntegrationBridge] = None
_bridge_lock = threading.Lock()

def initialize_vad_bridge(openai_api_key: str) -> VADIntegrationBridge:
    """
    Initialize the global VAD integration bridge.
    
    Args:
        openai_api_key: OpenAI API key for Whisper transcription
        
    Returns:
        VADIntegrationBridge instance
    """
    global _vad_bridge_instance
    
    with _bridge_lock:
        if _vad_bridge_instance is not None:
            logger.warning("VAD bridge already initialized, returning existing instance")
            return _vad_bridge_instance
        
        try:
            base_temp_dir = os.getenv('VAD_TEMP_DIR', 'tmp_vad_audio_sessions')
            _vad_bridge_instance = VADIntegrationBridge(
                openai_api_key=openai_api_key,
                base_temp_dir=base_temp_dir
            )
            
            logger.info("Global VAD integration bridge initialized successfully")
            return _vad_bridge_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize global VAD bridge: {e}", exc_info=True)
            raise

def get_vad_bridge() -> Optional[VADIntegrationBridge]:
    """Get the global VAD integration bridge instance."""
    return _vad_bridge_instance

def cleanup_vad_bridge():
    """Clean up the global VAD integration bridge."""
    global _vad_bridge_instance
    
    with _bridge_lock:
        if _vad_bridge_instance is not None:
            try:
                _vad_bridge_instance.cleanup_all_sessions()
                _vad_bridge_instance = None
                logger.info("Global VAD bridge cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up VAD bridge: {e}", exc_info=True)
        else:
            logger.info("No VAD bridge to clean up")

def is_vad_enabled() -> bool:
    """Check if VAD integration is enabled and available."""
    return os.getenv('ENABLE_VAD_TRANSCRIPTION', 'true').lower() == 'true' and _vad_bridge_instance is not None

def log_vad_configuration():
    """Log current VAD configuration for debugging."""
    from transcription_service import get_default_vad_aggressiveness
    default_vad = get_default_vad_aggressiveness()
    logger.info("=== VAD CONFIGURATION ===")
    logger.info(f"VAD Enabled Flag: {os.getenv('ENABLE_VAD_TRANSCRIPTION', 'true')}")
    logger.info(f"VAD Aggressiveness: {os.getenv('VAD_AGGRESSIVENESS', str(default_vad))} (provider default: {default_vad})")
    logger.info(f"VAD Segment Duration: {os.getenv('VAD_SEGMENT_DURATION', '15.0')}s")
    logger.info(f"VAD Temp Directory: {os.getenv('VAD_TEMP_DIR', 'tmp_vad_audio_sessions')}")
    logger.info(f"Bridge Instance: {'Initialized' if _vad_bridge_instance else 'Not Initialized'}")
    logger.info("=========================")
