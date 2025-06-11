import os
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from collections import defaultdict

# Defer imports to fix circular dependency
# from vad_transcription_service import VADTranscriptionManager, SessionAudioProcessor

# Configure logging
logger = logging.getLogger(__name__)

class VADIntegrationBridge:
    """
    Bridge class that integrates VAD transcription service with the existing API server.
    Manages the lifecycle of VAD sessions and coordinates with existing session data.
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
        vad_aggressiveness = int(os.getenv('VAD_AGGRESSIVENESS', '2'))  # Default to level 2
        
        try:
            self.vad_manager = VADTranscriptionManager(
                openai_api_key=openai_api_key,
                base_temp_dir=base_temp_dir,
                vad_aggressiveness=vad_aggressiveness
            )
            logger.info(f"VAD Integration Bridge initialized with aggressiveness {vad_aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize VAD manager: {e}")
            raise RuntimeError(f"VAD Integration Bridge initialization failed: {e}")
        
        # Integration state
        self.session_processors: Dict[str, "SessionAudioProcessor"] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self.bridge_lock = threading.RLock()
        
        # Statistics
        self.global_stats = {
            "sessions_created": 0,
            "sessions_destroyed": 0,
            "total_audio_chunks_processed": 0,
            "total_chunks_with_voice": 0,
            "total_chunks_transcribed": 0,
            "bridge_start_time": time.time()
        }
        
        logger.info("VAD Integration Bridge ready for session management")

    def _handle_transcription_result(self, result: Dict[str, Any]):
        """
        Callback function to handle transcription results from the VAD pipeline.
        This function is called by the SessionAudioProcessor.
        """
        session_id = result.get("session_id")
        if not session_id:
            logger.error("VAD Bridge: Received transcription result without a session_id.")
            return

        logger.info(f"VAD Bridge: Handling transcription result for session {session_id}")

        try:
            from transcription_service import process_audio_segment_and_update_s3
            
            with self.bridge_lock:
                if session_id not in self.session_metadata:
                    logger.error(f"VAD Bridge: Metadata not found for session {session_id}. Cannot process transcription.")
                    return
                
                main_session_data = self.session_metadata[session_id].get("existing_session_data")
                main_session_lock = self.session_metadata[session_id].get("main_session_lock")  # Get from stored metadata
                clean_wav_path = result.get("clean_wav_path")

                if not all([main_session_data, main_session_lock, clean_wav_path]):
                    logger.error(f"VAD Bridge: Missing critical data for S3 update. Data: {main_session_data is not None}, Lock: {main_session_lock is not None}, Path: {clean_wav_path}")
                    return

            # Call the unified processing function. It's thread-safe via the lock.
            process_audio_segment_and_update_s3(
                temp_segment_wav_path=clean_wav_path,
                session_data=main_session_data,
                session_lock=main_session_lock
            )
            logger.info(f"VAD Bridge: Dispatched result for session {session_id} to be saved to S3.")
            
            # Clean up the clean WAV file after processing
            try:
                if os.path.exists(clean_wav_path):
                    os.remove(clean_wav_path)
                    logger.debug(f"VAD Bridge: Cleaned up clean WAV file: {clean_wav_path}")
            except Exception as cleanup_e:
                logger.warning(f"VAD Bridge: Failed to clean up clean WAV file {clean_wav_path}: {cleanup_e}")

        except ImportError:
            logger.error(f"VAD Bridge: Failed to import process_audio_segment_and_update_s3.")
        except Exception as e:
            logger.error(f"VAD Bridge: Unexpected error handling transcription result: {e}", exc_info=True)


    def create_vad_session(self, session_id: str, existing_session_data: Dict[str, Any], 
                          main_session_lock: threading.RLock) -> bool:
        """
        Create a new VAD session integrated with existing session data.
        
        Args:
            session_id: Session identifier from existing system
            existing_session_data: Session data from active_sessions
            main_session_lock: Session lock from the main system
            
        Returns:
            True if session created successfully, False otherwise
        """
        logger.info(f"Creating VAD session for {session_id}")
        
        with self.bridge_lock:
            if session_id in self.session_processors:
                logger.warning(f"VAD session {session_id} already exists")
                return True
            
            try:
                # Extract configuration from existing session data
                language_setting = existing_session_data.get('language_setting_from_client', 'any')
                segment_duration = float(os.getenv('VAD_SEGMENT_DURATION', '15.0'))
                
                # Create VAD session processor
                processor = self.vad_manager.create_session(
                    session_id=session_id,
                    language_setting=language_setting,
                    segment_duration_target=segment_duration,
                    result_callback=self._handle_transcription_result # Pass the callback
                )
                
                # Store session references
                self.session_processors[session_id] = processor
                self.session_metadata[session_id] = {
                    "created_at": time.time(),
                    "language_setting": language_setting,
                    "segment_duration": segment_duration,
                    "existing_session_data": existing_session_data,
                    "main_session_lock": main_session_lock,  # Store the lock from main system
                    "is_active": True
                }
                
                # Start the processor
                processor.start()
                
                # Update global stats
                self.global_stats["sessions_created"] += 1
                
                logger.info(f"VAD session {session_id} created and started successfully")
                logger.info(f"Session config - Language: {language_setting}, Segment Duration: {segment_duration}s")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to create VAD session {session_id}: {e}", exc_info=True)
                return False

    def destroy_vad_session(self, session_id: str) -> bool:
        """
        Destroy a VAD session and clean up resources.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session destroyed successfully, False otherwise
        """
        logger.info(f"Destroying VAD session {session_id}")
        
        with self.bridge_lock:
            if session_id not in self.session_processors:
                logger.warning(f"VAD session {session_id} not found for destruction")
                return False
            
            try:
                # Get final statistics before destroying
                processor = self.session_processors[session_id]
                final_stats = processor.get_statistics()
                
                # Log session summary
                self._log_session_summary(session_id, final_stats)
                
                # Update global statistics
                self._update_global_stats_from_session(final_stats)
                
                # Destroy the session
                self.vad_manager.destroy_session(session_id)
                
                # Clean up local references
                del self.session_processors[session_id]
                del self.session_metadata[session_id]
                if session_id in self.session_locks:
                    del self.session_locks[session_id]
                
                # Update global stats
                self.global_stats["sessions_destroyed"] += 1
                
                logger.info(f"VAD session {session_id} destroyed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to destroy VAD session {session_id}: {e}", exc_info=True)
                return False

    def process_audio_blob(self, session_id: str, webm_blob_bytes: bytes) -> bool:
        """
        Process an audio blob through the VAD pipeline.
        
        Args:
            session_id: Session identifier
            webm_blob_bytes: Raw WebM audio data
            
        Returns:
            True if processing initiated successfully, False otherwise
        """
        logger.debug(f"Processing audio blob for session {session_id} ({len(webm_blob_bytes)} bytes)")
        
        with self.bridge_lock:
            if session_id not in self.session_processors:
                logger.error(f"VAD session {session_id} not found for audio processing")
                return False
            
            processor = self.session_processors[session_id]
            
            if not self.session_metadata[session_id].get("is_active", False):
                logger.warning(f"VAD session {session_id} is not active, skipping audio blob")
                return False
        
        try:
            # Process through VAD pipeline
            processor.add_audio_blob(webm_blob_bytes)
            
            logger.debug(f"Audio blob queued for VAD processing in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process audio blob for session {session_id}: {e}", exc_info=True)
            return False

    def pause_session(self, session_id: str) -> bool:
        """
        Pause audio processing for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if paused successfully, False otherwise
        """
        logger.info(f"Pausing VAD session {session_id}")
        
        with self.bridge_lock:
            if session_id not in self.session_metadata:
                logger.warning(f"VAD session {session_id} not found for pausing")
                return False
            
            self.session_metadata[session_id]["is_active"] = False
            logger.info(f"VAD session {session_id} paused")
            return True

    def resume_session(self, session_id: str) -> bool:
        """
        Resume audio processing for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if resumed successfully, False otherwise
        """
        logger.info(f"Resuming VAD session {session_id}")
        
        with self.bridge_lock:
            if session_id not in self.session_metadata:
                logger.warning(f"VAD session {session_id} not found for resuming")
                return False
            
            self.session_metadata[session_id]["is_active"] = True
            logger.info(f"VAD session {session_id} resumed")
            return True

    def get_session_statistics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics dictionary or None if session not found
        """
        with self.bridge_lock:
            if session_id not in self.session_processors:
                return None
            
            processor = self.session_processors[session_id]
            stats = processor.get_statistics()
            
            # Add bridge-specific metadata
            metadata = self.session_metadata.get(session_id, {})
            stats.update({
                "bridge_metadata": {
                    "created_at": metadata.get("created_at"),
                    "language_setting": metadata.get("language_setting"),
                    "segment_duration": metadata.get("segment_duration"),
                    "is_active": metadata.get("is_active")
                }
            })
            
            return stats

    def get_all_session_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active VAD sessions."""
        with self.bridge_lock:
            all_stats = {}
            for session_id in self.session_processors:
                stats = self.get_session_statistics(session_id)
                if stats:
                    all_stats[session_id] = stats
            return all_stats

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global bridge statistics."""
        with self.bridge_lock:
            runtime = time.time() - self.global_stats["bridge_start_time"]
            
            return {
                **self.global_stats,
                "runtime_seconds": runtime,
                "active_sessions": len(self.session_processors),
                "vad_manager_stats": {
                    "base_temp_dir": self.vad_manager.base_temp_dir,
                    "vad_aggressiveness": self.vad_manager.vad_aggressiveness
                }
            }

    def is_session_active(self, session_id: str) -> bool:
        """Check if a VAD session is active."""
        with self.bridge_lock:
            return (session_id in self.session_processors and 
                   self.session_metadata.get(session_id, {}).get("is_active", False))

    def cleanup_all_sessions(self):
        """Clean up all VAD sessions."""
        logger.info("Cleaning up all VAD sessions")
        
        with self.bridge_lock:
            session_ids = list(self.session_processors.keys())
        
        for session_id in session_ids:
            self.destroy_vad_session(session_id)
        
        logger.info(f"All {len(session_ids)} VAD sessions cleaned up")

    def _log_session_summary(self, session_id: str, final_stats: Dict[str, Any]):
        """Log a comprehensive session summary."""
        logger.info(f"=== VAD SESSION SUMMARY: {session_id} ===")
        
        metadata = self.session_metadata.get(session_id, {})
        created_at = metadata.get("created_at", 0)
        session_duration = time.time() - created_at
        
        logger.info(f"Session Duration: {session_duration:.1f}s")
        logger.info(f"Language Setting: {metadata.get('language_setting', 'unknown')}")
        logger.info(f"Segment Duration Target: {metadata.get('segment_duration', 'unknown')}s")
        
        logger.info(f"Audio Processing Statistics:")
        logger.info(f"  Chunks Received: {final_stats.get('chunks_received', 0)}")
        logger.info(f"  Chunks Processed: {final_stats.get('chunks_processed', 0)}")
        logger.info(f"  Chunks with Voice: {final_stats.get('chunks_with_voice', 0)}")
        logger.info(f"  Chunks Transcribed: {final_stats.get('chunks_transcribed', 0)}")
        logger.info(f"  Total Processing Time: {final_stats.get('total_processing_time_ms', 0):.1f}ms")
        logger.info(f"  Errors: {len(final_stats.get('errors', []))}")
        
        # Calculate efficiency metrics
        if final_stats.get('chunks_processed', 0) > 0:
            voice_detection_rate = (final_stats.get('chunks_with_voice', 0) / 
                                   final_stats.get('chunks_processed', 1)) * 100
            transcription_success_rate = (final_stats.get('chunks_transcribed', 0) / 
                                        final_stats.get('chunks_with_voice', 1)) * 100
            
            logger.info(f"Efficiency Metrics:")
            logger.info(f"  Voice Detection Rate: {voice_detection_rate:.1f}%")
            logger.info(f"  Transcription Success Rate: {transcription_success_rate:.1f}%")
        
        logger.info(f"=== END SESSION SUMMARY: {session_id} ===")

    def _update_global_stats_from_session(self, session_stats: Dict[str, Any]):
        """Update global statistics with session data."""
        self.global_stats["total_audio_chunks_processed"] += session_stats.get("chunks_processed", 0)
        self.global_stats["total_chunks_with_voice"] += session_stats.get("chunks_with_voice", 0)
        self.global_stats["total_chunks_transcribed"] += session_stats.get("chunks_transcribed", 0)


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
    # This check is now just for logging/info, the server will attempt to use it if initialized.
    return os.getenv('ENABLE_VAD_TRANSCRIPTION', 'true').lower() == 'true' and _vad_bridge_instance is not None

def log_vad_configuration():
    """Log current VAD configuration for debugging."""
    logger.info("=== VAD CONFIGURATION ===")
    logger.info(f"VAD Enabled Flag: {os.getenv('ENABLE_VAD_TRANSCRIPTION', 'true')}")
    logger.info(f"VAD Aggressiveness: {os.getenv('VAD_AGGRESSIVENESS', '2')}")
    logger.info(f"VAD Segment Duration: {os.getenv('VAD_SEGMENT_DURATION', '15.0')}s")
    logger.info(f"VAD Temp Directory: {os.getenv('VAD_TEMP_DIR', 'tmp_vad_audio_sessions')}")
    logger.info(f"Bridge Instance: {'Initialized' if _vad_bridge_instance else 'Not Initialized'}")
    logger.info("=========================")
