import os
import logging
import uuid
import time
import threading
import subprocess
from typing import Dict, Any, Optional, Tuple, Callable
import wave
import numpy as np

import webrtcvad
from flask import current_app # To access the global executor

# Try to import audio processing libraries
try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - advanced audio filtering disabled")

# Configure logging
logger = logging.getLogger(__name__)

class VADTranscriptionService:
    """
    Main service class for VAD-filtered real-time transcription.
    Handles the "fast path" of the pipeline: WebM conversion, enhancement, and VAD.
    """
    
    def __init__(self, openai_api_key: str, vad_aggressiveness: int = 2):
        """
        Initialize the VAD transcription service.
        
        Args:
            openai_api_key: OpenAI API key for Whisper transcription
            vad_aggressiveness: VAD aggressiveness level (0-3, higher = more strict)
        """
        self.openai_api_key = openai_api_key
        self.vad_aggressiveness = vad_aggressiveness
        
        try:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness level {vad_aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}", exc_info=True)
            raise RuntimeError(f"VAD initialization failed: {e}")
        
        self.target_sample_rate = 16000
        self.target_channels = 1
        self.vad_frame_duration_ms = 30
        self.vad_frame_bytes = int(self.target_sample_rate * 2 * self.vad_frame_duration_ms / 1000)
        
        logger.info(f"VAD Service initialized - Sample Rate: {self.target_sample_rate}Hz, "
                   f"Frame Duration: {self.vad_frame_duration_ms}ms, "
                   f"Frame Size: {self.vad_frame_bytes} bytes")

    def _get_audio_duration(self, wav_path: str, session_id: str) -> float:
        """Get audio duration using ffprobe with fallback to wave module."""
        try:
            ffprobe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", wav_path]
            result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            logger.debug(f"Session {session_id}: ffprobe detected duration: {duration:.2f}s")
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Session {session_id}: ffprobe duration detection failed: {e}")
        
        try:
            with wave.open(wav_path, 'rb') as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception as e:
            logger.warning(f"Session {session_id}: wave module duration detection failed: {e}")
        
        return 0.0

    def process_webm_to_wav(self, webm_blob_bytes: bytes, output_wav_path: str, 
                                session_id: str) -> Tuple[bool, float]:
        """
        Convert WebM blob bytes to WAV using ffmpeg via stdin pipe.
        
        Args:
            webm_blob_bytes: The complete WebM data as bytes.
            output_wav_path: Path where WAV output should be saved.
            session_id: Session ID for logging context.
            
        Returns:
            Tuple of (success: bool, duration_seconds: float)
        """
        logger.debug(f"Session {session_id}: Starting WebM to WAV conversion via pipe")
        
        try:
            # ffmpeg command that reads from stdin
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", "pipe:0",  # Read from stdin
                "-ac", str(self.target_channels),
                "-ar", str(self.target_sample_rate),
                "-acodec", "pcm_s16le",
                "-f", "wav",
                output_wav_path
            ]
            
            logger.info(f"Session {session_id}: Executing ffmpeg via pipe: {' '.join(ffmpeg_cmd)}")
            
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=webm_blob_bytes)

            if process.returncode != 0:
                logger.error(f"Session {session_id}: ffmpeg pipe conversion failed. RC: {process.returncode}")
                logger.error(f"Session {session_id}: ffmpeg stderr: {stderr.decode('utf-8', 'ignore')}")
                return False, 0.0

            duration = self._get_audio_duration(output_wav_path, session_id)
            if duration > 0:
                logger.info(f"Session {session_id}: WAV file created successfully from pipe - Duration: {duration:.2f}s")
                return True, duration
            else:
                logger.warning(f"Session {session_id}: WAV file created from pipe but duration detection failed")
                return True, 0.0
                
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error during piped conversion: {e}", exc_info=True)
            return False, 0.0

    def detect_voice_activity(self, wav_path: str, session_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Perform VAD on a WAV file and extract voiced frames."""
        voiced_audio_data = bytearray()
        try:
            with wave.open(wav_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
            
            total_frames = len(audio_data) // self.vad_frame_bytes
            voiced_frames = 0
            for i in range(0, len(audio_data), self.vad_frame_bytes):
                frame = audio_data[i:i + self.vad_frame_bytes]
                if len(frame) < self.vad_frame_bytes: continue
                
                if self.vad.is_speech(frame, self.target_sample_rate):
                    voiced_frames += 1
                    voiced_audio_data.extend(frame)
            
            logger.info(f"Session {session_id}: VAD Analysis Complete. Total: {total_frames}, Voiced: {voiced_frames}")
            return bytes(voiced_audio_data), {}
            
        except Exception as e:
            logger.error(f"Session {session_id}: VAD analysis failed: {e}", exc_info=True)
            return b"", {"error": str(e)}

    def apply_audio_enhancement(self, wav_path: str, session_id: str) -> str:
        """Apply audio enhancement pipeline."""
        if not SCIPY_AVAILABLE:
            return wav_path

        try:
            sample_rate, audio_data = wavfile.read(wav_path)
            enhanced_data = audio_data
            
            # Placeholder for actual enhancement logic (e.g., filtering)
            enhanced_data, _ = self._apply_spectral_filter(enhanced_data, sample_rate, session_id)
            
            enhanced_path = wav_path.replace('.wav', '_enhanced.wav')
            wavfile.write(enhanced_path, sample_rate, enhanced_data.astype(np.int16))
            logger.info(f"Session {session_id}: Audio enhancement applied, saved to {enhanced_path}")
            return enhanced_path
        except Exception as e:
            logger.error(f"Session {session_id}: Audio enhancement failed: {e}", exc_info=True)
            return wav_path

    def _apply_spectral_filter(self, audio_data: np.ndarray, sample_rate: int, session_id: str) -> Tuple[np.ndarray, bool]:
        """Apply spectral filtering."""
        try:
            audio_float = audio_data.astype(np.float32)
            nyquist = sample_rate / 2
            
            # High-pass filter
            sos_hp = signal.butter(4, 80 / nyquist, btype='high', output='sos')
            audio_float = signal.sosfilt(sos_hp, audio_float)
            
            # Notch filter for 60Hz hum
            b_notch, a_notch = signal.iirnotch(60 / nyquist, Q=30)
            audio_float = signal.lfilter(b_notch, a_notch, audio_float)

            logger.debug(f"Session {session_id}: Spectral filtering applied.")
            return np.clip(audio_float, -32768, 32767).astype(np.int16), True
        except Exception as e:
            logger.warning(f"Session {session_id}: Spectral filtering failed: {e}")
            return audio_data, False

    def process_audio_segment(self, webm_blob_bytes: bytes, session_id: str, 
                             temp_dir: str, language_setting: str, session_data: Dict, session_lock: threading.Lock):
        """
        Process a single audio segment through the fast path and offload the slow path.
        This version receives a complete WebM blob and pipes it to ffmpeg.
        """
        segment_id = uuid.uuid4().hex
        processing_start_time = time.time()
        
        # Define paths for intermediate and final files
        wav_path = os.path.join(temp_dir, f"segment_{segment_id}.wav")
        clean_wav_path = os.path.join(temp_dir, f"clean_segment_{segment_id}.wav")
        
        files_to_cleanup = [wav_path]

        try:
            # Convert WebM bytes to WAV file via pipe
            conversion_success, _ = self.process_webm_to_wav(webm_blob_bytes, wav_path, session_id)
            if not conversion_success:
                raise RuntimeError("WebM to WAV conversion via pipe failed")
            
            enhanced_wav_path = self.apply_audio_enhancement(wav_path, session_id)
            if enhanced_wav_path != wav_path:
                files_to_cleanup.append(enhanced_wav_path)
            
            voiced_audio_bytes, _ = self.detect_voice_activity(enhanced_wav_path, session_id)
            
            if len(voiced_audio_bytes) < (self.vad_frame_bytes * 10):
                logger.info(f"Session {session_id}: Insufficient voice detected in segment. Skipping transcription.")
            else:
                with wave.open(clean_wav_path, 'wb') as wf:
                    wf.setnchannels(self.target_channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.target_sample_rate)
                    wf.writeframes(voiced_audio_bytes)
                
                logger.info(f"Session {session_id}: Submitting transcription for {clean_wav_path} to executor.")
                from transcription_service import process_audio_segment_and_update_s3
                current_app.executor.submit(
                    process_audio_segment_and_update_s3,
                    temp_segment_wav_path=clean_wav_path,
                    session_data=session_data,
                    session_lock=session_lock
                )
        
        except Exception as e:
            logger.error(f"Session {session_id}: Error in fast-path processing for segment {segment_id}: {e}", exc_info=True)
            if os.path.exists(clean_wav_path):
                files_to_cleanup.append(clean_wav_path)
        
        finally:
            # Clean up intermediate files (the clean_wav_path is cleaned up by the worker thread)
            for file_path in files_to_cleanup:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Session {session_id}: Failed to clean up {file_path}: {cleanup_error}")
            
            logger.debug(f"Session {session_id}: Fast-path processing for segment {segment_id} completed in {(time.time() - processing_start_time)*1000:.1f}ms.")

class VADTranscriptionManager:
    """
    Manages VAD service instance and session directories. Simplified to remove thread management.
    """
    
    def __init__(self, openai_api_key: str, base_temp_dir: str = "tmp_vad_audio_sessions",
                 vad_aggressiveness: int = 2):
        self.base_temp_dir = base_temp_dir
        self.vad_aggressiveness = vad_aggressiveness
        os.makedirs(self.base_temp_dir, exist_ok=True)
        
        self.vad_service = VADTranscriptionService(
            openai_api_key=openai_api_key,
            vad_aggressiveness=vad_aggressiveness
        )
        logger.info("VAD Transcription Manager initialized (Executor Model)")

    def get_session_temp_dir(self, session_id: str) -> str:
        """Returns the temporary directory path for a given session."""
        return os.path.join(self.base_temp_dir, session_id)

    def cleanup_session_directory(self, session_id: str):
        """Clean up the temporary directory for a specific session."""
        session_temp_dir = self.get_session_temp_dir(session_id)
        if os.path.exists(session_temp_dir):
            try:
                import shutil
                shutil.rmtree(session_temp_dir)
                logger.info(f"Cleaned up temp directory: {session_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {session_temp_dir}: {e}")