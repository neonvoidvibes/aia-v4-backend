"""
VAD-Filtered Real-Time Transcription Service

This module implements a robust VAD-filtered transcription pipeline.
It uses a stateful, frame-by-frame VAD analysis with buffering to ensure
only clean, continuous speech segments are sent to the transcription API,
which is the most effective method to prevent contextual hallucinations.
"""

import os
import logging
import uuid
import time
import threading
import subprocess
from typing import Dict, Any, Tuple, Optional
import wave
import numpy as np
from collections import deque

import webrtcvad
from flask import current_app

# Try to import audio processing libraries
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - audio enhancement will be skipped.")

logger = logging.getLogger(__name__)

class VADTranscriptionService:
    """
    Handles the "fast path" of the VAD pipeline: WebM conversion, frame-by-frame VAD,
    and submission of clean speech segments to the transcription worker pool.
    """
    
    def __init__(self, openai_api_key: str, vad_aggressiveness: int = 3): # Increased default aggressiveness
        self.openai_api_key = openai_api_key
        self.vad_aggressiveness = vad_aggressiveness
        
        try:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness level {vad_aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}", exc_info=True)
            raise RuntimeError(f"VAD initialization failed: {e}")
        
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_bytes = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2) # 16-bit audio
        
        logger.info(f"VAD Service initialized - Sample Rate: {self.sample_rate}Hz, Frame Duration: {self.frame_duration_ms}ms")

    def process_webm_to_wav(self, webm_blob_bytes: bytes) -> Optional[np.ndarray]:
        """Converts a WebM byte stream to a 16kHz mono NumPy array via ffmpeg pipe."""
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", "pipe:0",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-f", "s16le", # Output raw PCM S16LE
                "pipe:1" # Output to stdout
            ]
            
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=webm_blob_bytes)

            if process.returncode != 0:
                logger.error(f"ffmpeg pipe failed. RC: {process.returncode}, Stderr: {stderr.decode('utf-8', 'ignore')}")
                return None

            return np.frombuffer(stdout, dtype=np.int16)
        except Exception as e:
            logger.error(f"Error in process_webm_to_wav: {e}", exc_info=True)
            return None

    def segment_and_submit_audio(
        self,
        audio_data_np: np.ndarray,
        session_id: str,
        temp_dir: str,
        session_data: Dict,
        session_lock: threading.Lock
    ):
        """
        Analyzes audio frame-by-frame, buffers voiced segments, and submits them for transcription.
        """
        padding_duration_ms = 300  # Add 300ms padding before and after speech
        min_speech_duration_ms = 1000 # Minimum speech duration to transcribe
        silence_timeout_ms = 800 # Process buffer after this much silence

        padding_frames = padding_duration_ms // self.frame_duration_ms
        ring_buffer = deque(maxlen=padding_frames)
        
        voiced_frames = []
        triggered = False
        last_voiced_frame_time = time.time()
        
        frames = self._frame_generator(audio_data_np)

        for i, (frame, is_speech) in enumerate(frames):
            if is_speech:
                last_voiced_frame_time = time.time()

            if not triggered:
                ring_buffer.append((frame, is_speech))
                if is_speech:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer if s])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                
                # Check for silence timeout
                if not is_speech and (time.time() - last_voiced_frame_time) * 1000 > silence_timeout_ms:
                    # End of a speech segment due to silence
                    speech_segment = b''.join(voiced_frames)
                    if len(speech_segment) > (min_speech_duration_ms / 1000) * self.sample_rate * 2:
                        logger.info(f"Submitting segment of {len(speech_segment)} bytes due to silence timeout.")
                        self._submit_voiced_segment(speech_segment, session_id, temp_dir, session_data, session_lock)
                    
                    triggered = False
                    voiced_frames = []
                    ring_buffer.clear()
        
        # Process any remaining buffered audio at the end
        if voiced_frames:
            speech_segment = b''.join(voiced_frames)
            if len(speech_segment) > (min_speech_duration_ms / 1000) * self.sample_rate * 2:
                logger.info(f"Submitting final buffered segment of {len(speech_segment)} bytes.")
                self._submit_voiced_segment(speech_segment, session_id, temp_dir, session_data, session_lock)

    def _frame_generator(self, audio_data: np.ndarray):
        """Generator that yields audio frames and VAD decision."""
        audio_bytes = audio_data.tobytes()
        for i in range(0, len(audio_bytes), self.frame_bytes):
            frame = audio_bytes[i:i + self.frame_bytes]
            if len(frame) == self.frame_bytes:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                yield frame, is_speech
    
    def _submit_voiced_segment(
        self,
        audio_bytes: bytes,
        session_id: str,
        temp_dir: str,
        session_data: Dict,
        session_lock: threading.Lock
    ):
        """Saves a clean speech segment to a WAV file and submits it to the worker pool."""
        segment_id = uuid.uuid4().hex
        clean_wav_path = os.path.join(temp_dir, f"clean_speech_{segment_id}.wav")

        try:
            with wave.open(clean_wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_bytes)

            logger.info(f"Session {session_id}: Submitting clean WAV {clean_wav_path} to executor.")
            
            # Use a copy to avoid race conditions with the main session dict
            task_session_data = session_data.copy()
            
            from transcription_service import process_audio_segment_and_update_s3
            current_app.executor.submit(
                process_audio_segment_and_update_s3,
                temp_segment_wav_path=clean_wav_path,
                session_data=task_session_data,
                session_lock=session_lock
            )
        except Exception as e:
            logger.error(f"Session {session_id}: Failed to save and submit voiced segment: {e}", exc_info=True)
            if os.path.exists(clean_wav_path):
                try: os.remove(clean_wav_path)
                except Exception as del_e: logger.error(f"Failed to clean up failed segment file {clean_wav_path}: {del_e}")
    
    def process_audio_segment(
        self,
        webm_blob_bytes: bytes,
        session_id: str,
        temp_dir: str,
        language_setting: str,
        session_data: Dict,
        session_lock: threading.Lock
    ):
        """
        Process a single audio segment: convert, analyze, and submit.
        """
        processing_start_time = time.time()
        
        # Convert entire blob to raw PCM data
        audio_data_np = self.process_webm_to_wav(webm_blob_bytes)
        if audio_data_np is None:
            logger.error(f"Session {session_id}: Failed to convert WebM to WAV, skipping segment.")
            return

        # Let the session data know the duration of the full chunk for offset calculations
        with session_lock:
             duration_seconds = len(audio_data_np) / self.sample_rate
             session_data['actual_segment_duration_seconds'] = duration_seconds
             logger.debug(f"Session {session_id}: Setting segment duration to {duration_seconds:.2f}s")
        
        # Segment the audio based on voice activity and submit to workers
        self.segment_and_submit_audio(
            audio_data_np,
            session_id,
            temp_dir,
            session_data,
            session_lock
        )

        logger.debug(f"Session {session_id}: Full segment processing completed in {(time.time() - processing_start_time)*1000:.1f}ms.")

class VADTranscriptionManager:
    """
    Manages VAD service instance and session directories.
    """
    
    def __init__(self, openai_api_key: str, base_temp_dir: str = "tmp_vad_audio_sessions",
                 vad_aggressiveness: int = 3):
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