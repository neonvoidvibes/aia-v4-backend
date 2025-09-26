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
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import wave
import numpy as np
from collections import deque

import webrtcvad
from flask import current_app
from event_bus import emit as emit_event

# Try to import audio processing libraries
try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - audio enhancement will be skipped.")

logger = logging.getLogger(__name__)


def _emit_session_event(session_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    emit_event(session_id, event_type, payload or {})

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
        Analyzes audio frame-by-frame, buffers voiced segments based on silence detection,
        and submits clean speech segments for transcription. This version uses frame counting
        for silence detection, which is robust for processing pre-recorded chunks.
        """
        padding_duration_ms = 300
        min_speech_duration_ms = 1000
        # A segment is terminated after this duration of silence.
        silence_timeout_ms = 800

        # Convert millisecond settings to frame counts
        padding_frames = padding_duration_ms // self.frame_duration_ms
        min_speech_frames = min_speech_duration_ms // self.frame_duration_ms
        silence_timeout_frames = silence_timeout_ms // self.frame_duration_ms

        ring_buffer = deque(maxlen=padding_frames)
        
        voiced_frames = []
        silent_frames_count = 0
        triggered = False
        submitted_segment = False

        frames = self._frame_generator(audio_data_np)

        for frame, is_speech in frames:
            if is_speech:
                silent_frames_count = 0 # Reset silence counter on speech
            else:
                silent_frames_count += 1 # Increment silence counter

            if not triggered: # Waiting for speech to start
                ring_buffer.append((frame, is_speech))
                if is_speech:
                    triggered = True
                    # Add all frames from the ring buffer (including padding) to start the segment
                    voiced_frames.extend(f for f, s in ring_buffer)
                    ring_buffer.clear()
            else: # Actively collecting a speech segment
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))

                # Check if speech segment has ended due to silence
                if silent_frames_count > silence_timeout_frames:
                    # We have a complete speech segment ending in silence.
                    # The voiced_frames buffer contains speech + padding + trailing silence.
                    # Trim the trailing silence before submitting.
                    speech_segment_bytes = b''.join(voiced_frames[:-silent_frames_count])
                    
                    if len(speech_segment_bytes) >= (min_speech_duration_ms / 1000 * self.sample_rate * 2):
                        logger.info(f"Submitting voiced segment of {len(speech_segment_bytes)} bytes due to silence timeout.")
                        self._submit_voiced_segment(speech_segment_bytes, session_id, temp_dir, session_data, session_lock)
                        submitted_segment = True

                    # Reset for the next speech segment
                    triggered = False
                    voiced_frames = []
                    ring_buffer.clear()
                    silent_frames_count = 0
        
        # After the loop, process any remaining buffered audio. This handles cases
        # where the 15-second chunk ends mid-speech.
        if voiced_frames:
            speech_segment_bytes = b''.join(voiced_frames)
            # Only submit if it's a meaningful amount of audio
            if len(speech_segment_bytes) >= (min_speech_duration_ms / 1000 * self.sample_rate * 2):
                logger.info(f"Submitting final buffered audio segment of {len(speech_segment_bytes)} bytes at end of chunk.")
                self._submit_voiced_segment(speech_segment_bytes, session_id, temp_dir, session_data, session_lock)
                submitted_segment = True

        return submitted_segment

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
        from pathlib import Path
        import tempfile
        from session_cleanup import ack_transcription_enqueue

        segment_id = uuid.uuid4().hex
        final_path = Path(temp_dir) / f"clean_speech_{segment_id}.wav"

        pending_incremented = False

        try:
            # Atomic write: temp file, fsync, then replace
            with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.wav') as tf:
                with wave.open(tf.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_bytes)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_path = Path(tf.name)

            # Atomic replace
            os.replace(tmp_path, final_path)
            ack_transcription_enqueue(session_id)  # increment refcount

            logger.info(f"Session {session_id}: Submitting clean WAV {final_path} to executor.")

            # Retrieve keys from session data to pass to the processing function
            openai_key = session_data.get("openai_api_key")
            anthropic_key = session_data.get("anthropic_api_key")

            try:
                with session_lock:
                    session_data["pending_segments"] = int(session_data.get("pending_segments", 0)) + 1
                    pending_incremented = True
            except Exception as pending_err:
                logger.debug(f"Session {session_id}: Failed to increment pending segments: {pending_err}")

            from transcription_service import process_audio_segment_and_update_s3
            current_app.executor.submit(
                process_audio_segment_and_update_s3,
                temp_segment_wav_path=str(final_path),
                session_data=session_data,
                session_lock=session_lock,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key
            )
        except Exception as e:
            logger.error(f"Session {session_id}: Failed to save and submit voiced segment: {e}", exc_info=True)
            if pending_incremented:
                try:
                    with session_lock:
                        session_data["pending_segments"] = max(0, int(session_data.get("pending_segments", 1)) - 1)
                except Exception as pending_err:
                    logger.debug(f"Session {session_id}: Failed to decrement pending after submit error: {pending_err}")
            if os.path.exists(final_path):
                try: os.remove(final_path)
                except Exception as del_e: logger.error(f"Failed to clean up failed segment file {final_path}: {del_e}")
    
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
        silence_seconds = None

        # Convert entire blob to raw PCM data
        audio_data_np = self.process_webm_to_wav(webm_blob_bytes)
        if audio_data_np is None:
            logger.error(f"Session {session_id}: Failed to convert WebM to WAV, skipping segment.")
            return

        chunk_duration = 0.0
        try:
            chunk_duration = max(0.0, len(audio_data_np) / float(self.sample_rate))
        except Exception:
            chunk_duration = 0.0

        # The duration of the full webm blob is no longer used to increment the total session duration.
        # Instead, the precise duration of each voiced WAV segment will be calculated in the
        # transcription_service and used to update the total duration there, ensuring timestamp accuracy.
        
        # Segment the audio based on voice activity and submit to workers
        submitted_segment = self.segment_and_submit_audio(
            audio_data_np,
            session_id,
            temp_dir,
            session_data,
            session_lock
        )

        now_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        with session_lock:
            session_data["last_audio_ts"] = now_iso
            if submitted_segment:
                session_data["silence_seconds"] = 0
            else:
                increment = max(1, int(round(chunk_duration))) if chunk_duration > 0 else 1
                session_data["silence_seconds"] = int(session_data.get("silence_seconds", 0)) + increment
                silence_seconds = int(session_data["silence_seconds"])

        if not submitted_segment and silence_seconds is not None:
            _emit_session_event(session_id, "silence_tick", {"seconds": silence_seconds})

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
