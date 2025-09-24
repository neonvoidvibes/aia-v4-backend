"""
Session cleanup with reference counting for atomic VAD writes.
"""
import os
import shutil
import time
import logging
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Reference counting for session cleanup
_PENDING: Dict[str, int] = defaultdict(int)
_LAST_ACTIVITY: Dict[str, float] = {}

def ack_transcription_enqueue(session_id: str):
    """Increment reference count when a transcription is enqueued."""
    _PENDING[session_id] += 1
    _LAST_ACTIVITY[session_id] = time.time()
    logger.debug(f"Session {session_id}: transcription enqueued, pending count: {_PENDING[session_id]}")

def release_transcription_ref(session_id: str):
    """Decrement reference count when a transcription is completed."""
    if _PENDING.get(session_id, 0) > 0:
        _PENDING[session_id] -= 1
    _LAST_ACTIVITY[session_id] = time.time()
    logger.debug(f"Session {session_id}: transcription completed, pending count: {_PENDING[session_id]}")

def maybe_cleanup_session(session_id: str, idle_grace_s: int = None):
    """Only delete when no pending work, and session idle."""
    if idle_grace_s is None:
        idle_grace_s = int(os.getenv('TRANSCRIBE_IDLE_GRACE_S', '60'))

    # Only delete when no pending work, and session idle
    if _PENDING.get(session_id, 0) == 0:
        last = _LAST_ACTIVITY.get(session_id, 0)
        if time.time() - last >= idle_grace_s:
            audio_tmp_dir = os.getenv('AUDIO_TMP_DIR', 'tmp_vad_audio_sessions')
            session_dir = f"{audio_tmp_dir}/{session_id}"
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
                logger.info(f"Session {session_id}: cleaned up session directory after {idle_grace_s}s grace period")
            _PENDING.pop(session_id, None)
            _LAST_ACTIVITY.pop(session_id, None)

def mark_session_terminated(session_id: str):
    """Call this on WS disconnect/auth fail; do NOT delete immediately."""
    _LAST_ACTIVITY[session_id] = time.time()
    logger.debug(f"Session {session_id}: marked as terminated")

def session_id_from_path(path: str) -> Optional[str]:
    """Extract session ID from a file path."""
    try:
        # Path format: tmp_vad_audio_sessions/{session_id}/clean_speech_*.wav
        parts = path.split(os.sep)
        if 'tmp_vad_audio_sessions' in parts:
            idx = parts.index('tmp_vad_audio_sessions')
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception as e:
        logger.error(f"Failed to extract session ID from path {path}: {e}")
    return None

def get_session_pending_count(session_id: str) -> int:
    """Get the pending transcription count for a session."""
    return _PENDING.get(session_id, 0)