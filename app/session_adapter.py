from __future__ import annotations
import os
from typing import Optional, Dict
import threading
from app.pipeline_wiring import build_pipeline
from core.types import AudioBlob
from core.wav_norm import to_mono16k_pcm
from storage.log_store import WriteAheadLog

class SessionAdapter:
    def __init__(self, *, dg_client, whisper_client, s3_client, bucket: str, base_prefix: str) -> None:
        self.pipeline = build_pipeline(dg_client=dg_client, whisper_client=whisper_client,
                                       s3_client=s3_client, bucket=bucket, base_prefix=base_prefix)
        self._wal = WriteAheadLog(s3_client, bucket=bucket, base_prefix=base_prefix)
        self._seq: Dict[str, int] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def on_segment(self, *, session_id: str, raw_path: str, captured_ts: float,
                   duration_s: float, language: Optional[str]) -> None:
        with self._global_lock:
            lock = self._locks.setdefault(session_id, threading.Lock())
        with lock:
            # seq++
            seq = self._seq.get(session_id, -1) + 1
            self._seq[session_id] = seq
        # normalize to mono/16k wav
        wav_path = f"tmp/sessions/{session_id}/blobs/{seq:012d}.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        to_mono16k_pcm(raw_path, wav_path)
        # append-only process
        blob = AudioBlob(session_id=session_id, seq=seq, captured_ts=captured_ts,
                         wav_path=wav_path, duration_s=duration_s)
        _ = self.pipeline.process_blob(blob, language=language)

    def on_finalize(self, *, session_id: str) -> None:
        self._wal.publish_manifest(session_id=session_id)
        with self._global_lock:
            self._seq.pop(session_id, None)
            self._locks.pop(session_id, None)

