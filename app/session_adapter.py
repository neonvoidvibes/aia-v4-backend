from __future__ import annotations
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Dict

from app.pipeline_wiring import build_pipeline
from core.types import AudioBlob, TranscriptChunk
from core.wav_norm import to_mono16k_pcm
from storage.log_store import WriteAheadLog
from utils.transcript_format import format_transcript_line


class SessionAdapter:
    """Thread-safe adapter that normalizes blobs, feeds the pipeline, and keeps legacy text files in sync."""

    def __init__(self, *, dg_client, whisper_client, s3_client, bucket: str, base_prefix: str) -> None:
        self.pipeline = build_pipeline(dg_client=dg_client, whisper_client=whisper_client,
                                       s3_client=s3_client, bucket=bucket, base_prefix=base_prefix)
        self._wal = WriteAheadLog(s3_client, bucket=bucket, base_prefix=base_prefix)
        self._s3 = s3_client
        self._bucket = bucket
        self._seq: Dict[str, int] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._legacy_keys: Dict[str, str] = {}
        self._global_lock = threading.Lock()

    def register_session(self, session_id: str, transcript_key: str) -> None:
        with self._global_lock:
            self._legacy_keys[session_id] = transcript_key

    def on_segment(self, *, session_id: str, raw_path: str, captured_ts: float,
                   duration_s: float, language: Optional[str]) -> Optional[TranscriptChunk]:
        with self._global_lock:
            lock = self._locks.setdefault(session_id, threading.Lock())
        with lock:
            seq = self._seq.get(session_id, -1) + 1
            self._seq[session_id] = seq
        wav_path = f"tmp/sessions/{session_id}/blobs/{seq:012d}.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        to_mono16k_pcm(raw_path, wav_path)

        blob = AudioBlob(session_id=session_id, seq=seq, captured_ts=captured_ts,
                         wav_path=wav_path, duration_s=duration_s)
        chunk = self.pipeline.process_blob(blob, language=language)
        if chunk:
            with lock:
                self._append_legacy_transcript(session_id, chunk)
        return chunk

    def on_finalize(self, *, session_id: str) -> None:
        self._wal.publish_manifest(session_id=session_id)
        with self._global_lock:
            self._seq.pop(session_id, None)
            self._locks.pop(session_id, None)
            self._legacy_keys.pop(session_id, None)

    def _append_legacy_transcript(self, session_id: str, chunk: TranscriptChunk) -> None:
        key = self._legacy_keys.get(session_id)
        if not key:
            return
        timestamp = datetime.fromtimestamp(chunk.captured_ts, tz=timezone.utc).strftime("[%H:%M:%S]")
        line = format_transcript_line(timestamp, chunk.text, chunk.meta)
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            existing = obj["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
            existing = ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        updated = f"{existing}{line}\n"
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=updated.encode("utf-8"))
