from __future__ import annotations
import os
import re
import shutil
import threading
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List, Tuple

from app.pipeline_wiring import build_pipeline
from core.types import AudioBlob, TranscriptChunk
from core.wav_norm import to_mono16k_pcm
from storage.log_store import WriteAheadLog
from utils.transcript_format import format_transcript_line
from utils.silence_gate import evaluate_silence, config_from_env, SilenceGateResult

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\b\w+\b")

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
        self._legacy_prefix_counts: Dict[str, Dict[str, int]] = {}
        self._legacy_timezones: Dict[str, str] = {}
        self._global_lock = threading.Lock()
        self._silence_gate_enabled = os.getenv('ENABLE_SILENCE_GATE', 'true').lower() not in {'0', 'false', 'no'}
        self._trim_margin_frames = max(0, int(os.getenv('SILENCE_GATE_TRIM_MARGIN_FRAMES', '2')))
        if self._silence_gate_enabled:
            default_cfg = config_from_env()
            logger.info(
                "SessionAdapter: silence gate enabled (default min_ratio=%.3f, rms_floor=%.1f, confirm=%s, trim_margin_frames=%s)",
                default_cfg.min_speech_ratio,
                default_cfg.rms_floor,
                default_cfg.confirm_silence_windows,
                self._trim_margin_frames,
            )
        else:
            logger.info("SessionAdapter: silence gate disabled via ENABLE_SILENCE_GATE")

    @staticmethod
    def _resolve_aggressiveness(value: Optional[int]) -> int:
        try:
            return max(0, min(3, int(value)))
        except (TypeError, ValueError):
            return 2

    def silence_gate_enabled(self) -> bool:
        return self._silence_gate_enabled

    def gate_config_for(self, aggressiveness: Optional[int]) -> SilenceGateConfig:
        aggr = self._resolve_aggressiveness(aggressiveness)
        return config_from_env(aggr)

    def trim_margin_frames(self) -> int:
        return self._trim_margin_frames

    def _trim_wav_to_voiced(
        self,
        *,
        session_id: str,
        seq: int,
        wav_path: str,
        result: SilenceGateResult,
    ) -> None:
        start_frame = result.voiced_start_frame
        end_frame = result.voiced_end_frame
        if start_frame is None or end_frame is None:
            return
        if result.frame_bytes <= 0 or result.frame_count <= 0:
            return

        margin = self._trim_margin_frames
        expanded_start = max(0, start_frame - margin)
        expanded_end = min(result.frame_count - 1, end_frame + margin)
        if expanded_start == 0 and expanded_end >= result.frame_count - 1:
            return  # Nothing to trim

        start_byte = expanded_start * result.frame_bytes
        end_byte = min(result.frame_count * result.frame_bytes, (expanded_end + 1) * result.frame_bytes)
        if end_byte <= start_byte:
            return

        try:
            import wave
            with wave.open(wav_path, 'rb') as wf:
                params = wf.getparams()
                wf.rewind()
                data = wf.readframes(wf.getnframes())
            trimmed = data[start_byte:end_byte]
            if len(trimmed) == len(data) or not trimmed:
                return
            with wave.open(wav_path, 'wb') as wf_out:
                wf_out.setparams(params)
                wf_out.writeframes(trimmed)
            logger.info(
                "Session %s: trimmed wav seq=%s start_frame=%s end_frame=%s margin=%s original_bytes=%s trimmed_bytes=%s",
                session_id,
                seq,
                expanded_start,
                expanded_end,
                margin,
                len(data),
                len(trimmed),
            )
        except Exception as exc:
            logger.warning(
                "Session %s: failed to trim wav seq=%s (%s)",
                session_id,
                seq,
                exc,
            )

    def register_session(self, session_id: str, transcript_key: str, timezone_name: Optional[str] = None) -> None:
        with self._global_lock:
            self._legacy_keys[session_id] = transcript_key
            self._legacy_prefix_counts.pop(session_id, None)
            if timezone_name:
                self._legacy_timezones[session_id] = timezone_name
            else:
                self._legacy_timezones.pop(session_id, None)
            logger.debug("Session %s: registered legacy transcript %s", session_id, transcript_key)

    def on_segment(self, *, session_id: str, raw_path: str, captured_ts: float,
                   duration_s: float, language: Optional[str],
                   vad_aggressiveness: Optional[int] = None,
                   input_format: str = 'file') -> Optional[TranscriptChunk]:
        with self._global_lock:
            lock = self._locks.setdefault(session_id, threading.Lock())
        with lock:
            seq = self._seq.get(session_id, -1) + 1
            self._seq[session_id] = seq
        wav_path = f"tmp/sessions/{session_id}/blobs/{seq:012d}.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        if input_format == 'wav16k':
            src = os.path.abspath(raw_path)
            dst = os.path.abspath(wav_path)
            if src != dst:
                shutil.copyfile(src, dst)
            logger.debug("Session %s: accepted PCM segment seq=%s without re-encoding", session_id, seq)
        else:
            logger.debug("Session %s: normalizing segment seq=%s via ffmpeg", session_id, seq)
            to_mono16k_pcm(raw_path, wav_path)

        if self._silence_gate_enabled:
            aggr = self._resolve_aggressiveness(vad_aggressiveness)
            gate_config = config_from_env(aggr)
            gate_result: SilenceGateResult = evaluate_silence(
                wav_path,
                aggressiveness=aggr,
                config=gate_config,
            )
            decision_message = (
                f"Session {session_id}: silence gate {{status}} seq={seq} "
                f"ratio={gate_result.speech_ratio:.3f} rms={gate_result.avg_rms:.1f} "
                f"frames={gate_result.frame_count} speech_frames={gate_result.speech_frames} "
                f"reason={gate_result.reason} aggr={gate_result.aggressiveness} "
                f"min_ratio={gate_config.min_speech_ratio:.3f} rms_floor={gate_config.rms_floor:.1f} "
                f"confirm={gate_config.confirm_silence_windows}"
            )
            if gate_result.is_speech:
                logger.info(decision_message.format(status="passed"))
                if gate_result.voiced_start_frame is not None and gate_result.voiced_end_frame is not None:
                    self._trim_wav_to_voiced(
                        session_id=session_id,
                        seq=seq,
                        wav_path=wav_path,
                        result=gate_result,
                    )
            else:
                logger.info(decision_message.format(status="dropped"))
                self._wal.log_silence_drop(
                    session_id=session_id,
                    seq=seq,
                    captured_ts=captured_ts,
                    aggressiveness=gate_result.aggressiveness,
                    speech_ratio=gate_result.speech_ratio,
                    avg_rms=gate_result.avg_rms,
                    frame_count=gate_result.frame_count,
                    speech_frames=gate_result.speech_frames,
                    reason=gate_result.reason,
                    min_speech_ratio=gate_config.min_speech_ratio,
                    rms_floor=gate_config.rms_floor,
                    confirm_windows=gate_config.confirm_silence_windows,
                    local_wav_path=wav_path,
                )
                return None

        blob = AudioBlob(session_id=session_id, seq=seq, captured_ts=captured_ts,
                         wav_path=wav_path, duration_s=duration_s)
        chunk = self.pipeline.process_blob(blob, language=language)
        if chunk:
            with lock:
                self._append_legacy_transcript(session_id, chunk)
        else:
            logger.debug("Session %s: chunk seq=%s dropped (pipeline returned None)", session_id, seq)
        return chunk

    def on_finalize(self, *, session_id: str) -> None:
        self._wal.publish_manifest(session_id=session_id)
        with self._global_lock:
            self._seq.pop(session_id, None)
            self._locks.pop(session_id, None)
            self._legacy_keys.pop(session_id, None)
            self._legacy_prefix_counts.pop(session_id, None)
            self._legacy_timezones.pop(session_id, None)
        logger.debug("Session %s: finalized", session_id)

    def _append_legacy_transcript(self, session_id: str, chunk: TranscriptChunk) -> None:
        key = self._legacy_keys.get(session_id)
        if not key:
            logger.debug("Session %s: no legacy key; skipping chunk seq=%s", session_id, chunk.seq)
            return
        advisory = {}
        if isinstance(chunk.meta, dict):
            advisory = chunk.meta.get("advisory") or {}
        if advisory.get("is_near_dup"):
            logger.debug("Session %s: skipping near-duplicate chunk seq=%s for legacy transcript", session_id, chunk.seq)
            return

        clean_text = self._strip_repeated_prefix(session_id, chunk.text)
        if not clean_text:
            logger.debug("Session %s: chunk seq=%s reduced to empty after prefix strip", session_id, chunk.seq)
            return

        timestamp = self._format_timestamp(session_id, chunk.captured_ts)
        line = format_transcript_line(timestamp, clean_text, chunk.meta)
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            existing = obj["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
            existing = ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        updated = f"{existing}{line}\n"
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=updated.encode("utf-8"))
        logger.debug("Session %s: appended chunk seq=%s to legacy transcript (len=%s)", session_id, chunk.seq, len(updated))

    def _format_timestamp(self, session_id: str, captured_ts: float) -> str:
        """Return a [HH:MM:SS ZZZ] timestamp string in the session's timezone."""
        tz_name = self._legacy_timezones.get(session_id) or "UTC"
        try:
            tzinfo = ZoneInfo(tz_name)
        except Exception:
            logger.debug("Session %s: invalid timezone '%s', falling back to UTC", session_id, tz_name)
            tzinfo = timezone.utc
            tz_name = "UTC"

        utc_dt = datetime.fromtimestamp(captured_ts, tz=timezone.utc)
        local_dt = utc_dt.astimezone(tzinfo)
        tz_abbr = local_dt.tzname() or tz_name or "UTC"
        return f"[{local_dt.strftime('%H:%M:%S')} {tz_abbr}]"

    def _strip_repeated_prefix(self, session_id: str, text: str) -> str:
        stripped = text.lstrip()
        tokens, spans = self._tokenize_with_spans(stripped)
        if not tokens:
            return stripped

        prefix_len = min(len(tokens), 2) if tokens else 0
        if prefix_len == 0:
            return stripped
        prefix_tokens = tokens[:prefix_len]
        prefix_key = " ".join(prefix_tokens)

        counts = self._legacy_prefix_counts.setdefault(session_id, {})
        seen_before = counts.get(prefix_key, 0)
        counts[prefix_key] = seen_before + 1

        logger.debug(
            "Session %s: prefix '%s' tokens=%s seen=%s text='%s'",
            session_id,
            prefix_key,
            prefix_tokens,
            seen_before,
            stripped[:60],
        )

        if seen_before == 0:
            return stripped

        if len(tokens) <= prefix_len:
            logger.debug("Session %s: prefix removal would yield empty for text '%s'", session_id, stripped)
            return ""

        cut = spans[prefix_len - 1][1]
        remainder = stripped[cut:].lstrip(" ,.!?-–—")
        logger.debug("Session %s: trimmed prefix -> '%s'", session_id, remainder[:60])
        return remainder or ""

    @staticmethod
    def _tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for match in _TOKEN_PATTERN.finditer(text):
            start, end = match.span()
            tokens.append(match.group().lower())
            spans.append((start, end))
        return tokens, spans
