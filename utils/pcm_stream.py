"""PCM streaming helpers shared between the WebSocket handler and tests."""

from __future__ import annotations

import struct
from typing import Any, Callable, Dict, Optional

PCM_FRAME_MAGIC = 0x314D4350  # 'PCM1' little-endian
PCM_FRAME_HEADER_BYTES = 32


DispatchFn = Callable[[str, Dict[str, Any], bytes, float, int, int], None]


def handle_pcm_frame(
    session_id: str,
    session_data: Dict[str, Any],
    frame_bytes: bytes,
    dispatch_fn: DispatchFn,
    *,
    logger: Optional[Any] = None,
    default_segment_target_ms: int = 3000,
) -> bool:
    """Parse an incoming PCM frame and dispatch accumulated segments.

    Returns True when the frame is recognised as PCM (even if dropped) so callers
    can skip WebM handling, or False when the payload should fall through to the
    legacy pipeline.
    """

    log = logger if logger is not None else _NoopLogger()

    if len(frame_bytes) < PCM_FRAME_HEADER_BYTES:
        log.warning(
            "Session %s: Received undersized PCM frame (%s bytes)",
            session_id,
            len(frame_bytes),
        )
        stats = session_data.setdefault(
            "pcm_frame_stats", {"frames_received": 0, "frames_dropped": 0, "out_of_order": 0}
        )
        stats["frames_dropped"] = stats.get("frames_dropped", 0) + 1
        return True

    try:
        (
            magic,
            seq,
            timestamp_ms,
            frame_samples,
            frame_duration_ms,
            sample_rate,
            channels,
            format_code,
            payload_length,
        ) = struct.unpack("<IIdHHIHHI", frame_bytes[:PCM_FRAME_HEADER_BYTES])
    except struct.error as exc:
        log.warning("Session %s: Failed to unpack PCM frame header: %s", session_id, exc)
        return False

    if magic != PCM_FRAME_MAGIC:
        return False

    stats = session_data.setdefault(
        "pcm_frame_stats", {"frames_received": 0, "frames_dropped": 0, "out_of_order": 0}
    )
    stats["frames_received"] = stats.get("frames_received", 0) + 1

    available_payload = len(frame_bytes) - PCM_FRAME_HEADER_BYTES
    if payload_length <= 0 or payload_length > available_payload:
        log.warning(
            "Session %s: PCM frame payload truncated (expected %s bytes, have %s)",
            session_id,
            payload_length,
            available_payload,
        )
        stats["frames_dropped"] = stats.get("frames_dropped", 0) + 1
        return True

    if format_code != 1:
        log.warning(
            "Session %s: Unsupported PCM format code %s, expected 1 (pcm16)",
            session_id,
            format_code,
        )
        stats["frames_dropped"] = stats.get("frames_dropped", 0) + 1
        return True

    payload = frame_bytes[PCM_FRAME_HEADER_BYTES : PCM_FRAME_HEADER_BYTES + payload_length]
    last_seq = session_data.get("pcm_last_seq", 0)
    expected_seq = last_seq + 1
    if seq != expected_seq:
        stats["out_of_order"] = stats.get("out_of_order", 0) + 1
        log.warning(
            "Session %s: PCM frame out of order (expected %s, got %s)",
            session_id,
            expected_seq,
            seq,
        )
    session_data["pcm_last_seq"] = seq
    session_data["pcm_last_frame_ts"] = timestamp_ms

    buffer = session_data.setdefault("pcm_frame_buffer", bytearray())
    buffer.extend(payload)

    effective_frame_samples = frame_samples or (payload_length // 2)
    if frame_duration_ms <= 0 and sample_rate:
        frame_duration_ms = int((effective_frame_samples / sample_rate) * 1000)

    session_data["pcm_accumulated_duration_ms"] = session_data.get("pcm_accumulated_duration_ms", 0.0) + max(
        frame_duration_ms, 0
    )
    session_data["pcm_samples_buffered"] = session_data.get("pcm_samples_buffered", 0) + effective_frame_samples
    session_data["pcm_sample_rate"] = sample_rate or session_data.get("pcm_sample_rate", 16000)
    session_data["pcm_channels"] = channels or session_data.get("pcm_channels", 1)

    target_ms = session_data.get("pcm_segment_target_ms", default_segment_target_ms)
    if session_data["pcm_accumulated_duration_ms"] >= max(target_ms, frame_duration_ms or 1):
        pcm_bytes = bytes(buffer)
        if pcm_bytes:
            buffer.clear()
            total_samples = session_data.get("pcm_samples_buffered", len(pcm_bytes) // 2)
            session_data["pcm_samples_buffered"] = 0
            session_data["pcm_accumulated_duration_ms"] = 0.0
            sample_rate = session_data.get("pcm_sample_rate", 16000)
            channels = session_data.get("pcm_channels", 1)
            try:
                duration_s = total_samples / sample_rate if sample_rate else len(pcm_bytes) / (16000 * 2)
            except Exception:
                duration_s = len(pcm_bytes) / (16000 * 2)
            dispatch_fn(session_id, session_data, pcm_bytes, duration_s, sample_rate, channels)
    return True


class _NoopLogger:
    def warning(self, *args, **kwargs):  # pragma: no cover - debug aid
        pass

    def info(self, *args, **kwargs):  # pragma: no cover - debug aid
        pass

    def debug(self, *args, **kwargs):  # pragma: no cover - debug aid
        pass
