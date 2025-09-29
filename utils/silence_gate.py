from __future__ import annotations

import array
import contextlib
import logging
import math
import os
import wave
from dataclasses import dataclass
from typing import Optional

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - fallback when dependency missing
    webrtcvad = None

logger = logging.getLogger(__name__)

_FRAME_MS_DEFAULT = 30


@dataclass(frozen=True)
class SilenceGateConfig:
    """Configuration for the silence gate evaluation."""

    aggressiveness: int
    min_speech_ratio: float
    rms_floor: float
    confirm_silence_windows: int


@dataclass(frozen=True)
class SilenceGateResult:
    """Result of running the silence gate on a WAV segment."""

    is_speech: bool
    speech_ratio: float
    avg_rms: float
    frame_count: int
    speech_frames: int
    aggressiveness: int
    reason: str
    frame_ms: int = _FRAME_MS_DEFAULT
    sample_rate: int = 16000
    frame_bytes: int = 0
    voiced_start_frame: Optional[int] = None
    voiced_end_frame: Optional[int] = None


def _sanitize_aggressiveness(level: Optional[int]) -> int:
    try:
        value = int(level) if level is not None else 2
    except (TypeError, ValueError):
        value = 2
    return max(0, min(3, value))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def config_from_env(aggressiveness: Optional[int] = None) -> SilenceGateConfig:
    """Build a gate configuration based on environment knobs."""

    level = _sanitize_aggressiveness(aggressiveness)
    base_ratio = _env_float("SILENCE_GATE_MIN_RATIO", 0.2)
    ratio_step = _env_float("SILENCE_GATE_MIN_RATIO_STEP", 0.05)
    min_ratio = min(0.95, max(0.0, base_ratio + ratio_step * max(0, level - 1)))

    base_rms = _env_float("SILENCE_GATE_RMS_FLOOR", 180.0)
    rms_step = _env_float("SILENCE_GATE_RMS_FLOOR_STEP", 40.0)
    rms_floor = max(0.0, base_rms + rms_step * max(0, level - 1))

    confirm_windows = max(1, _env_int("SILENCE_GATE_CONFIRM_WINDOWS", 2))
    if os.getenv("SILENCE_GATE_STRICT_MODE", "false").lower() in {"1", "true", "yes"}:
        confirm_windows = max(1, confirm_windows - 1)

    return SilenceGateConfig(
        aggressiveness=level,
        min_speech_ratio=min_ratio,
        rms_floor=rms_floor,
        confirm_silence_windows=confirm_windows,
    )


def _frame_rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    pcm = array.array("h")
    pcm.frombytes(frame)
    if pcm.typecode != "h":  # pragma: no cover - defensive
        pcm = array.array("h", pcm)
    if not pcm:
        return 0.0
    squares = (sample * sample for sample in pcm)
    mean_square = sum(squares) / len(pcm)
    return math.sqrt(mean_square)


def _generate_frames(raw: bytes, sample_rate: int, frame_ms: int) -> list[bytes]:
    bytes_per_sample = 2
    frame_size = int(sample_rate * (frame_ms / 1000.0)) * bytes_per_sample
    if frame_size <= 0:
        return []
    total = len(raw)
    frames = []
    for offset in range(0, total - frame_size + 1, frame_size):
        frames.append(raw[offset: offset + frame_size])
    return frames


def evaluate_silence(
    wav_path: str,
    *,
    aggressiveness: Optional[int] = None,
    config: Optional[SilenceGateConfig] = None,
    frame_ms: int = _FRAME_MS_DEFAULT,
) -> SilenceGateResult:
    """Evaluate the WAV segment and decide if it contains meaningful speech."""

    if config is None:
        config = config_from_env(aggressiveness)
    else:
        config = SilenceGateConfig(
            aggressiveness=_sanitize_aggressiveness(config.aggressiveness),
            min_speech_ratio=max(0.0, min(1.0, config.min_speech_ratio)),
            rms_floor=max(0.0, config.rms_floor),
            confirm_silence_windows=max(1, config.confirm_silence_windows),
        )

    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
    except Exception as exc:  # pragma: no cover - corrupted input
        logger.warning("Silence gate failed to read WAV '%s': %s", wav_path, exc)
        return SilenceGateResult(
            is_speech=True,
            speech_ratio=1.0,
            avg_rms=0.0,
            frame_count=0,
            speech_frames=0,
            aggressiveness=config.aggressiveness,
            reason="read_error",
            sample_rate=16000,
            frame_bytes=0,
        )

    if channels != 1 or sample_width != 2:
        # Gate expects mono 16-bit PCM; fail open for safety.
        total_rms = _frame_rms(raw)
        return SilenceGateResult(
            is_speech=True,
            speech_ratio=1.0,
            avg_rms=total_rms,
            frame_count=0,
            speech_frames=0,
            aggressiveness=config.aggressiveness,
            reason="unsupported_format",
            sample_rate=sample_rate,
            frame_bytes=0,
        )

    frames = _generate_frames(raw, sample_rate, frame_ms)
    frame_count = len(frames)
    frame_size = len(frames[0]) if frames else int(sample_rate * (frame_ms / 1000.0)) * 2
    if frame_count == 0:
        overall_rms = _frame_rms(raw)
        return SilenceGateResult(
            is_speech=True,
            speech_ratio=1.0,
            avg_rms=overall_rms,
            frame_count=0,
            speech_frames=0,
            aggressiveness=config.aggressiveness,
            reason="insufficient_audio",
            sample_rate=sample_rate,
            frame_bytes=0,
        )

    vad_available = webrtcvad is not None
    speech_frames = 0
    max_consecutive_silence = 0
    consecutive_silence = 0
    rms_total = 0.0

    vad_decisions: list[bool] = []
    if vad_available:
        vad = webrtcvad.Vad(config.aggressiveness)
    else:
        vad = None

    for frame in frames:
        frame_rms = _frame_rms(frame)
        rms_total += frame_rms
        is_speech_frame = False
        if vad is not None:
            try:
                is_speech_frame = vad.is_speech(frame, sample_rate)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("VAD error for frame (sample_rate=%s): %s", sample_rate, exc)
                is_speech_frame = False
        else:
            # Fallback: treat high-RMS frames as speech when VAD unavailable.
            is_speech_frame = frame_rms >= config.rms_floor

        vad_decisions.append(is_speech_frame)
        if is_speech_frame:
            speech_frames += 1
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            max_consecutive_silence = max(max_consecutive_silence, consecutive_silence)

    speech_ratio = speech_frames / frame_count
    avg_rms = rms_total / frame_count

    voiced_start = next((i for i, flag in enumerate(vad_decisions) if flag), None)
    voiced_end = (
        (frame_count - 1 - next((i for i, flag in enumerate(reversed(vad_decisions)) if flag), None))
        if speech_frames > 0 else None
    )

    if speech_frames == 0:
        return SilenceGateResult(
            is_speech=False,
            speech_ratio=0.0,
            avg_rms=avg_rms,
            frame_count=frame_count,
            speech_frames=0,
            aggressiveness=config.aggressiveness,
            reason="no_voiced_frames",
            frame_ms=frame_ms,
            sample_rate=sample_rate,
            frame_bytes=frame_size,
        )

    if speech_ratio < config.min_speech_ratio and avg_rms < config.rms_floor:
        final_reason = "vad_unavailable" if not vad_available else "below_thresholds"
        return SilenceGateResult(
            is_speech=False,
            speech_ratio=speech_ratio,
            avg_rms=avg_rms,
            frame_count=frame_count,
            speech_frames=speech_frames,
            aggressiveness=config.aggressiveness,
            reason=final_reason,
            frame_ms=frame_ms,
            sample_rate=sample_rate,
            frame_bytes=frame_size,
        )

    pass_reason = "speech_ratio" if speech_ratio >= config.min_speech_ratio else "rms_floor"
    if max_consecutive_silence < config.confirm_silence_windows and pass_reason != "speech_ratio":
        pass_reason = "hysteresis"

    return SilenceGateResult(
        is_speech=True,
        speech_ratio=speech_ratio,
        avg_rms=avg_rms,
        frame_count=frame_count,
        speech_frames=speech_frames,
        aggressiveness=config.aggressiveness,
        reason=pass_reason,
        frame_ms=frame_ms,
        sample_rate=sample_rate,
        frame_bytes=frame_size,
        voiced_start_frame=voiced_start,
        voiced_end_frame=voiced_end,
    )


__all__ = [
    "SilenceGateConfig",
    "SilenceGateResult",
    "config_from_env",
    "evaluate_silence",
]
