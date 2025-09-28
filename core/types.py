from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any
import time

ASRError = Literal["provider_error", "asr_empty", "transport"]

@dataclass(frozen=True)
class AudioBlob:
    session_id: str
    seq: int                  # monotonic from capture
    captured_ts: float        # client capture clock (epoch seconds)
    wav_path: str             # normalized mono/16k WAV on disk
    duration_s: float

@dataclass(frozen=True)
class ASRSegment:
    text: str
    start_s: float            # relative to blob start
    end_s: float

@dataclass(frozen=True)
class ASRResult:
    ok: bool
    segments: List[ASRSegment]
    raw_text: str
    provider: str
    meta: Dict[str, Any]
    error: Optional[ASRError] = None

@dataclass(frozen=True)
class TranscriptChunk:
    session_id: str
    seq: int                  # copy of AudioBlob.seq (ordering key)
    captured_ts: float
    text: str                 # final text to append (never empty if ASR had text)
    provider: str
    meta: Dict[str, Any]
    byte_len: int
    created_ts: float = time.time()

@dataclass(frozen=True)
class AdvisoryFlags:
    is_near_dup: bool = False
    notes: Optional[str] = None
