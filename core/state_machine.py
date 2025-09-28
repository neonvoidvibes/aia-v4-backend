from __future__ import annotations
from typing import Optional, Callable
from core.types import AudioBlob, ASRResult, TranscriptChunk
from providers.base import ASRProvider
from core.advisory import NearDupCache, advisory_flags

class TranscriptionPipeline:
    """
    Minimal, append-only pipeline:
      ingest blob -> ASR -> normalize -> append chunk (never empty if ASR non-empty)
      Ordering by blob.seq. Filters are advisory; they never delete content.
    """
    def __init__(self, provider_primary: ASRProvider, provider_fallback: Optional[ASRProvider],
                 wal_append: Callable[[TranscriptChunk], str],
                 on_chunk_committed: Optional[Callable[[TranscriptChunk, str], None]] = None) -> None:
        self._p1 = provider_primary
        self._p2 = provider_fallback
        self._wal_append = wal_append
        self._cb = on_chunk_committed
        self._dups = NearDupCache(max_items=64)

    @staticmethod
    def _normalize(text: str) -> str:
        # Non-destructive: trim outer whitespace only
        return text.strip()

    def _call_provider(self, wav: str, lang: Optional[str]) -> ASRResult:
        r = self._p1.transcribe_file(wav, lang)
        if r.ok:
            return r
        if (not r.ok) and r.error in ("asr_empty", "provider_error") and self._p2:
            r2 = self._p2.transcribe_file(wav, lang)
            if r2.ok:
                return r2
        return r

    def process_blob(self, blob: AudioBlob, language: Optional[str]) -> Optional[TranscriptChunk]:
        asr = self._call_provider(blob.wav_path, language)
        # Legal drops: true silence or provider failure
        if not asr.ok and asr.error == "asr_empty":
            return None
        if not asr.ok and asr.error == "provider_error":
            return None

        text = self._normalize(asr.raw_text)
        if not text:
            # ASR produced only whitespace -> treat as silence
            return None

        flags = advisory_flags(asr, blob.seq, self._dups)
        # Near-dup? We still append (log is authoritative). View layers may choose to hide dup lines.
        chunk = TranscriptChunk(
            session_id=blob.session_id,
            seq=blob.seq,
            captured_ts=blob.captured_ts,
            text=text,
            provider=asr.provider,
            meta={"advisory": flags.__dict__, "segments": len(asr.segments)},
            byte_len=len(text.encode("utf-8")),
        )
        key = self._wal_append(chunk)
        if self._cb:
            self._cb(chunk, key)
        return chunk

