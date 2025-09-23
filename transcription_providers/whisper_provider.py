from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from .base import TranscriptionProvider, TranscriptionResult, Segment

_USE_INTERNAL_HELPER = False
try:
    from services.transcription_helpers import transcribe_with_openai as _openai_helper  # pragma: no cover
    _USE_INTERNAL_HELPER = True
except Exception:
    pass

class WhisperProvider(TranscriptionProvider):
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-transcribe"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if not _USE_INTERNAL_HELPER:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)

    def _normalize_segments(self, segments: List[Dict[str, Any]]) -> List[Segment]:
        out: List[Segment] = []
        for s in segments or []:
            start = float(s.get("start", s.get("start_time", 0.0)))
            end = float(s.get("end", s.get("end_time", start)))
            text = (s.get("text") or s.get("transcript") or "").strip()
            out.append({
                "start": start, "end": end, "text": text,
                "avg_logprob": float(s.get("avg_logprob", 0.0)),
                "compression_ratio": float(s.get("compression_ratio", 0.0)),
                "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
            })
        return out

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None) -> Optional[TranscriptionResult]:
        if _USE_INTERNAL_HELPER:
            data = _openai_helper(path=path, language=language, prompt=prompt)
            if not data: return None
            return {"text": (data.get("text") or "").strip(), "segments": self._normalize_segments(data.get("segments") or [])}

        with open(path, "rb") as f:
            resp = self._client.audio.transcriptions.create(  # type: ignore
                file=f, model=self.model, language=language, prompt=prompt, response_format="verbose_json", temperature=0
            )
        text = (getattr(resp, "text", None) or getattr(resp, "get", lambda *_: None)("text") or "").strip()  # type: ignore
        segments = getattr(resp, "segments", None) or (isinstance(resp, dict) and resp.get("segments") or [])
        return {"text": text, "segments": self._normalize_segments(segments or [])}