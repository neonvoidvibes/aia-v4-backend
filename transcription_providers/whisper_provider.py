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
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "whisper-1"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if not _USE_INTERNAL_HELPER:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)

    def _normalize_segments(self, segments: List[Any]) -> List[Segment]:
        out: List[Segment] = []
        for s in segments or []:
            # Handle both dict objects (internal helper) and Pydantic objects (OpenAI API)
            if hasattr(s, 'get'):  # Dictionary-like object
                start = float(s.get("start", s.get("start_time", 0.0)))
                end = float(s.get("end", s.get("end_time", start)))
                text = (s.get("text") or s.get("transcript") or "").strip()
                avg_logprob = float(s.get("avg_logprob", 0.0))
                compression_ratio = float(s.get("compression_ratio", 0.0))
                no_speech_prob = float(s.get("no_speech_prob", 0.0))
            else:  # Pydantic object (OpenAI API response)
                start = float(getattr(s, "start", getattr(s, "start_time", 0.0)))
                end = float(getattr(s, "end", getattr(s, "end_time", start)))
                text = (getattr(s, "text", None) or getattr(s, "transcript", None) or "").strip()
                avg_logprob = float(getattr(s, "avg_logprob", 0.0))
                compression_ratio = float(getattr(s, "compression_ratio", 0.0))
                no_speech_prob = float(getattr(s, "no_speech_prob", 0.0))

            out.append({
                "start": start, "end": end, "text": text,
                "avg_logprob": avg_logprob,
                "compression_ratio": compression_ratio,
                "no_speech_prob": no_speech_prob,
            })
        return out

    def _align_words_from_segments(self, segments: List[Segment]) -> List[Dict[str, Any]]:
        """Provide 'words' with start/end/conf so the detector can run consistently."""
        words = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue

            # Simple word splitting with approximate timing
            words_in_seg = text.split()
            if not words_in_seg:
                continue

            seg_duration = seg.get('end', 0) - seg.get('start', 0)
            word_duration = seg_duration / len(words_in_seg) if len(words_in_seg) > 0 else 0

            for i, word in enumerate(words_in_seg):
                word_start = seg.get('start', 0) + (i * word_duration)
                word_end = word_start + word_duration
                words.append({
                    "text": word,
                    "start": word_start,
                    "end": word_end,
                    "confidence": 1.0  # Whisper doesn't provide word-level confidence
                })

        return words

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None, vad_aggressiveness: Optional[int] = None) -> Optional[TranscriptionResult]:
        # Handle language parameter: OpenAI Whisper uses None for auto-detect, not 'any'
        openai_language = None if not language or language == "any" else language

        if _USE_INTERNAL_HELPER:
            data = _openai_helper(path=path, language=openai_language, prompt=prompt)
            if not data: return None
            segments = self._normalize_segments(data.get("segments") or [])
            words = self._align_words_from_segments(segments)
            return {"text": (data.get("text") or "").strip(), "segments": segments, "words": words}

        with open(path, "rb") as f:
            resp = self._client.audio.transcriptions.create(  # type: ignore
                file=f, model=self.model, language=openai_language, prompt=prompt, response_format="verbose_json", temperature=0
            )
        text = (getattr(resp, "text", None) or getattr(resp, "get", lambda *_: None)("text") or "").strip()  # type: ignore
        segments = getattr(resp, "segments", None) or (isinstance(resp, dict) and resp.get("segments") or [])
        normalized_segments = self._normalize_segments(segments or [])
        words = self._align_words_from_segments(normalized_segments)
        return {"text": text, "segments": normalized_segments, "words": words}