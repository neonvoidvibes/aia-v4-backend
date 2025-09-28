from __future__ import annotations
import os
from typing import Optional, Dict, Any
from openai import OpenAI

class OpenAIWhisper:
    """Thin wrapper around OpenAI Whisper transcription."""

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1") -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._model = model
        self._client = OpenAI(api_key=self._api_key)

    def transcribe_file(self, wav_path: str, language: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        openai_language = None if not language or language == "any" else language
        with open(wav_path, "rb") as fh:
            resp = self._client.audio.transcriptions.create(
                file=fh,
                model=self._model,
                language=openai_language,
                temperature=kwargs.get("temperature", 0.0),
                response_format="verbose_json",
            )
        text = (getattr(resp, "text", None) or "").strip()
        duration = float(getattr(resp, "duration", kwargs.get("duration", 0.0)) or 0.0)
        segments = getattr(resp, "segments", None)
        if segments is None:
            normalized_segments = []
        else:
            normalized_segments = []
            for seg in segments:
                if hasattr(seg, "dict"):
                    normalized_segments.append(seg.dict())
                elif isinstance(seg, dict):
                    normalized_segments.append(seg)
        return {"text": text, "duration": duration, "segments": normalized_segments}
