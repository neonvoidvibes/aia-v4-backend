from __future__ import annotations
from typing import Optional, Dict, Any, List
from core.types import ASRResult, ASRSegment
from providers.base import ASRProvider

class WhisperProvider(ASRProvider):
    name = "whisper"

    def __init__(self, client) -> None:
        self._client = client

    def transcribe_file(self, wav_path: str, language: Optional[str]) -> ASRResult:
        try:
            resp: Dict[str, Any] = self._client.transcribe_file(
                wav_path=wav_path,
                language=language or "sv",
                model="whisper-1",
                temperature=0.0,
            )
        except Exception:
            return ASRResult(ok=False, segments=[], raw_text="", provider=self.name,
                             meta={}, error="provider_error")

        text = (resp.get("text") or "").strip()
        if not text:
            return ASRResult(ok=False, segments=[], raw_text="", provider=self.name,
                             meta=resp, error="asr_empty")
        seg = ASRSegment(text=text, start_s=0.0, end_s=max(0.0, float(resp.get("duration", 0.0))))
        return ASRResult(ok=True, segments=[seg], raw_text=text, provider=self.name, meta=resp)
