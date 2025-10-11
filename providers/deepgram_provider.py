from __future__ import annotations
from typing import Optional, Dict, Any, List
from core.types import ASRResult, ASRSegment
from providers.base import ASRProvider

# NOTE: This stub assumes an existing deepgram client wrapper exists elsewhere.
# Replace _dg_call(...) with your actual SDK call; keep the output mapping here.

def _as_segments(dg_json: Dict[str, Any]) -> List[ASRSegment]:
    segs: List[ASRSegment] = []
    for ch in dg_json.get("results", {}).get("channels", []):
        for alt in ch.get("alternatives", []):
            words = alt.get("words", [])
            if words:
                start = words[0].get("start", 0.0)
                end = words[-1].get("end", 0.0)
            else:
                start = 0.0
                end = 0.0
            segs.append(ASRSegment(text=alt.get("transcript", "").strip(),
                                   start_s=float(start), end_s=float(end)))
    return segs

class DeepgramProvider(ASRProvider):
    name = "deepgram"

    def __init__(self, client) -> None:
        self._client = client

    def transcribe_file(self, wav_path: str, language: Optional[str]) -> ASRResult:
        try:
            # Replace with your SDK call. Return same schema.
            resp: Dict[str, Any] = self._client.transcribe_file(
                wav_path=wav_path,
                language=language,
                model="nova-3",
                smart_format=True,
                words=True,
                vad_mode=1,
                punctuate=True,
                timestamps=True,
            )
        except Exception:
            return ASRResult(ok=False, segments=[], raw_text="", provider=self.name,
                             meta={}, error="provider_error")

        segs = _as_segments(resp)
        raw_text = " ".join([s.text for s in segs]).strip()
        if not raw_text:
            return ASRResult(ok=False, segments=[], raw_text="", provider=self.name,
                             meta=resp, error="asr_empty")
        return ASRResult(ok=True, segments=segs, raw_text=raw_text, provider=self.name, meta=resp)
