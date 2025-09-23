from __future__ import annotations
import os, requests
from typing import Any, Dict, List, Optional
from .base import TranscriptionProvider, TranscriptionResult, Segment

class DeepgramProvider(TranscriptionProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "nova-2-general", smart_format: bool = True, punctuate: bool = True):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key: raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.model = model
        self.smart_format = smart_format
        self.punctuate = punctuate
        self._endpoint = "https://api.deepgram.com/v1/listen"

    def _request(self, path: str, language: Optional[str]) -> Dict[str, Any]:
        params = {"model": self.model, "smart_format": str(self.smart_format).lower(), "punctuate": str(self.punctuate).lower()}
        if language: params["language"] = language
        headers = {"Authorization": f"Token {self.api_key}"}
        with open(path, "rb") as f:
            r = requests.post(self._endpoint, headers=headers, params=params, data=f)
        if r.status_code >= 300: raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _extract(self, data: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
        try:
            alt0 = data["results"]["channels"][0]["alternatives"][0]
            return (alt0.get("transcript","").strip(), alt0.get("words") or [])
        except Exception:
            return ("", [])

    def _words_to_segments(self, words: List[Dict[str, Any]], gap_s: float = 0.6) -> List[Segment]:
        segs: List[Segment] = []
        if not words: return segs
        cur = {"start": float(words[0]["start"]), "end": float(words[0]["end"]), "text": [words[0]["word"]]}
        for prev, w in zip(words, words[1:]):
            w_start, w_end = float(w["start"]), float(w["end"])
            if (w_start - float(prev["end"])) > gap_s:
                segs.append({"start": cur["start"], "end": cur["end"], "text": " ".join(cur["text"]).strip(),
                             "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0})
                cur = {"start": w_start, "end": w_end, "text": [w["word"]]}
            else:
                cur["end"] = w_end
                cur["text"].append(w["word"])
        segs.append({"start": cur["start"], "end": cur["end"], "text": " ".join(cur["text"]).strip(),
                     "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0})
        return segs

    def transcribe_file(self, path: str, language: Optional[str] = None, prompt: Optional[str] = None) -> Optional[TranscriptionResult]:
        data = self._request(path, language)
        text, words = self._extract(data)
        return {"text": text, "segments": self._words_to_segments(words)}