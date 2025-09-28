from __future__ import annotations
import os
from typing import Optional, Dict, Any
import httpx

class DeepgramSDK:
    """Minimal Deepgram REST client returning raw JSON responses."""

    def __init__(self, api_key: Optional[str] = None, endpoint: str = "https://api.deepgram.com/v1/listen") -> None:
        self._api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self._endpoint = endpoint

    def transcribe_file(self, wav_path: str, language: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        params.update({k: v for k, v in kwargs.items() if v is not None})
        if language and language != "any":
            params.setdefault("language", language)
        else:
            params.setdefault("detect_language", "true")

        for key, value in list(params.items()):
            if isinstance(value, bool):
                params[key] = str(value).lower()

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }

        timeout = float(os.getenv("DEEPGRAM_TIMEOUT_MS", "6000")) / 1000.0
        with open(wav_path, "rb") as fh, httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            response = client.post(self._endpoint, headers=headers, params=params, data=fh)
            response.raise_for_status()
            return response.json()
