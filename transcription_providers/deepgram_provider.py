from __future__ import annotations
import os, requests, logging
from typing import Any, Dict, List, Optional
from .base import TranscriptionProvider, TranscriptionResult, Segment

logger = logging.getLogger(__name__)

class DeepgramProvider(TranscriptionProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3", smart_format: bool = True, punctuate: bool = True):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            logger.error("DEEPGRAM_API_KEY is not set")
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self.model = model
        self.smart_format = smart_format
        self.punctuate = punctuate
        self._endpoint = "https://api.deepgram.com/v1/listen"
        logger.info(f"DeepgramProvider initialized with model={model} (multilingual capable), api_key={'*' * 10}{'...' if len(self.api_key) > 10 else ''}")

    def _request(self, path: str, language: Optional[str]) -> Dict[str, Any]:
        # Deepgram parameters - enable word timestamps and smart formatting
        params = {
            "model": self.model,
            "smart_format": str(self.smart_format).lower(),
            "punctuate": str(self.punctuate).lower(),
            "words": "true",  # Enable word-level timestamps
            "timestamps": "true"  # Enable timestamps
        }
        # Handle language parameter for nova-3 multilingual support
        if language and language != "any":
            params["language"] = language
        else:
            # For nova-3, enable multilingual detection when language is "any"
            params["detect_language"] = "true"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"  # Specify content type
        }

        logger.info(f"Deepgram request: file={path}, params={params}")

        try:
            with open(path, "rb") as f:
                r = requests.post(self._endpoint, headers=headers, params=params, data=f)

            logger.info(f"Deepgram response: status={r.status_code}")

            if r.status_code >= 300:
                logger.error(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")
                raise RuntimeError(f"Deepgram HTTP {r.status_code}: {r.text[:500]}")

            response_data = r.json()
            logger.info(f"Deepgram response data keys: {list(response_data.keys())}")
            return response_data
        except Exception as e:
            logger.error(f"Deepgram request failed: {e}")
            raise

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
        try:
            logger.info(f"Deepgram transcribe_file called: path={path}, language={language}")
            data = self._request(path, language)
            text, words = self._extract(data)
            segments = self._words_to_segments(words)

            logger.info(f"Deepgram transcription result: text_length={len(text)}, segments_count={len(segments)}")
            if text:
                logger.info(f"Deepgram transcript preview: {text[:100]}...")

            result = {"text": text, "segments": segments}
            return result
        except Exception as e:
            logger.error(f"Deepgram transcribe_file failed: {e}")
            return None